//! Direct in-place dense LU, QR and Cholesky over a caller-owned native matrix.
//!
//! `lu!`, `qr!` and `cholesky!` with caller scratch run faer's low-level
//! `*_in_place` kernels on the input matrix's own storage, which then holds the
//! packed factor (Julia `F.factors === A`). The LU row permutation and the QR
//! block Householder coefficients live in Julia-owned buffers passed on every
//! call; nothing here retains a pointer across calls. Every validation that does
//! not need the numerical result happens before the first write to the matrix.
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::linalg::householder;
use faer::linalg::lu::partial_pivoting as plu;
use faer::linalg::qr::no_pivoting as nqr;
use faer::perm::PermRef;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

const LU: i32 = 0;
const QR: i32 = 2;
const LLT: i32 = 4;

/// Operation codes shared with Julia.
const FACTOR: i32 = 0;
const SOLVE: i32 = 1;
const TRANSPOSE: i32 = 2;
const ADJOINT: i32 = 3;

fn block_size<T: ComplexField>(kind: i32, m: usize, n: usize) -> usize {
    if kind == QR {
        nqr::factor::recommended_block_size::<T>(m, n)
    } else {
        0
    }
}

fn requirement<T: ComplexField>(
    kind: i32,
    op: i32,
    m: usize,
    n: usize,
    k: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    let bs = block_size::<T>(kind, m, n);
    Ok(match (kind, op) {
        (LU, FACTOR) => plu::factor::lu_in_place_scratch::<usize, T>(m, n, par, Default::default()),
        (LU, SOLVE) => plu::solve::solve_in_place_scratch::<usize, T>(n, k, par),
        (LU, TRANSPOSE | ADJOINT) => {
            plu::solve::solve_transpose_in_place_scratch::<usize, T>(n, k, par)
        }
        (QR, FACTOR) => nqr::factor::qr_in_place_scratch::<T>(m, n, bs, par, Default::default()),
        (QR, SOLVE) => nqr::solve::solve_lstsq_in_place_scratch::<T>(m, n, bs, k, par),
        (QR, TRANSPOSE | ADJOINT) => {
            nqr::solve::solve_transpose_in_place_scratch::<T>(n, bs, k, par)
        }
        (LLT, FACTOR) => llt::factor::cholesky_in_place_scratch::<T>(n, par, Default::default()),
        (LLT, SOLVE | TRANSPOSE | ADJOINT) => llt::solve::solve_in_place_scratch::<T>(n, k, par),
        _ => return Err("unknown direct factorization kind or operation".into()),
    })
}

fn requirement_tag(
    tag: i32,
    kind: i32,
    op: i32,
    m: usize,
    n: usize,
    k: usize,
    par: ParShape,
) -> Result<(StackReq, usize), String> {
    Ok(match tag {
        0 => (
            requirement::<f32>(kind, op, m, n, k, par)?,
            block_size::<f32>(kind, m, n),
        ),
        1 => (
            requirement::<f64>(kind, op, m, n, k, par)?,
            block_size::<f64>(kind, m, n),
        ),
        2 => (
            requirement::<faer::c32>(kind, op, m, n, k, par)?,
            block_size::<faer::c32>(kind, m, n),
        ),
        3 => (
            requirement::<faer::c64>(kind, op, m, n, k, par)?,
            block_size::<faer::c64>(kind, m, n),
        ),
        _ => return Err("unsupported scalar type".into()),
    })
}

/// Writes `[bytes, alignment, qr_block_size]` for one direct operation.
#[julia]
pub fn direct_factor_requirements(
    kind: i32,
    op: i32,
    m: usize,
    n: usize,
    k: usize,
    tag: i32,
    team: Option<&OwnedTeam>,
    out: &mut [usize],
) -> Result<(), String> {
    guarded(|| {
        if out.len() != 3 {
            return Err("direct factor requirements output must have length three".into());
        }
        let (req, bs) = with_team(team, |par| {
            requirement_tag(tag, kind, op, m, n, k, par.shape())
        })?;
        out[0] = req.size_bytes();
        out[1] = req.align_bytes();
        out[2] = bs;
        Ok(())
    })
}

/// Borrow the Julia-owned QR coefficient buffer (`rows x cols`, column-major).
///
/// Safety: Julia passes a live, GC-preserved buffer of exactly this shape that
/// does not alias the packed matrix.
unsafe fn coeff_mut<'a, T: ComplexField>(p: *mut u8, rows: usize, cols: usize) -> MatMut<'a, T> {
    MatMut::from_raw_parts_mut(p.cast::<T>(), rows, cols, 1, rows as isize)
}
unsafe fn coeff_ref<'a, T: ComplexField>(p: *const u8, rows: usize, cols: usize) -> MatRef<'a, T> {
    MatRef::from_raw_parts(p.cast::<T>(), rows, cols, 1, rows as isize)
}

fn check_aux<T: ComplexField>(
    kind: i32,
    m: usize,
    n: usize,
    perm: usize,
    perm_inv: usize,
    coeff: *const u8,
    coeff_rows: usize,
) -> Result<(), String> {
    match kind {
        LU if perm != m || perm_inv != m => Err("LU permutation length mismatch".into()),
        QR if coeff_rows != block_size::<T>(QR, m, n) || coeff.is_null() => {
            Err("QR coefficient buffer shape mismatch".into())
        }
        LLT if m != n => Err("Cholesky requires a square matrix".into()),
        LU | QR | LLT => Ok(()),
        _ => Err("unknown direct factorization kind".into()),
    }
}

fn triangle_finite<T: ComplexField>(a: MatRef<'_, T>, lower: bool) -> bool {
    (0..a.ncols()).all(|j| {
        let (start, end) = if lower { (j, a.nrows()) } else { (0, j + 1) };
        (start..end).all(|i| is_finite(&a[(i, j)]))
    })
}

#[allow(clippy::too_many_arguments)]
fn factor<T: ComplexField>(
    mut a: MatMut<'_, T>,
    kind: i32,
    lower: bool,
    perm: &mut [usize],
    perm_inv: &mut [usize],
    coeff: *mut u8,
    coeff_rows: usize,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<usize, String> {
    let (m, n) = a.shape();
    check_aux::<T>(kind, m, n, perm.len(), perm_inv.len(), coeff, coeff_rows)?;
    let finite = if kind == LLT {
        triangle_finite(a.as_ref(), lower)
    } else {
        a.as_ref().is_all_finite()
    };
    if !finite {
        return Err("input must be finite".into());
    }
    let req = requirement::<T>(kind, FACTOR, m, n, 0, par.shape())?;
    let stack = scratch.stack(req)?;
    // Nothing above writes the matrix; numerical failure below may leave it
    // partially factored, as Julia's LAPACK-backed `lu!`/`cholesky!` do.
    Ok(match kind {
        LU => {
            plu::factor::lu_in_place(a.as_mut(), perm, perm_inv, par, stack, Default::default());
            (0..m.min(n))
                .find(|&i| a[(i, i)] == zero::<T>())
                .map_or(0, |i| i + 1)
        }
        QR => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_mut::<T>(coeff, coeff_rows, m.min(n)) };
            nqr::factor::qr_in_place(a, c, par, stack, Default::default());
            0
        }
        _ => {
            // Reproduce the input `Llt::new` sees: the selected triangle with a
            // real diagonal, mirrored into the lower triangle, strict upper zero.
            for j in 0..n {
                a[(j, j)] = from_real::<T>(&real(&a[(j, j)]));
                for i in j + 1..n {
                    if !lower {
                        a[(i, j)] = conj(&a[(j, i)]);
                    }
                    a[(j, i)] = zero();
                }
            }
            match llt::factor::cholesky_in_place(
                a.as_mut(),
                Default::default(),
                par,
                stack,
                Default::default(),
            ) {
                Ok(_) => {
                    // The kernel may touch the strict upper triangle; `Llt::new`
                    // zeroes it after factoring as well.
                    for j in 0..n {
                        for i in j + 1..n {
                            a[(j, i)] = if lower { zero() } else { conj(&a[(i, j)]) };
                            if !lower {
                                a[(i, j)] = zero();
                            }
                        }
                    }
                    0
                }
                Err(llt::factor::LltError::NonPositivePivot { index }) => index + 1,
            }
        }
    })
}

/// The lower-oriented Cholesky factor stored in `a` (lower: as is; upper: `Uᴴ`).
fn cholesky_lower<T: ComplexField>(a: MatRef<'_, T>, lower: bool) -> Mat<T> {
    let n = a.nrows();
    Mat::from_fn(n, n, |i, j| {
        if i < j {
            zero()
        } else if lower {
            copy(&a[(i, j)])
        } else {
            conj(&a[(j, i)])
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn solve<T: ComplexField>(
    a: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    kind: i32,
    op: i32,
    lower: bool,
    perm: &[usize],
    perm_inv: &[usize],
    coeff: *const u8,
    coeff_rows: usize,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<(), String> {
    let (m, n) = a.shape();
    check_aux::<T>(kind, m, n, perm.len(), perm_inv.len(), coeff, coeff_rows)?;
    if !(SOLVE..=ADJOINT).contains(&op) {
        return Err("unknown direct solve operation".into());
    }
    let square = m == n;
    let rows_ok = match (kind, op) {
        (QR, SOLVE) => m >= n && rhs.nrows() == m,
        _ => square && rhs.nrows() == n,
    };
    if !rows_ok {
        return Err(
            "direct solve requires a square factor (or tall QR) and matching RHS rows".into(),
        );
    }
    let k = rhs.ncols();
    let req = requirement::<T>(kind, op, m, n, k, par.shape())?;
    let stack = scratch.stack(req)?;
    let conj_op = if op == ADJOINT { Conj::Yes } else { Conj::No };
    match kind {
        LU => {
            let p = PermRef::new_checked(perm, perm_inv, n);
            if op == SOLVE {
                plu::solve::solve_in_place_with_conj(a, a, p, Conj::No, rhs, par, stack);
            } else {
                plu::solve::solve_transpose_in_place_with_conj(a, a, p, conj_op, rhs, par, stack);
            }
        }
        QR => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_ref::<T>(coeff, coeff_rows, m.min(n)) };
            if op == SOLVE {
                nqr::solve::solve_lstsq_in_place_with_conj(a, c, a, Conj::No, rhs, par, stack);
            } else {
                nqr::solve::solve_transpose_in_place_with_conj(a, c, a, conj_op, rhs, par, stack);
            }
        }
        _ => {
            // A = L Lᴴ is self-adjoint; Aᵀ = conj(A) solves with conjugated L.
            let conj_l = if op == TRANSPOSE { Conj::Yes } else { Conj::No };
            if lower {
                llt::solve::solve_in_place_with_conj(a, conj_l, rhs, par, stack);
            } else {
                // Upper storage holds U = Lᴴ, so L = Uᴴ = conj(Uᵀ).
                llt::solve::solve_in_place_with_conj(
                    a.transpose(),
                    conj_l.compose(Conj::Yes),
                    rhs,
                    par,
                    stack,
                );
            }
        }
    }
    Ok(())
}

/// faer's Cholesky reconstruct/inverse fill the lower triangle only.
fn self_adjoint_from_lower<T: ComplexField>(mut a: MatMut<'_, T>) {
    for j in 0..a.ncols() {
        a[(j, j)] = from_real::<T>(&real(&a[(j, j)]));
        for i in 0..j {
            a[(i, j)] = conj(&a[(j, i)]);
        }
    }
}

fn identity<T: ComplexField>(m: usize, n: usize) -> Mat<T> {
    Mat::from_fn(m, n, |i, j| if i == j { one() } else { zero() })
}

/// Part codes follow the retained-factor convention: 0 reconstruct, 1 L,
/// 2 U, 3 Q, 4 R, 6 thin Q, 8 Householder basis, 9 QR coefficients,
/// 10 inverse, 11 diagonal.
#[allow(clippy::too_many_arguments)]
fn part<T: ComplexField>(
    a: MatRef<'_, T>,
    kind: i32,
    code: i32,
    lower: bool,
    perm: &[usize],
    perm_inv: &[usize],
    coeff: *const u8,
    coeff_rows: usize,
) -> Result<Mat<T>, String> {
    let (m, n) = a.shape();
    let size = m.min(n);
    check_aux::<T>(kind, m, n, perm.len(), perm_inv.len(), coeff, coeff_rows)?;
    let par = Par::Seq;
    let buffer = |req: StackReq| MemBuffer::try_new(req).map_err(|e| format!("{e:?}"));
    Ok(match (kind, code) {
        (LU, 1) => Mat::from_fn(m, size, |i, j| {
            if i == j {
                one()
            } else if i > j {
                copy(&a[(i, j)])
            } else {
                zero()
            }
        }),
        (LU, 2) => Mat::from_fn(
            size,
            n,
            |i, j| if i <= j { copy(&a[(i, j)]) } else { zero() },
        ),
        (LU, 0) => {
            let p = PermRef::new_checked(perm, perm_inv, m);
            let l = part::<T>(a, kind, 1, lower, perm, perm_inv, coeff, coeff_rows)?;
            let u = part::<T>(a, kind, 2, lower, perm, perm_inv, coeff, coeff_rows)?;
            let mut out = Mat::zeros(m, n);
            let mut mem = buffer(plu::reconstruct::reconstruct_scratch::<usize, T>(
                m,
                n,
                par.shape(),
            ))?;
            plu::reconstruct::reconstruct(
                out.as_mut(),
                l.as_ref(),
                u.as_ref(),
                p,
                par,
                MemStack::new(&mut mem),
            );
            out
        }
        (LU, 10) if m == n => {
            let p = PermRef::new_checked(perm, perm_inv, n);
            let mut out = Mat::zeros(n, n);
            let mut mem = buffer(plu::inverse::inverse_scratch::<usize, T>(n, par.shape()))?;
            plu::inverse::inverse(out.as_mut(), a, a, p, par, MemStack::new(&mut mem));
            out
        }
        (QR, 4) => Mat::from_fn(
            size,
            n,
            |i, j| if i <= j { copy(&a[(i, j)]) } else { zero() },
        ),
        (QR, 8) => Mat::from_fn(m, size, |i, j| {
            if i == j {
                one()
            } else if i > j {
                copy(&a[(i, j)])
            } else {
                zero()
            }
        }),
        (QR, 3 | 6 | 0) => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_ref::<T>(coeff, coeff_rows, size) };
            let mut out = match code {
                3 => identity::<T>(m, m),
                6 => identity::<T>(m, size),
                _ => {
                    let mut r = Mat::zeros(m, n);
                    r.as_mut()
                        .get_mut(..size, ..)
                        .copy_from_triangular_upper(a.get(..size, ..));
                    r
                }
            };
            let cols = out.ncols();
            let mut mem = buffer(
                householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                    m, coeff_rows, cols,
                ),
            )?;
            householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                a.get(.., ..size),
                c,
                Conj::No,
                out.as_mut(),
                par,
                MemStack::new(&mut mem),
            );
            out
        }
        // The coefficients may hold faer's infinite rank-deficiency markers,
        // which Julia-side FaerMatrix construction would reject as non-finite.
        // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
        (QR, 9) => unsafe { coeff_ref::<T>(coeff, coeff_rows, size) }.to_owned(),
        (QR, 10) if m == n => {
            let c = unsafe { coeff_ref::<T>(coeff, coeff_rows, size) };
            let mut out = Mat::zeros(n, n);
            let mut mem = buffer(nqr::inverse::inverse_scratch::<T>(
                n,
                coeff_rows,
                par.shape(),
            ))?;
            nqr::inverse::inverse(out.as_mut(), a, c, a, par, MemStack::new(&mut mem));
            out
        }
        (LU | QR | LLT, 11) => Mat::from_fn(size, 1, |i, _| copy(&a[(i, i)])),
        (LLT, 1) => cholesky_lower(a, lower),
        (LLT, 2) => cholesky_lower(a, lower).adjoint().to_owned(),
        (LLT, 0) => {
            let l = cholesky_lower(a, lower);
            let mut out = Mat::zeros(n, n);
            let mut mem = buffer(llt::reconstruct::reconstruct_scratch::<T>(n, par.shape()))?;
            llt::reconstruct::reconstruct(out.as_mut(), l.as_ref(), par, MemStack::new(&mut mem));
            self_adjoint_from_lower(out.as_mut());
            out
        }
        (LLT, 10) => {
            let l = cholesky_lower(a, lower);
            let mut out = Mat::zeros(n, n);
            let mut mem = buffer(llt::inverse::inverse_scratch::<T>(n, par.shape()))?;
            llt::inverse::inverse(out.as_mut(), l.as_ref(), par, MemStack::new(&mut mem));
            self_adjoint_from_lower(out.as_mut());
            out
        }
        _ => return Err("direct factorization has no requested part".into()),
    })
}

fn typed_shape(data: &MatrixData) -> ((usize, usize), i32) {
    match data {
        MatrixData::F32(x) => (x.shape(), 0),
        MatrixData::F64(x) => (x.shape(), 1),
        MatrixData::C32(x) => (x.shape(), 2),
        MatrixData::C64(x) => (x.shape(), 3),
    }
}
fn check_typed(data: &MatrixData, tag: i32, m: usize, n: usize) -> Result<(), String> {
    if typed_shape(data) != ((m, n), tag) {
        return Err("direct factor native scalar type/dimensions do not match".into());
    }
    Ok(())
}
fn owned_shape(a: &Mat<f64>, tag: i32, m: usize, n: usize) -> Result<(), String> {
    if tag != 1 || a.shape() != (m, n) {
        return Err("direct factor native scalar type/dimensions do not match".into());
    }
    Ok(())
}

#[julia]
impl OwnedMatrix {
    /// Factor this matrix's own storage in place; returns Julia `info`.
    #[julia]
    pub fn direct_factor(
        &mut self,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &mut [usize],
        perm_inv: &mut [usize],
        coeff: *mut u8,
        coeff_rows: usize,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            owned_shape(&self.matrix, tag, m, n)?;
            let a = &mut self.matrix;
            with_team(team, |par| {
                factor(
                    a.as_mut(),
                    kind,
                    lower,
                    perm,
                    perm_inv,
                    coeff,
                    coeff_rows,
                    scratch,
                    par,
                )
            })
        })
    }
    /// Solve in place on `rhs` with the packed factor stored in `self`.
    #[julia]
    pub fn direct_solve(
        &self,
        rhs: &mut OwnedMatrix,
        kind: i32,
        op: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &[usize],
        perm_inv: &[usize],
        coeff: *const u8,
        coeff_rows: usize,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            owned_shape(&self.matrix, tag, m, n)?;
            let a = self.matrix.as_ref();
            let b = rhs.matrix.as_mut();
            with_team(team, |par| {
                solve(
                    a, b, kind, op, lower, perm, perm_inv, coeff, coeff_rows, scratch, par,
                )
            })
        })
    }
    /// Fill `self` with a part of the packed factor stored in `packed`.
    #[julia]
    pub fn direct_part(
        &mut self,
        packed: &OwnedMatrix,
        kind: i32,
        code: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &[usize],
        perm_inv: &[usize],
        coeff: *const u8,
        coeff_rows: usize,
    ) -> Result<(), String> {
        guarded(|| {
            owned_shape(&packed.matrix, tag, m, n)?;
            self.matrix = part(
                packed.matrix.as_ref(),
                kind,
                code,
                lower,
                perm,
                perm_inv,
                coeff,
                coeff_rows,
            )?;
            Ok(())
        })
    }
}

#[julia]
impl TypedMatrix {
    /// Factor this matrix's own storage in place; returns Julia `info`.
    #[julia]
    pub fn direct_factor(
        &mut self,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &mut [usize],
        perm_inv: &mut [usize],
        coeff: *mut u8,
        coeff_rows: usize,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            check_typed(data, tag, m, n)?;
            macro_rules! go {
                ($a:expr) => {
                    with_team(team, |par| {
                        factor(
                            $a.as_mut(),
                            kind,
                            lower,
                            perm,
                            perm_inv,
                            coeff,
                            coeff_rows,
                            scratch,
                            par,
                        )
                    })
                };
            }
            match data {
                MatrixData::F32(a) => go!(a),
                MatrixData::F64(a) => go!(a),
                MatrixData::C32(a) => go!(a),
                MatrixData::C64(a) => go!(a),
            }
        })
    }
    /// Solve in place on `rhs` with the packed factor stored in `self`.
    #[julia]
    pub fn direct_solve(
        &self,
        rhs: &mut TypedMatrix,
        kind: i32,
        op: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &[usize],
        perm_inv: &[usize],
        coeff: *const u8,
        coeff_rows: usize,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let data = self.data()?;
            check_typed(data, tag, m, n)?;
            let rhs_data = rhs.data.as_mut().ok_or("matrix is not initialized")?;
            macro_rules! go {
                ($a:expr, $b:expr) => {{
                    let a = $a.as_ref();
                    let b = $b.as_mut();
                    with_team(team, |par| {
                        solve(
                            a, b, kind, op, lower, perm, perm_inv, coeff, coeff_rows, scratch, par,
                        )
                    })
                }};
            }
            match (data, rhs_data, tag) {
                (MatrixData::F32(a), MatrixData::F32(b), 0) => go!(a, b),
                (MatrixData::F64(a), MatrixData::F64(b), 1) => go!(a, b),
                (MatrixData::C32(a), MatrixData::C32(b), 2) => go!(a, b),
                (MatrixData::C64(a), MatrixData::C64(b), 3) => go!(a, b),
                _ => Err("direct solve scalar types do not match".into()),
            }
        })
    }
    /// Fill `self` with a part of the packed factor stored in `packed`.
    #[julia]
    pub fn direct_part(
        &mut self,
        packed: &TypedMatrix,
        kind: i32,
        code: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        perm: &[usize],
        perm_inv: &[usize],
        coeff: *const u8,
        coeff_rows: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let data = packed.data()?;
            check_typed(data, tag, m, n)?;
            macro_rules! go {
                ($a:expr) => {
                    part(
                        $a.as_ref(),
                        kind,
                        code,
                        lower,
                        perm,
                        perm_inv,
                        coeff,
                        coeff_rows,
                    )?
                };
            }
            let out = match data {
                MatrixData::F32(a) => MatrixData::F32(go!(a)),
                MatrixData::F64(a) => MatrixData::F64(go!(a)),
                MatrixData::C32(a) => MatrixData::C32(go!(a)),
                MatrixData::C64(a) => MatrixData::C64(go!(a)),
            };
            self.data = Some(out);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;
