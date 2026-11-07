//! Direct in-place dense factorizations over a caller-owned native matrix.
//!
//! `lu!`, `full_piv_lu!`, `qr!`, `cholesky!`, `ldlt!` and `bunchkaufman!` with
//! caller scratch run faer's low-level `*_in_place` kernels on the input
//! matrix's own storage, which then holds the packed factor (Julia
//! `F.factors === A`). Permutations, the QR block Householder coefficients and
//! the Bunch–Kaufman off-diagonal live in Julia-owned buffers passed on every
//! call; nothing here retains a pointer across calls. Every validation that does
//! not need the numerical result happens before the first write to the matrix.
//!
//! Auxiliary buffers by kind: LU `perm`/`perm_inv` hold the row permutation
//! (length m); full-pivot LU holds `[row (m); column (n)]` in each; Bunch–Kaufman
//! holds the symmetric permutation (n) and uses the coefficient buffer as an
//! n×1 column with the block diagonal's subdiagonal (lower storage) or its
//! superdiagonal, the conjugate (upper storage).
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::{lblt, ldlt, llt};
use faer::linalg::householder;
use faer::linalg::lu::full_pivoting as flu;
use faer::linalg::lu::partial_pivoting as plu;
use faer::linalg::qr::no_pivoting as nqr;
use faer::perm::PermRef;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Col, ColMut, ColRef, Conj, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

// Kind codes match the retained-factor construction codes.
const LU: i32 = 0;
const FULL_LU: i32 = 1;
const QR: i32 = 2;
const LLT: i32 = 4;
const LDLT: i32 = 5;
const LBLT: i32 = 6;
const LBLT_ROOK: i32 = 8;

/// Operation codes shared with Julia. Right solves `X op(A) = B` run on the
/// transposed view of B, which the in-place kernels accept as any stride:
/// `X A = B` is `Aᵀ Xᵀ = Bᵀ`, `X Aᵀ = B` is `A Xᵀ = Bᵀ` and `X Aᴴ = B` is
/// `conj(A) Xᵀ = Bᵀ`.
const FACTOR: i32 = 0;
const SOLVE: i32 = 1;
const TRANSPOSE: i32 = 2;
const ADJOINT: i32 = 3;
const RIGHT_SOLVE: i32 = 4;
const RIGHT_TRANSPOSE: i32 = 5;
const RIGHT_ADJOINT: i32 = 6;
/// Overwrite the packed factor with the inverse of the factored matrix.
const INVERSE: i32 = 7;

/// `(transposed, conjugated)` of the left solve an operation code runs.
fn solve_form(op: i32) -> Option<(bool, bool)> {
    Some(match op {
        SOLVE | RIGHT_TRANSPOSE => (false, false),
        TRANSPOSE | RIGHT_SOLVE => (true, false),
        ADJOINT => (true, true),
        RIGHT_ADJOINT => (false, true),
        _ => return None,
    })
}

fn self_adjoint(kind: i32) -> bool {
    matches!(kind, LLT | LDLT | LBLT | LBLT_ROOK)
}
fn is_lblt(kind: i32) -> bool {
    kind == LBLT || kind == LBLT_ROOK
}

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
    if op == INVERSE {
        return inverse_requirement::<T>(kind, m, n, bs, par);
    }
    // A right solve needs the scratch of the left solve it runs; conjugation
    // does not change faer's workspace.
    let op = match solve_form(op) {
        Some((true, _)) if op != ADJOINT => TRANSPOSE,
        Some((false, _)) => SOLVE,
        _ => op,
    };
    let solve = SOLVE..=ADJOINT;
    Ok(match (kind, op) {
        (LU, FACTOR) => plu::factor::lu_in_place_scratch::<usize, T>(m, n, par, Default::default()),
        (LU, SOLVE) => plu::solve::solve_in_place_scratch::<usize, T>(n, k, par),
        (LU, TRANSPOSE | ADJOINT) => {
            plu::solve::solve_transpose_in_place_scratch::<usize, T>(n, k, par)
        }
        (FULL_LU, FACTOR) => {
            flu::factor::lu_in_place_scratch::<usize, T>(m, n, par, Default::default())
        }
        (FULL_LU, SOLVE) => flu::solve::solve_in_place_scratch::<usize, T>(n, k, par),
        (FULL_LU, TRANSPOSE | ADJOINT) => {
            flu::solve::solve_transpose_in_place_scratch::<usize, T>(n, k, par)
        }
        (QR, FACTOR) => nqr::factor::qr_in_place_scratch::<T>(m, n, bs, par, Default::default()),
        (QR, SOLVE) => nqr::solve::solve_lstsq_in_place_scratch::<T>(m, n, bs, k, par),
        (QR, TRANSPOSE | ADJOINT) => {
            nqr::solve::solve_transpose_in_place_scratch::<T>(n, bs, k, par)
        }
        (LLT, FACTOR) => llt::factor::cholesky_in_place_scratch::<T>(n, par, Default::default()),
        (LLT, op) if solve.contains(&op) => llt::solve::solve_in_place_scratch::<T>(n, k, par),
        (LDLT, FACTOR) => ldlt::factor::cholesky_in_place_scratch::<T>(n, par, Default::default()),
        (LDLT, op) if solve.contains(&op) => ldlt::solve::solve_in_place_scratch::<T>(n, k, par),
        // The same parameters as the factorization itself: the blocked
        // variant's workspace depends on the block size.
        (LBLT | LBLT_ROOK, FACTOR) => lblt::factor::cholesky_in_place_scratch::<usize, T>(
            n,
            par,
            crate::lblt::params::<T>(kind == LBLT_ROOK).into(),
        ),
        (LBLT | LBLT_ROOK, op) if solve.contains(&op) => {
            lblt::solve::solve_in_place_scratch::<usize, T>(n, k, par)
        }
        _ => return Err("unknown direct factorization kind or operation".into()),
    })
}

/// The packed factor's copy (plus Bunch–Kaufman's subdiagonal) followed by
/// faer's inverse workspace, which writes into the matrix itself.
fn inverse_requirement<T: ComplexField>(
    kind: i32,
    m: usize,
    n: usize,
    bs: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    if m != n {
        return Err("an inverse requires a square factor".into());
    }
    let inverse = match kind {
        LU => plu::inverse::inverse_scratch::<usize, T>(n, par),
        FULL_LU => flu::inverse::inverse_scratch::<usize, T>(n, par),
        QR => nqr::inverse::inverse_scratch::<T>(n, bs, par),
        LLT => llt::inverse::inverse_scratch::<T>(n, par),
        LDLT => ldlt::inverse::inverse_scratch::<T>(n, par),
        LBLT | LBLT_ROOK => lblt::inverse::inverse_scratch::<usize, T>(n, par),
        _ => return Err("unknown direct factorization kind or operation".into()),
    };
    let sub = if is_lblt(kind) { n } else { 0 };
    Ok(StackReq::all_of(&[
        faer::linalg::temp_mat_scratch::<T>(n, n),
        faer::linalg::temp_mat_scratch::<T>(sub, 1),
        inverse,
    ]))
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
/// Borrow the Julia-owned Bunch–Kaufman off-diagonal column (length `n`).
/// Safety: as for [`coeff_mut`]; a null pointer is accepted only when `n == 0`.
unsafe fn column_mut<'a, T: ComplexField>(p: *mut u8, n: usize) -> ColMut<'a, T> {
    if n == 0 {
        ColMut::from_slice_mut(&mut [])
    } else {
        ColMut::from_slice_mut(std::slice::from_raw_parts_mut(p.cast::<T>(), n))
    }
}
unsafe fn column_ref<'a, T: ComplexField>(p: *const u8, n: usize) -> ColRef<'a, T> {
    if n == 0 {
        ColRef::from_slice(&[])
    } else {
        ColRef::from_slice(std::slice::from_raw_parts(p.cast::<T>(), n))
    }
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
    if self_adjoint(kind) && m != n {
        return Err("self-adjoint factorizations require a square matrix".into());
    }
    match kind {
        LU if perm != m || perm_inv != m => Err("LU permutation length mismatch".into()),
        FULL_LU if perm != m + n || perm_inv != m + n => {
            Err("full-pivot LU permutation length mismatch".into())
        }
        QR if coeff_rows != block_size::<T>(QR, m, n) || coeff.is_null() => {
            Err("QR coefficient buffer shape mismatch".into())
        }
        LBLT | LBLT_ROOK if perm != n || perm_inv != n => {
            Err("Bunch-Kaufman permutation length mismatch".into())
        }
        LBLT | LBLT_ROOK if coeff_rows != n || (n > 0 && coeff.is_null()) => {
            Err("Bunch-Kaufman off-diagonal buffer shape mismatch".into())
        }
        LU | FULL_LU | QR | LLT | LDLT | LBLT | LBLT_ROOK => Ok(()),
        _ => Err("unknown direct factorization kind".into()),
    }
}

fn triangle_finite<T: ComplexField>(a: MatRef<'_, T>, lower: bool) -> bool {
    (0..a.ncols()).all(|j| {
        let (start, end) = if lower { (j, a.nrows()) } else { (0, j + 1) };
        (start..end).all(|i| is_finite(&a[(i, j)]))
    })
}

fn first_zero_pivot<T: ComplexField>(a: MatRef<'_, T>) -> usize {
    (0..a.nrows().min(a.ncols()))
        .find(|&i| a[(i, i)] == zero::<T>())
        .map_or(0, |i| i + 1)
}

/// Reproduce the input faer's self-adjoint solvers see: the selected triangle
/// with a real diagonal, mirrored into the lower triangle, strict upper zero.
fn lower_hermitian_in_place<T: ComplexField>(mut a: MatMut<'_, T>, lower: bool) {
    let n = a.nrows();
    for j in 0..n {
        a[(j, j)] = from_real::<T>(&real(&a[(j, j)]));
        for i in j + 1..n {
            if !lower {
                a[(i, j)] = conj(&a[(j, i)]);
            }
            a[(j, i)] = zero();
        }
    }
}
/// The kernels may touch the strict upper triangle; zero it as the retained
/// factors do, or move the lower factor to the upper triangle as its adjoint.
fn store_factor_triangle<T: ComplexField>(mut a: MatMut<'_, T>, lower: bool) {
    let n = a.nrows();
    for j in 0..n {
        for i in j + 1..n {
            a[(j, i)] = if lower { zero() } else { conj(&a[(i, j)]) };
            if !lower {
                a[(i, j)] = zero();
            }
        }
    }
}

/// Visit each diagonal block of B (diagonal `diag`, subdiagonal `off`) with its
/// start and determinant until `f` returns false. Does not allocate.
fn for_each_block<T: ComplexField>(
    diag: ColRef<'_, T>,
    off: ColRef<'_, T>,
    mut f: impl FnMut(usize, T) -> bool,
) {
    let n = diag.nrows();
    let mut i = 0;
    while i < n {
        let block = i + 1 < n && off[i] != zero::<T>();
        let determinant = if block {
            sub(&mul(&diag[i], &diag[i + 1]), &mul(&off[i], &conj(&off[i])))
        } else {
            copy(&diag[i])
        };
        if !f(i, determinant) {
            return;
        }
        i += if block { 2 } else { 1 };
    }
}
/// Julia `info` of a block diagonal: one plus the first singular block's start.
fn singular_block<T: ComplexField>(diag: ColRef<'_, T>, off: ColRef<'_, T>) -> usize {
    let mut info = 0;
    for_each_block(diag, off, |i, det| {
        if det == zero::<T>() {
            info = i + 1;
        }
        info == 0
    });
    info
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
    let finite = if self_adjoint(kind) {
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
            first_zero_pivot(a.as_ref())
        }
        FULL_LU => {
            let (row, col) = perm.split_at_mut(m);
            let (row_inv, col_inv) = perm_inv.split_at_mut(m);
            flu::factor::lu_in_place(
                a.as_mut(),
                row,
                row_inv,
                col,
                col_inv,
                par,
                stack,
                Default::default(),
            );
            first_zero_pivot(a.as_ref())
        }
        QR => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_mut::<T>(coeff, coeff_rows, m.min(n)) };
            nqr::factor::qr_in_place(a, c, par, stack, Default::default());
            0
        }
        LLT => {
            lower_hermitian_in_place(a.as_mut(), lower);
            match llt::factor::cholesky_in_place(
                a.as_mut(),
                Default::default(),
                par,
                stack,
                Default::default(),
            ) {
                Ok(_) => {
                    store_factor_triangle(a, lower);
                    0
                }
                Err(llt::factor::LltError::NonPositivePivot { index }) => index + 1,
            }
        }
        LDLT => {
            lower_hermitian_in_place(a.as_mut(), lower);
            // faer stores D on the diagonal and the unit-lower L below it.
            match ldlt::factor::cholesky_in_place(
                a.as_mut(),
                Default::default(),
                par,
                stack,
                Default::default(),
            ) {
                Ok(_) => {
                    store_factor_triangle(a, lower);
                    0
                }
                Err(ldlt::factor::LdltError::ZeroPivot { index }) => index + 1,
            }
        }
        _ => {
            lower_hermitian_in_place(a.as_mut(), lower);
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let mut sub = unsafe { column_mut::<T>(coeff, n) };
            // 1×1 pivots leave their subdiagonal entries untouched.
            sub.fill(zero());
            lblt::factor::cholesky_in_place(
                a.as_mut(),
                sub.as_mut().as_diagonal_mut(),
                perm,
                perm_inv,
                par,
                stack,
                crate::lblt::params::<T>(kind == LBLT_ROOK).into(),
            );
            let info = singular_block(a.as_ref().diagonal().column_vector(), sub.as_ref());
            store_factor_triangle(a, lower);
            if !lower {
                // Upper storage keeps Bᴴ's view: the superdiagonal of B.
                for i in 0..n {
                    sub[i] = conj(&sub[i]);
                }
            }
            info
        }
    })
}

/// The lower-oriented packed factor stored in `a` (lower: as is; upper: its
/// adjoint). Its diagonal holds L (Cholesky), D (LDLᴴ) or B's diagonal (LBLᴴ).
fn packed_lower<T: ComplexField>(a: MatRef<'_, T>, lower: bool) -> Mat<T> {
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
fn unit_lower<T: ComplexField>(a: MatRef<'_, T>, rows: usize, cols: usize) -> Mat<T> {
    Mat::from_fn(rows, cols, |i, j| {
        if i == j {
            one()
        } else if i > j {
            copy(&a[(i, j)])
        } else {
            zero()
        }
    })
}
/// B's subdiagonal from the stored off-diagonal column (see the module docs).
fn lblt_subdiag<T: ComplexField>(stored: ColRef<'_, T>, lower: bool) -> Col<T> {
    Col::from_fn(stored.nrows(), |i| {
        if lower {
            copy(&stored[i])
        } else {
            conj(&stored[i])
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
    let (transposed, conjugated) = solve_form(op).ok_or("unknown direct solve operation")?;
    let rhs = if op >= RIGHT_SOLVE {
        rhs.transpose_mut()
    } else {
        rhs
    };
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
    let conj_op = if conjugated { Conj::Yes } else { Conj::No };
    // A self-adjoint A has Aᵀ = conj(A) and Aᴴ = A: solve with the factor,
    // conjugated when exactly one of the two applies.
    let conj_sa = if transposed != conjugated {
        Conj::Yes
    } else {
        Conj::No
    };
    // Upper storage holds Lᴴ, so `a.transpose()` is conj(L).
    let (l, conj_l) = if lower {
        (a, conj_sa)
    } else {
        (a.transpose(), conj_sa.compose(Conj::Yes))
    };
    match kind {
        LU => {
            let p = PermRef::new_checked(perm, perm_inv, n);
            if !transposed {
                plu::solve::solve_in_place_with_conj(a, a, p, conj_op, rhs, par, stack);
            } else {
                plu::solve::solve_transpose_in_place_with_conj(a, a, p, conj_op, rhs, par, stack);
            }
        }
        FULL_LU => {
            let row = PermRef::new_checked(&perm[..n], &perm_inv[..n], n);
            let col = PermRef::new_checked(&perm[n..], &perm_inv[n..], n);
            if !transposed {
                flu::solve::solve_in_place_with_conj(a, a, row, col, conj_op, rhs, par, stack);
            } else {
                flu::solve::solve_transpose_in_place_with_conj(
                    a, a, row, col, conj_op, rhs, par, stack,
                );
            }
        }
        QR => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_ref::<T>(coeff, coeff_rows, m.min(n)) };
            if !transposed {
                nqr::solve::solve_lstsq_in_place_with_conj(a, c, a, conj_op, rhs, par, stack);
            } else {
                nqr::solve::solve_transpose_in_place_with_conj(a, c, a, conj_op, rhs, par, stack);
            }
        }
        LLT => llt::solve::solve_in_place_with_conj(l, conj_l, rhs, par, stack),
        LDLT => ldlt::solve::solve_in_place_with_conj(l, a.diagonal(), conj_l, rhs, par, stack),
        _ => {
            // Upper storage keeps conj(subdiag), matching conj(L) and conj(B):
            // solving conj(conj(A)) with them is solving A.
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let sub = unsafe { column_ref::<T>(coeff, n) };
            let p = PermRef::new_checked(perm, perm_inv, n);
            lblt::solve::solve_in_place_with_conj(
                l,
                a.diagonal(),
                sub.as_diagonal(),
                conj_l,
                p,
                rhs,
                par,
                stack,
            );
        }
    }
    Ok(())
}

/// Overwrite the packed factor in `a` with the inverse of the factored matrix
/// (both triangles for the self-adjoint kinds, as Julia's `inv!(::Cholesky)`).
/// faer's inverses read the factor while writing their output, so the factor
/// is first copied into the caller scratch; nothing else is allocated here.
/// Julia checks the factorization `info` before the call; a QR factor with a
/// zero R diagonal returns its 1-based index before any write (0 otherwise).
#[allow(clippy::too_many_arguments)]
fn invert<T: ComplexField>(
    mut a: MatMut<'_, T>,
    kind: i32,
    lower: bool,
    perm: &[usize],
    perm_inv: &[usize],
    coeff: *const u8,
    coeff_rows: usize,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<usize, String> {
    use faer::mat::AsMatMut;
    let (m, n) = a.shape();
    check_aux::<T>(kind, m, n, perm.len(), perm_inv.len(), coeff, coeff_rows)?;
    if kind == QR {
        let i = first_zero_pivot(a.as_ref());
        if i != 0 {
            return Ok(i);
        }
    }
    let bs = block_size::<T>(kind, m, n);
    let stack = scratch.stack(inverse_requirement::<T>(kind, m, n, bs, par.shape())?)?;
    let (mut f, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
    let mut f = f.as_mat_mut();
    let lblt = is_lblt(kind);
    let (mut sub, stack) =
        faer::linalg::temp_mat_zeroed::<T, _, _>(if lblt { n } else { 0 }, 1, stack);
    let mut sub = sub.as_mat_mut();
    if self_adjoint(kind) {
        // The lower-oriented packed factor, as `packed_lower`.
        for j in 0..n {
            for i in j..n {
                f[(i, j)] = if lower {
                    copy(&a[(i, j)])
                } else {
                    conj(&a[(j, i)])
                };
            }
        }
    } else {
        f.copy_from(a.as_ref());
    }
    if lblt {
        // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
        let stored = unsafe { column_ref::<T>(coeff, n) };
        for i in 0..n {
            sub[(i, 0)] = if lower {
                copy(&stored[i])
            } else {
                conj(&stored[i])
            };
        }
    }
    let f = f.as_ref();
    match kind {
        LU => {
            let p = PermRef::new_checked(perm, perm_inv, n);
            plu::inverse::inverse(a.as_mut(), f, f, p, par, stack);
        }
        FULL_LU => {
            let row = PermRef::new_checked(&perm[..n], &perm_inv[..n], n);
            let col = PermRef::new_checked(&perm[n..], &perm_inv[n..], n);
            flu::inverse::inverse(a.as_mut(), f, f, row, col, par, stack);
        }
        QR => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let c = unsafe { coeff_ref::<T>(coeff, coeff_rows, n) };
            nqr::inverse::inverse(a.as_mut(), f, c, f, par, stack);
        }
        LLT => llt::inverse::inverse(a.as_mut(), f, par, stack),
        LDLT => ldlt::inverse::inverse(a.as_mut(), f, f.diagonal(), par, stack),
        _ => {
            let p = PermRef::new_checked(perm, perm_inv, n);
            let s = sub.as_ref().col(0).as_diagonal();
            lblt::inverse::inverse(a.as_mut(), f, f.diagonal(), s, p, par, stack);
        }
    }
    if self_adjoint(kind) {
        self_adjoint_from_lower(a);
    }
    Ok(0)
}

/// faer's self-adjoint reconstruct/inverse fill the lower triangle only.
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

/// `[log|det|, phase]` through the native diagonal measure (#148) applied to
/// one pivot per factor of det(A), times the permutation signs.
fn logabsdet<T: ComplexField<Real: crate::measures::Ln>>(
    a: MatRef<'_, T>,
    kind: i32,
    lower: bool,
    perm: &[usize],
    coeff: *const u8,
) -> Result<Mat<T>, String> {
    let (m, n) = a.shape();
    if m != n {
        return Err("determinant requires a square factor".into());
    }
    let d = |i: usize| copy(&a[(i, i)]);
    let (pivots, sign) = match kind {
        LU => (
            Mat::from_fn(n, 1, |i, _| d(i)),
            crate::permutation_sign(perm),
        ),
        FULL_LU => (
            Mat::from_fn(n, 1, |i, _| d(i)),
            crate::permutation_sign(&perm[..n]) * crate::permutation_sign(&perm[n..]),
        ),
        // det(L Lᴴ) = Π lᵢᵢ², without forming squares that could overflow.
        LLT => (Mat::from_fn(2 * n, 1, |i, _| d(i % n.max(1))), 1.0),
        LDLT => (Mat::from_fn(n, 1, |i, _| d(i)), 1.0),
        LBLT | LBLT_ROOK => {
            // det(P L B Lᴴ Pᵀ) = det(B): one entry per diagonal block.
            // SAFETY: shape checked by the caller's `check_aux`.
            let off = lblt_subdiag(unsafe { column_ref::<T>(coeff, n) }, lower);
            let mut blocks = Vec::with_capacity(n);
            for_each_block(a.diagonal().column_vector(), off.as_ref(), |_, det| {
                blocks.push(det);
                true
            });
            (Mat::from_fn(blocks.len(), 1, |i, _| copy(&blocks[i])), 1.0)
        }
        _ => return Err("direct factorization has no determinant".into()),
    };
    let mut out = Mat::<T>::zeros(2, 1);
    crate::measures::measure(
        pivots.as_ref(),
        false,
        5,
        2,
        Par::Seq,
        out.as_ptr_mut().cast(),
    )?;
    out[(1, 0)] = mul(&out[(1, 0)], &from_f64::<T>(sign));
    Ok(out)
}

/// Part codes follow the retained-factor convention: 0 reconstruct, 1 L,
/// 2 U, 3 Q, 4 R, 5 D, 6 thin Q, 7 lower-oriented packed (self-adjoint kinds),
/// 8 Householder basis, 9 QR coefficients, 10 inverse, 11 diagonal,
/// 12 `[log|det|, phase]`.
#[allow(clippy::too_many_arguments)]
fn part<T: ComplexField<Real: crate::measures::Ln>>(
    a: MatRef<'_, T>,
    kind: i32,
    code: i32,
    lower: bool,
    perm: &[usize],
    perm_inv: &[usize],
    coeff: *const u8,
    coeff_rows: usize,
    par: Par<'_>,
) -> Result<Mat<T>, String> {
    let (m, n) = a.shape();
    let size = m.min(n);
    check_aux::<T>(kind, m, n, perm.len(), perm_inv.len(), coeff, coeff_rows)?;
    let buffer = |req: StackReq| MemBuffer::try_new(req).map_err(|e| format!("{e:?}"));
    let lu_kind = kind == LU || kind == FULL_LU;
    Ok(match (kind, code) {
        (_, 12) => logabsdet(a, kind, lower, perm, coeff)?,
        (LU | FULL_LU, 1) => unit_lower(a, m, size),
        (LU | FULL_LU, 2) => Mat::from_fn(
            size,
            n,
            |i, j| if i <= j { copy(&a[(i, j)]) } else { zero() },
        ),
        (LU, 0) => {
            let p = PermRef::new_checked(perm, perm_inv, m);
            let l = unit_lower(a, m, size);
            let u = part::<T>(
                a,
                kind,
                2,
                lower,
                perm,
                perm_inv,
                coeff,
                coeff_rows,
                Par::Seq,
            )?;
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
        (FULL_LU, 0) => {
            let row = PermRef::new_checked(&perm[..m], &perm_inv[..m], m);
            let col = PermRef::new_checked(&perm[m..], &perm_inv[m..], n);
            let l = unit_lower(a, m, size);
            let u = part::<T>(
                a,
                kind,
                2,
                lower,
                perm,
                perm_inv,
                coeff,
                coeff_rows,
                Par::Seq,
            )?;
            let mut out = Mat::zeros(m, n);
            let mut mem = buffer(flu::reconstruct::reconstruct_scratch::<usize, T>(
                m,
                n,
                par.shape(),
            ))?;
            flu::reconstruct::reconstruct(
                out.as_mut(),
                l.as_ref(),
                u.as_ref(),
                row,
                col,
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
        (FULL_LU, 10) if m == n => {
            let row = PermRef::new_checked(&perm[..n], &perm_inv[..n], n);
            let col = PermRef::new_checked(&perm[n..], &perm_inv[n..], n);
            let mut out = Mat::zeros(n, n);
            let mut mem = buffer(flu::inverse::inverse_scratch::<usize, T>(n, par.shape()))?;
            flu::inverse::inverse(out.as_mut(), a, a, row, col, par, MemStack::new(&mut mem));
            out
        }
        (QR, 4) => Mat::from_fn(
            size,
            n,
            |i, j| if i <= j { copy(&a[(i, j)]) } else { zero() },
        ),
        (QR, 8) => unit_lower(a, m, size),
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
        (_, 11) if lu_kind || kind == QR || self_adjoint(kind) => {
            Mat::from_fn(size, 1, |i, _| copy(&a[(i, i)]))
        }
        (LLT, 1) | (LLT | LDLT | LBLT | LBLT_ROOK, 7) => packed_lower(a, lower),
        (LLT, 2) => packed_lower(a, lower).adjoint().to_owned(),
        (LDLT | LBLT | LBLT_ROOK, 1) => unit_lower(packed_lower(a, lower).as_ref(), n, n),
        (LDLT, 5) => Mat::from_fn(n, n, |i, j| if i == j { copy(&a[(i, i)]) } else { zero() }),
        (LBLT | LBLT_ROOK, 5) => {
            // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
            let sub = lblt_subdiag(unsafe { column_ref::<T>(coeff, n) }, lower);
            Mat::from_fn(n, n, |i, j| {
                if i == j {
                    copy(&a[(i, i)])
                } else if i == j + 1 {
                    copy(&sub[j])
                } else if j == i + 1 {
                    conj(&sub[i])
                } else {
                    zero()
                }
            })
        }
        (LLT | LDLT | LBLT | LBLT_ROOK, 0 | 10) => {
            let l = packed_lower(a, lower);
            let mut out = Mat::zeros(n, n);
            let sub = if is_lblt(kind) {
                // SAFETY: shape checked by `check_aux`; Julia preserves the buffer.
                lblt_subdiag(unsafe { column_ref::<T>(coeff, n) }, lower)
            } else {
                Col::zeros(0)
            };
            let req = match (kind, code) {
                (LLT, 0) => llt::reconstruct::reconstruct_scratch::<T>(n, par.shape()),
                (LLT, _) => llt::inverse::inverse_scratch::<T>(n, par.shape()),
                (LDLT, 0) => ldlt::reconstruct::reconstruct_scratch::<T>(n, par.shape()),
                (LDLT, _) => ldlt::inverse::inverse_scratch::<T>(n, par.shape()),
                (_, 0) => lblt::reconstruct::reconstruct_scratch::<usize, T>(n, par.shape()),
                _ => lblt::inverse::inverse_scratch::<usize, T>(n, par.shape()),
            };
            let mut mem = buffer(req)?;
            let stack = MemStack::new(&mut mem);
            let (o, lr, d) = (out.as_mut(), l.as_ref(), l.diagonal());
            match (kind, code) {
                (LLT, 0) => llt::reconstruct::reconstruct(o, lr, par, stack),
                (LLT, _) => llt::inverse::inverse(o, lr, par, stack),
                (LDLT, 0) => ldlt::reconstruct::reconstruct(o, lr, d, par, stack),
                (LDLT, _) => ldlt::inverse::inverse(o, lr, d, par, stack),
                (_, c) => {
                    let p = PermRef::new_checked(perm, perm_inv, n);
                    let s = sub.as_ref().as_diagonal();
                    if c == 0 {
                        lblt::reconstruct::reconstruct(o, lr, d, s, p, par, stack);
                    } else {
                        lblt::inverse::inverse(o, lr, d, s, p, par, stack);
                    }
                }
            }
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
    /// Overwrite the packed factor stored in `self` with the inverse; returns
    /// a QR factor's zero-pivot index (before any write) or 0.
    #[julia]
    pub fn direct_inverse(
        &mut self,
        kind: i32,
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
    ) -> Result<usize, String> {
        guarded(|| {
            owned_shape(&self.matrix, tag, m, n)?;
            let a = self.matrix.as_mut();
            with_team(team, |par| {
                invert(
                    a, kind, lower, perm, perm_inv, coeff, coeff_rows, scratch, par,
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            owned_shape(&packed.matrix, tag, m, n)?;
            self.matrix = with_team(team, |par| {
                part(
                    packed.matrix.as_ref(),
                    kind,
                    code,
                    lower,
                    perm,
                    perm_inv,
                    coeff,
                    coeff_rows,
                    par,
                )
            })?;
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
    /// Overwrite the packed factor stored in `self` with the inverse; returns
    /// a QR factor's zero-pivot index (before any write) or 0.
    #[julia]
    pub fn direct_inverse(
        &mut self,
        kind: i32,
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
    ) -> Result<usize, String> {
        guarded(|| {
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            check_typed(data, tag, m, n)?;
            macro_rules! go {
                ($a:expr) => {
                    with_team(team, |par| {
                        invert(
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let data = packed.data()?;
            check_typed(data, tag, m, n)?;
            macro_rules! go {
                ($a:expr) => {
                    with_team(team, |par| {
                        part(
                            $a.as_ref(),
                            kind,
                            code,
                            lower,
                            perm,
                            perm_inv,
                            coeff,
                            coeff_rows,
                            par,
                        )
                    })?
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
