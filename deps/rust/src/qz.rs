//! Generalized Hessenberg reduction and native real/complex QZ.
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::{guarded, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{gevd, matmul::matmul};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField, RealField};
use faer::{c32, c64, Accum, Mat, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

struct Pair<T: ComplexField> {
    s: Mat<T>,
    t: Mat<T>,
    q: Mat<T>,
    z: Mat<T>,
    ascl: Option<T::Real>,
    bscl: Option<T::Real>,
}
fn prepare<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    mut par: Par,
    stack: &mut MemStack,
) -> Result<Pair<T>, String> {
    if a.nrows() != a.ncols() || a.shape() != b.shape() {
        return Err("QZ requires matching square matrices".into());
    }
    finite(a)?;
    finite(b)?;
    let n = a.nrows();
    let mut s = a.to_owned();
    let mut t = b.to_owned();
    let ascl = crate::schur::scale(&mut s);
    let bscl = crate::schur::scale(&mut t);
    let mut q = Mat::identity(n, n);
    let mut z = Mat::identity(n, n);
    if n > 0 {
        let block = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n, n);
        {
            let (mut coeff, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(block, n, stack);
            faer::linalg::qr::no_pivoting::factor::qr_in_place(
                t.as_mut(),
                coeff.as_mat_mut(),
                par.reborrow(),
                stack,
                Default::default(),
            );
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                t.as_ref(), coeff.as_mat_ref(), faer::Conj::No, q.as_mut(), par.reborrow(), stack);
        }
        // QR's basis shares the packed R storage; only after Q is formed may its
        // strict lower triangle be discarded.
        for j in 0..n {
            for i in j + 1..n {
                t[(i, j)] = zero::<T>();
            }
        }
        {
            let (mut projection, _) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
            matmul(
                projection.as_mat_mut(),
                Accum::Replace,
                q.adjoint(),
                s.as_ref(),
                one::<T>(),
                par.reborrow(),
            );
            s.copy_from(projection.as_mat_ref());
        }
        let params = crate::schur_tuning::generalized_hessenberg_params::<T>();
        gevd::gen_hessenberg::generalized_hessenberg(
            s.as_mut(),
            t.as_mut(),
            Some(q.as_mut()),
            Some(z.as_mut()),
            par,
            stack,
            params,
        );
    }
    Ok(Pair {
        s,
        t,
        q,
        z,
        ascl,
        bscl,
    })
}
fn from_reduced<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    q: MatRef<'_, T>,
    z: MatRef<'_, T>,
) -> Result<Pair<T>, String> {
    if a.nrows() != a.ncols()
        || a.shape() != b.shape()
        || a.shape() != q.shape()
        || a.shape() != z.shape()
    {
        return Err("H, T, Q and Z must have matching square dimensions".into());
    }
    for m in [a, b, q, z] {
        finite(m)?;
    }
    for j in 0..a.ncols() {
        for i in j + 1..a.nrows() {
            if b[(i, j)] != zero::<T>() || (i > j + 1 && a[(i, j)] != zero::<T>()) {
                return Err("input pair is not in generalized Hessenberg form".into());
            }
        }
    }
    let mut s = a.to_owned();
    let mut t = b.to_owned();
    let ascl = crate::schur::scale(&mut s);
    let bscl = crate::schur::scale(&mut t);
    Ok(Pair {
        s,
        t,
        q: q.to_owned(),
        z: z.to_owned(),
        ascl,
        bscl,
    })
}
fn finish<T: ComplexField>(p: &mut Pair<T>) -> Result<(), String> {
    crate::schur::restore(&mut p.s, p.ascl.take());
    crate::schur::restore(&mut p.t, p.bscl.take());
    for m in [&p.s, &p.t, &p.q, &p.z] {
        finite(m.as_ref())?;
    }
    Ok(())
}
// Selected faer 2d4ea4e includes #253's even-window query fix and #254's
// fully-deflated sentinel fix. #260 is still unmerged: the local Sylvester
// solve owns four index arrays rather than the query's one. Retain the bound
// for those live allocations, including even-rounded recursive windows, and
// keep the public recommendation as a lower bound for future revisions.
pub(crate) fn real_qz_scratch<T: RealField>(
    n: usize,
    params: gevd::GeneralizedSchurParams,
    par: ParShape,
) -> StackReq {
    if n < usize::max(15, params.blocking_threshold) {
        return StackReq::EMPTY;
    }
    let floor = (n - 3) / 3;
    let nw = floor + floor % 2;
    let ns = (params.recommended_shift_count)(n, n);
    let cost = (params.relative_cost_estimate_of_shift_chase_to_matmul)(n, n);
    let extra =
        (ns as f64 / (1.0 + 2.0 * ns as f64 / (cost as f64 / 100.0 * n as f64)).sqrt()) as usize;
    let panel = ns + (extra.saturating_sub(1) / 4) * 4 + 4;
    let temp = faer::linalg::temp_mat_scratch::<T>;
    let local = StackReq::all_of(&[
        temp(4, 4),
        temp(4, 4),
        temp(8, 8),
        temp(8, 1),
        StackReq::all_of(&[StackReq::new::<usize>(8); 4]),
        StackReq::any_of(&[
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(
                8,
                8,
                Par::Seq.shape(),
                Default::default(),
            ),
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T>(
                n,
                1,
                Par::Seq.shape(),
            ),
        ]),
    ]);
    let aed = StackReq::any_of(&[
        real_qz_scratch::<T>(nw, params, par),
        local,
        temp(nw, n),
        temp(n, nw),
    ]);
    StackReq::any_of(&[
        gevd::qz_real::hessenberg_to_qz_scratch::<T>(n, par, params),
        StackReq::all_of(&[temp(nw, nw), temp(nw, nw), aed]),
        StackReq::all_of(&[temp(panel, panel), temp(panel, panel), temp(n, 2 * ns)]),
    ])
}

trait QzScalar: ComplexField + Sized {
    type Alpha;
    fn qz_scratch(n: usize, par: ParShape) -> StackReq;
    unsafe fn qz(
        p: &mut Pair<Self>,
        alpha: *mut u8,
        beta: *mut u8,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<(), String>;
}
macro_rules! real_qz {
    ($r:ty,$c:ty) => {
        impl QzScalar for $r {
            type Alpha = $c;
            fn qz_scratch(n: usize, par: ParShape) -> StackReq {
                if n == 0 {
                    return StackReq::EMPTY;
                }
                StackReq::all_of(&[StackReq::new::<Self>(n); 3]).and(real_qz_scratch::<Self>(
                    n,
                    crate::schur_tuning::qz_params::<Self>(),
                    par,
                ))
            }
            unsafe fn qz(
                p: &mut Pair<Self>,
                alpha: *mut u8,
                beta: *mut u8,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                let n = p.s.nrows();
                if n == 0 {
                    return finish(p);
                }
                let (mut ar, stack) = stack.make_with::<Self>(n, |_| Self::NAN);
                let (mut ai, stack) = stack.make_with::<Self>(n, |_| Self::NAN);
                let (mut be, stack) = stack.make_with::<Self>(n, |_| Self::NAN);
                let params: gevd::GeneralizedSchurParams = crate::schur_tuning::qz_params::<Self>();
                gevd::qz_real::hessenberg_to_qz(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    faer::ColMut::from_slice_mut(&mut ar),
                    faer::ColMut::from_slice_mut(&mut ai),
                    faer::ColMut::from_slice_mut(&mut be),
                    gevd::ComputeEigenvectors::Yes,
                    par,
                    params,
                    stack,
                );
                let ia = recip(&p.ascl.clone().unwrap_or(1.0));
                let ib = recip(&p.bscl.clone().unwrap_or(1.0));
                let mut previous_pair = false;
                for i in 0..n {
                    if !ar[i].is_finite() || !ai[i].is_finite() || !be[i].is_finite() {
                        return Err("QZ did not converge to finite homogeneous eigenvalues".into());
                    }
                    let pair = i + 1 < n && p.s[(i + 1, i)] != 0.0;
                    if previous_pair && pair {
                        return Err("QZ did not converge to quasi-triangular form".into());
                    }
                    previous_pair = pair;
                    let a = <$c>::new(ar[i] * ia, ai[i] * ia);
                    let b = be[i] * ib;
                    if !a.re.is_finite() || !a.im.is_finite() || !b.is_finite() {
                        return Err("QZ eigenvalue rescaling overflow".into());
                    }
                    alpha.cast::<$c>().add(i).write(a);
                    beta.cast::<Self>().add(i).write(b);
                }
                // The kernels may use entries below the active bands as workspace.
                for j in 0..n {
                    for i in j + 2..n {
                        p.s[(i, j)] = 0.0;
                    }
                    for i in j + 1..n {
                        p.t[(i, j)] = 0.0;
                    }
                }
                finish(p)
            }
        }
    };
}
macro_rules! complex_qz {
    ($c:ty) => {
        impl QzScalar for $c {
            type Alpha = Self;
            fn qz_scratch(n: usize, par: ParShape) -> StackReq {
                if n == 0 {
                    return StackReq::EMPTY;
                }
                StackReq::all_of(&[StackReq::new::<Self>(n); 2]).and(
                    gevd::qz_cplx::hessenberg_to_qz_scratch::<Self>(
                        n,
                        par,
                        crate::schur_tuning::qz_params::<Self>(),
                    ),
                )
            }
            unsafe fn qz(
                p: &mut Pair<Self>,
                alpha: *mut u8,
                beta: *mut u8,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                let n = p.s.nrows();
                if n == 0 {
                    return finish(p);
                }
                let (mut a, stack) = stack
                    .make_with::<Self>(n, |_| Self::new(<Self as ComplexField>::Real::NAN, 0.0));
                let (mut b, stack) = stack
                    .make_with::<Self>(n, |_| Self::new(<Self as ComplexField>::Real::NAN, 0.0));
                let params: gevd::GeneralizedSchurParams = crate::schur_tuning::qz_params::<Self>();
                gevd::qz_cplx::hessenberg_to_qz(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    faer::ColMut::from_slice_mut(&mut a),
                    faer::ColMut::from_slice_mut(&mut b),
                    gevd::ComputeEigenvectors::Yes,
                    par,
                    params,
                    stack,
                );
                for j in 0..n.saturating_sub(1) {
                    if p.s[(j + 1, j)] != Self::new(0.0, 0.0) {
                        return Err("QZ did not converge to triangular form".into());
                    }
                }
                let ia = recip(&p.ascl.clone().unwrap_or(1.0));
                let ib = recip(&p.bscl.clone().unwrap_or(1.0));
                for i in 0..n {
                    let av = mul_real(&a[i], &ia);
                    let bv = mul_real(&b[i], &ib);
                    if !is_finite(&av) || !is_finite(&bv) {
                        return Err("QZ did not converge to finite homogeneous eigenvalues".into());
                    }
                    alpha.cast::<Self>().add(i).write(av);
                    beta.cast::<Self>().add(i).write(bv);
                }
                for j in 0..n {
                    for i in j + 1..n {
                        p.s[(i, j)] = Self::new(0.0, 0.0);
                        p.t[(i, j)] = Self::new(0.0, 0.0);
                    }
                }
                finish(p)
            }
        }
    };
}
real_qz!(f32, c32);
real_qz!(f64, c64);
complex_qz!(c32);
complex_qz!(c64);
unsafe fn compute<T: QzScalar>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    initial: Option<(MatRef<'_, T>, MatRef<'_, T>)>,
    alpha: *mut u8,
    beta: *mut u8,
    only_hessenberg: bool,
    n: usize,
) -> Result<Pair<T>, String> {
    if a.nrows() != n || a.ncols() != n || b.shape() != a.shape() {
        return Err("QZ current native dimensions do not match requested output".into());
    }
    let stage = if only_hessenberg {
        1
    } else if initial.is_some() {
        2
    } else {
        0
    };
    let mut memory = MemBuffer::try_new(construction::requirement::<T>(n, stage, ParShape::Seq)?)
        .map_err(|e| format!("{e:?}"))?;
    compute_in(
        a,
        b,
        initial,
        alpha,
        beta,
        only_hessenberg,
        n,
        Par::Seq,
        MemStack::new(&mut memory),
    )
}
unsafe fn compute_in<T: QzScalar>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    initial: Option<(MatRef<'_, T>, MatRef<'_, T>)>,
    alpha: *mut u8,
    beta: *mut u8,
    only_hessenberg: bool,
    n: usize,
    mut par: Par,
    stack: &mut MemStack,
) -> Result<Pair<T>, String> {
    if a.nrows() != n {
        return Err("QZ current native dimension does not match the requested output".into());
    }
    if !only_hessenberg {
        homogeneous_buffers::<T::Alpha, T>(alpha, beta, n)?;
    }
    let mut p = if let Some((q, z)) = initial {
        from_reduced(a, b, q, z)?
    } else {
        prepare(a, b, par.reborrow(), stack)?
    };
    if only_hessenberg {
        finish(&mut p)?;
    } else {
        T::qz(&mut p, alpha, beta, par, stack)?;
    }
    Ok(p)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn qz_from(
        &mut self,
        a: &OwnedMatrix,
        b: &OwnedMatrix,
        out_t: &mut OwnedMatrix,
        out_q: &mut OwnedMatrix,
        out_z: &mut OwnedMatrix,
        alpha: *mut u8,
        beta: *mut u8,
        only_hessenberg: bool,
        in_q: *const u8,
        in_z: *const u8,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            if tag != 1 {
                return Err("QZ requested scalar type does not match its native owner".into());
            }
            let initial = if in_q.is_null() && in_z.is_null() {
                None
            } else if !in_q.is_null() && !in_z.is_null() {
                Some((
                    (&*in_q.cast::<OwnedMatrix>()).matrix.as_ref(),
                    (&*in_z.cast::<OwnedMatrix>()).matrix.as_ref(),
                ))
            } else {
                return Err("Q and Z must both be provided".into());
            };
            let p = compute(
                a.matrix.as_ref(),
                b.matrix.as_ref(),
                initial,
                alpha,
                beta,
                only_hessenberg,
                n,
            )?;
            self.matrix = p.s;
            (out_t).matrix = p.t;
            (out_q).matrix = p.q;
            (out_z).matrix = p.z;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn qz_from(
        &mut self,
        a: &TypedMatrix,
        b: &TypedMatrix,
        out_t: &mut TypedMatrix,
        out_q: &mut TypedMatrix,
        out_z: &mut TypedMatrix,
        alpha: *mut u8,
        beta: *mut u8,
        only_hessenberg: bool,
        in_q: *const u8,
        in_z: *const u8,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let a = (a).data()?;
            let b = (b).data()?;
            let initial = if in_q.is_null() && in_z.is_null() {
                None
            } else if !in_q.is_null() && !in_z.is_null() {
                Some((
                    (&*in_q.cast::<TypedMatrix>()).data()?,
                    (&*in_z.cast::<TypedMatrix>()).data()?,
                ))
            } else {
                return Err("Q and Z must both be provided".into());
            };
            macro_rules! run {
                ($a:expr,$b:expr,$variant:ident) => {{
                    let init = match initial {
                        None => None,
                        Some((MatrixData::$variant(q), MatrixData::$variant(z))) => {
                            Some((q.as_ref(), z.as_ref()))
                        }
                        _ => return Err("generalized Hessenberg scalar types must match".into()),
                    };
                    let p = compute(
                        $a.as_ref(),
                        $b.as_ref(),
                        init,
                        alpha,
                        beta,
                        only_hessenberg,
                        n,
                    )?;
                    (out_t).data = Some(MatrixData::$variant(p.t));
                    (out_q).data = Some(MatrixData::$variant(p.q));
                    (out_z).data = Some(MatrixData::$variant(p.z));
                    MatrixData::$variant(p.s)
                }};
            }
            self.data = Some(match (a, b) {
                (MatrixData::F32(a), MatrixData::F32(b)) if tag == 0 => run!(a, b, F32),
                (MatrixData::F64(a), MatrixData::F64(b)) if tag == 1 => run!(a, b, F64),
                (MatrixData::C32(a), MatrixData::C32(b)) if tag == 2 => run!(a, b, C32),
                (MatrixData::C64(a), MatrixData::C64(b)) if tag == 3 => run!(a, b, C64),
                _ => return Err("QZ scalar types must match the requested scalar type".into()),
            });
            Ok(())
        })
    }
}

trait QzReorderScalar: ComplexField {
    type Alpha;
    fn values_scratch(n: usize) -> StackReq {
        if Self::IS_REAL {
            faer::linalg::temp_mat_scratch::<Self::Real>(n, 3)
        } else {
            StackReq::EMPTY
        }
    }
    fn block_size(s: MatRef<'_, Self>, i: usize) -> usize {
        if Self::IS_REAL && i + 1 < s.nrows() && s[(i + 1, i)] != zero() {
            2
        } else {
            1
        }
    }
    fn move_block(
        p: &mut Pair<Self>,
        from: usize,
        to: usize,
        stack: &mut MemStack,
    ) -> Result<(), String>;
    // SAFETY: The caller provides aligned, writable, nonoverlapping n-entry
    // Julia vectors of the corresponding Alpha/Self types and preserves them.
    unsafe fn write_values(
        p: &Pair<Self>,
        alpha: *mut u8,
        beta: *mut u8,
        stack: &mut MemStack,
    ) -> Result<(), String>;
}
macro_rules! real_qz_reorder {
    ($r:ty,$c:ty) => {
        impl QzReorderScalar for $r {
            type Alpha = $c;
            fn move_block(
                p: &mut Pair<Self>,
                from: usize,
                to: usize,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                gevd::reorder_schur_real(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    from,
                    to,
                    Par::Seq,
                    stack,
                )
                .map(|_| ())
                .map_err(|e| {
                    format!(
                        "generalized Schur reorder rejected at row {}",
                        e.position + 1
                    )
                })
            }
            unsafe fn write_values(
                p: &Pair<Self>,
                alpha: *mut u8,
                beta: *mut u8,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                let n = p.s.nrows();
                let (mut values, _) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, 3, stack);
                let (mut ab, mut be) = values.as_mat_mut().split_at_col_mut(2);
                let (ar, ai) = ab.rb_mut().two_cols_mut(0, 1);
                gevd::schur_eigenvalues_real(
                    p.s.as_ref(),
                    p.t.as_ref(),
                    ar,
                    ai,
                    be.rb_mut().col_mut(0),
                )
                .map_err(|e| {
                    format!(
                        "generalized Schur eigenvalue extraction rejected at row {}",
                        e.position + 1
                    )
                })?;
                let values = values.as_mat_ref();
                if !values.is_all_finite() {
                    return Err("nonfinite generalized Schur homogeneous eigenvalues".into());
                }
                for i in 0..n {
                    // SAFETY: checked output buffers are preserved by the Julia caller.
                    unsafe {
                        alpha
                            .cast::<$c>()
                            .add(i)
                            .write(<$c>::new(values[(i, 0)], values[(i, 1)]));
                        beta.cast::<Self>().add(i).write(values[(i, 2)]);
                    }
                }
                Ok(())
            }
        }
    };
}
real_qz_reorder!(f32, c32);
real_qz_reorder!(f64, c64);
macro_rules! complex_qz_reorder {
    ($t:ty) => {
        impl QzReorderScalar for $t {
            type Alpha = Self;
            fn move_block(
                p: &mut Pair<Self>,
                from: usize,
                to: usize,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                gevd::reorder_schur_cplx(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    from,
                    to,
                    Par::Seq,
                    stack,
                )
                .map(|_| ())
                .map_err(|e| {
                    format!(
                        "generalized Schur reorder rejected at row {}",
                        e.position + 1
                    )
                })
            }
            unsafe fn write_values(
                p: &Pair<Self>,
                alpha: *mut u8,
                beta: *mut u8,
                _stack: &mut MemStack,
            ) -> Result<(), String> {
                for i in 0..p.s.nrows() {
                    // SAFETY: checked output buffers are preserved by the Julia caller.
                    unsafe {
                        alpha.cast::<Self>().add(i).write(p.s[(i, i)]);
                        beta.cast::<Self>().add(i).write(p.t[(i, i)]);
                    }
                }
                Ok(())
            }
        }
    };
}
complex_qz_reorder!(c32);
complex_qz_reorder!(c64);

fn reorder_requirement<T: QzReorderScalar>(n: usize) -> StackReq {
    StackReq::all_of(&[
        StackReq::new::<u8>(n),
        StackReq::any_of(&[gevd::reorder_schur_scratch::<T>(n), T::values_scratch(n)]),
    ])
}
#[julia]
pub fn qz_reorder_requirements(n: usize, tag: i32, out: &mut [usize]) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| {
        let req = match tag {
            0 => reorder_requirement::<f32>(n),
            1 => reorder_requirement::<f64>(n),
            2 => reorder_requirement::<c32>(n),
            3 => reorder_requirement::<c64>(n),
            _ => return Err("invalid QZ reorder scalar tag".into()),
        };
        reorder_buffer::<usize>(out.cast(), 2)?;
        // SAFETY: the Julia caller preserves a two-entry UInt output buffer;
        // its alignment and span have been checked above.
        unsafe {
            out.write(req.size_bytes());
            out.add(1).write(req.align_bytes());
        }
        Ok(())
    })
}
fn reorder_requirement_for<T: QzReorderScalar>(s: MatRef<'_, T>) -> StackReq {
    reorder_requirement::<T>(s.nrows())
}
fn reorder_buffer<T>(p: *const u8, n: usize) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let bytes = n
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("QZ buffer byte length overflow")?;
    if p.is_null()
        || (p as usize) % std::mem::align_of::<T>() != 0
        || (p as usize).checked_add(bytes).is_none()
    {
        return Err("QZ buffer is null, unaligned or has an invalid address span".into());
    }
    Ok(())
}
fn homogeneous_buffers<A, B>(alpha: *mut u8, beta: *mut u8, n: usize) -> Result<(), String> {
    reorder_buffer::<A>(alpha, n)?;
    reorder_buffer::<B>(beta, n)?;
    if n > 0 {
        let a = alpha as usize;
        let b = beta as usize;
        // The individual span checks prove these lengths/endpoints fit usize.
        if a < b + n * std::mem::size_of::<B>() && b < a + n * std::mem::size_of::<A>() {
            return Err("QZ homogeneous output buffers must not overlap".into());
        }
    }
    Ok(())
}
// SAFETY: select, alpha and beta refer to preserved Julia buffers of n entries;
// their types, alignment and address spans are checked here. No pointer escapes.
unsafe fn reorder_pair<T: QzReorderScalar>(
    s: MatRef<'_, T>,
    t: MatRef<'_, T>,
    q: MatRef<'_, T>,
    z: MatRef<'_, T>,
    select: *const u8,
    n: usize,
    alpha: *mut u8,
    beta: *mut u8,
    stack: &mut MemStack,
) -> Result<Pair<T>, String> {
    if s.nrows() != s.ncols()
        || s.shape() != t.shape()
        || s.shape() != q.shape()
        || s.shape() != z.shape()
        || s.nrows() != n
    {
        return Err("QZ factors and selection must have matching square dimensions".into());
    }
    for matrix in [s, t, q, z] {
        finite(matrix)?;
    }
    for j in 0..n {
        for i in j + 1..n {
            if t[(i, j)] != zero() || (i > j + usize::from(T::IS_REAL) && s[(i, j)] != zero()) {
                return Err("QZ factors must be in generalized Schur form".into());
            }
        }
    }
    reorder_buffer::<u8>(select, n)?;
    homogeneous_buffers::<T::Alpha, T>(alpha, beta, n)?;
    // SAFETY: n agrees with all factor dimensions and the caller's preserved
    // Julia selection vector; empty buffers need not have a nonnull pointer.
    let source = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(select, n) }
    };
    let (mut flags, stack) = stack.make_with::<u8>(n, |i| source[i]);
    if flags.iter().any(|&flag| flag > 1) {
        return Err("QZ selection must contain Boolean values".into());
    }
    let mut here = 0;
    while here < n {
        let size = T::block_size(s, here);
        if size == 2 {
            if here + 2 < n && s[(here + 2, here + 1)] != zero() {
                return Err("QZ real conjugate blocks must not overlap".into());
            }
            let chosen = flags[here] | flags[here + 1];
            flags[here] = chosen;
            flags[here + 1] = chosen;
        }
        here += size;
    }
    let mut p = Pair {
        s: s.to_owned(),
        t: t.to_owned(),
        q: q.to_owned(),
        z: z.to_owned(),
        ascl: None,
        bscl: None,
    };
    let mut here = 0;
    let mut next = 0;
    while here < n {
        let size = T::block_size(p.s.as_ref(), here);
        if flags[here] != 0 {
            if here != next {
                T::move_block(&mut p, here, next, stack)?;
                flags[next..here + size].rotate_right(size);
            }
            next += size;
        }
        here += size;
    }
    for matrix in [&p.s, &p.t, &p.q, &p.z] {
        finite(matrix.as_ref())?;
    }
    // SAFETY: output types, length and alignment were checked before any work;
    // all factor and eigenvalue checks finish before writing either buffer.
    unsafe {
        T::write_values(&p, alpha, beta, stack)?;
    }
    Ok(p)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn qz_reorder_from(
        &mut self,
        s: &OwnedMatrix,
        t: &OwnedMatrix,
        q: &OwnedMatrix,
        z: &OwnedMatrix,
        out_t: &mut OwnedMatrix,
        out_q: &mut OwnedMatrix,
        out_z: &mut OwnedMatrix,
        select: *const u8,
        n: usize,
        alpha: *mut u8,
        beta: *mut u8,
        tag: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("QZ requested scalar type does not match its native owner".into());
            }
            // SAFETY: Julia allocates and preserves the checked n-entry buffers.
            let p = unsafe {
                reorder_pair(
                    s.matrix.as_ref(),
                    t.matrix.as_ref(),
                    q.matrix.as_ref(),
                    z.matrix.as_ref(),
                    select,
                    n,
                    alpha,
                    beta,
                    scratch.stack(reorder_requirement::<f64>(n))?,
                )?
            };
            self.matrix = p.s;
            out_t.matrix = p.t;
            out_q.matrix = p.q;
            out_z.matrix = p.z;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn qz_reorder_from(
        &mut self,
        s: &TypedMatrix,
        t: &TypedMatrix,
        q: &TypedMatrix,
        z: &TypedMatrix,
        out_t: &mut TypedMatrix,
        out_q: &mut TypedMatrix,
        out_z: &mut TypedMatrix,
        select: *const u8,
        n: usize,
        alpha: *mut u8,
        beta: *mut u8,
        tag: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! run {
                ($s:expr,$t:expr,$q:expr,$z:expr,$variant:ident) => {{
                    // SAFETY: Julia allocates and preserves the checked buffers
                    // with scalar types matching this native variant.
                    let p = unsafe {
                        reorder_pair(
                            $s.as_ref(),
                            $t.as_ref(),
                            $q.as_ref(),
                            $z.as_ref(),
                            select,
                            n,
                            alpha,
                            beta,
                            scratch.stack(reorder_requirement_for($s.as_ref()))?,
                        )?
                    };
                    out_t.data = Some(MatrixData::$variant(p.t));
                    out_q.data = Some(MatrixData::$variant(p.q));
                    out_z.data = Some(MatrixData::$variant(p.z));
                    MatrixData::$variant(p.s)
                }};
            }
            self.data = Some(match (s.data()?, t.data()?, q.data()?, z.data()?) {
                (
                    MatrixData::F32(s),
                    MatrixData::F32(t),
                    MatrixData::F32(q),
                    MatrixData::F32(z),
                ) if tag == 0 => run!(s, t, q, z, F32),
                (
                    MatrixData::F64(s),
                    MatrixData::F64(t),
                    MatrixData::F64(q),
                    MatrixData::F64(z),
                ) if tag == 1 => run!(s, t, q, z, F64),
                (
                    MatrixData::C32(s),
                    MatrixData::C32(t),
                    MatrixData::C32(q),
                    MatrixData::C32(z),
                ) if tag == 2 => run!(s, t, q, z, C32),
                (
                    MatrixData::C64(s),
                    MatrixData::C64(t),
                    MatrixData::C64(q),
                    MatrixData::C64(z),
                ) if tag == 3 => run!(s, t, q, z, C64),
                _ => return Err("QZ scalar types must match the requested scalar type".into()),
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod reorder_tests {
    use super::*;
    #[test]
    fn buffer_validation_rejects_null_alignment_and_span() {
        assert!(reorder_buffer::<f64>(std::ptr::null(), 0).is_ok());
        assert!(reorder_buffer::<f64>(std::ptr::null(), 1).is_err());
        let storage = [0u64; 2];
        let unaligned = storage.as_ptr().cast::<u8>().wrapping_add(1);
        assert!(reorder_buffer::<f64>(unaligned, 1).is_err());
        assert!(reorder_buffer::<f64>(storage.as_ptr().cast(), usize::MAX).is_err());
    }
    #[test]
    fn dimension_check_precedes_raw_buffer_access() {
        let s = Mat::<f64>::identity(2, 2);
        // SAFETY: all pointers are null deliberately; the shape/length mismatch
        // must reject before attempting any raw access or slice construction.
        let result = unsafe {
            reorder_pair(
                s.as_ref(),
                s.as_ref(),
                s.as_ref(),
                s.as_ref(),
                std::ptr::null(),
                usize::MAX,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                MemStack::new(&mut []),
            )
        };
        assert!(result.err().unwrap().contains("dimensions"));
    }
    #[test]
    fn overlapping_metadata_is_rejected_before_writes() {
        let s = Mat::<f64>::identity(2, 2);
        let mask = [0u8; 2];
        let mut output = [c64::new(42.0, 0.0); 2];
        let ptr = output.as_mut_ptr().cast::<u8>();
        // SAFETY: both spans are within output; overlap is deliberately invalid
        // and must be detected before reading or writing the output buffers.
        let result = unsafe {
            reorder_pair(
                s.as_ref(),
                s.as_ref(),
                s.as_ref(),
                s.as_ref(),
                mask.as_ptr(),
                2,
                ptr,
                ptr,
                MemStack::new(&mut []),
            )
        };
        assert!(result.err().unwrap().contains("overlap"));
        assert_eq!(output, [c64::new(42.0, 0.0); 2]);
    }
}

#[cfg(test)]
mod adoption_tests;

mod construction;
