//! Generalized Hessenberg reduction and native real/complex QZ.
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{gevd, matmul::matmul};
use faer::traits::{math_utils::*, ComplexField, RealField};
use faer::{c32, c64, Accum, Col, Mat, MatRef, Par};
use rustcall_julia_macros::julia;

struct Pair<T: ComplexField> {
    s: Mat<T>,
    t: Mat<T>,
    q: Mat<T>,
    z: Mat<T>,
    ascl: Option<T::Real>,
    bscl: Option<T::Real>,
}
fn prepare<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Result<Pair<T>, String> {
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
        let qr = t.qr(Par::Seq);
        q = qr.compute_Q(Par::Seq);
        t = qr.R().to_owned();
        let source = s;
        s = Mat::zeros(n, n);
        matmul(
            s.as_mut(),
            Accum::Replace,
            q.adjoint(),
            source.as_ref(),
            one::<T>(),
            Par::Seq,
        );
        let params: gevd::gen_hessenberg::GeneralizedHessenbergParams = faer::auto!(T);
        let mut mem = MemBuffer::new(gevd::gen_hessenberg::generalized_hessenberg_scratch::<T>(
            n, params,
        ));
        gevd::gen_hessenberg::generalized_hessenberg(
            s.as_mut(),
            t.as_mut(),
            Some(q.as_mut()),
            Some(z.as_mut()),
            Par::Seq,
            MemStack::new(&mut mem),
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
// The inspected faer real-QZ scratch helper uses floor((n-3)/3), but the
// blocked kernel rounds that maximum AED window upward to an even size. Its
// local Sylvester solve also owns four index arrays rather than one. Derive
// the live allocation bound using those actual sizes, while retaining the
// public recommendation as a lower bound for future faer revisions.
pub(crate) fn real_qz_scratch<T: RealField>(
    n: usize,
    params: gevd::GeneralizedSchurParams,
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
        real_qz_scratch::<T>(nw, params),
        local,
        temp(nw, n),
        temp(n, nw),
    ]);
    StackReq::any_of(&[
        gevd::qz_real::hessenberg_to_qz_scratch::<T>(n, Par::Seq.shape(), params),
        StackReq::all_of(&[temp(nw, nw), temp(nw, nw), aed]),
        StackReq::all_of(&[temp(panel, panel), temp(panel, panel), temp(n, 2 * ns)]),
    ])
}

trait QzScalar: ComplexField + Sized {
    unsafe fn qz(p: &mut Pair<Self>, alpha: *mut u8, beta: *mut u8) -> Result<(), String>;
}
macro_rules! real_qz {
    ($r:ty,$c:ty) => {
        impl QzScalar for $r {
            unsafe fn qz(p: &mut Pair<Self>, alpha: *mut u8, beta: *mut u8) -> Result<(), String> {
                let n = p.s.nrows();
                if n == 0 {
                    return finish(p);
                }
                let mut ar = Col::from_fn(n, |_| Self::NAN);
                let mut ai = Col::from_fn(n, |_| Self::NAN);
                let mut be = Col::from_fn(n, |_| Self::NAN);
                let params: gevd::GeneralizedSchurParams = faer::auto!(Self);
                let mut mem = MemBuffer::new(real_qz_scratch::<Self>(n, params));
                gevd::qz_real::hessenberg_to_qz(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    ar.as_mut(),
                    ai.as_mut(),
                    be.as_mut(),
                    gevd::ComputeEigenvectors::Yes,
                    Par::Seq,
                    params,
                    MemStack::new(&mut mem),
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
            unsafe fn qz(p: &mut Pair<Self>, alpha: *mut u8, beta: *mut u8) -> Result<(), String> {
                let n = p.s.nrows();
                if n == 0 {
                    return finish(p);
                }
                let mut a = Col::from_fn(n, |_| Self::new(<Self as ComplexField>::Real::NAN, 0.0));
                let mut b = Col::from_fn(n, |_| Self::new(<Self as ComplexField>::Real::NAN, 0.0));
                let params: gevd::GeneralizedSchurParams = faer::auto!(Self);
                let mut mem = MemBuffer::new(gevd::qz_cplx::hessenberg_to_qz_scratch::<Self>(
                    n,
                    Par::Seq.shape(),
                    params,
                ));
                gevd::qz_cplx::hessenberg_to_qz(
                    p.s.as_mut(),
                    p.t.as_mut(),
                    Some(p.q.as_mut()),
                    Some(p.z.as_mut()),
                    a.as_mut(),
                    b.as_mut(),
                    gevd::ComputeEigenvectors::Yes,
                    Par::Seq,
                    params,
                    MemStack::new(&mut mem),
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
) -> Result<Pair<T>, String> {
    let mut p = if let Some((q, z)) = initial {
        from_reduced(a, b, q, z)?
    } else {
        prepare(a, b)?
    };
    if only_hessenberg {
        finish(&mut p)?;
    } else {
        T::qz(&mut p, alpha, beta)?;
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
    ) -> Result<(), String> {
        guarded(|| unsafe {
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
                    let p = compute($a.as_ref(), $b.as_ref(), init, alpha, beta, only_hessenberg)?;
                    (out_t).data = Some(MatrixData::$variant(p.t));
                    (out_q).data = Some(MatrixData::$variant(p.q));
                    (out_z).data = Some(MatrixData::$variant(p.z));
                    MatrixData::$variant(p.s)
                }};
            }
            self.data = Some(match (a, b) {
                (MatrixData::F32(a), MatrixData::F32(b)) => run!(a, b, F32),
                (MatrixData::F64(a), MatrixData::F64(b)) => run!(a, b, F64),
                (MatrixData::C32(a), MatrixData::C32(b)) => run!(a, b, C32),
                (MatrixData::C64(a), MatrixData::C64(b)) => run!(a, b, C64),
                _ => return Err("QZ scalar types must match".into()),
            });
            Ok(())
        })
    }
}
