//! Partial spectra: declared kernel/materialization scratch and scoped teams.
//!
//! Every request runs faer's native drivers. Small/full-projection self-adjoint
//! and SVD requests take faer's own full-subspace branch (operator
//! materialization plus dense EVD/SVD, tensor4all/faer#204 and tensor4all/faer#213), whose
//! workspace is part of the `*_scratch` requirement. Results are written directly
//! into the caller's span-checked buffers without Rust-side staging.
use super::{Data, Operator, OwnedLinearOperator};
use crate::teams::with_team;
use crate::views::{source, Borrowed, Spec};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
use faer::mat::AsMatMut;
use faer::matrix_free::eigen as pe;
use faer::matrix_free::{BiLinOp, LinOp};
use faer::prelude::{IntoConst, Reborrow, ReborrowMut};
use faer::traits::RealField;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, MatMut, MatRef, Par, ParShape};
use num_complex::Complex;
use rustcall_julia_macros::julia;
fn params(min: usize, max: usize, restarts: usize) -> pe::PartialEigenParams {
    let mut p = pe::PartialEigenParams::default();
    p.min_dim = min;
    p.max_dim = max;
    p.max_restarts = restarts;
    p
}
fn requirement<T: ComplexField>(
    a: &Operator<T>,
    kind: i32,
    k: usize,
    min: usize,
    max: usize,
    restarts: usize,
    p: ParShape,
) -> Result<StackReq, String> {
    let (m, n) = a.shape();
    if !(0..3).contains(&kind) || k > m.min(n) || (kind != 2 && m != n) {
        return Err("partial spectrum shape/kind/count mismatch".into());
    }
    let limit = if kind == 2 { m.min(n) } else { n };
    if min.max(32).max(k).min(limit) > max.max(64).max(k.saturating_mul(2)).min(limit) {
        return Err("partial spectrum min_dim exceeds maximum projection dimension".into());
    }
    // faer's full-subspace branch ignores max_restarts; Colmat keeps restarts=0
    // as an explicit "do not search" request with no workspace.
    if k == 0 || restarts == 0 {
        return Ok(StackReq::EMPTY);
    }
    let param = params(min, max, restarts);
    let kernel = match kind {
        0 => pe::partial_eigen_scratch(a, k, p, param),
        // faer declares the full-subspace branch's two n x n buffers itself
        // since tensor4all/faer#305 (the under-declaration reported as #301).
        1 => pe::partial_self_adjoint_eigen_scratch(a, k, p, param),
        _ => pe::partial_svd_scratch(a, k, p, param),
    };
    Ok(kernel.or(residual_requirement(a, kind, p)))
}
fn checked_requirement(
    a: &OwnedLinearOperator,
    kind: i32,
    k: usize,
    m: usize,
    n: usize,
    tag: i32,
    min: usize,
    max: usize,
    restarts: usize,
    p: ParShape,
) -> Result<StackReq, String> {
    super::apply_workspace::validate(a, m, n, tag)?;
    match a.data()? {
        Data::F32(a) => requirement(a, kind, k, min, max, restarts, p),
        Data::F64(a) => requirement(a, kind, k, min, max, restarts, p),
        Data::C32(a) => requirement(a, kind, k, min, max, restarts, p),
        Data::C64(a) => requirement(a, kind, k, min, max, restarts, p),
    }
}
fn span<T>(ptr: *const u8, count: usize) -> Result<(), String> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("partial buffer byte count overflow")?;
    if count > 0
        && (ptr.is_null()
            || (ptr as usize) % std::mem::align_of::<T>() != 0
            || (ptr as usize).checked_add(bytes).is_none()
            || bytes > isize::MAX as usize)
    {
        return Err("invalid partial spectrum buffer address/alignment/span".into());
    }
    Ok(())
}
/// Mutable column-major view of a caller buffer; empty views never touch `ptr`.
///
/// # Safety
/// For nonempty shapes `ptr` must be span/alignment checked for `rows*cols`
/// entries and must not alias any other live view.
unsafe fn view<'a, T>(ptr: *mut u8, rows: usize, cols: usize) -> MatMut<'a, T> {
    let ptr = if rows == 0 || cols == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    } else {
        ptr.cast::<T>()
    };
    MatMut::from_raw_parts_mut(ptr, rows, cols, 1, rows.max(1) as isize)
}
/// Mutable slice of a caller buffer; empty slices never touch `ptr`.
///
/// # Safety
/// Same contract as [`view`].
unsafe fn slice<'a, T>(ptr: *mut u8, len: usize) -> &'a mut [T] {
    if len == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(ptr.cast::<T>(), len)
    }
}
/// Column-major `rows x cols` view whose transpose is the caller's `cols x rows`
/// buffer, so faer's right singular vectors land in `Vt` layout.
///
/// # Safety
/// Same contract as [`view`].
unsafe fn transposed_view<'a, T>(ptr: *mut u8, rows: usize, cols: usize) -> MatMut<'a, T> {
    let ptr = if rows == 0 || cols == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    } else {
        ptr.cast::<T>()
    };
    MatMut::from_raw_parts_mut(ptr, rows, cols, cols.max(1) as isize, 1)
}
/// Independent residual workspace, used after the kernel has returned.
fn residual_requirement<T: ComplexField>(a: &Operator<T>, kind: i32, p: ParShape) -> StackReq {
    let (m, n) = a.shape();
    match kind {
        // Real operators act on the real and imaginary parts of complex vectors.
        0 if T::IS_REAL => StackReq::all_of(&[
            temp_mat_scratch::<T>(n, 2),
            temp_mat_scratch::<T>(n, 2),
            a.apply_scratch(1, p),
        ]),
        0 | 1 => StackReq::all_of(&[temp_mat_scratch::<T>(n, 1), a.apply_scratch(1, p)]),
        _ => StackReq::all_of(&[
            temp_mat_scratch::<T>(m, 1),
            temp_mat_scratch::<T>(n, 1),
            StackReq::any_of(&[a.apply_scratch(1, p), a.transpose_apply_scratch(1, p)]),
        ]),
    }
}
/// `out[j] = ||A v_j - lambda_j v_j||` for vectors of the operator scalar.
fn eigen_residuals<T: ComplexField>(
    a: &Operator<T>,
    vectors: MatRef<'_, T>,
    values: &[T],
    out: &mut [T::Real],
    mut par: Par<'_>,
    stack: &mut MemStack,
) {
    let n = vectors.nrows();
    let (mut r, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
    let mut r = r.as_mat_mut();
    for j in 0..out.len() {
        a.apply(r.rb_mut(), vectors.get(.., j..j + 1), par.reborrow(), stack);
        for i in 0..n {
            r[(i, 0)] = sub(&r[(i, 0)], &mul(&values[j], &vectors[(i, j)]));
        }
        out[j] = r.norm_l2();
    }
}
/// Residuals of complex eigenpairs of a real operator, applying `A` to the real
/// and imaginary parts so the operator is never promoted.
fn real_eigen_residuals<R: RealField>(
    a: &Operator<R>,
    vectors: MatRef<'_, Complex<R>>,
    values: &[Complex<R>],
    out: &mut [R],
    mut par: Par<'_>,
    stack: &mut MemStack,
) {
    let n = vectors.nrows();
    let (mut x, stack) = temp_mat_zeroed::<R, _, _>(n, 2, stack);
    let mut x = x.as_mat_mut();
    let (mut y, stack) = temp_mat_zeroed::<R, _, _>(n, 2, stack);
    let mut y = y.as_mat_mut();
    for j in 0..out.len() {
        for i in 0..n {
            x[(i, 0)] = copy(&vectors[(i, j)].re);
            x[(i, 1)] = copy(&vectors[(i, j)].im);
        }
        // One column at a time, so callback operators keep reusing their
        // single-column marshalling buffers.
        for c in 0..2 {
            a.apply(
                y.rb_mut().get_mut(.., c..c + 1),
                x.rb().get(.., c..c + 1),
                par.reborrow(),
                stack,
            );
        }
        let (re, im) = (&values[j].re, &values[j].im);
        // (A - lambda)(u + iv) = (Au - re u + im v) + i(Av - re v - im u).
        for i in 0..n {
            let (u, v) = (copy(&x[(i, 0)]), copy(&x[(i, 1)]));
            y[(i, 0)] = add(&sub(&y[(i, 0)], &mul(re, &u)), &mul(im, &v));
            y[(i, 1)] = sub(&sub(&y[(i, 1)], &mul(re, &v)), &mul(im, &u));
        }
        out[j] = y.norm_l2();
    }
}
/// `out[j] = max(||A v_j - s_j u_j||, ||A' u_j - s_j v_j||)`.
fn svd_residuals<T: ComplexField>(
    a: &Operator<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
    values: &[T],
    out: &mut [T::Real],
    mut par: Par<'_>,
    stack: &mut MemStack,
) {
    let (m, n) = (left.nrows(), right.nrows());
    let (mut f, stack) = temp_mat_zeroed::<T, _, _>(m, 1, stack);
    let mut f = f.as_mat_mut();
    let (mut b, stack) = temp_mat_zeroed::<T, _, _>(n, 1, stack);
    let mut b = b.as_mat_mut();
    for j in 0..out.len() {
        a.apply(f.rb_mut(), right.get(.., j..j + 1), par.reborrow(), stack);
        a.adjoint_apply(b.rb_mut(), left.get(.., j..j + 1), par.reborrow(), stack);
        for i in 0..m {
            f[(i, 0)] = sub(&f[(i, 0)], &mul(&values[j], &left[(i, j)]));
        }
        for i in 0..n {
            b[(i, 0)] = sub(&b[(i, 0)], &mul(&values[j], &right[(i, j)]));
        }
        let (x, y) = (f.norm_l2(), b.norm_l2());
        out[j] = if x >= y { x } else { y };
    }
}
macro_rules! solve {
    ($name:ident,$t:ty,$c:ty,$variant:ident,$eig:ident) => {
        fn $name(
            a: &Operator<$t>,
            kind: i32,
            k: usize,
            start: *const u8,
            start_desc: *const isize,
            tolerance: f64,
            min: usize,
            max: usize,
            restarts: usize,
            vectors: *mut u8,
            values: *mut u8,
            right: *mut u8,
            residuals: *mut u8,
            stack: &mut MemStack,
            mut par: Par<'_>,
        ) -> Result<usize, String> {
            type R = <$t as ComplexField>::Real;
            let (m, n) = a.shape();
            let tolerance = from_f64::<R>(tolerance);
            if !tolerance.is_finite() {
                return Err("partial tolerance exceeds native precision range".into());
            }
            let left_len = m.checked_mul(k).ok_or("partial output size overflow")?;
            let right_len = n.checked_mul(k).ok_or("partial output size overflow")?;
            if kind == 0 {
                span::<$c>(vectors, left_len)?;
                span::<$c>(values, k)?;
            } else {
                span::<$t>(vectors, left_len)?;
                span::<$t>(values, k)?;
            }
            if kind == 2 {
                span::<$t>(right, right_len)?;
            }
            span::<R>(residuals, k)?;
            // The start is read (in place for native and borrowed operands) into
            // one normalized native column; a null descriptor selects the
            // deterministic default 1 + i/n.
            let mut v = if start_desc.is_null() {
                Col::<$t>::from_fn(n, |i| from_f64::<$t>(1.0 + i as f64 / n.max(1) as f64))
            } else {
                // SAFETY: Julia preserves the descriptor and its owner for this call.
                let s = unsafe { Spec::read(start_desc)? };
                let src = match unsafe { source(start, s)? } {
                    Borrowed::$variant(x) => x,
                    _ => return Err("partial spectrum start scalar mismatch".into()),
                };
                let src = if src.ncols() == 1 {
                    src
                } else {
                    src.transpose()
                };
                if src.nrows() != n || src.ncols() != 1 {
                    return Err("partial spectrum start length mismatch".into());
                }
                Col::<$t>::from_fn(n, |i| {
                    if s.conjugate {
                        conj(&src[(i, 0)])
                    } else {
                        copy(&src[(i, 0)])
                    }
                })
            };
            if k == 0 || restarts == 0 {
                return Ok(0);
            }
            let norm = v.norm_l2();
            if !norm.is_finite() || norm == zero::<R>() {
                return Err("partial spectrum start must have finite nonzero norm".into());
            }
            let scale = recip(&norm);
            for i in 0..n {
                v[i] = mul_real(&v[i], &scale);
            }
            let param = params(min, max, restarts);
            // SAFETY: the Julia caller passes freshly allocated, mutually disjoint
            // output arrays of exactly these span-checked dimensions, and the start
            // vector has already been copied. faer writes only the leading converged
            // columns; the caller reads only the returned count.
            let count = unsafe {
                match kind {
                    0 => pe::partial_eigen(
                        view::<$c>(vectors, n, k),
                        slice::<$c>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par.reborrow(),
                        stack,
                        param,
                    ),
                    1 => pe::partial_self_adjoint_eigen(
                        view::<$t>(vectors, n, k),
                        slice::<$t>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par.reborrow(),
                        stack,
                        param,
                    ),
                    _ => pe::partial_svd(
                        view::<$t>(vectors, m, k),
                        transposed_view::<$t>(right, n, k),
                        slice::<$t>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par.reborrow(),
                        stack,
                        param,
                    ),
                }
            }
            .n_converged_eigen
            .min(k);
            // Independent residuals, the conjugation into Vt and the Vt row
            // compaction all happen in the caller's buffers. faer already emits
            // the converged pairs sorted by decreasing magnitude.
            // SAFETY: same span-checked, disjoint caller buffers as above; the
            // kernel's views have been dropped.
            unsafe {
                let out = &mut slice::<R>(residuals, k)[..count];
                match kind {
                    0 => $eig(
                        a,
                        view::<$c>(vectors, n, k).into_const().get(.., ..count),
                        &slice::<$c>(values, k)[..count],
                        out,
                        par.reborrow(),
                        stack,
                    ),
                    1 => eigen_residuals(
                        a,
                        view::<$t>(vectors, n, k).into_const().get(.., ..count),
                        &slice::<$t>(values, k)[..count],
                        out,
                        par.reborrow(),
                        stack,
                    ),
                    _ => {
                        let mut v = transposed_view::<$t>(right, n, k);
                        svd_residuals(
                            a,
                            view::<$t>(vectors, m, k).into_const().get(.., ..count),
                            v.rb().get(.., ..count),
                            &slice::<$t>(values, k)[..count],
                            out,
                            par.reborrow(),
                            stack,
                        );
                        if !<$t as ComplexField>::IS_REAL {
                            for j in 0..count {
                                for i in 0..n {
                                    v[(i, j)] = conj(&v[(i, j)]);
                                }
                            }
                        }
                        // Keep the leading `count` rows of the k x n buffer as a
                        // dense count x n prefix; destinations never pass sources.
                        if count < k {
                            let base = right.cast::<$t>();
                            for i in 0..n {
                                std::ptr::copy(base.add(i * k), base.add(i * count), count);
                            }
                        }
                    }
                }
            }
            Ok(count)
        }
    };
}
solve!(run_f32, f32, c32, F32, real_eigen_residuals);
solve!(run_f64, f64, c64, F64, real_eigen_residuals);
solve!(run_c32, c32, c32, C32, eigen_residuals);
solve!(run_c64, c64, c64, C64, eigen_residuals);
fn execute(
    a: &OwnedLinearOperator,
    kind: i32,
    k: usize,
    m: usize,
    n: usize,
    tag: i32,
    start: *const u8,
    start_desc: *const isize,
    tolerance: f64,
    min: usize,
    max: usize,
    restarts: usize,
    vectors: *mut u8,
    values: *mut u8,
    right: *mut u8,
    residuals: *mut u8,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<usize, String> {
    let req = checked_requirement(a, kind, k, m, n, tag, min, max, restarts, par.shape())?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err("partial tolerance must be finite and nonnegative".into());
    }
    let stack = scratch.stack(req)?;
    match a.data()? {
        Data::F32(a) => run_f32(
            a, kind, k, start, start_desc, tolerance, min, max, restarts, vectors, values, right,
            residuals, stack, par,
        ),
        Data::F64(a) => run_f64(
            a, kind, k, start, start_desc, tolerance, min, max, restarts, vectors, values, right,
            residuals, stack, par,
        ),
        Data::C32(a) => run_c32(
            a, kind, k, start, start_desc, tolerance, min, max, restarts, vectors, values, right,
            residuals, stack, par,
        ),
        Data::C64(a) => run_c64(
            a, kind, k, start, start_desc, tolerance, min, max, restarts, vectors, values, right,
            residuals, stack, par,
        ),
    }
}

#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn partial_requirements(
        &self,
        kind: i32,
        k: usize,
        m: usize,
        n: usize,
        tag: i32,
        min: usize,
        max: usize,
        restarts: usize,
        out: *mut usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                let req =
                    checked_requirement(self, kind, k, m, n, tag, min, max, restarts, par.shape())?;
                span::<usize>(out.cast(), 2)?;
                if out.is_null() {
                    return Err("null partial query output".into());
                }
                unsafe {
                    out.write(req.size_bytes());
                    out.add(1).write(req.align_bytes());
                }
                Ok(())
            })
        })
    }
    #[julia]
    pub fn partial_workspace(
        &self,
        kind: i32,
        k: usize,
        m: usize,
        n: usize,
        tag: i32,
        start: *const u8,
        start_desc: *const isize,
        tolerance: f64,
        min: usize,
        max: usize,
        restarts: usize,
        vectors: *mut u8,
        values: *mut u8,
        right: *mut u8,
        residuals: *mut u8,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            with_team(team, |par| {
                execute(
                    self, kind, k, m, n, tag, start, start_desc, tolerance, min, max, restarts,
                    vectors, values, right, residuals, scratch, par,
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::Base;
    use faer::Mat;
    /// Descriptor of a contiguous borrowed Julia vector (owner kind 4).
    fn raw_start(n: usize, tag: isize) -> [isize; 14] {
        let n = n as isize;
        [4, 0, 1, n, 0, 1, 1, 0, 0, n, 1, 1, n, tag]
    }
    #[test]
    fn non_finite_full_subspace_reports_zero_and_preserves_outputs() {
        for kind in [1, 2] {
            let op = OwnedLinearOperator {
                data: Some(Data::F64(Operator::new(Base::Dense(Mat::<f64>::from_fn(
                    4,
                    4,
                    |_, _| f64::NAN,
                ))))),
            };
            let mut info = [0usize; 2];
            op.partial_requirements(kind, 2, 4, 4, 1, 0, 0, 20, info.as_mut_ptr(), None)
                .unwrap();
            let mut scratch = OwnedScratch::new();
            scratch.scratch_reserve(info[0], info[1]).unwrap();
            let start = [1.0; 4];
            let mut vectors = [19.0; 8];
            let mut values = [23.0; 2];
            let mut right = [29.0; 8];
            let mut residuals = [31.0; 2];
            let count = op
                .partial_workspace(
                    kind,
                    2,
                    4,
                    4,
                    1,
                    start.as_ptr().cast(),
                    raw_start(4, 1).as_ptr(),
                    1e-8,
                    0,
                    0,
                    20,
                    vectors.as_mut_ptr().cast(),
                    values.as_mut_ptr().cast(),
                    right.as_mut_ptr().cast(),
                    residuals.as_mut_ptr().cast(),
                    &mut scratch,None
                )
                .unwrap();
            assert_eq!(count, 0);
            assert_eq!(vectors, [19.0; 8]);
            assert_eq!(values, [23.0; 2]);
            assert_eq!(right, [29.0; 8]);
            assert_eq!(residuals, [31.0; 2]);
        }
    }
    macro_rules! test_scalar {
        ($test:ident,$t:ty,$c:ty,$variant:ident,$tag:expr,$tol:expr) => {
            #[test]
            fn $test() {
                for width in [1, 2] {
                    let mut team = OwnedTeam::new();
                    team.team_initialize(width).unwrap();
                    for kind in 0..3 {
                        for (m, n, max) in [
                            (5, 5, 0),
                            (96, 96, 0),
                            (96, 96, 96),
                            (5, 3, 0),
                            (3, 5, 0),
                            (104, 96, 0),
                        ] {
                            if kind != 2 && m != n {
                                continue;
                            }
                            let dense = Mat::<$t>::from_fn(m, n, |i, j| {
                                if i == j {
                                    from_f64((i + 2) as f64)
                                } else {
                                    zero()
                                }
                            });
                            let op = OwnedLinearOperator {
                                data: Some(Data::$variant(Operator::new(Base::Dense(
                                    dense.clone(),
                                )))),
                            };
                            let req = team
                                .execute(|p| {
                                    checked_requirement(
                                        &op,
                                        kind,
                                        2,
                                        m,
                                        n,
                                        $tag,
                                        0,
                                        max,
                                        30,
                                        p.shape(),
                                    )
                                })
                                .unwrap()
                                .unwrap();
                            let mut scratch = OwnedScratch::new();
                            scratch
                                .scratch_reserve(req.size_bytes(), req.align_bytes())
                                .unwrap();
                            let start = vec![one::<$t>(); n];
                            let mut u = vec![zero::<$t>(); m * 2];
                            let mut uc = vec![<$c>::new(0.0, 0.0); m * 2];
                            let mut v = vec![zero::<$t>(); n * 2];
                            let mut r = vec![<$t as ComplexField>::Real::from(7u8); 2];
                            let mut s = vec![zero::<$t>(); 2];
                            let mut sc = vec![<$c>::new(0.0, 0.0); 2];
                            let left = if kind == 0 {
                                uc.as_mut_ptr().cast()
                            } else {
                                u.as_mut_ptr().cast()
                            };
                            let values = if kind == 0 {
                                sc.as_mut_ptr().cast()
                            } else {
                                s.as_mut_ptr().cast()
                            };
                            let count = team
                                .execute(|par| {
                                    execute(
                                        &op,
                                        kind,
                                        2,
                                        m,
                                        n,
                                        $tag,
                                        start.as_ptr().cast(),
                                        raw_start(n, $tag).as_ptr(),
                                        $tol,
                                        0,
                                        max,
                                        30,
                                        left,
                                        values,
                                        v.as_mut_ptr().cast(),
                                        r.as_mut_ptr().cast(),
                                        &mut scratch,
                                        par,
                                    )
                                })
                                .unwrap()
                                .unwrap();
                            assert_eq!(count, 2);
                            let mut found: Vec<f64> = if kind == 0 {
                                sc.iter().map(|z| z.re as f64).collect()
                            } else {
                                s.iter().map(|z| real(z) as f64).collect()
                            };
                            found.sort_by(|a, b| b.partial_cmp(a).unwrap());
                            assert!(
                                (found[0] - (m.min(n) + 1) as f64).abs()
                                    < 100.0 * $tol * m.max(n) as f64
                            );
                            assert!(
                                (found[1] - m.min(n) as f64).abs() < 100.0 * $tol * m.max(n) as f64
                            );
                            for j in 0..2 {
                                assert!((r[j] as f64) < 100.0 * $tol * m.max(n) as f64);
                                if kind == 0 {
                                    let residual = Mat::<$c>::from_fn(m, 1, |i, _| {
                                        let d = if i < n { (i + 2) as f64 } else { 0.0 };
                                        uc[i + j * m] * <$c>::new(d as _, 0.0)
                                            - sc[j] * uc[i + j * m]
                                    });
                                    assert!(
                                        (residual.norm_l2() as f64)
                                            < 100.0 * $tol * m.max(n) as f64
                                    );
                                } else {
                                    let residual = Mat::<$t>::from_fn(m, 1, |i, _| {
                                        let x = if kind == 2 {
                                            if i < n {
                                                // Rust writes Vt (k x n), conjugated.
                                                conj(&v[j + i * 2])
                                            } else {
                                                zero()
                                            }
                                        } else {
                                            u[i + j * m]
                                        };
                                        from_f64::<$t>((i + 2) as f64) * x - s[j] * u[i + j * m]
                                    });
                                    assert!(
                                        (residual.norm_l2() as f64)
                                            < 100.0 * $tol * m.max(n) as f64
                                    );
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    test_scalar!(partial_resources_f32, f32, c32, F32, 0, 2e-4);
    test_scalar!(partial_resources_f64, f64, c64, F64, 1, 1e-8);
    test_scalar!(partial_resources_c32, c32, c32, C32, 2, 2e-4);
    test_scalar!(partial_resources_c64, c64, c64, C64, 3, 1e-8);
    #[test]
    fn invalid_resources_preserve_partial_outputs() {
        let op = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::Dense(Mat::<f64>::identity(
                5, 5,
            ))))),
        };
        assert!(op
            .partial_requirements(1, 2, 5, 5, 1, 0, 0, 20, std::ptr::null_mut(), None)
            .is_err());
        let mut info = [0usize; 2];
        op.partial_requirements(1, 2, 5, 5, 1, 0, 0, 20, info.as_mut_ptr(), None)
            .unwrap();
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0] - 1, info[1]).unwrap();
        let start = [1.0; 5];
        let mut vectors = [19.0; 10];
        let mut values = [23.0; 2];
        let mut right = [29.0; 10];
        let mut residuals = [31.0; 2];
        assert!(op
            .partial_workspace(
                1,
                2,
                5,
                5,
                1,
                start.as_ptr().cast(),
                raw_start(5, 1).as_ptr(),
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                residuals.as_mut_ptr().cast(),
                &mut scratch, None
            )
            .is_err());
        assert_eq!(vectors, [19.0; 10]);
        assert_eq!(values, [23.0; 2]);
        assert_eq!(right, [29.0; 10]);
        scratch.scratch_reserve(info[0], info[1]).unwrap();
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.team_close().unwrap();
        assert!(op
            .partial_workspace(
                1,
                2,
                5,
                5,
                1,
                start.as_ptr().cast(),
                raw_start(5, 1).as_ptr(),
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                residuals.as_mut_ptr().cast(),
                &mut scratch, Some(&team)
            )
            .is_err());
        assert_eq!(vectors, [19.0; 10]);
        assert_eq!(values, [23.0; 2]);
        assert_eq!(right, [29.0; 10]);
        assert!(op
            .partial_workspace(
                1,
                2,
                4,
                4,
                1,
                start.as_ptr().cast(),
                raw_start(5, 1).as_ptr(),
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                residuals.as_mut_ptr().cast(),
                &mut scratch, None
            )
            .is_err());
        assert_eq!(vectors, [19.0; 10]);
        assert_eq!(values, [23.0; 2]);
    }
}
