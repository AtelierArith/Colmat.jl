//! Partial spectra: declared kernel/materialization scratch and scoped teams.
//!
//! Every request runs faer's native drivers. Small/full-projection self-adjoint
//! and SVD requests take faer's own full-subspace branch (operator
//! materialization plus dense EVD/SVD, tensor4all/faer#204 and tensor4all/faer#213), whose
//! workspace is part of the `*_scratch` requirement. Results are written directly
//! into the caller's span-checked buffers without Rust-side staging.
use super::{Data, Operator, OwnedLinearOperator};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::temp_mat_scratch;
use faer::matrix_free::eigen as pe;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, MatMut, Par, ParShape};
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
    Ok(match kind {
        0 => pe::partial_eigen_scratch(a, k, p, param),
        1 => {
            let req = pe::partial_self_adjoint_eigen_scratch(a, k, p, param);
            // faer's full-subspace scratch omits the n x n eigenvector buffer that
            // `self_adjoint_eigen::full::partial_full_spectrum` allocates, so an
            // exact-size buffer panics in `self_adjoint_evd` (n = 64, k = 2).
            if max.max(64).max(k.saturating_mul(2)).min(n) >= n {
                req.and(temp_mat_scratch::<T>(n, n))
            } else {
                req
            }
        }
        _ => pe::partial_svd_scratch(a, k, p, param),
    })
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
macro_rules! solve {
    ($name:ident,$t:ty,$c:ty) => {
        fn $name(
            a: &Operator<$t>,
            kind: i32,
            k: usize,
            start: *const u8,
            tolerance: f64,
            min: usize,
            max: usize,
            restarts: usize,
            vectors: *mut u8,
            values: *mut u8,
            right: *mut u8,
            stack: &mut MemStack,
            par: Par<'_>,
        ) -> Result<usize, String> {
            let (m, n) = a.shape();
            let tolerance = from_f64::<<$t as ComplexField>::Real>(tolerance);
            if !tolerance.is_finite() {
                return Err("partial tolerance exceeds native precision range".into());
            }
            let left_len = m.checked_mul(k).ok_or("partial output size overflow")?;
            let right_len = n.checked_mul(k).ok_or("partial output size overflow")?;
            span::<$t>(start, n)?;
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
            if k == 0 || restarts == 0 {
                return Ok(0);
            }
            // SAFETY: caller owns n initialized input entries, aligned/span-checked above.
            let v = Col::<$t>::from_fn(n, |i| unsafe { start.cast::<$t>().add(i).read() });
            if !v.norm_l2().is_finite() || v.norm_l2() == 0.0 {
                return Err("partial spectrum start must have finite nonzero norm".into());
            }
            let param = params(min, max, restarts);
            // SAFETY: the Julia caller passes freshly allocated, mutually disjoint
            // output arrays of exactly these span-checked dimensions, and the start
            // vector has already been copied. faer writes only the leading converged
            // columns; the caller reads only the returned count.
            let info = unsafe {
                match kind {
                    0 => pe::partial_eigen(
                        view::<$c>(vectors, n, k),
                        slice::<$c>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par,
                        stack,
                        param,
                    ),
                    1 => pe::partial_self_adjoint_eigen(
                        view::<$t>(vectors, n, k),
                        slice::<$t>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par,
                        stack,
                        param,
                    ),
                    _ => pe::partial_svd(
                        view::<$t>(vectors, m, k),
                        view::<$t>(right, n, k),
                        slice::<$t>(values, k),
                        a,
                        v.as_ref(),
                        tolerance as _,
                        par,
                        stack,
                        param,
                    ),
                }
            };
            Ok(info.n_converged_eigen.min(k))
        }
    };
}
solve!(run_f32, f32, c32);
solve!(run_f64, f64, c64);
solve!(run_c32, c32, c32);
solve!(run_c64, c64, c64);
fn execute(
    a: &OwnedLinearOperator,
    kind: i32,
    k: usize,
    m: usize,
    n: usize,
    tag: i32,
    start: *const u8,
    tolerance: f64,
    min: usize,
    max: usize,
    restarts: usize,
    vectors: *mut u8,
    values: *mut u8,
    right: *mut u8,
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
            a, kind, k, start, tolerance, min, max, restarts, vectors, values, right, stack, par,
        ),
        Data::F64(a) => run_f64(
            a, kind, k, start, tolerance, min, max, restarts, vectors, values, right, stack, par,
        ),
        Data::C32(a) => run_c32(
            a, kind, k, start, tolerance, min, max, restarts, vectors, values, right, stack, par,
        ),
        Data::C64(a) => run_c64(
            a, kind, k, start, tolerance, min, max, restarts, vectors, values, right, stack, par,
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
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
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
        tolerance: f64,
        min: usize,
        max: usize,
        restarts: usize,
        vectors: *mut u8,
        values: *mut u8,
        right: *mut u8,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| {
            let par = Par::Seq;
            execute(
                self, kind, k, m, n, tag, start, tolerance, min, max, restarts, vectors, values,
                right, scratch, par,
            )
        })
    }
    #[julia]
    pub fn partial_requirements_team(
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
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
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
            })?
        })
    }
    #[julia]
    pub fn partial_workspace_team(
        &self,
        kind: i32,
        k: usize,
        m: usize,
        n: usize,
        tag: i32,
        start: *const u8,
        tolerance: f64,
        min: usize,
        max: usize,
        restarts: usize,
        vectors: *mut u8,
        values: *mut u8,
        right: *mut u8,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
    ) -> Result<usize, String> {
        guarded(|| {
            team.execute(|par| {
                execute(
                    self, kind, k, m, n, tag, start, tolerance, min, max, restarts, vectors,
                    values, right, scratch, par,
                )
            })?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::Base;
    use faer::Mat;
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
            op.partial_requirements(kind, 2, 4, 4, 1, 0, 0, 20, info.as_mut_ptr())
                .unwrap();
            let mut scratch = OwnedScratch::new();
            scratch.scratch_reserve(info[0], info[1]).unwrap();
            let start = [1.0; 4];
            let mut vectors = [19.0; 8];
            let mut values = [23.0; 2];
            let mut right = [29.0; 8];
            let count = op
                .partial_workspace(
                    kind,
                    2,
                    4,
                    4,
                    1,
                    start.as_ptr().cast(),
                    1e-8,
                    0,
                    0,
                    20,
                    vectors.as_mut_ptr().cast(),
                    values.as_mut_ptr().cast(),
                    right.as_mut_ptr().cast(),
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(count, 0);
            assert_eq!(vectors, [19.0; 8]);
            assert_eq!(values, [23.0; 2]);
            assert_eq!(right, [29.0; 8]);
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
                                        $tol,
                                        0,
                                        max,
                                        30,
                                        left,
                                        values,
                                        v.as_mut_ptr().cast(),
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
                                                v[i + j * n]
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
            .partial_requirements(1, 2, 5, 5, 1, 0, 0, 20, std::ptr::null_mut())
            .is_err());
        let mut info = [0usize; 2];
        op.partial_requirements(1, 2, 5, 5, 1, 0, 0, 20, info.as_mut_ptr())
            .unwrap();
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0] - 1, info[1]).unwrap();
        let start = [1.0; 5];
        let mut vectors = [19.0; 10];
        let mut values = [23.0; 2];
        let mut right = [29.0; 10];
        assert!(op
            .partial_workspace(
                1,
                2,
                5,
                5,
                1,
                start.as_ptr().cast(),
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                &mut scratch
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
            .partial_workspace_team(
                1,
                2,
                5,
                5,
                1,
                start.as_ptr().cast(),
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                &mut scratch,
                &team
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
                1e-8,
                0,
                0,
                20,
                vectors.as_mut_ptr().cast(),
                values.as_mut_ptr().cast(),
                right.as_mut_ptr().cast(),
                &mut scratch
            )
            .is_err());
        assert_eq!(vectors, [19.0; 10]);
        assert_eq!(values, [23.0; 2]);
    }
}
