//! Native mean/variance kernels applied to borrowed native matrices and views.
//! Only a scalar-type promotion or a full reduction of a non-contiguous view
//! stages data; every other call reads the owner storage through a `MatRef`.
use crate::guarded;
use crate::views::{source, Borrowed, Spec};
use faer::stats::{self, NanHandling};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Row, RowMut, RowRef};
use rustcall_julia_macros::julia;

/// Exact scalar conversions between the four native scalar types.
pub(crate) trait Native: ComplexField + Copy {
    const TAG: i32;
    fn parts(&self) -> (f64, f64);
    fn from_parts(re: f64, im: f64) -> Self;
    fn real_from_f64(x: f64) -> Self::Real;
    fn real_to_f64(x: &Self::Real) -> f64;
}
macro_rules! real_native {
    ($t:ty, $tag:expr) => {
        impl Native for $t {
            const TAG: i32 = $tag;
            fn parts(&self) -> (f64, f64) {
                (*self as f64, 0.0)
            }
            fn from_parts(re: f64, _: f64) -> Self {
                re as $t
            }
            fn real_from_f64(x: f64) -> $t {
                x as $t
            }
            fn real_to_f64(x: &$t) -> f64 {
                *x as f64
            }
        }
    };
}
macro_rules! complex_native {
    ($t:ty, $r:ty, $tag:expr) => {
        impl Native for $t {
            const TAG: i32 = $tag;
            fn parts(&self) -> (f64, f64) {
                (self.re as f64, self.im as f64)
            }
            fn from_parts(re: f64, im: f64) -> Self {
                <$t>::new(re as $r, im as $r)
            }
            fn real_from_f64(x: f64) -> $r {
                x as $r
            }
            fn real_to_f64(x: &$r) -> f64 {
                *x as f64
            }
        }
    };
}
real_native!(f32, 0);
real_native!(f64, 1);
complex_native!(faer::c32, f32, 2);
complex_native!(faer::c64, f64, 3);

/// Lossless promotion pairs, as produced by Julia's `promote_type`.
pub(crate) fn widens(from: i32, to: i32) -> bool {
    from == to || matches!((from, to), (0, 1) | (0, 2) | (0, 3) | (1, 3) | (2, 3))
}

/// Reduction axis of the borrowed `m x n` matrix.
pub(crate) const NONE: i32 = 0;
pub(crate) const ROWS: i32 = 1;
pub(crate) const COLS: i32 = 2;
pub(crate) const ALL: i32 = 3;

pub(crate) fn expected_len(m: usize, n: usize, axis: i32) -> Result<usize, String> {
    Ok(match axis {
        NONE => m.checked_mul(n).ok_or("statistics size overflow")?,
        ROWS => n,
        COLS => m,
        ALL => 1,
        _ => return Err("unknown statistics axis".into()),
    })
}

fn empty_mean<T: ComplexField>() -> T {
    // The same value faer gives for an empty slice: zero scaled by 1/0.
    mul_real(&zero::<T>(), &recip(&zero::<T::Real>()))
}

/// Statistics' supplied-mean arithmetic for one included entry.
fn singleton<T: Native>(x: &T, mean: &T, corrected: bool) -> T::Real {
    let value = T::real_to_f64(&abs2(&(x - mean)));
    T::real_from_f64(value / if corrected { 0.0 } else { 1.0 })
}

// faer bug (tensor4all/faer#317): the stats kernels reverse a negative output-axis stride on
// the input only, so outputs land in reversed order. Flip input, output and
// means together to hand faer a non-negative column stride.
fn mean_into<T: ComplexField>(out: RowMut<'_, T>, a: MatRef<'_, T>, policy: NanHandling) {
    if a.col_stride() < 0 {
        stats::row_mean(out.reverse_cols_mut(), a.reverse_cols(), policy);
    } else {
        stats::row_mean(out, a, policy);
    }
}
fn varm_into<T: ComplexField>(
    out: RowMut<'_, T::Real>,
    a: MatRef<'_, T>,
    means: RowRef<'_, T>,
    policy: NanHandling,
) {
    if a.col_stride() < 0 {
        stats::row_varm(
            out.reverse_cols_mut(),
            a.reverse_cols(),
            means.reverse_cols(),
            policy,
        );
    } else {
        stats::row_varm(out, a, means, policy);
    }
}

/// Column-wise reduction of `a`; one output per column.
unsafe fn columns<T: Native>(
    a: MatRef<'_, T>,
    output: *mut u8,
    means: *const u8,
    variance: bool,
    corrected: bool,
    ignore: bool,
) {
    let n = a.ncols();
    let policy = if ignore {
        NanHandling::Ignore
    } else {
        NanHandling::Propagate
    };
    if !variance {
        let out = RowMut::from_raw_parts_mut(output.cast::<T>(), n, 1);
        mean_into(out, a, policy);
        return;
    }
    let computed;
    let means = if means.is_null() {
        let mut m = Row::<T>::zeros(n);
        mean_into(m.as_mut(), a, policy);
        computed = m;
        computed.as_ref()
    } else {
        RowRef::from_raw_parts(means.cast::<T>(), n, 1)
    };
    let mut out = RowMut::from_raw_parts_mut(output.cast::<T::Real>(), n, 1);
    varm_into(out.as_mut(), a, means, policy);
    // faer divides by N-1 and returns zero for a single entry; Statistics
    // divides singleton sums by zero and scales population variances by 1/N.
    for j in 0..n {
        let col = a.col(j);
        let (count, first) = if ignore {
            let mut count = 0usize;
            let mut first = None;
            for i in 0..col.nrows() {
                if !is_nan(&col[i]) {
                    count += 1;
                    first.get_or_insert(i);
                }
            }
            (count, first)
        } else {
            (col.nrows(), (col.nrows() > 0).then_some(0))
        };
        if count == 0 {
            out[j] = nan();
        } else if count == 1 {
            out[j] = singleton(&col[first.unwrap()], &means[j], corrected);
        } else if !corrected {
            let scale = (count - 1) as f64 / count as f64;
            out[j] = T::real_from_f64(T::real_to_f64(&out[j]) * scale);
        }
    }
}

/// Elementwise statistics: every entry is its own singleton slice.
unsafe fn elements<T: Native>(
    a: MatRef<'_, T>,
    output: *mut u8,
    means: *const u8,
    variance: bool,
    corrected: bool,
    ignore: bool,
) {
    let m = a.nrows();
    for j in 0..a.ncols() {
        for i in 0..m {
            let k = i + j * m;
            let x = a[(i, j)];
            let excluded = ignore && is_nan(&x);
            if !variance {
                let value = if excluded { empty_mean::<T>() } else { x };
                output.cast::<T>().add(k).write(value);
                continue;
            }
            let value = if excluded {
                nan()
            } else {
                let mean = if means.is_null() {
                    x
                } else {
                    *means.cast::<T>().add(k)
                };
                singleton(&x, &mean, corrected)
            };
            output.cast::<T::Real>().add(k).write(value);
        }
    }
}

unsafe fn reduce<T: Native>(
    a: MatRef<'_, T>,
    output: *mut u8,
    means: *const u8,
    axis: i32,
    variance: bool,
    corrected: bool,
    ignore: bool,
) -> Result<(), String> {
    match axis {
        NONE => elements(a, output, means, variance, corrected, ignore),
        ROWS => columns(a, output, means, variance, corrected, ignore),
        COLS => columns(a.transpose(), output, means, variance, corrected, ignore),
        ALL => {
            let (m, n) = a.shape();
            let len = m.checked_mul(n).ok_or("statistics size overflow")?;
            let staged;
            // Borrow the storage as one column whenever its layout allows it.
            let column = if n == 1 {
                a
            } else if m == 1 {
                a.transpose()
            } else if len == 0 {
                MatRef::from_raw_parts(a.as_ptr(), 0, 1, 1, 0)
            } else if Some(a.col_stride()) == a.row_stride().checked_mul(m as isize) {
                MatRef::from_raw_parts(a.as_ptr(), len, 1, a.row_stride(), 0)
            } else {
                staged = Mat::from_fn(len, 1, |k, _| a[(k % m, k / m)]);
                staged.as_ref()
            };
            columns(column, output, means, variance, corrected, ignore)
        }
        _ => return Err("unknown statistics axis".into()),
    }
    Ok(())
}

/// Reduce `a` in the working scalar `W`, converting only when promoted.
unsafe fn promoted<S: Native, W: Native>(
    a: MatRef<'_, S>,
    output: *mut u8,
    means: *const u8,
    axis: i32,
    variance: bool,
    corrected: bool,
    ignore: bool,
) -> Result<(), String> {
    if S::TAG == W::TAG {
        // Same scalar type: reinterpret the borrowed view without copying.
        let a = MatRef::from_raw_parts(
            a.as_ptr().cast::<W>(),
            a.nrows(),
            a.ncols(),
            a.row_stride(),
            a.col_stride(),
        );
        return reduce(a, output, means, axis, variance, corrected, ignore);
    }
    if !widens(S::TAG, W::TAG) {
        return Err("statistics working scalar must promote the data scalar".into());
    }
    // A supplied mean promoted the scalar type; faer's kernels need one type.
    let converted = Mat::<W>::from_fn(a.nrows(), a.ncols(), |i, j| {
        let (re, im) = a[(i, j)].parts();
        W::from_parts(re, im)
    });
    reduce(
        converted.as_ref(),
        output,
        means,
        axis,
        variance,
        corrected,
        ignore,
    )
}

unsafe fn dispatch<S: Native>(
    a: MatRef<'_, S>,
    work: i32,
    output: *mut u8,
    means: *const u8,
    axis: i32,
    variance: bool,
    corrected: bool,
    ignore: bool,
) -> Result<(), String> {
    match work {
        0 => promoted::<S, f32>(a, output, means, axis, variance, corrected, ignore),
        1 => promoted::<S, f64>(a, output, means, axis, variance, corrected, ignore),
        2 => promoted::<S, faer::c32>(a, output, means, axis, variance, corrected, ignore),
        3 => promoted::<S, faer::c64>(a, output, means, axis, variance, corrected, ignore),
        _ => Err("unsupported statistics scalar".into()),
    }
}

/// Mean (`variance == false`, output scalar `W`) or variance (output
/// `W::Real`) of a native owner/view. `means` is null or holds `W` values
/// with the output length. `output_len` is checked against the shape.
#[julia]
pub fn statistics_view_reduce(
    owner: *const u8,
    desc: *const isize,
    output: *mut u8,
    output_len: usize,
    means: *const u8,
    work: i32,
    axis: i32,
    variance: bool,
    corrected: bool,
    ignore: bool,
) -> Result<(), String> {
    guarded(|| unsafe {
        let spec = Spec::read(desc)?;
        if spec.conjugate {
            return Err("statistics do not accept conjugated views".into());
        }
        let borrowed = source(owner, spec)?;
        let (m, n) = match &borrowed {
            Borrowed::F32(a) => a.shape(),
            Borrowed::F64(a) => a.shape(),
            Borrowed::C32(a) => a.shape(),
            Borrowed::C64(a) => a.shape(),
        };
        if output_len != expected_len(m, n, axis)? {
            return Err("statistics output length mismatch".into());
        }
        if !variance && !means.is_null() {
            return Err("means are only used for variances".into());
        }
        match borrowed {
            Borrowed::F32(a) => dispatch(a, work, output, means, axis, variance, corrected, ignore),
            Borrowed::F64(a) => dispatch(a, work, output, means, axis, variance, corrected, ignore),
            Borrowed::C32(a) => dispatch(a, work, output, means, axis, variance, corrected, ignore),
            Borrowed::C64(a) => dispatch(a, work, output, means, axis, variance, corrected, ignore),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run<T: Native>(
        a: MatRef<'_, T>,
        axis: i32,
        variance: bool,
        corrected: bool,
        ignore: bool,
    ) -> Vec<f64> {
        let len = expected_len(a.nrows(), a.ncols(), axis).unwrap();
        if variance {
            let mut out = vec![T::real_from_f64(0.0); len];
            unsafe {
                reduce(
                    a,
                    out.as_mut_ptr().cast(),
                    std::ptr::null(),
                    axis,
                    true,
                    corrected,
                    ignore,
                )
                .unwrap()
            };
            out.iter().map(T::real_to_f64).collect()
        } else {
            let mut out = vec![T::from_parts(0.0, 0.0); len];
            unsafe {
                reduce(
                    a,
                    out.as_mut_ptr().cast(),
                    std::ptr::null(),
                    axis,
                    false,
                    corrected,
                    ignore,
                )
                .unwrap()
            };
            out.iter().map(|x| x.parts().0).collect()
        }
    }

    #[test]
    fn reversed_views_keep_slice_order() {
        let a = Mat::<f64>::from_fn(3, 4, |i, j| (i * 10 + j * j) as f64);
        for ignore in [false, true] {
            let reversed = a.as_ref().reverse_rows().reverse_cols();
            let dense = reversed.to_owned();
            for axis in [NONE, ROWS, COLS, ALL] {
                for variance in [false, true] {
                    for corrected in [false, true] {
                        let x = run(reversed, axis, variance, corrected, ignore);
                        let y = run(dense.as_ref(), axis, variance, corrected, ignore);
                        for (x, y) in x.iter().zip(&y) {
                            assert!(
                                x.is_nan() && y.is_nan()
                                    || (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                                "{x} {y}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn singleton_and_population_scaling_match_statistics() {
        let a = Mat::<f64>::from_fn(1, 3, |_, j| [1.0, 2.0, 4.0][j]);
        let population = run(a.as_ref(), COLS, true, false, false)[0];
        assert!((population - 14.0 / 9.0).abs() < 1e-12);
        assert!(run(a.as_ref(), ROWS, true, true, false)
            .iter()
            .all(|x| x.is_nan()));
        assert_eq!(run(a.as_ref(), ROWS, true, false, false), vec![0.0; 3]);
        let empty = Mat::<f64>::zeros(0, 2);
        assert!(run(empty.as_ref(), ALL, true, true, false)[0].is_nan());
        assert!(run(empty.as_ref(), ALL, false, true, false)[0].is_nan());
    }

    #[test]
    fn promotion_converts_only_widening_pairs() {
        assert!(widens(0, 3) && widens(2, 3) && widens(1, 1));
        assert!(!widens(1, 0) && !widens(1, 2) && !widens(3, 2));
        let a = Mat::<f32>::from_fn(2, 2, |i, j| (i + 2 * j) as f32);
        let mut out = [0.0f64; 1];
        unsafe {
            promoted::<f32, f64>(
                a.as_ref(),
                out.as_mut_ptr().cast(),
                std::ptr::null(),
                ALL,
                true,
                true,
                false,
            )
            .unwrap();
            assert!(promoted::<f64, f32>(
                Mat::<f64>::zeros(1, 1).as_ref(),
                out.as_mut_ptr().cast(),
                std::ptr::null(),
                ALL,
                true,
                true,
                false,
            )
            .is_err());
        }
        assert!((out[0] - 5.0 / 3.0).abs() < 1e-12);
    }

    fn allocations<T: Native>(
        a: MatRef<'_, T>,
        work: i32,
        axis: i32,
        variance: bool,
        ignore: bool,
        means: *const u8,
    ) -> usize {
        let len = expected_len(a.nrows(), a.ncols(), axis).unwrap();
        let mut out = vec![0.0f64; 2 * len];
        let out = out.as_mut_ptr().cast::<u8>();
        crate::allocation_probe::count(|| unsafe {
            dispatch(a, work, out, means, axis, variance, true, ignore).unwrap();
        })
    }

    // Rust-side counts: borrowed same-type kernels allocate nothing; only the
    // computed means row, a promoted copy, a staged full-reduction column and
    // faer's own NaN-count buffer (ignore, per-row) remain.
    #[test]
    fn allocations_are_limited_to_required_scratch() {
        let a = Mat::<f64>::from_fn(64, 48, |i, j| (i * 3 + j) as f64);
        let means = vec![1.0f64; 2 * 64 * 48];
        let means = means.as_ptr().cast::<u8>();
        let null = std::ptr::null();
        let reversed = a.as_ref().reverse_rows().reverse_cols();
        let strided = a.as_ref().subrows(0, 32).subcols(0, 24);
        for (view, contiguous) in [(a.as_ref(), true), (reversed, true), (strided, false)] {
            for axis in [NONE, ROWS, COLS, ALL] {
                let staged = usize::from(axis == ALL && !contiguous);
                for ignore in [false, true] {
                    let faer = usize::from(ignore && axis == COLS);
                    assert_eq!(
                        allocations(view, 1, axis, false, ignore, null),
                        staged + faer
                    );
                    assert_eq!(
                        allocations(view, 1, axis, true, ignore, means),
                        staged + faer
                    );
                    let computed = usize::from(axis != NONE);
                    assert_eq!(
                        allocations(view, 1, axis, true, ignore, null),
                        staged + computed + 2 * faer
                    );
                    assert_eq!(allocations(view, 3, axis, true, ignore, means), 1 + faer);
                }
            }
        }
    }
}
