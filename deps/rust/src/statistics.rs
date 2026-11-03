//! Native mean/variance kernels for exact Julia primitive scalar buffers.
use crate::guarded;
use faer::stats::{self, NanHandling};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Col, MatRef, Row};
use rustcall_julia_macros::julia;
unsafe fn run<T: ComplexField>(
    input: *const u8,
    m: usize,
    n: usize,
    output: *mut u8,
    means: *const u8,
    axis: i32,
    variance: bool,
    ignore: bool,
) -> Result<(), String> {
    let a = MatRef::from_raw_parts(input.cast::<T>(), m, n, 1, m as isize);
    let policy = if ignore {
        NanHandling::Ignore
    } else {
        NanHandling::Propagate
    };
    if axis == 1 {
        if variance {
            let means = faer::RowRef::from_raw_parts(means.cast::<T>(), n, 1);
            let mut out = Row::<T::Real>::zeros(n);
            stats::row_varm(out.as_mut(), a, means, policy);
            for j in 0..n {
                output.cast::<T::Real>().add(j).write(copy(&out[j]));
            }
        } else {
            let mut out = Row::<T>::zeros(n);
            stats::row_mean(out.as_mut(), a, policy);
            for j in 0..n {
                output.cast::<T>().add(j).write(copy(&out[j]));
            }
        }
    } else if axis == 2 {
        if variance {
            let means = faer::ColRef::from_raw_parts(means.cast::<T>(), m, 1);
            let mut out = Col::<T::Real>::zeros(m);
            stats::col_varm(out.as_mut(), a, means, policy);
            for i in 0..m {
                output.cast::<T::Real>().add(i).write(copy(&out[i]));
            }
        } else {
            let mut out = Col::<T>::zeros(m);
            stats::col_mean(out.as_mut(), a, policy);
            for i in 0..m {
                output.cast::<T>().add(i).write(copy(&out[i]));
            }
        }
    } else {
        return Err("statistics axis must be 1 or 2".into());
    }
    Ok(())
}
#[julia]
pub fn statistics_reduce(
    input: *const u8,
    m: usize,
    n: usize,
    output: *mut u8,
    means: *const u8,
    tag: i32,
    axis: i32,
    variance: bool,
    ignore: bool,
) -> Result<(), String> {
    guarded(|| unsafe {
        match tag {
            0 => run::<f32>(input, m, n, output, means, axis, variance, ignore),
            1 => run::<f64>(input, m, n, output, means, axis, variance, ignore),
            2 => run::<faer::c32>(input, m, n, output, means, axis, variance, ignore),
            3 => run::<faer::c64>(input, m, n, output, means, axis, variance, ignore),
            _ => Err("unsupported statistics scalar".into()),
        }
    })
}
