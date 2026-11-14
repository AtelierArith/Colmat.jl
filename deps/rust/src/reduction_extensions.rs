//! BLAS value reductions and minimum-absolute-value indices.
use crate::kernels::Scalar;
use colmat::{traits::RealField, MatRef};
use paste::paste;

#[derive(Clone, Copy)]
enum Operation {
    AbsMax,
    AbsMin,
    Max,
    Min,
    Sum,
}
unsafe fn real_reduce<T: Scalar + RealField>(
    n: isize,
    x: *const T,
    inc: isize,
    operation: Operation,
) -> T {
    if n <= 0 || inc < 0 {
        return T::z();
    }
    if inc == 0 {
        return match operation {
            Operation::Sum => T::z(),
            Operation::AbsMax | Operation::AbsMin => T::real((*x).abs1()),
            _ => *x,
        };
    }
    let values = MatRef::from_raw_parts(x, n as usize, 1, inc, 0);
    match operation {
        Operation::Sum => values.sum(),
        Operation::Max => values.max().unwrap(),
        Operation::Min => values.min().unwrap(),
        Operation::AbsMax | Operation::AbsMin => {
            let absolute: Vec<T> = (0..n)
                .map(|i| T::real((*x.offset(i * inc)).abs1()))
                .collect();
            let values = MatRef::from_raw_parts(absolute.as_ptr(), n as usize, 1, 1, 0);
            if matches!(operation, Operation::AbsMax) {
                values.max().unwrap()
            } else {
                values.min().unwrap()
            }
        }
    }
}
unsafe fn index_min<T: Scalar, R: Scalar + RealField>(n: isize, x: *const T, inc: isize) -> isize {
    if n <= 0 || inc <= 0 {
        return 0;
    }
    // Round abs1 to the input's real precision before comparing. In particular,
    // a complex-f32 addition may tie after rounding even if f64 sums differ.
    let absolute: Vec<R> = (0..n)
        .map(|i| R::real((*x.offset(i * inc)).abs1()))
        .collect();
    let values = MatRef::from_raw_parts(absolute.as_ptr(), n as usize, 1, 1, 0);
    let minimum = values.min().unwrap();
    // Native min keeps a leading NaN; it has no equality match, so preserve the
    // first index in that case too. Fortran returns a 1-based logical index.
    absolute
        .iter()
        .position(|&value| value == minimum)
        .unwrap_or(0) as isize
        + 1
}
macro_rules! value_export {
    ($prefix:ident,$t:ty,$i:ty,$suffix:tt,$family:ident,$operation:ident)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$prefix $family $suffix>](n:*const $i,x:*const $t,inc:*const $i)->$t {
            real_reduce(*n as isize,x,*inc as isize,Operation::$operation)
        }
    }};
}
macro_rules! real_exports {
    ($prefix:ident,$t:ty,$i:ty,$suffix:tt) => {
        value_export!($prefix, $t, $i, $suffix, amax, AbsMax);
        value_export!($prefix, $t, $i, $suffix, amin, AbsMin);
        value_export!($prefix, $t, $i, $suffix, max, Max);
        value_export!($prefix, $t, $i, $suffix, min, Min);
        value_export!($prefix, $t, $i, $suffix, sum, Sum);
    };
}
macro_rules! index_export {
    ($prefix:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<i $prefix amin $suffix>](n:*const $i,x:*const $t,inc:*const $i)->$i {
            index_min::<$t,$r>(*n as isize,x,*inc as isize) as $i
        }
    }};
}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
index_export!(s, f32, f32, i32, _);
index_export!(d, f64, f64, i32, _);
index_export!(c, colmat::c32, f32, i32, _);
index_export!(z, colmat::c64, f64, i32, _);
index_export!(s, f32, f32, i64, _64_);
index_export!(d, f64, f64, i64, _64_);
index_export!(c, colmat::c32, f32, i64, _64_);
index_export!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null;
    #[test]
    fn unused_inputs_are_not_read() {
        unsafe {
            for op in [
                Operation::AbsMax,
                Operation::AbsMin,
                Operation::Max,
                Operation::Min,
                Operation::Sum,
            ] {
                assert_eq!(real_reduce::<f64>(0, null(), 1, op), 0.0);
                assert_eq!(real_reduce::<f64>(3, null(), -1, op), 0.0);
            }
            assert_eq!(real_reduce::<f64>(3, null(), 0, Operation::Sum), 0.0);
            assert_eq!(index_min::<colmat::c64, f64>(3, null(), 0), 0);
            assert_eq!(index_min::<colmat::c64, f64>(3, null(), -1), 0);
        }
    }
    #[test]
    fn complex_abs1_rounding_preserves_first_tie() {
        unsafe {
            let x = [
                colmat::c32::new(1.0, f32::EPSILON / 2.0),
                colmat::c32::new(1.0, 0.0),
            ];
            assert_eq!(index_min::<colmat::c32, f32>(2, x.as_ptr(), 1), 1);
        }
    }
}
