//! Julia-compatible ordering for real matrix extrema, including signed zero.
use crate::{
    guarded,
    typed::{MatrixData, TypedMatrix},
};
use faer::MatRef;
use rustcall_julia_macros::julia;

macro_rules! extremum {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(a: MatRef<'_, $t>, minimum: bool) -> Result<$t, String> {
            if a.nrows() == 0 || a.ncols() == 0 {
                return Err("empty matrix has no extrema".into());
            }
            let mut value = a[(0, 0)];
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    let x = a[(i, j)];
                    if x.is_nan() {
                        return Ok(x);
                    }
                    let order = x.total_cmp(&value);
                    if (minimum && order.is_lt()) || (!minimum && order.is_gt()) {
                        value = x;
                    }
                }
            }
            Ok(value)
        }
    };
}
extremum!(extremum_f32, f32);
extremum!(extremum_f64, f64);

#[julia]
pub fn real_extremum_f32(source: &TypedMatrix, minimum: bool) -> Result<f32, String> {
    guarded(|| match source.data()? {
        MatrixData::F32(a) => extremum_f32(a.as_ref(), minimum),
        _ => Err("Float32 extrema require Float32 native storage".into()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    macro_rules! cases {
        ($test:ident, $t:ty, $function:ident) => {
            #[test]
            fn $test() {
                for values in [[0.0 as $t, -0.0], [-0.0, 0.0]] {
                    let a = Mat::from_fn(1, 2, |_, j| values[j]);
                    assert!($function(a.as_ref(), true).unwrap().is_sign_negative());
                    assert!(!$function(a.as_ref(), false).unwrap().is_sign_negative());
                }
                let a = Mat::<$t>::from_fn(3, 2, |i, j| (3 * j + i) as $t - 2.0);
                assert_eq!($function(a.as_ref(), true).unwrap(), -2.0);
                assert_eq!($function(a.as_ref(), false).unwrap(), 3.0);
                for minimum in [false, true] {
                    let nan =
                        Mat::<$t>::from_fn(
                            2,
                            2,
                            |i, j| if (i, j) == (1, 1) { <$t>::NAN } else { 1.0 },
                        );
                    assert!($function(nan.as_ref(), minimum).unwrap().is_nan());
                    assert!($function(Mat::<$t>::zeros(0, 2).as_ref(), minimum).is_err());
                }
            }
        };
    }
    cases!(real_f32_ordering, f32, extremum_f32);
    cases!(real_f64_ordering, f64, extremum_f64);
}
