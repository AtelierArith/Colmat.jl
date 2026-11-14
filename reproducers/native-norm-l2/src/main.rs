use colmat::{c32, c64, Mat};

fn main() {
    let tiny32 = f32::from_bits(1);
    let tiny64 = f64::from_bits(1);
    let large32 = f32::MAX / 8.0;
    let large64 = f64::MAX / 8.0;
    let mut failed = 0;
    macro_rules! check {
        ($name:expr,$value:expr,$expected:expr,$tol:expr) => {{
            let actual = $value;
            let expected = $expected;
            let valid = actual.is_finite()
                && actual > 0.0
                && ((actual as f64) / (expected as f64) - 1.0).abs() <= $tol;
            println!(
                "{}: actual={:?}, expected={:?}, valid={}",
                $name, actual, expected, valid
            );
            if !valid {
                failed += 1;
            }
        }};
    }
    // A 7x7 constant matrix has Frobenius norm exactly 7*abs(value).
    // All expected results are finite; large targets use the scalar-rounded value.
    check!(
        "f32 positive control",
        Mat::<f32>::from_fn(7, 7, |_, _| 1.0).norm_l2(),
        7.0f32,
        1e-6
    );
    check!(
        "f64 positive control",
        Mat::<f64>::from_fn(7, 7, |_, _| 1.0).norm_l2(),
        7.0f64,
        1e-14
    );
    check!(
        "f32 subnormal",
        Mat::<f32>::from_fn(7, 7, |_, _| tiny32).norm_l2(),
        7.0 * tiny32,
        1e-6
    );
    check!(
        "f64 subnormal",
        Mat::<f64>::from_fn(7, 7, |_, _| tiny64).norm_l2(),
        7.0 * tiny64,
        1e-14
    );
    check!(
        "f32 large",
        Mat::<f32>::from_fn(7, 7, |_, _| large32).norm_l2(),
        7.0 * large32,
        1e-6
    );
    check!(
        "f64 large",
        Mat::<f64>::from_fn(7, 7, |_, _| large64).norm_l2(),
        7.0 * large64,
        1e-14
    );
    check!(
        "c32 subnormal",
        Mat::<c32>::from_fn(7, 7, |_, _| c32::new(tiny32, 0.0)).norm_l2(),
        7.0 * tiny32,
        1e-6
    );
    check!(
        "c64 subnormal",
        Mat::<c64>::from_fn(7, 7, |_, _| c64::new(tiny64, 0.0)).norm_l2(),
        7.0 * tiny64,
        1e-14
    );
    check!(
        "c32 large",
        Mat::<c32>::from_fn(7, 7, |_, _| c32::new(large32, 0.0)).norm_l2(),
        7.0 * large32,
        1e-6
    );
    check!(
        "c64 large",
        Mat::<c64>::from_fn(7, 7, |_, _| c64::new(large64, 0.0)).norm_l2(),
        7.0 * large64,
        1e-14
    );
    assert_eq!(
        failed, 0,
        "native norm_l2 violates finite representable Frobenius norms"
    );
}
