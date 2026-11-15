use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::svd::{bidiagonal_dqds, bidiagonal_dqds_scratch};

fn main() {
    let mut bad = false;
    macro_rules! check {
        ($t:ty, $small:expr) => {{
            let small = $small as $t;
            let mut d = [small, 1.0];
            let original = d;
            let e = [0.25 as $t];
            let mut memory = MemBuffer::new(bidiagonal_dqds_scratch::<$t>(2));
            let result = bidiagonal_dqds(&mut d, &e, MemStack::new(&mut memory));
            // For this scale, the O(small^2) correction is far below epsilon.
            // Use the determinant/product identity, avoiding squared tiny data.
            let expected_small = small / (1.0 as $t).hypot(0.25);
            let relative_error = (d[1] / expected_small - 1.0).abs();
            println!("{}: result={result:?}, actual={:e}, expected={:e}, relative_error={relative_error:e}", stringify!($t), d[1], expected_small);
            bad |= if result.is_ok() {
                relative_error > 20.0 * <$t>::EPSILON
            } else { d != original };
        }};
    }
    check!(f32, 1e-10); // Nearby positive control.
    check!(f32, 1e-22);
    check!(f64, 1e-100); // Nearby positive control.
    check!(f64, 1e-161);
    assert!(
        !bad,
        "dqds reported success with inaccurate small singular values"
    );
}
