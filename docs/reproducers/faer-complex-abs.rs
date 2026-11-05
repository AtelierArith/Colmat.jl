use faer_traits::{c32, c64, math_utils::abs};
fn main() {
    let mut bad = false;
    for x in [f32::from_bits(1 << 19), f32::from_bits(1 << 9)] {
        let z = c32::new(0.0, x);
        let actual = abs(&z);
        let expected = z.re.hypot(z.im);
        println!(
            "c32 x={x:e} faer_abs={actual:e} hypot={expected:e} ratio={}",
            actual / x
        );
        bad |= actual != expected;
    }
    for x in [f64::from_bits(1 << 44), f64::from_bits(1 << 4)] {
        let z = c64::new(0.0, x);
        let actual = abs(&z);
        let expected = z.re.hypot(z.im);
        println!(
            "c64 x={x:e} faer_abs={actual:e} hypot={expected:e} ratio={}",
            actual / x
        );
        bad |= actual != expected;
    }
    assert!(
        !bad,
        "faer complex magnitude disagrees with the exactly representable pure-axis magnitude"
    );
}
