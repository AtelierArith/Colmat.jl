use faer::{c64, linalg::jacobi::JacobiRotation};
fn main() {
    let p = c64::new(3.0, 2.0);
    let (j, r) = JacobiRotation::rotg(p, c64::new(0.0, 0.0));
    println!("c={}, s={}, r={}, expected r={}", j.c, j.s, r, p);
    assert_eq!(r, p);
}
