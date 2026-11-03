use faer::{c64, linalg::jacobi::JacobiRotation, mat};
fn main() {
    let z = |r, i| c64::new(r, i);
    let j = JacobiRotation {
        c: z(0.6, 0.0),
        s: z(0.0, 0.8),
    };
    let (a, b, c, d) = j.apply_on_the_right_2x2(z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0), z(1.0, 0.0));
    let actual = mat![[a, b], [c, d]];
    let expected = mat![[j.c, -j.s.conj()], [j.s, j.c]];
    let error = (&actual - &expected).norm_l2();
    println!("norm(I*J - actual)={error}");
    assert!(error < 1e-14);
}
