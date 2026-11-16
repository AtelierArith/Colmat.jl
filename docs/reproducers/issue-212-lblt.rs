use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::cholesky::lblt::{factor::*, solve::*};
use colmat::{diag::Diag, Mat, Par, ParShape};

fn check(n: usize, exponent: i32) -> bool {
    let s = if exponent == 0 {
        1.0
    } else {
        f64::from_bits(1u64 << 34)
    };
    assert!(s > 0.0);
    assert!(exponent == 0 || s.is_subnormal());
    let original = Mat::from_fn(n, n, |i, j| {
        if i / 2 == j / 2 && i != j {
            if n > 2 && i / 2 == n / 2 - 1 {
                1.0
            } else {
                s
            }
        } else {
            0.0
        }
    });
    let mut a = original.clone();
    let mut sub = Diag::<f64>::zeros(n);
    let mut p = vec![0usize; n];
    let mut inverse = vec![0usize; n];
    let mut params: colmat::Spec<LbltParams, f64> = Default::default();
    params.config.pivoting = PivotingStrategy::Rook;
    let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, f64>(
        n,
        ParShape::Seq,
        params,
    ));
    let (_, perm) = cholesky_in_place(
        a.as_mut(),
        sub.as_mut(),
        &mut p,
        &mut inverse,
        Par::Seq,
        MemStack::new(&mut mem),
        params,
    );
    let factor_finite = (0..n).all(|i| sub[i].is_finite() && (0..n).all(|j| a[(i, j)].is_finite()));
    let mut rhs = original;
    let mut mem = MemBuffer::new(solve_in_place_scratch::<usize, f64>(n, n, ParShape::Seq));
    solve_in_place(
        a.as_ref(),
        a.as_ref().diagonal(),
        sub.as_ref(),
        perm,
        rhs.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    let error = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .map(|(i, j)| (rhs[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs())
        .fold(0.0f64, |x, y| {
            if y.is_finite() {
                x.max(y)
            } else {
                f64::INFINITY
            }
        });
    println!("n={n} exponent={exponent} factor_finite={factor_finite} identity_error={error}");
    factor_finite && error < 1e-14
}
fn main() {
    let mut passed = true;
    for (n, exponent) in [(2, 0), (4, 0), (4, -1040), (2, -1040)] {
        passed &= check(n, exponent);
    }
    assert!(passed, "expected finite factors and identity solutions");
}
