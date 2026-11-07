// faer's generalized EVD returns alpha = beta = 0 (an "indeterminate" pair)
// for most eigenvalues of a regular pencil whose A has a tiny but normal norm.
// B is strictly diagonally dominant, hence nonsingular, so every eigenvalue is
// finite and every beta must be nonzero. The same pencil with A scaled by a
// power of two into the normal range is solved correctly.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors::No;
use faer::linalg::gevd;
use faer::{c64, Col, Mat, Par};

fn entry(i: usize, j: usize, b: bool) -> f64 {
    if b {
        if i == j {
            2.0 + 0.1 * i as f64
        } else {
            0.05 * (((3 * i + 11 * j + 1) % 97) as f64 / 97.0 - 0.5)
        }
    } else {
        ((7 * i + 13 * j + 5) % 101) as f64 / 101.0 - 0.5
    }
}
fn real(n: usize, scale: f64) -> usize {
    let mut a = Mat::<f64>::from_fn(n, n, |i, j| entry(i, j, false) * scale);
    let mut b = Mat::<f64>::from_fn(n, n, |i, j| entry(i, j, true));
    let (mut re, mut im, mut beta) = (Col::zeros(n), Col::zeros(n), Col::zeros(n));
    let mut mem = MemBuffer::new(gevd::gevd_scratch::<f64>(
        n,
        No,
        No,
        Par::Seq.shape(),
        Default::default(),
    ));
    gevd::gevd_real(
        a.as_mut(),
        b.as_mut(),
        re.as_diagonal_mut(),
        im.as_diagonal_mut(),
        beta.as_diagonal_mut(),
        None,
        None,
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    (0..n).filter(|&j| beta[j] == 0.0).count()
}
fn complex(n: usize, scale: f64) -> usize {
    let mut a = Mat::<c64>::from_fn(n, n, |i, j| c64::new(entry(i, j, false) * scale, 0.0));
    let mut b = Mat::<c64>::from_fn(n, n, |i, j| c64::new(entry(i, j, true), 0.0));
    let (mut alpha, mut beta) = (Col::<c64>::zeros(n), Col::<c64>::zeros(n));
    let mut mem = MemBuffer::new(gevd::gevd_scratch::<c64>(
        n,
        No,
        No,
        Par::Seq.shape(),
        Default::default(),
    ));
    gevd::gevd_cplx(
        a.as_mut(),
        b.as_mut(),
        alpha.as_diagonal_mut(),
        beta.as_diagonal_mut(),
        None,
        None,
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    (0..n).filter(|&j| beta[j] == c64::new(0.0, 0.0)).count()
}
fn main() {
    let n = 8;
    let mut failed = false;
    for (label, scale) in [
        ("2^-60", 2f64.powi(-60)),
        ("1e-200", 1e-200),
        ("1e-300", 1e-300),
    ] {
        let (r, c) = (real(n, scale), complex(n, scale));
        println!(
            "n = {n}, A scaled by {label}: zero beta real f64 = {r}/{n}, complex c64 = {c}/{n}"
        );
        failed |= r + c > 0;
    }
    assert!(
        !failed,
        "a nonsingular B must give nonzero beta for every eigenvalue"
    );
}
