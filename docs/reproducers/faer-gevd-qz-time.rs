// Wall time of faer's values-only real GEVD (`gevd_real`, default params,
// Par::Seq) on deterministic pencils whose B is strongly diagonally dominant.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors::No;
use faer::linalg::gevd;
use faer::{Col, Mat, Par};
use std::time::Instant;

fn main() {
    for n in [64, 74, 75, 80, 96, 120, 149, 150, 160] {
        let a0 = Mat::<f64>::from_fn(n, n, |i, j| {
            ((7 * i + 13 * j + 5) % 101) as f64 / 101.0 - 0.5
        });
        let b0 = Mat::<f64>::from_fn(n, n, |i, j| {
            let x = ((3 * i + 11 * j + 1) % 97) as f64 / 97.0 - 0.5;
            if i == j {
                x + 2.0 * n as f64
            } else {
                x
            }
        });
        let mut mem = MemBuffer::new(gevd::gevd_scratch::<f64>(
            n,
            No,
            No,
            Par::Seq.shape(),
            Default::default(),
        ));
        let (mut re, mut im, mut beta) = (Col::zeros(n), Col::zeros(n), Col::zeros(n));
        let mut best = f64::INFINITY;
        for _ in 0..2 {
            let (mut a, mut b) = (a0.clone(), b0.clone());
            let t = Instant::now();
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
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("n = {n:3}: gevd_real values-only {:.4} s", best);
    }
}
