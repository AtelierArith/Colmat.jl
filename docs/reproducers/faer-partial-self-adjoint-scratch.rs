//! `partial_self_adjoint_eigen_scratch` under-declares the full-subspace branch.
//! Usage: `cargo run --release -- exact` (panics) or `-- padded` (positive control).
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::matrix_free::eigen::{
    partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch, PartialEigenParams,
};
use faer::{Col, Mat, Par};

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "exact".into());
    let n = 64;
    // Symmetric with a known spectrum: A = Q diag(1..=n) Q^T, Q from a QR.
    let x = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 17) as f64 - 8.0 + (i == j) as u8 as f64 * 20.0);
    let q = x.qr(Par::Seq).compute_Q(Par::Seq);
    let d = Mat::<f64>::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
    let a = &q * &d * q.transpose();
    let k = 4;
    let params = PartialEigenParams::default();
    let mut req = partial_self_adjoint_eigen_scratch::<f64>(&a.as_ref(), k, Par::Seq.shape(), params);
    if mode == "padded" {
        req = req.and(faer::linalg::temp_mat_scratch::<f64>(n, n));
    }
    println!("mode={mode} declared bytes={}", req.size_bytes());
    let mut buf = MemBuffer::new(StackReq::new::<u8>(0).and(req));
    let mut vecs = Mat::<f64>::zeros(n, k);
    let mut vals = vec![0.0f64; k];
    let v0 = Col::<f64>::from_fn(n, |i| 1.0 + i as f64);
    let info = partial_self_adjoint_eigen(
        vecs.as_mut(),
        &mut vals,
        &a.as_ref(),
        v0.as_ref(),
        1e-12,
        Par::Seq,
        MemStack::new(&mut buf),
        params,
    );
    let converged = info.n_converged_eigen;
    let mut worst = 0.0f64;
    for j in 0..converged {
        let r = &a * vecs.col(j) - vecs.col(j) * vals[j];
        worst = worst.max(r.norm_l2());
    }
    println!("converged={converged} values={vals:?} max residual={worst:e}");
    assert!(converged == k && worst < 1e-10);
}
