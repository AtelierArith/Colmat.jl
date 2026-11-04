use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::gevd::{self, ComputeEigenvectors};
use faer::{Col, Mat, Par, ParShape};
fn sample(i: usize, j: usize) -> f64 {
    let mut x = (i as u64 + 7).wrapping_mul(0x9E3779B97F4A7C15) ^ (j as u64 + 19);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
extern "C" fn window(n: usize, _active: usize) -> usize {
    let w = n.saturating_sub(3) / 3;
    w + w % 2
}
fn one_norm(a: &Mat<f64>) -> f64 {
    (0..a.ncols())
        .map(|j| (0..a.nrows()).map(|i| a[(i, j)].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}
fn main() {
    let n = 18;
    let a = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            1.0 + i as f64 / n as f64
        } else if i == j + 1 {
            0.2
        } else if i < j {
            0.02 * sample(i, j)
        } else {
            0.0
        }
    });
    let b = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if i < j {
            0.01 * sample(j, i)
        } else {
            0.0
        }
    });
    let mut s = a.clone();
    let mut t = b.clone();
    let mut q = Mat::<f64>::identity(n, n);
    let mut z = q.clone();
    let mut ar = Col::<f64>::from_fn(n, |_| f64::NAN);
    let mut ai = ar.clone();
    let mut beta = ar.clone();
    let mut params: gevd::GeneralizedSchurParams = faer::auto!(f64);
    params.blocking_threshold = 15;
    let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, ParShape::Seq, params);
    println!("queried {} bytes", req.size_bytes());
    let mut memory = MemBuffer::new(req);
    gevd::qz_real::hessenberg_to_qz(
        s.as_mut(),
        t.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        ar.as_mut(),
        ai.as_mut(),
        beta.as_mut(),
        ComputeEigenvectors::Yes,
        Par::Seq,
        params,
        MemStack::new(&mut memory),
    );
    assert!(ar.is_all_finite() && ai.is_all_finite() && beta.is_all_finite());
    let ea = &q * &s * z.adjoint() - &a;
    let eb = &q * &t * z.adjoint() - &b;
    // DGET54's normalization for moderate finite pencils; no overflow scaling
    // is needed for this fixed fixture. Reconstructed factors are full matrices.
    let ratio = one_norm(&ea).max(one_norm(&eb))
        / one_norm(&a).max(one_norm(&b))
        / (2.0 * n as f64 * f64::EPSILON);
    let qe = q.adjoint() * &q - Mat::<f64>::identity(n, n);
    let ze = z.adjoint() * &z - Mat::<f64>::identity(n, n);
    println!(
        "coupled_ratio={ratio:e} Q_orthogonality={} Z_orthogonality={}",
        qe.norm_l2(),
        ze.norm_l2()
    );
    assert!(
        ratio < 20.0,
        "coupled reconstruction exceeds the LAPACK-style threshold"
    );
}
