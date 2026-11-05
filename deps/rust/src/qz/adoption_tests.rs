//! #254 adoption: the row-zero fully-deflated window must not underflow.
//! Fixture from docs/reproducers/faer-qz-aed-sentinel.rs, whose pinned parent is
//! 19053d8. Keep #260's wrapper scratch correction while testing the merged fix.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd::{self, ComputeEigenvectors};
use faer::traits::{math_utils::*, RealField};
use faer::{Col, Mat, Par};
fn sample(i: usize, j: usize) -> f64 {
    let mut x = (i as u64 + 7).wrapping_mul(0x9E3779B97F4A7C15) ^ (j as u64 + 19);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
fn check<T: RealField>() {
    let n = 78;
    let original_a = Mat::<T>::from_fn(n, n, |i, j| {
        from_f64::<T>(if i == j {
            1.0 + i as f64 / n as f64
        } else if i == j + 1 {
            0.2
        } else if i < j {
            0.02 * sample(i, j)
        } else {
            0.0
        })
    });
    let original_b = Mat::<T>::from_fn(n, n, |i, j| {
        from_f64::<T>(if i == j {
            2.0
        } else if i < j {
            0.01 * sample(j, i)
        } else {
            0.0
        })
    });
    let mut a = original_a.clone();
    let mut b = original_b.clone();
    let mut q = Mat::<T>::identity(n, n);
    let mut z = q.clone();
    let mut re = Col::<T>::zeros(n);
    let mut im = re.clone();
    let mut beta = re.clone();
    let params = faer::auto!(T);
    let mut memory = MemBuffer::new(super::real_qz_scratch::<T>(n, params));
    gevd::qz_real::hessenberg_to_qz(
        a.as_mut(),
        b.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        re.as_mut(),
        im.as_mut(),
        beta.as_mut(),
        ComputeEigenvectors::Yes,
        Par::Seq,
        params,
        MemStack::new(&mut memory),
    );
    let pair_error = (&q * &a * z.transpose() - &original_a).norm_l2()
        + (&q * &b * z.transpose() - &original_b).norm_l2();
    let scale =
        from_f64::<T>(n as f64) * eps::<T>() * (original_a.norm_l2() + original_b.norm_l2());
    let pair_ratio = pair_error / scale;
    let unit = Mat::<T>::identity(n, n);
    let q_ratio = (q.transpose() * &q - &unit).norm_l2() / (from_f64::<T>(n as f64) * eps::<T>());
    let z_ratio = (z.transpose() * &z - &unit).norm_l2() / (from_f64::<T>(n as f64) * eps::<T>());
    assert!(
        pair_ratio < from_f64::<T>(20.0),
        "coupled ratio {pair_ratio:?}"
    );
    assert!(q_ratio < from_f64::<T>(20.0), "Q ratio {q_ratio:?}");
    assert!(z_ratio < from_f64::<T>(20.0), "Z ratio {z_ratio:?}");
}
#[test]
fn fully_deflated_row_zero_f32() {
    check::<f32>();
}
#[test]
fn fully_deflated_row_zero_f64() {
    check::<f64>();
}
