use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{partial_svd, partial_svd_scratch, PartialEigenParams};
use faer::{Col, Mat, Par};
fn main() {
    let a = Mat::<f64>::identity(4, 4);
    let mut u = Mat::<f64>::zeros(4, 1);
    let mut v = Mat::<f64>::zeros(4, 1);
    let mut s = [0.0];
    let v0 = Col::<f64>::from_fn(4, |i| (i+1) as f64);
    let params = PartialEigenParams::default();
    let mut mem = MemBuffer::new(partial_svd_scratch(&a, 1, Par::Seq.shape(), params));
    let info = partial_svd(u.as_mut(), v.as_mut(), &mut s, &a, v0.as_ref(), 1e-10,
        Par::Seq, MemStack::new(&mut mem), params);
    assert_eq!(info.n_converged_eigen, 1);
    assert!((s[0]-1.0).abs()<1e-8);
}
