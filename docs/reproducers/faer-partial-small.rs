use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{
    partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch, PartialEigenParams,
};
use faer::{Col, Mat, Par};
fn main() {
    let a = Mat::<f64>::identity(4, 4);
    let mut vectors = Mat::<f64>::zeros(4, 1);
    let mut values = [0.0];
    let v0 = Col::<f64>::from_fn(4, |i| (i + 1) as f64);
    let params = PartialEigenParams::default();
    let mut mem = MemBuffer::new(partial_self_adjoint_eigen_scratch(
        &a,
        1,
        Par::Seq.shape(),
        params,
    ));
    partial_self_adjoint_eigen(
        vectors.as_mut(),
        &mut values,
        &a,
        v0.as_ref(),
        1e-10,
        Par::Seq,
        MemStack::new(&mut mem),
        params,
    );
}
