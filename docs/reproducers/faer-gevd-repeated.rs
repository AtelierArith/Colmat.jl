use faer::{Mat, Par};
fn main() {
    let a = Mat::<f64>::identity(2, 2);
    let b = Mat::<f64>::identity(2, 2);
    let f = a.generalized_eigen(&b, Par::Seq).unwrap();
    println!("vectors={:?}", f.U());
    assert!(f.U().is_all_finite(), "identity pair must have finite eigenvectors");
}
