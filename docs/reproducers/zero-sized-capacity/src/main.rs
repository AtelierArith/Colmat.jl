use faer::Mat;
fn main() {
    assert_eq!(Mat::<()>::with_capacity(0, 0).shape(), (0, 0));
    assert_eq!(Mat::<f64>::with_capacity(1, 1).shape(), (0, 0));
    println!("controls: empty ZST and nonzero f64 reservation succeeded");
    let matrix = Mat::<()>::with_capacity(1, 1);
    assert_eq!(matrix.shape(), (0, 0));
}
