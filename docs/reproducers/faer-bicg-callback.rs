use faer::{Mat, Par, matrix_free::{IdentityPrecond, bicgstab::*}, dyn_stack::{MemBuffer, MemStack}};
fn main() {
    let a = IdentityPrecond { dim: 4 };
    let b = Mat::<f64>::from_fn(4, 1, |_, _| 1.0);
    let mut x = Mat::<f64>::zeros(4, 1);
    let mut count = 0;
    let mut mem = MemBuffer::new(bicgstab_scratch::<f64>(a, a, a, 1, Par::Seq.shape()));
    let info = bicgstab(x.as_mut(), a, a, a, b.as_ref(), BicgParams::default(), |_| { count += 1; }, Par::Seq, MemStack::new(&mut mem)).unwrap();
    println!("iter_count={}, callback_count={}, residual={}", info.iter_count, count, (&x-&b).norm_l2());
    assert_eq!(count, info.iter_count, "successful final update must reach callback");
}
