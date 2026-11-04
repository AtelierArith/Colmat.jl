use faer::{Mat, Par, matrix_free::{IdentityPrecond, lsmr::*}, dyn_stack::{MemBuffer, MemStack}};
fn main() {
    let mut failed = 0;
    for scale in [1e-300, 1e300] {
        let a = Mat::<f64>::from_fn(4, 4, |i,j| if i==j {scale} else {0.0});
        let b = Mat::<f64>::from_fn(4, 1, |_,_| 1.0);
        let mut x = Mat::<f64>::zeros(4, 1);
        let m = IdentityPrecond { dim: 4 };
        let mut mem = MemBuffer::new(lsmr_scratch(m, &a, 1, Par::Seq.shape()));
        let info = lsmr(x.as_mut(), m, &a, b.as_ref(), LsmrParams {max_iters:2,..Default::default()}, |_| {}, Par::Seq, MemStack::new(&mut mem));
        println!("scale={scale:e}, info={info:?}, x00={}, true_residual={}",x[(0,0)],(&a*&x-&b).norm_l2());
        let actual=(&a*&x-&b).norm_l2();
        if !actual.is_finite() || actual>1e-12 { failed+=1; }
        let expected=Mat::<f64>::from_fn(4,1,|_,_|1.0/scale);
        assert!((&a*&expected-&b).norm_l2()<1e-14,"representable reference solution");
    }
    assert_eq!(failed,0,"LSMR must not report convergence with an invalid iterate");
}
