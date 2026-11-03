use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{
    matrix_free::lsmr::{lsmr, lsmr_scratch, LsmrParams},
    Mat, Par,
};
fn main() {
    let a = Mat::<f64>::identity(2, 2);
    let b = Mat::<f64>::from_fn(2, 1, |_, _| 1.0);
    let mut x = Mat::<f64>::zeros(2, 1);
    let mut mem = MemBuffer::new(lsmr_scratch(&a, &a, 1, Par::Seq.shape()));
    let result = lsmr(
        x.as_mut(),
        &a,
        &a,
        b.as_ref(),
        LsmrParams {
            abs_tolerance: 10.0,
            rel_tolerance: 0.0,
            max_iters: 0,
            ..Default::default()
        },
        |_| {},
        Par::Seq,
        MemStack::new(&mut mem),
    );
    println!("initial normal residual={}, result={result:?}", b.norm_l2());
    assert!(result.is_ok());
}
