use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd::gen_hessenberg::{self, GeneralizedHessenbergParams};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Mat, Par};
fn run<T: ComplexField>(f: T, g: T) -> bool {
    let mut a = Mat::<T>::identity(3, 3);
    a[(1, 0)] = f;
    a[(2, 0)] = g;
    let mut b = Mat::<T>::identity(3, 3);
    let mut q = b.clone();
    let mut z = b.clone();
    let params: GeneralizedHessenbergParams = faer::auto!(T);
    let mut memory = MemBuffer::new(gen_hessenberg::generalized_hessenberg_scratch::<T>(
        3, params,
    ));
    gen_hessenberg::generalized_hessenberg(
        a.as_mut(),
        b.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        Par::Seq,
        MemStack::new(&mut memory),
        params,
    );
    let finite = [&a, &b, &q, &z].into_iter().all(|m| m.is_all_finite());
    println!("dtype={} all_finite={finite}", core::any::type_name::<T>());
    finite
}
fn main() {
    let results = [
        run::<f32>(f32::from_bits(1), 1.0),
        run::<f64>(1e-320, 1.0),
        run::<c32>(
            c32::new(0.0, 0.0),
            c32::new(f32::from_bits(1), f32::from_bits(1)),
        ),
        run::<c64>(c64::new(0.0, 0.0), c64::new(1e-320, 1e-320)),
    ];
    assert!(
        results.into_iter().all(|x| x),
        "finite input produced nonfinite factors"
    );
}
