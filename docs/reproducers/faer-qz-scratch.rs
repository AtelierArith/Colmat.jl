// Executed on t4a-faer 0.29.0 / 19053d8; see faer-qz-scratch.md.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd::{self, ComputeEigenvectors};
use faer::{Col, Mat, Par, ParShape};
extern "C" fn window(_: usize, _: usize) -> usize {
    32
}
fn main() {
    let n = 96;
    let mut a = Mat::<f64>::from_fn(n, n, |i, j| {
        if i <= j + 1 {
            ((17 * i + 13 * j + 7) % 101) as f64 / 101.0 - 0.5
        } else {
            0.0
        }
    });
    let mut b = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if i < j {
            0.01 * ((7 * i + 11 * j + 3) % 97) as f64 / 97.0
        } else {
            0.0
        }
    });
    let mut q = Mat::<f64>::identity(n, n);
    let mut z = q.clone();
    let mut re = Col::<f64>::zeros(n);
    let mut im = re.clone();
    let mut beta = re.clone();
    let mut params: gevd::GeneralizedSchurParams = faer::auto!(f64);
    params.recommended_deflation_window = window;
    let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, ParShape::Seq, params);
    println!("n={n} scratch_bytes={}", req.size_bytes());
    let mut memory = MemBuffer::new(req);
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
    println!("completed");
}
