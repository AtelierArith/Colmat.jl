use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::gevd::{self, ComputeEigenvectors};
use faer::{Col, Mat, Par, ParShape};
fn sample(i: usize, j: usize) -> f64 {
    let mut x = (i as u64 + 7).wrapping_mul(0x9E3779B97F4A7C15) ^ (j as u64 + 19);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
fn main() {
    let n = 78;
    let mut a = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            1.0 + i as f64 / n as f64
        } else if i == j + 1 {
            0.2
        } else if i < j {
            0.02 * sample(i, j)
        } else {
            0.0
        }
    });
    let mut b = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else if i < j {
            0.01 * sample(j, i)
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
    let req = gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, ParShape::Seq, params);
    // Deliberately provide ample valid storage to isolate the AED index failure
    // from the separately reported workspace query defect (#252).
    let mut memory = MemBuffer::new(StackReq::all_of(&[req, StackReq::new::<u8>(1_048_576)]));
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
