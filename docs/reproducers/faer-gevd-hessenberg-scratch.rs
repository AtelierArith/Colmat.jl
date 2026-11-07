// faer's `gevd_scratch` sizes the generalized Hessenberg step with `auto!`
// (block size 32) whatever `GevdParams::hessenberg` says, while `gevd_real`
// runs that step with the caller's params. A block size above the default
// therefore needs more stack than the public query reports.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{self, gen_hessenberg, GevdParams};
use faer::{Col, Mat, Par};

fn main() {
    let n = 200;
    let mut params: GevdParams = faer::auto!(f64);
    params.hessenberg.block_size = 64;
    let query = gevd::gevd_scratch::<f64>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::Yes,
        Par::Seq.shape(),
        params.into(),
    );
    let hessenberg = gen_hessenberg::generalized_hessenberg_scratch::<f64>(n, params.hessenberg);
    println!(
        "n = {n}, hessenberg.block_size = {}: gevd_scratch = {} B, \
         generalized_hessenberg_scratch(params.hessenberg) = {} B",
        params.hessenberg.block_size,
        query.size_bytes(),
        hessenberg.size_bytes()
    );
    let mut a = Mat::<f64>::from_fn(n, n, |i, j| {
        ((7 * i + 13 * j + 5) % 101) as f64 / 101.0 - 0.5
    });
    let mut b = Mat::<f64>::from_fn(n, n, |i, j| {
        if i == j {
            2.0
        } else {
            0.01 * (((3 * i + 11 * j + 1) % 97) as f64 / 97.0 - 0.5)
        }
    });
    let mut re = Col::<f64>::zeros(n);
    let mut im = Col::<f64>::zeros(n);
    let mut beta = Col::<f64>::zeros(n);
    let mut v = Mat::<f64>::zeros(n, n);
    let mut memory = MemBuffer::new(query);
    gevd::gevd_real(
        a.as_mut(),
        b.as_mut(),
        re.as_diagonal_mut(),
        im.as_diagonal_mut(),
        beta.as_diagonal_mut(),
        None,
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut memory),
        params.into(),
    )
    .unwrap();
    println!("gevd_real succeeded with the queried scratch");
}
