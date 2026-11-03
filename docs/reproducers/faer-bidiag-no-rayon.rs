use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::{householder as hh, svd::bidiag};
use faer::{Conj, Mat, Par};
fn main() {
    let (m, n, bs) = (131, 129, 1);
    let a = Mat::<f64>::from_fn(m, n, |i, j| {
        (((i * 73 + j * 31 + i * j * 7) % 1009) as f64) / 504.0 - 1.0
    });
    let mut reduced = a.clone();
    let mut hl = Mat::<f64>::zeros(bs, n);
    let mut hr = Mat::<f64>::zeros(bs, n - 1);
    let mut mem = MemBuffer::new(bidiag::bidiag_in_place_scratch::<f64>(
        m,
        n,
        Par::Seq.shape(),
        Default::default(),
    ));
    // Make the omitted write deterministic, rather than depending on heap contents.
    mem.fill(std::mem::MaybeUninit::new(0));
    bidiag::bidiag_in_place(
        reduced.as_mut(),
        hl.as_mut(),
        hr.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    );
    let mut check = a.clone();
    let mut mem = MemBuffer::new(
        hh::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(
            m, bs, n,
        ),
    );
    hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        reduced.as_ref(),
        hl.as_ref(),
        Conj::No,
        check.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    let mut mem = MemBuffer::new(
        hh::apply_block_householder_sequence_on_the_right_in_place_scratch::<f64>(n - 1, bs, m),
    );
    hh::apply_block_householder_sequence_on_the_right_in_place_with_conj(
        reduced.as_ref().submatrix(0, 1, n - 1, n - 1).transpose(),
        hr.as_ref(),
        Conj::No,
        check.as_mut().subcols_mut(1, n - 1),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    let b = Mat::<f64>::from_fn(m, n, |i, j| {
        if i == j || j == i + 1 {
            reduced[(i, j)]
        } else {
            0.0
        }
    });
    let residual = (&check - &b).norm_l2() / a.norm_l2();
    println!("relative residual: {residual:.6e}");
    assert!(residual.is_finite() && residual < 1e-10);
}
