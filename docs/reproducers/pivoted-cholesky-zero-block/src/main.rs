use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt_pivoting::factor::{
    cholesky_in_place, cholesky_in_place_scratch, PivLltParams,
};
use faer::{Mat, Par};

fn main() {
    let block_size: usize = std::env::args()
        .nth(1)
        .expect("block_size")
        .parse()
        .unwrap();
    let params = PivLltParams {
        block_size,
        ..Default::default()
    };
    let mut a = Mat::<f64>::full(1, 1, 1.0);
    let mut forward = [0usize];
    let mut inverse = [0usize];
    let req = cholesky_in_place_scratch::<usize, f64>(1, Par::Seq.shape(), params.into());
    let mut memory = MemBuffer::new(req);
    eprintln!("entering kernel: block_size={block_size}, A=[1]");
    let (info, _) = cholesky_in_place(
        a.as_mut(),
        &mut forward,
        &mut inverse,
        Par::Seq,
        MemStack::new(&mut memory),
        params.into(),
    )
    .unwrap();
    assert_eq!(info.rank, 1);
    assert_eq!(a[(0, 0)], 1.0);
    println!("completed: rank={}, L=[{}]", info.rank, a[(0, 0)]);
}
