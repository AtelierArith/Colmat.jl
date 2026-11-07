// Allocation probe for `LuRef::factors` (auto/supernodal sparse LU L/U export).
//
// Counts heap allocations and bytes made by one `factors()` call and compares
// them with the two returned CSC matrices (three buffers each). Any surplus is
// the intermediate triplet vectors and their sort/dedup buffers.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams, NumericLu};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::Par;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size(), Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(n, Relaxed);
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn matrix(n: usize) -> SparseColMat<usize, f64> {
    let mut t = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if i == j {
                t.push(Triplet::new(i, j, 4.0 + i as f64));
            } else if (i + j) % 3 == 0 || i.abs_diff(j) == 1 {
                t.push(Triplet::new(i, j, ((17 * i + 13 * j) % 19) as f64 - 9.0));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &t).unwrap()
}

fn main() {
    let a = matrix(30);
    for (name, threshold) in [
        ("simplicial", SupernodalThreshold::FORCE_SIMPLICIAL),
        ("supernodal", SupernodalThreshold::FORCE_SUPERNODAL),
    ] {
        let params = LuSymbolicParams {
            supernodal_flop_ratio_threshold: threshold,
            ..Default::default()
        };
        let symbolic = factorize_symbolic_lu(a.symbolic(), params).unwrap();
        let mut numeric = NumericLu::<usize, f64>::new();
        let mut mem = MemBuffer::new(
            symbolic.factorize_numeric_lu_scratch::<f64>(Par::Seq.shape(), Default::default()),
        );
        let lu = symbolic
            .factorize_numeric_lu(
                &mut numeric,
                a.as_ref(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .unwrap();
        let (c0, b0) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
        let (l, u) = lu.factors().unwrap();
        let (count, bytes) = (COUNT.load(Relaxed) - c0, BYTES.load(Relaxed) - b0);
        let word = std::mem::size_of::<usize>();
        let output = |m: &SparseColMat<usize, f64>| {
            (m.ncols() + 1) * word + m.compute_nnz() * (word + std::mem::size_of::<f64>())
        };
        println!(
            "{name}: nnz(L)={} nnz(U)={} allocations={count} bytes={bytes} \
             output_buffers=6 output_bytes={}",
            l.compute_nnz(),
            u.compute_nnz(),
            output(&l) + output(&u),
        );
    }
}
