// faer's sparse symbolic LU/QR with an explicit column ordering still sizes
// (and allocates) the COLAMD workspace that only the ordering-less path uses.
use faer::sparse::linalg::SupernodalThreshold;
use faer::perm::Perm;
use faer::sparse::linalg::{colamd, lu, qr};
use faer::sparse::SymbolicSparseColMat;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static LARGEST: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LARGEST.fetch_max(l.size(), Relaxed);
        TOTAL.fetch_add(l.size(), Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;
fn reset() {
    LARGEST.store(0, Relaxed);
    TOTAL.store(0, Relaxed);
}

fn main() {
    let n = 2000usize;
    // Deterministic pattern: diagonal plus three off-diagonals per column.
    let mut cols: Vec<Vec<usize>> = (0..n)
        .map(|j| {
            let mut c = vec![j, (j * 7 + 3) % n, (j * 13 + 5) % n, (j * 31 + 11) % n];
            c.sort();
            c.dedup();
            c
        })
        .collect();
    let mut ptr = vec![0usize];
    let mut idx = Vec::new();
    for c in cols.drain(..) {
        idx.extend(c);
        ptr.push(idx.len());
    }
    let nnz = idx.len();
    let a = SymbolicSparseColMat::new_checked(n, n, ptr, None, idx);
    let identity = Perm::<usize>::new_checked(
        (0..n).collect::<Vec<_>>().into_boxed_slice(),
        (0..n).collect::<Vec<_>>().into_boxed_slice(),
        n,
    );
    let colamd_bytes = colamd::order_scratch::<usize>(n, n, nnz).size_bytes();
    println!("n = {n}, nnz = {nnz}, colamd::order_scratch = {colamd_bytes} B");
    for supernodal in [false, true] {
        let threshold = if supernodal {
            SupernodalThreshold::FORCE_SUPERNODAL
        } else {
            SupernodalThreshold::FORCE_SIMPLICIAL
        };
        let mut p = lu::LuSymbolicParams::default();
        p.supernodal_flop_ratio_threshold = threshold;
        reset();
        let _ = lu::factorize_symbolic_lu_with_column_ordering(a.as_ref(), p, Some(identity.as_ref())).unwrap();
        println!("LU {}: explicit ordering -> largest allocation {} B, total {} B",
            if supernodal { "supernodal" } else { "simplicial" }, LARGEST.load(Relaxed), TOTAL.load(Relaxed));
        let mut q = qr::QrSymbolicParams::default();
        q.supernodal_flop_ratio_threshold = threshold;
        reset();
        let _ = qr::factorize_symbolic_qr_with_column_ordering(a.as_ref(), q, Some(identity.as_ref())).unwrap();
        println!("QR {}: explicit ordering -> largest allocation {} B, total {} B",
            if supernodal { "supernodal" } else { "simplicial" }, LARGEST.load(Relaxed), TOTAL.load(Relaxed));
    }
}
