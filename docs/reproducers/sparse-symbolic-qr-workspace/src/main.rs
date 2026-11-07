use faer::perm::Perm;
use faer::sparse::linalg::qr::{factorize_symbolic_qr_with_column_ordering, QrSymbolicParams};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

static ON: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);
static SIZES: [AtomicUsize; 64] = [const { AtomicUsize::new(0) }; 64];
struct Probe;
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ON.load(Relaxed) {
            let k = COUNT.fetch_add(1, Relaxed);
            if k < SIZES.len() {
                SIZES[k].store(l.size(), Relaxed);
            }
            LARGEST.fetch_max(l.size(), Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Probe = Probe;

fn main() {
    let (m, n) = (200usize, 100usize);
    let e: Vec<_> = (0..n)
        .flat_map(|j| {
            [(j, 1.0), (j + 100, 0.5), ((j * 7) % m, 0.25)].map(|(row, val)| Triplet {
                row,
                col: j,
                val,
            })
        })
        .collect();
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(m, n, &e).unwrap();
    let id: Box<[usize]> = (0..n).collect();
    let id = Perm::new_checked(id.clone(), id, n);
    // faer sizes its buffer as max(COLAMD, analysis); print both terms.
    let nnz = a.compute_nnz();
    let colamd = faer::sparse::linalg::colamd::order_scratch::<usize>(m, n, nnz).size_bytes();
    println!("m={m} n={n} nnz={nnz} colamd::order_scratch={colamd} bytes");
    let mut excess = Vec::new();
    for t in [
        SupernodalThreshold::FORCE_SIMPLICIAL,
        SupernodalThreshold::FORCE_SUPERNODAL,
    ] {
        let params = QrSymbolicParams {
            supernodal_flop_ratio_threshold: t,
            ..Default::default()
        };
        COUNT.store(0, Relaxed);
        LARGEST.store(0, Relaxed);
        ON.store(true, Relaxed);
        let s = factorize_symbolic_qr_with_column_ordering(a.symbolic(), params, Some(id.as_ref()));
        ON.store(false, Relaxed);
        let s = s.unwrap();
        let count = COUNT.load(Relaxed);
        let largest = LARGEST.load(Relaxed);
        let sizes: Vec<usize> = SIZES[..count.min(SIZES.len())]
            .iter()
            .map(|x| x.load(Relaxed))
            .collect();
        println!(
            "threshold={} len_idx={} allocations={count} largest={largest} bytes sizes={sizes:?}",
            t.0,
            s.len_idx(),
        );
        // Every returned owner array holds at most max(m, n + 1, len_idx)
        // indices; a larger allocation is the internal analysis workspace,
        // which a caller-stack variant would take from `MemStack` instead.
        let owned = std::mem::size_of::<usize>() * m.max(n + 1).max(s.len_idx());
        if largest > owned {
            excess.push((largest, owned));
        }
    }
    assert!(
        excess.is_empty(),
        "symbolic QR allocated internal workspace (largest, owned bound): {excess:?}"
    );
}
