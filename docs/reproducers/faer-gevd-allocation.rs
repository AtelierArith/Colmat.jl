// Counts heap allocations made inside faer's low-level generalized EVD
// drivers (`gevd_real`/`gevd_cplx`) when they run on exactly the queried
// `gevd_scratch` buffer. Setup (inputs, outputs, scratch) is outside the count.
// The first call of a process in each configuration may initialise lazily
// allocated library state, so it is reported separately.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors as Vectors;
use faer::linalg::gevd;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, Mat, MatMut, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct Probe;
fn record(size: usize) {
    if ACTIVE.load(Relaxed) {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(size, Relaxed);
    }
}
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record(n);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Probe = Probe;

trait Scalar: ComplexField {
    fn point(i: usize, j: usize, b: bool) -> Self;
    fn run(a: MatMut<'_, Self>, b: MatMut<'_, Self>, side: &str, s: &mut MemStack) -> usize;
}
fn real_entry(i: usize, j: usize, b: bool) -> f64 {
    if b {
        if i == j {
            2.0 + 0.1 * i as f64
        } else {
            0.05 * (((3 * i + 11 * j + 1) % 97) as f64 / 97.0 - 0.5)
        }
    } else {
        ((7 * i + 13 * j + 5) % 101) as f64 / 101.0 - 0.5
    }
}
fn outputs<T: ComplexField>(n: usize, side: &str) -> (Option<Mat<T>>, Option<Mat<T>>) {
    let l = matches!(side, "left" | "both").then(|| Mat::<T>::zeros(n, n));
    let r = matches!(side, "right" | "both").then(|| Mat::<T>::zeros(n, n));
    (l, r)
}
macro_rules! real {
    ($t:ty) => {
        impl Scalar for $t {
            fn point(i: usize, j: usize, b: bool) -> Self {
                real_entry(i, j, b) as Self
            }
            fn run(
                mut a: MatMut<'_, Self>,
                mut b: MatMut<'_, Self>,
                side: &str,
                s: &mut MemStack,
            ) -> usize {
                let n = a.nrows();
                let (mut re, mut im, mut beta) = (
                    Col::<Self>::zeros(n),
                    Col::<Self>::zeros(n),
                    Col::<Self>::zeros(n),
                );
                let (mut l, mut r) = outputs::<Self>(n, side);
                ACTIVE.store(true, Relaxed);
                gevd::gevd_real(
                    a.as_mut(),
                    b.as_mut(),
                    re.as_diagonal_mut(),
                    im.as_diagonal_mut(),
                    beta.as_diagonal_mut(),
                    l.as_mut().map(|m| m.as_mut()),
                    r.as_mut().map(|m| m.as_mut()),
                    Par::Seq,
                    s,
                    Default::default(),
                )
                .unwrap();
                ACTIVE.store(false, Relaxed);
                (0..n)
                    .filter(|&j| beta[j] != 0.0 && re[j].is_finite())
                    .count()
            }
        }
    };
}
macro_rules! complex {
    ($t:ty, $r:ty) => {
        impl Scalar for $t {
            fn point(i: usize, j: usize, b: bool) -> Self {
                let x = real_entry(i, j, b) as $r;
                Self::new(x, if b { 0.0 } else { 0.3 * x })
            }
            fn run(
                mut a: MatMut<'_, Self>,
                mut b: MatMut<'_, Self>,
                side: &str,
                s: &mut MemStack,
            ) -> usize {
                let n = a.nrows();
                let (mut alpha, mut beta) = (Col::<Self>::zeros(n), Col::<Self>::zeros(n));
                let (mut l, mut r) = outputs::<Self>(n, side);
                ACTIVE.store(true, Relaxed);
                gevd::gevd_cplx(
                    a.as_mut(),
                    b.as_mut(),
                    alpha.as_diagonal_mut(),
                    beta.as_diagonal_mut(),
                    l.as_mut().map(|m| m.as_mut()),
                    r.as_mut().map(|m| m.as_mut()),
                    Par::Seq,
                    s,
                    Default::default(),
                )
                .unwrap();
                ACTIVE.store(false, Relaxed);
                (0..n).filter(|&j| beta[j] != Self::new(0.0, 0.0)).count()
            }
        }
    };
}
real!(f32);
real!(f64);
complex!(c32, f32);
complex!(c64, f64);
fn check<T: Scalar>(name: &str, scales: [f64; 3]) {
    for n in [32, 96] {
        for side in ["values", "right", "left", "both"] {
            let flag = |on: bool| if on { Vectors::Yes } else { Vectors::No };
            let query = gevd::gevd_scratch::<T>(
                n,
                flag(matches!(side, "left" | "both")),
                flag(matches!(side, "right" | "both")),
                Par::Seq.shape(),
                Default::default(),
            );
            let mut mem = MemBuffer::new(query);
            for (case, scale) in ["ordinary", "large", "small"].into_iter().zip(scales) {
                let scale = from_f64::<T::Real>(scale);
                let a0 = Mat::<T>::from_fn(n, n, |i, j| mul_real(&T::point(i, j, false), &scale));
                let b0 = Mat::<T>::from_fn(n, n, |i, j| T::point(i, j, true));
                let mut counts = Vec::new();
                for _ in 0..4 {
                    let (mut a, mut b) = (a0.clone(), b0.clone());
                    COUNT.store(0, Relaxed);
                    BYTES.store(0, Relaxed);
                    let finite = T::run(a.as_mut(), b.as_mut(), side, MemStack::new(&mut mem));
                    assert_eq!(finite, n);
                    counts.push((COUNT.load(Relaxed), BYTES.load(Relaxed)));
                }
                println!(
                "{name} n={n:3} {side:6} {case:8} scratch={:7} B first call (allocations, bytes)={:?}, next three={:?}",
                query.size_bytes(),
                counts[0],
                &counts[1..]
            );
            }
        }
    }
}
fn main() {
    check::<f32>("f32", [1.0, 1e30, 1e-20]);
    check::<f64>("f64", [1.0, 1e300, 1e-200]);
    check::<c32>("c32", [1.0, 1e30, 1e-20]);
    check::<c64>("c64", [1.0, 1e300, 1e-200]);
}
