use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Mat, MatMut, MatRef, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct Probe;
unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ACTIVE.load(Relaxed) {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size(), Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if ACTIVE.load(Relaxed) {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size(), Relaxed);
        }
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ACTIVE.load(Relaxed) {
            COUNT.fetch_add(1, Relaxed);
            BYTES.fetch_add(n, Relaxed);
        }
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: Probe = Probe;
trait Scalar: ComplexField {
    fn point(i: usize, j: usize) -> Self;
    fn run(t: MatRef<'_, Self>, q: MatRef<'_, Self>, x: MatMut<'_, Self>, s: &mut MemStack);
}
macro_rules! real {
    ($t:ty) => {
        impl Scalar for $t {
            fn point(i: usize, j: usize) -> Self {
                if i == j {
                    (i + 1) as Self
                } else if i < j {
                    0.03
                } else {
                    0.0
                }
            }
            fn run(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                x: MatMut<'_, Self>,
                s: &mut MemStack,
            ) {
                evd::eigenvectors_from_schur_real(t, Some(q), x, Par::Seq, s, Default::default())
                    .unwrap();
            }
        }
    };
}
macro_rules! complex {
    ($t:ty, $r:ty) => {
        impl Scalar for $t {
            fn point(i: usize, j: usize) -> Self {
                if i == j {
                    Self::new((i + 1) as $r, 0.1 as $r * i as $r)
                } else if i < j {
                    Self::new(0.03, 0.02)
                } else {
                    Self::new(0.0, 0.0)
                }
            }
            fn run(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                x: MatMut<'_, Self>,
                s: &mut MemStack,
            ) {
                evd::eigenvectors_from_schur_cplx(t, Some(q), x, Par::Seq, s, Default::default())
                    .unwrap();
            }
        }
    };
}
real!(f32);
real!(f64);
complex!(c32, f32);
complex!(c64, f64);
fn check<T: Scalar>(name: &str, scales: [f64; 3]) {
    let n = 32;
    let normal = Mat::<T>::from_fn(n, n, T::point);
    let q = Mat::<T>::identity(n, n);
    let mut x = Mat::<T>::zeros(n, n);
    let mut mem = MemBuffer::new(evd::eigenvectors_from_schur_scratch::<T>(n));
    for (case, scale) in ["ordinary", "large", "small"].into_iter().zip(scales) {
        let scale = from_f64::<T::Real>(scale);
        let a = Mat::<T>::from_fn(n, n, |i, j| mul_real(&normal[(i, j)], &scale));
        T::run(a.as_ref(), q.as_ref(), x.as_mut(), MemStack::new(&mut mem));
        for rep in 0..3 {
            COUNT.store(0, Relaxed);
            BYTES.store(0, Relaxed);
            ACTIVE.store(true, Relaxed);
            T::run(a.as_ref(), q.as_ref(), x.as_mut(), MemStack::new(&mut mem));
            ACTIVE.store(false, Relaxed);
            let (count, bytes) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
            let mut residual = &normal * &x;
            for j in 0..n {
                for i in 0..n {
                    residual[(i, j)] = sub(&residual[(i, j)], &mul(&x[(i, j)], &normal[(j, j)]));
                }
            }
            assert!(
                residual.norm_l2()
                    <= from_f64::<T::Real>(256.0 * n as f64) * eps::<T::Real>() * normal.norm_l2()
            );
            for j in 0..n {
                assert!(
                    abs(&sub(&x.col(j).norm_l2(), &one::<T::Real>()))
                        < from_f64::<T::Real>(256.0) * eps::<T::Real>()
                );
            }
            assert_eq!(count, if case == "ordinary" { 0 } else { 1 });
            println!("{name} {case} rep={rep} allocations={count} requested_bytes={bytes}");
        }
    }
}
fn main() {
    check::<f32>("f32", [1.0, 1e30, 1e-30]);
    check::<f64>("f64", [1.0, 1e300, 1e-300]);
    check::<c32>("c32", [1.0, 1e30, 1e-30]);
    check::<c64>("c64", [1.0, 1e300, 1e-300]);
}
