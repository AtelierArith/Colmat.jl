use faer::linalg::householder::upgrade_householder_factor;
use faer::traits::{math_utils::*, ComplexField};
use faer::utils::team::stats;
use faer::{c32, c64, Mat, Par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
struct Counting;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
fn record(n: usize) {
    if ACTIVE.load(Ordering::SeqCst) {
        COUNT.fetch_add(1, Ordering::SeqCst);
        BYTES.fetch_add(n, Ordering::SeqCst);
    }
}
unsafe impl GlobalAlloc for Counting {
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
static GLOBAL: Counting = Counting;
trait Value: ComplexField {
    fn double(x: Self::Real) -> f64;
}
impl Value for f32 {
    fn double(x: f32) -> f64 {
        x as f64
    }
}
impl Value for f64 {
    fn double(x: f64) -> f64 {
        x
    }
}
impl Value for c32 {
    fn double(x: f32) -> f64 {
        x as f64
    }
}
impl Value for c64 {
    fn double(x: f64) -> f64 {
        x
    }
}
fn run<T: Value>(name: &str) {
    let (m, n) = (8192, 32);
    let basis = Mat::<T>::from_fn(m, n, |i, j| {
        if i < j {
            zero()
        } else if i == j {
            one()
        } else {
            from_f64(0.01 * ((i + 3 * j) as f64).sin())
        }
    });
    let tau: Vec<T> = (0..n)
        .map(|j| {
            from_f64::<T>(0.5) * from_real::<T>(&(basis.col(j).norm_l2() * basis.col(j).norm_l2()))
        })
        .collect();
    let mut factor = Mat::<T>::zeros(n, n);
    for width in [1usize, 2] {
        let pool = threadteam::Pool::builder().threads(width).build().unwrap();
        let team = pool.into_team(width).unwrap();
        let mut exec = team.exec(width);
        exec.team(|t|{
   let mut par=Par::Team(t);
   stats::enable(true);
   for _ in 0..3{factor.fill(zero());for j in 0..n{factor[(j,j)]=tau[j].clone();}upgrade_householder_factor(factor.as_mut(),basis.as_ref(),n,1,par.reborrow());}
   for repetition in 0..3{
    factor.fill(zero());for j in 0..n{factor[(j,j)]=tau[j].clone();}
    let _=stats::take();COUNT.store(0,Ordering::SeqCst);BYTES.store(0,Ordering::SeqCst);ACTIVE.store(true,Ordering::SeqCst);
    upgrade_householder_factor(factor.as_mut(),basis.as_ref(),n,1,par.reborrow());
    ACTIVE.store(false,Ordering::SeqCst);let count=COUNT.load(Ordering::SeqCst);let bytes=BYTES.load(Ordering::SeqCst);let s=stats::take();let parallel=s.run.iter().skip(2).sum::<usize>()+s.split.iter().skip(2).sum::<usize>();
    let mut error=0.0f64;
    for j in 0..n{for i in 0..j{let expected=faer::linalg::matmul::dot::inner_prod(basis.col(i).transpose(),faer::Conj::Yes,basis.col(j),faer::Conj::No);let delta=abs(&(factor[(i,j)].clone()-expected));error=error.max(T::double(delta));}}
    println!("{name} width={width} repeat={repetition} allocations={count} requested_bytes={bytes} parallel_sections={parallel} gram_abs_error={error:e}");
    assert!(error<1e-3);
   }
   stats::enable(false);
  }).unwrap();
    }
}
fn main() {
    run::<f32>("f32");
    run::<f64>("f64");
    run::<c32>("c32");
    run::<c64>("c64");
}
