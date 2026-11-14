use colmat::Par;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

static WIDTH: AtomicUsize = AtomicUsize::new(1);
thread_local! {
    // Each calling thread owns its pool: simultaneous Julia calls do not share
    // an exclusive execution token, and recursive calls can run sequentially.
    static POOL: RefCell<(usize, Option<threadteam::Pool>)> = const { RefCell::new((1, None)) };
}

#[no_mangle]
pub extern "C" fn colmat_blas_set_num_threads(n: i32) {
    if n > 0 {
        WIDTH.store(n as usize, Ordering::Relaxed);
    }
}
#[no_mangle]
pub extern "C" fn colmat_blas_get_num_threads() -> i32 {
    WIDTH.load(Ordering::Relaxed) as i32
}
// Register these names with LBT, which appends the symbol suffix for each ABI.
#[no_mangle]
pub extern "C" fn colmat_blas_set_num_threads64_(n: i32) {
    colmat_blas_set_num_threads(n);
}
#[no_mangle]
pub extern "C" fn colmat_blas_get_num_threads64_() -> i32 {
    colmat_blas_get_num_threads()
}

pub fn with_par<R>(f: impl FnOnce(Par<'_>) -> R) -> R {
    let n = WIDTH.load(Ordering::Relaxed);
    if n == 1 {
        return f(Par::Seq);
    }
    POOL.with(|slot| {
        let Ok(mut held) = slot.try_borrow_mut() else {
            return f(Par::Seq);
        };
        if held.0 != n {
            drop(held.1.take());
            held.0 = n;
            held.1 = threadteam::Pool::builder()
                .threads(n)
                .thread_name(|i| format!("colmat-blas-{i}"))
                .acquire_timeout(std::time::Duration::from_secs(30))
                .build()
                .ok();
        }
        let Some(pool) = held.1.take() else {
            return f(Par::Seq);
        };
        let team = match pool.into_team(n) {
            Ok(team) => team,
            Err(err) => {
                held.1 = Some(err.into_pool());
                return f(Par::Seq);
            }
        };
        let result = {
            let mut exec = team.exec(n);
            exec.team(|token| f(Par::Team(token)))
                .expect("owned BLAS team")
        };
        held.1 = Some(team.into_pool());
        result
    })
}
