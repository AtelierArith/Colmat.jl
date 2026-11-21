use colmat::Par;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

static WIDTH: AtomicUsize = AtomicUsize::new(1);
thread_local! {
    // Each calling thread owns its team: simultaneous Julia calls do not share
    // an exclusive execution token, and recursive calls can run sequentially.
    // The team is kept across BLAS calls. Rebuilding a pool and reserving its
    // members costs milliseconds (every member acknowledges and is placed), so
    // doing it once per factorization dominated short calls; a token from the
    // resident team is a lock and a dispatch.
    static TEAM: RefCell<(usize, Option<threadteam::Team>)> = const { RefCell::new((1, None)) };
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
        // Release the resident members so a single-threaded setting leaves no
        // parked threads or affinity placement behind.
        TEAM.with(|slot| {
            if let Ok(mut held) = slot.try_borrow_mut() {
                held.1 = None;
            }
        });
        return f(Par::Seq);
    }
    TEAM.with(|slot| {
        let Ok(mut held) = slot.try_borrow_mut() else {
            return f(Par::Seq);
        };
        if held.0 != n || held.1.as_ref().is_none_or(|team| team.size().get() != n) {
            held.1 = None;
            held.0 = n;
            held.1 = threadteam::Pool::builder()
                .threads(n)
                .thread_name(|i| format!("colmat-blas-{i}"))
                .acquire_timeout(std::time::Duration::from_secs(30))
                .build()
                .ok()
                .and_then(|pool| pool.into_team(n).ok());
        }
        let Some(team) = held.1.as_ref() else {
            return f(Par::Seq);
        };
        let mut exec = team.exec(n);
        exec.team(|token| f(Par::Team(token)))
            .expect("owned BLAS team")
    })
}
