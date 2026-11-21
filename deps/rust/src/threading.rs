use colmat::Par;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Work (real element updates; one complex update counts four) at or below
/// which the corresponding operation runs on `Par::Seq`.
///
/// Entering the resident team opens a root scope: the leader reads and sets its
/// affinity, every reserved member acknowledges and places itself, and the
/// leader restores its mask on exit. That rendezvous costs about 8 us at 4
/// threads and 9 us at 8 threads, independent of the operation, so a call whose
/// parallel work is shorter than that is faster without a team. The limits
/// below are the measured crossovers on the benchmark host (Zen 4, 4 and 8
/// threads, f64/c64) rounded to the sequential side, so a shape in the
/// benchmark suite never takes the slower path:
///
/// * `gemm`: `m*n*k`; n=64 (262k) is 2x faster sequentially, n=96 (884k) is a
///   tie at 4 threads and 1.3x faster threaded at 8. The limit keeps n=96
///   threaded.
/// * `lu`: `m*n*min(m,n)`; n=160 (4.1M) is 12% faster sequentially, n=192
///   (7.1M) is a 5% win threaded, n=256 (16.8M) is 19% faster threaded.
/// * `cholesky`: `n^3`; n=192 (7.1M) is a tie, n=256 (16.8M) is 25% faster
///   threaded.
/// * `trsm`/triangular solve: `m*n*min(m,n)`; n=96 (884k) is 6% faster
///   sequentially, n=128 (2.1M) is 15% faster threaded.
/// * one QR panel: `rows*cols*block`; n=224 panels (1.37M) are 8% faster
///   sequentially, n=256 panels (1.8M) are a tie, n=320+ are faster threaded.
///
/// A `par::Team` is still used for every larger call, so the pinned team's
/// throughput advantage above these sizes is unchanged.
pub const GEMM_WORK: usize = 700_000;
pub const LU_WORK: usize = 8_000_000;
pub const CHOLESKY_WORK: usize = 9_000_000;
pub const TRSM_WORK: usize = 2_000_000;
pub const QR_PANEL_WORK: usize = 1_500_000;
/// One dense self-adjoint eigendecomposition (reduction, tridiagonal solve and
/// back-transformation), estimated as `n^3` real updates.
///
/// The resident team's rendezvous is the dominant cost of a 32-square
/// decomposition: measured on the benchmark host, n=32 runs about 5% faster
/// sequentially at 4 and 8 threads, while n=48 already pays the entry back at 8
/// threads and n=64 is 20-30% faster threaded. Complex scalars have their own
/// limit because a complex 32-square is only at the fixed-cost boundary, not
/// past it, while a complex 48-square is clearly threaded.
pub const SELFADJOINT_WORK: usize = 100_000;
pub const SELFADJOINT_WORK_COMPLEX: usize = 300_000;
/// One dense SVD (bidiagonalization, bidiagonal solve and vector
/// back-transformation), estimated as `m*n*min(m,n)` real updates.
///
/// Like the self-adjoint driver, a 32-square SVD is slower on the resident team
/// (the entry alone is about 9 us, more than the factorization takes); 48
/// squares already pay it back at 8 threads and 64 squares gain 5-15%.
pub const SVD_WORK: usize = 100_000;
pub const SVD_WORK_COMPLEX: usize = 300_000;
/// One general eigendecomposition (Hessenberg, Schur and eigenvectors).
///
/// The Schur iteration is sequential for `n < 75` (`lahqr`), and the parallel
/// sections of the reduction and eigenvector back substitution are short
/// relative to the resident team's per-call rendezvous until the matrix is
/// large. Measured on the benchmark host, serial wins at n=256, threading wins
/// from n=512 (about 6% with 4 threads and 15% with 8), and n=1024 is 10%
/// faster threaded with both. The limit therefore keeps n<=256 sequential and
/// threads the larger suite sizes.
pub const GENERAL_EVD_WORK: usize = 100_000_000;

/// Element updates of a dense multiply-like operation, with four per update
/// for a complex scalar. Saturates instead of overflowing on absurd shapes.
pub fn real_work(a: usize, b: usize, c: usize, complex: bool) -> usize {
    a.saturating_mul(b)
        .saturating_mul(c)
        .saturating_mul(if complex { 4 } else { 1 })
}

static WIDTH: AtomicUsize = AtomicUsize::new(1);
thread_local! {
    // Each calling thread owns its team: simultaneous Julia calls do not share
    // an exclusive execution token, and recursive calls can run sequentially.
    // The team is kept across BLAS calls. Rebuilding a pool and reserving its
    // members costs milliseconds (every member acknowledges and is placed), so
    // doing it once per factorization dominated short calls; a token from the
    // resident team is a lock and a dispatch.
    static TEAM: RefCell<(usize, Option<threadteam::Team>)> = const { RefCell::new((1, None)) };
    // Nonzero while `with_par_if` runs a call whose work is below its limit.
    // Every nested `with_par` in that call must then stay sequential: taking an
    // outer team scope and then letting an inner helper acquire its own token
    // would either deadlock (same thread re-entering the team gate) or pay the
    // entry the fast path is avoiding.
    static SERIAL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct SerialScope;

impl SerialScope {
    fn enter() -> Self {
        SERIAL_DEPTH.with(|depth| depth.set(depth.get() + 1));
        SerialScope
    }
}

impl Drop for SerialScope {
    fn drop(&mut self) {
        SERIAL_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
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
    if SERIAL_DEPTH.with(|depth| depth.get()) != 0 {
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

/// Run `f` on the resident team only when `work` exceeds `limit`; otherwise run
/// it, and every nested `with_par` inside it, on `Par::Seq`.
///
/// `work` is in real element updates (see [`real_work`]), so callers pass the
/// shape-derived flop estimate they already use for blocking decisions. The
/// sequential branch also releases a team left over from an earlier wider
/// setting, exactly like the one-thread branch of [`with_par`].
pub fn with_par_if<R>(work: usize, limit: usize, f: impl FnOnce(Par<'_>) -> R) -> R {
    if work > limit {
        return with_par(f);
    }
    if WIDTH.load(Ordering::Relaxed) == 1 {
        return with_par(f);
    }
    let _serial = SerialScope::enter();
    f(Par::Seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_work_scales_complex_and_saturates() {
        assert_eq!(real_work(32, 64, 96, false), 32 * 64 * 96);
        // One complex multiply-add counts as four real updates.
        assert_eq!(real_work(32, 64, 96, true), 4 * 32 * 64 * 96);
        assert_eq!(real_work(usize::MAX, 2, 2, false), usize::MAX);
        assert_eq!(real_work(usize::MAX, 2, 2, true), usize::MAX);
    }

    #[test]
    fn small_work_uses_sequential_token_for_the_whole_callee() {
        // Below the limit the callee gets `Seq` and any helper that would
        // otherwise acquire its own team must stay `Seq` as well. The depth
        // guard must return to zero when the callee returns.
        with_par_if(1, usize::MAX, |outer| {
            assert!(matches!(outer, Par::Seq));
            with_par(|nested| assert!(matches!(nested, Par::Seq)));
        });
        assert_eq!(SERIAL_DEPTH.with(|depth| depth.get()), 0);
    }

    #[test]
    fn sequential_region_unwinds_on_panic() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_par_if(1, usize::MAX, |_| panic!("callee failed"));
        }));
        assert!(result.is_err());
        assert_eq!(SERIAL_DEPTH.with(|depth| depth.get()), 0);
    }
}
