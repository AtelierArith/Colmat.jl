//! Caller tuning for faer's multishift QR (Schur) and QZ drivers.
//!
//! `SchurParams`/`GeneralizedSchurParams` take the shift count, deflation
//! window and shift-chase cost as `extern "C" fn` pointers, which cannot
//! capture Julia values. Overrides are therefore stored here and read by fixed
//! functions. The Julia layer sets them only while holding its native-call
//! lock for a whole `with_schur_tuning` scope and always restores the defaults,
//! so no other call observes them. Zero means "keep faer's default".
//!
//! The same scope also carries `GeneralizedHessenbergParams` for the
//! generalized Hessenberg reduction that starts QZ construction, so its block
//! size reaches both the construction query and the kernel. The general GEVD
//! driver receives the same overrides through `gevd_params`; its workspace
//! query adds the tuned Hessenberg requirement that faer's `gevd_scratch`
//! omits.
use crate::guarded;
use faer::linalg::evd::SchurParams;
use faer::linalg::gevd::gen_hessenberg::GeneralizedHessenbergParams;
use faer::linalg::gevd::{GeneralizedSchurParams, GevdParams};
use faer::traits::ComplexField;
use faer::Spec;
use rustcall_julia_macros::julia;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static SHIFTS: AtomicUsize = AtomicUsize::new(0);
static WINDOW: AtomicUsize = AtomicUsize::new(0);
static BLOCKING: AtomicUsize = AtomicUsize::new(0);
static NIBBLE: AtomicUsize = AtomicUsize::new(0);
static COST: AtomicUsize = AtomicUsize::new(0);
static GEN_HESSENBERG_BLOCK: AtomicUsize = AtomicUsize::new(0);
static GEN_HESSENBERG_BLOCKING: AtomicUsize = AtomicUsize::new(0);
const ALL: [&AtomicUsize; 7] = [
    &SHIFTS,
    &WINDOW,
    &BLOCKING,
    &NIBBLE,
    &COST,
    &GEN_HESSENBERG_BLOCK,
    &GEN_HESSENBERG_BLOCKING,
];

extern "C" fn fixed_shifts(_n: usize, nh: usize) -> usize {
    // Keep the count even and inside the active block, as faer's default does.
    let s = SHIFTS.load(Relaxed).min(nh.saturating_sub(1)).max(2);
    s - s % 2
}
extern "C" fn fixed_window(_n: usize, nh: usize) -> usize {
    WINDOW.load(Relaxed).min(nh).max(2)
}
extern "C" fn fixed_cost(_n: usize, _nh: usize) -> usize {
    COST.load(Relaxed)
}

/// faer's defaults with the active overrides applied.
pub(crate) fn schur_params<T: ComplexField>() -> Spec<SchurParams, T> {
    let mut p: SchurParams = faer::auto!(T);
    if SHIFTS.load(Relaxed) != 0 {
        p.recommended_shift_count = fixed_shifts;
    }
    if WINDOW.load(Relaxed) != 0 {
        p.recommended_deflation_window = fixed_window;
    }
    let b = BLOCKING.load(Relaxed);
    if b != 0 {
        p.blocking_threshold = b;
    }
    let k = NIBBLE.load(Relaxed);
    if k != 0 {
        p.nibble_threshold = k;
    }
    p.into()
}
pub(crate) fn qz_params<T: ComplexField>() -> GeneralizedSchurParams {
    let mut p: GeneralizedSchurParams = faer::auto!(T);
    if SHIFTS.load(Relaxed) != 0 {
        p.recommended_shift_count = fixed_shifts;
    }
    if WINDOW.load(Relaxed) != 0 {
        p.recommended_deflation_window = fixed_window;
    }
    if COST.load(Relaxed) != 0 {
        p.relative_cost_estimate_of_shift_chase_to_matmul = fixed_cost;
    }
    let b = BLOCKING.load(Relaxed);
    if b != 0 {
        p.blocking_threshold = b;
    }
    let k = NIBBLE.load(Relaxed);
    if k != 0 {
        p.nibble_threshold = k;
    }
    p
}
/// faer's generalized Hessenberg defaults with the active overrides applied.
/// A block size of one (or at least the order) selects the unblocked kernel.
pub(crate) fn generalized_hessenberg_params<T: ComplexField>() -> GeneralizedHessenbergParams {
    let mut p: GeneralizedHessenbergParams = faer::auto!(T);
    let b = GEN_HESSENBERG_BLOCK.load(Relaxed);
    if b != 0 {
        p.block_size = b;
    }
    let t = GEN_HESSENBERG_BLOCKING.load(Relaxed);
    if t != 0 {
        p.blocking_threshold = t;
    }
    p
}

/// faer's GEVD defaults with the scope's generalized Hessenberg and QZ
/// overrides. `gevd_scratch` sizes the Hessenberg step with `auto!` whatever
/// `hessenberg` says, so callers must also reserve
/// `generalized_hessenberg_scratch(n, params.hessenberg)`
/// (docs/reproducers/faer-gevd-hessenberg-scratch.md).
pub(crate) fn gevd_params<T: ComplexField>() -> GevdParams {
    let mut p: GevdParams = faer::auto!(T);
    p.hessenberg = generalized_hessenberg_params::<T>();
    p.schur = qz_params::<T>();
    p
}

/// Install overrides (zero keeps the default); `nibble` is a percentage.
#[julia]
pub fn set_schur_tuning(
    shifts: usize,
    window: usize,
    blocking: usize,
    nibble: usize,
    cost: usize,
    gen_hessenberg_block: usize,
    gen_hessenberg_blocking: usize,
) -> Result<(), String> {
    guarded(|| {
        if shifts == 1 {
            return Err("shift count must be zero (default) or at least two".into());
        }
        if window == 1 {
            return Err("deflation window must be zero (default) or at least two".into());
        }
        if nibble > 100 {
            return Err("nibble threshold is a percentage between 0 and 100".into());
        }
        let values = [
            shifts,
            window,
            blocking,
            nibble,
            cost,
            gen_hessenberg_block,
            gen_hessenberg_blocking,
        ];
        for (a, v) in ALL.iter().zip(values) {
            a.store(v, Relaxed);
        }
        Ok(())
    })
}
/// Current overrides in `set_schur_tuning` order.
#[julia]
pub fn schur_tuning_state(out: &mut [usize]) -> Result<(), String> {
    guarded(|| {
        if out.len() != ALL.len() {
            return Err("tuning state needs seven entries".into());
        }
        for (o, v) in out.iter_mut().zip(ALL) {
            *o = v.load(Relaxed);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::evd;
    use faer::{Mat, Par};
    #[test]
    fn overrides_apply_and_reset() {
        set_schur_tuning(4, 6, 10, 30, 0, 5, 7).unwrap();
        let g = generalized_hessenberg_params::<f64>();
        assert_eq!((g.block_size, g.blocking_threshold), (5, 7));
        let p = schur_params::<f64>();
        assert_eq!(
            ((p.recommended_shift_count)(100, 100), p.blocking_threshold),
            (4, 10)
        );
        assert_eq!(
            (
                (p.recommended_deflation_window)(100, 100),
                p.nibble_threshold
            ),
            (6, 30)
        );
        assert_eq!((p.recommended_shift_count)(100, 3), 2);
        // A tuned Schur of a 120x120 Hessenberg matrix still reconstructs it.
        let n = 120;
        let h0 = Mat::from_fn(n, n, |i, j| {
            if i > j + 1 {
                0.0
            } else {
                ((i * 7 + j * 3) % 11) as f64 - 5.0
            }
        });
        let mut h = h0.clone();
        let mut z = Mat::<f64>::identity(n, n);
        let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
        let req =
            evd::hessenberg_schur_scratch::<f64>(n, true, Par::Seq.shape(), schur_params::<f64>());
        let mut mem = faer::dyn_stack::MemBuffer::new(req);
        evd::hessenberg_schur_real(
            h.as_mut(),
            Some(z.as_mut()),
            faer::ColMut::from_slice_mut(&mut wr),
            faer::ColMut::from_slice_mut(&mut wi),
            Par::Seq,
            faer::dyn_stack::MemStack::new(&mut mem),
            schur_params::<f64>(),
        )
        .unwrap();
        let back = &z * &h * z.transpose();
        assert!((&back - &h0).norm_max() < 1e-9 * h0.norm_max());
        assert!(set_schur_tuning(1, 0, 0, 0, 0, 0, 0).is_err());
        assert!(set_schur_tuning(0, 0, 0, 101, 0, 0, 0).is_err());
        set_schur_tuning(0, 0, 0, 0, 0, 0, 0).unwrap();
        let mut s = [9usize; 7];
        schur_tuning_state(&mut s).unwrap();
        assert_eq!(s, [0; 7]);
        let g = generalized_hessenberg_params::<f64>();
        assert_eq!((g.block_size, g.blocking_threshold), (32, 256 * 256));
    }
}
