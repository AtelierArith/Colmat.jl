//! Caller tuning for faer's multishift QR (Schur) and QZ drivers.
//!
//! `SchurParams`/`GeneralizedSchurParams` take the shift count, deflation
//! window and shift-chase cost as `extern "C" fn` pointers, which cannot
//! capture Julia values. Overrides are therefore stored here and read by fixed
//! functions. The Julia layer sets them only while holding its native-call
//! lock for a whole `with_schur_tuning` scope and always restores the defaults,
//! so no other call observes them. Zero means "keep faer's default".
use crate::guarded;
use faer::linalg::evd::SchurParams;
use faer::linalg::gevd::GeneralizedSchurParams;
use faer::traits::ComplexField;
use faer::Spec;
use rustcall_julia_macros::julia;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static SHIFTS: AtomicUsize = AtomicUsize::new(0);
static WINDOW: AtomicUsize = AtomicUsize::new(0);
static BLOCKING: AtomicUsize = AtomicUsize::new(0);
static NIBBLE: AtomicUsize = AtomicUsize::new(0);
static COST: AtomicUsize = AtomicUsize::new(0);

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

/// Install overrides (zero keeps the default); `nibble` is a percentage.
#[julia]
pub fn set_schur_tuning(
    shifts: usize,
    window: usize,
    blocking: usize,
    nibble: usize,
    cost: usize,
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
        SHIFTS.store(shifts, Relaxed);
        WINDOW.store(window, Relaxed);
        BLOCKING.store(blocking, Relaxed);
        NIBBLE.store(nibble, Relaxed);
        COST.store(cost, Relaxed);
        Ok(())
    })
}
/// Current overrides as `[shifts, window, blocking, nibble, cost]`.
#[julia]
pub fn schur_tuning_state(out: &mut [usize]) -> Result<(), String> {
    guarded(|| {
        if out.len() != 5 {
            return Err("tuning state needs five entries".into());
        }
        for (o, v) in out
            .iter_mut()
            .zip([&SHIFTS, &WINDOW, &BLOCKING, &NIBBLE, &COST])
        {
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
        set_schur_tuning(4, 6, 10, 30, 0).unwrap();
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
        assert!(set_schur_tuning(1, 0, 0, 0, 0).is_err());
        assert!(set_schur_tuning(0, 0, 0, 101, 0).is_err());
        set_schur_tuning(0, 0, 0, 0, 0).unwrap();
        let mut s = [9usize; 5];
        schur_tuning_state(&mut s).unwrap();
        assert_eq!(s, [0; 5]);
    }
}
