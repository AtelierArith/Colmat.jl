//! Caller controls for faer's dense self-adjoint factorizations.
//!
//! Every Colmat LLᴴ, LDLᴴ and LBLᴴ construction (retained, typed, borrowed,
//! team and direct in-place) and the matching scratch queries read their faer
//! parameters here, so query and execution agree.
//!
//! * Tuning (`LltParams`, `LdltParams`, `LbltParams.{block_size,
//!   par_threshold}`) only changes performance. It lives here for one Julia
//!   `with_cholesky_tuning` scope, which holds the native-call lock and
//!   restores the previous settings.
//! * Call options (dynamic regularization, LBLᴴ pivoting strategy) change the
//!   result. Julia installs them for a single `cholesky`/`ldlt`/`bunchkaufman`
//!   call under the same lock and clears them before it returns.
use crate::guarded;
use faer::linalg::cholesky::lblt::factor::{LbltParams, PivotingStrategy};
use faer::linalg::cholesky::ldlt::factor::{LdltParams, LdltRegularization};
use faer::linalg::cholesky::llt::factor::{LltParams, LltRegularization};
use faer::traits::{math_utils::from_f64, ComplexField};
use faer::utils::cost::ParThreshold;
use faer::Auto;
use rustcall_julia_macros::julia;

const TUNING_LEN: usize = 6;

/// `None` keeps faer's default, in `set_selfadjoint_tuning` order.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Tuning {
    llt_recursion: Option<usize>,
    llt_block: Option<usize>,
    ldlt_recursion: Option<usize>,
    ldlt_block: Option<usize>,
    lblt_block: Option<usize>,
    lblt_par: Option<usize>,
}
impl Tuning {
    fn all(&self) -> [Option<usize>; TUNING_LEN] {
        [
            self.llt_recursion,
            self.llt_block,
            self.ldlt_recursion,
            self.ldlt_block,
            self.lblt_block,
            self.lblt_par,
        ]
    }
}
const NO_TUNING: Tuning = Tuning {
    llt_recursion: None,
    llt_block: None,
    ldlt_recursion: None,
    ldlt_block: None,
    lblt_block: None,
    lblt_par: None,
};

/// Options of one factorization call. Regularization is active only when both
/// `delta` and `epsilon` are positive (as in faer); Julia validates that.
#[derive(Clone, Default, PartialEq, Debug)]
struct CallOptions {
    pivoting: Option<PivotingStrategy>,
    delta: f64,
    epsilon: f64,
    signs: Option<Vec<i8>>,
}
const NO_CALL: CallOptions = CallOptions {
    pivoting: None,
    delta: 0.0,
    epsilon: 0.0,
    signs: None,
};

// Julia serializes every native call, so the state is process-wide. Rust unit
// tests run on parallel threads; each test thread has its own state there.
#[cfg(not(test))]
mod state {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use std::sync::Mutex;
    static TUNING: Mutex<Tuning> = Mutex::new(NO_TUNING);
    static CALL: Mutex<CallOptions> = Mutex::new(NO_CALL);
    /// faer's `dynamic_regularization_count` of the latest LLᴴ/LDLᴴ construction.
    static LAST_COUNT: AtomicUsize = AtomicUsize::new(0);
    pub(super) fn tuning() -> Tuning {
        *TUNING.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(super) fn set_tuning(t: Tuning) {
        *TUNING.lock().unwrap_or_else(|e| e.into_inner()) = t;
    }
    pub(super) fn call() -> CallOptions {
        CALL.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub(super) fn set_call(c: CallOptions) {
        *CALL.lock().unwrap_or_else(|e| e.into_inner()) = c;
    }
    pub(super) fn count() -> usize {
        LAST_COUNT.load(Relaxed)
    }
    pub(super) fn set_count(c: usize) {
        LAST_COUNT.store(c, Relaxed);
    }
}
#[cfg(test)]
mod state {
    use super::*;
    use std::cell::{Cell, RefCell};
    thread_local! {
        static TUNING: Cell<Tuning> = const { Cell::new(NO_TUNING) };
        static CALL: RefCell<CallOptions> = const { RefCell::new(NO_CALL) };
        static LAST_COUNT: Cell<usize> = const { Cell::new(0) };
    }
    pub(super) fn tuning() -> Tuning {
        TUNING.with(|t| t.get())
    }
    pub(super) fn set_tuning(t: Tuning) {
        TUNING.with(|x| x.set(t));
    }
    pub(super) fn call() -> CallOptions {
        CALL.with(|c| c.borrow().clone())
    }
    pub(super) fn set_call(c: CallOptions) {
        CALL.with(|x| *x.borrow_mut() = c);
    }
    pub(super) fn count() -> usize {
        LAST_COUNT.with(|c| c.get())
    }
    pub(super) fn set_count(c: usize) {
        LAST_COUNT.with(|x| x.set(c));
    }
}
use state::{call, tuning};

pub(crate) fn llt_params<T: ComplexField>() -> LltParams {
    let t = tuning();
    let mut p = <LltParams as Auto<T>>::auto();
    if let Some(r) = t.llt_recursion {
        p.recursion_threshold = r;
    }
    if let Some(b) = t.llt_block {
        p.block_size = b;
    }
    p
}
pub(crate) fn ldlt_params<T: ComplexField>() -> LdltParams {
    let t = tuning();
    let mut p = <LdltParams as Auto<T>>::auto();
    if let Some(r) = t.ldlt_recursion {
        p.recursion_threshold = r;
    }
    if let Some(b) = t.ldlt_block {
        p.block_size = b;
    }
    p
}
/// The pivoting strategy of a block LDLᴴ call: the call option, otherwise
/// `Rook` for Julia's `rook=true`, otherwise faer's default `PartialDiag`.
pub(crate) fn lblt_strategy<T: ComplexField>(rook: bool) -> PivotingStrategy {
    match call().pivoting {
        Some(p) => p,
        None if rook => PivotingStrategy::Rook,
        None => <LbltParams as Auto<T>>::auto().pivoting,
    }
}
pub(crate) fn lblt_params<T: ComplexField>(rook: bool) -> LbltParams {
    let t = tuning();
    let mut p = <LbltParams as Auto<T>>::auto();
    p.pivoting = lblt_strategy::<T>(rook);
    if let Some(b) = t.lblt_block {
        p.block_size = b;
    }
    if let Some(v) = t.lblt_par {
        p.par_threshold = ParThreshold::from(v);
    }
    p
}

pub(crate) fn llt_regularization<T: ComplexField>() -> LltRegularization<T::Real> {
    let c = call();
    LltRegularization {
        dynamic_regularization_delta: from_f64::<T::Real>(c.delta),
        dynamic_regularization_epsilon: from_f64::<T::Real>(c.epsilon),
    }
}
/// Run `f` with the LDLᴴ regularization of the current call; the expected
/// signs must cover the whole order `n`.
pub(crate) fn with_ldlt_regularization<T: ComplexField, R>(
    n: usize,
    f: impl FnOnce(LdltRegularization<'_, T::Real>) -> R,
) -> Result<R, String> {
    let c = call();
    if let Some(s) = &c.signs {
        if s.len() != n {
            return Err("regularization_signs must have one entry per row".into());
        }
    }
    Ok(f(LdltRegularization {
        dynamic_regularization_signs: c.signs.as_deref(),
        dynamic_regularization_delta: from_f64::<T::Real>(c.delta),
        dynamic_regularization_epsilon: from_f64::<T::Real>(c.epsilon),
    }))
}
/// Regularized LLᴴ in place, or `None` without regularization (then call
/// faer's `llt::factor::cholesky_in_place` with [`llt_params`]).
///
/// faer's regularized LLᴴ at the pinned revision scales a replaced pivot's
/// column, diagonal included, by `1/sqrt(delta)`, so it stores the original
/// pivot over `sqrt(delta)` on the diagonal and the blocked recursion then
/// solves with that wrong diagonal (docs/reproducers/
/// faer-selfadjoint-factor-params). The same kernel's LDLᴴ mode with every
/// expected sign `+1` applies exactly the LLᴴ replacement rule and count and
/// stores D separately, so Colmat factors LDLᴴ with the LLᴴ block sizes and
/// scales the unit factor by `sqrt(D)`. The workspace is the same as LLᴴ's.
/// Returns faer's count, or the failing pivot index for non-finite input.
pub(crate) fn regularized_llt_in_place<T: ComplexField>(
    mut a: faer::MatMut<'_, T>,
    par: faer::Par<'_>,
    stack: &mut faer::dyn_stack::MemStack,
) -> Option<Result<usize, usize>> {
    use faer::linalg::cholesky::ldlt::factor::{cholesky_in_place, LdltError};
    use faer::prelude::ReborrowMut;
    use faer::traits::math_utils::{from_real, mul_real, real, sqrt};
    let c = call();
    if !(c.delta > 0.0 && c.epsilon > 0.0) {
        return None;
    }
    let n = a.nrows();
    let signs = vec![1i8; n];
    let p = llt_params::<T>();
    let mut params = <LdltParams as Auto<T>>::auto();
    params.recursion_threshold = p.recursion_threshold;
    params.block_size = p.block_size;
    let result = cholesky_in_place(
        a.rb_mut(),
        LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: from_f64::<T::Real>(c.delta),
            dynamic_regularization_epsilon: from_f64::<T::Real>(c.epsilon),
        },
        par,
        stack,
        params.into(),
    );
    Some(match result {
        Ok(info) => {
            for j in 0..n {
                // Every D is positive: above epsilon, or replaced by delta.
                let s = sqrt(&real(&a[(j, j)]));
                a[(j, j)] = from_real(&s);
                for i in j + 1..n {
                    a[(i, j)] = mul_real(&a[(i, j)], &s);
                }
            }
            Ok(info.dynamic_regularization_count)
        }
        Err(LdltError::ZeroPivot { index }) => Err(index),
    })
}
pub(crate) fn reset_regularization_count() {
    state::set_count(0);
}
pub(crate) fn record_regularization_count(count: usize) {
    state::set_count(count);
}

/// Strategy codes shared with Julia; zero keeps the kind's default.
fn strategy(code: i32) -> Result<Option<PivotingStrategy>, String> {
    Ok(match code {
        0 => None,
        1 => Some(PivotingStrategy::Partial),
        2 => Some(PivotingStrategy::PartialDiag),
        3 => Some(PivotingStrategy::Rook),
        4 => Some(PivotingStrategy::RookDiag),
        5 => Some(PivotingStrategy::Full),
        _ => return Err("unknown block LDLT pivoting strategy".into()),
    })
}

/// `values` holds the six settings and `set` marks which are overridden, in
/// the order LLᴴ recursion threshold, LLᴴ block size, LDLᴴ recursion
/// threshold, LDLᴴ block size, LBLᴴ block size and LBLᴴ parallel floor
/// (a `FaerParThreshold` value).
#[julia]
pub fn set_selfadjoint_tuning(values: &[usize], set: &[u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != TUNING_LEN || set.len() != TUNING_LEN {
            return Err("self-adjoint factor tuning needs six values and flags".into());
        }
        let pick = |i: usize| (set[i] != 0).then_some(values[i]);
        // faer's recursive LLᴴ/LDLᴴ never advances with a zero block size or
        // recursion threshold.
        for i in 0..4 {
            if pick(i) == Some(0) {
                return Err(
                    "LLT/LDLT recursion thresholds and block sizes must be positive".into(),
                );
            }
        }
        state::set_tuning(Tuning {
            llt_recursion: pick(0),
            llt_block: pick(1),
            ldlt_recursion: pick(2),
            ldlt_block: pick(3),
            lblt_block: pick(4),
            lblt_par: pick(5),
        });
        Ok(())
    })
}
/// Writes the active settings and their flags in `set_selfadjoint_tuning` order.
#[julia]
pub fn selfadjoint_tuning_state(values: &mut [usize], set: &mut [u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != TUNING_LEN || set.len() != TUNING_LEN {
            return Err("self-adjoint factor tuning needs six values and flags".into());
        }
        for (i, v) in tuning().all().into_iter().enumerate() {
            values[i] = v.unwrap_or(0);
            set[i] = v.is_some() as u8;
        }
        Ok(())
    })
}
/// Install the options of one factorization call. `signs` holds `i8` values
/// (-1, 0, 1) reinterpreted as bytes and is used only when `has_signs`.
#[julia]
pub fn set_selfadjoint_call_options(
    pivoting: i32,
    delta: f64,
    epsilon: f64,
    signs: &[u8],
    has_signs: bool,
) -> Result<(), String> {
    guarded(|| {
        let pivoting = strategy(pivoting)?;
        if !(delta >= 0.0 && epsilon >= 0.0 && delta.is_finite() && epsilon.is_finite()) {
            return Err("regularization delta and epsilon must be finite and nonnegative".into());
        }
        let signs = if has_signs {
            let s: Vec<i8> = signs.iter().map(|&b| b as i8).collect();
            if s.iter().any(|v| !(-1..=1).contains(v)) {
                return Err("regularization signs must be -1, 0 or 1".into());
            }
            Some(s)
        } else {
            None
        };
        state::set_call(CallOptions {
            pivoting,
            delta,
            epsilon,
            signs,
        });
        reset_regularization_count();
        Ok(())
    })
}
/// faer's regularization count of the latest LLᴴ/LDLᴴ construction (zero
/// without regularization or after a failed factorization).
#[julia]
pub fn selfadjoint_regularization_count() -> Result<usize, String> {
    Ok(state::count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_overrides_apply_and_reset() {
        assert!(set_selfadjoint_tuning(&[0; 6], &[1, 0, 0, 0, 0, 0]).is_err());
        assert!(set_selfadjoint_tuning(&[0; 6], &[0, 0, 0, 1, 0, 0]).is_err());
        set_selfadjoint_tuning(&[3, 5, 7, 9, 1, usize::MAX], &[1; 6]).unwrap();
        let llt = llt_params::<f64>();
        let ldlt = ldlt_params::<f64>();
        let lblt = lblt_params::<f64>(false);
        assert_eq!((llt.recursion_threshold, llt.block_size), (3, 5));
        assert_eq!((ldlt.recursion_threshold, ldlt.block_size), (7, 9));
        assert_eq!(lblt.block_size, 1);
        assert_eq!(lblt.par_threshold, ParThreshold::Never);
        let (mut v, mut s) = ([9usize; 6], [9u8; 6]);
        selfadjoint_tuning_state(&mut v, &mut s).unwrap();
        assert_eq!((v, s), ([3, 5, 7, 9, 1, usize::MAX], [1; 6]));
        set_selfadjoint_tuning(&[0; 6], &[0; 6]).unwrap();
        let llt = llt_params::<f64>();
        assert_eq!((llt.recursion_threshold, llt.block_size), (64, 128));
        assert_eq!(lblt_params::<f64>(false).block_size, 64);
    }

    #[test]
    fn call_options_select_strategy_and_validate() {
        assert_eq!(lblt_strategy::<f64>(false), PivotingStrategy::PartialDiag);
        assert_eq!(lblt_strategy::<f64>(true), PivotingStrategy::Rook);
        set_selfadjoint_call_options(5, 0.0, 0.0, &[], false).unwrap();
        assert_eq!(lblt_strategy::<f64>(false), PivotingStrategy::Full);
        assert!(set_selfadjoint_call_options(9, 0.0, 0.0, &[], false).is_err());
        assert!(set_selfadjoint_call_options(0, -1.0, 0.0, &[], false).is_err());
        assert!(set_selfadjoint_call_options(0, 1.0, 1.0, &[2], true).is_err());
        set_selfadjoint_call_options(0, 1.0, 0.5, &[1, 255], true).unwrap();
        let r = with_ldlt_regularization::<f64, _>(2, |r| {
            (
                r.dynamic_regularization_signs.map(|s| s.to_vec()),
                r.dynamic_regularization_delta,
            )
        })
        .unwrap();
        assert_eq!(r, (Some(vec![1, -1]), 1.0));
        assert!(with_ldlt_regularization::<f64, _>(3, |_| ()).is_err());
        set_selfadjoint_call_options(0, 0.0, 0.0, &[], false).unwrap();
        assert_eq!(call(), CallOptions::default());
    }
}
