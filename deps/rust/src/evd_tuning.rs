//! Caller tuning for the parameters inside faer's EVD drivers.
//!
//! The general driver takes `EvdParams` (Hessenberg reduction, multishift QR
//! from `schur_tuning`, EVD-from-Schur recursion) and the self-adjoint driver
//! `SelfAdjointEvdParams` (tridiagonalization parallel floor, divide-and-
//! conquer recursion). Like `schur_tuning`, overrides live here for one Julia
//! `with_evd_tuning` scope, which holds the native-call lock and restores them.
use crate::guarded;
use faer::linalg::evd::hessenberg::HessenbergParams;
use faer::linalg::evd::tridiag::TridiagParams;
use faer::linalg::evd::{EvdFromSchurParams, EvdParams, SelfAdjointEvdParams};
use faer::traits::ComplexField;
use faer::utils::cost::ParThreshold;
use faer::{Auto, Spec};
use rustcall_julia_macros::julia;
use std::sync::Mutex;

/// `None` keeps faer's default. Parallel floors use `FaerParThreshold` values
/// (zero derived, `usize::MAX` never parallel).
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Tuning {
    hessenberg_par: Option<usize>,
    hessenberg_blocking: Option<usize>,
    evd_recursion: Option<usize>,
    self_adjoint_recursion: Option<usize>,
    tridiagonal_par: Option<usize>,
}
static TUNING: Mutex<Tuning> = Mutex::new(Tuning {
    hessenberg_par: None,
    hessenberg_blocking: None,
    evd_recursion: None,
    self_adjoint_recursion: None,
    tridiagonal_par: None,
});
fn current() -> Tuning {
    *TUNING.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn evd_params<T: ComplexField>() -> Spec<EvdParams, T> {
    let t = current();
    let mut hessenberg = <HessenbergParams as Auto<T>>::auto();
    if let Some(p) = t.hessenberg_par {
        hessenberg.par_threshold = ParThreshold::from(p);
    }
    if let Some(b) = t.hessenberg_blocking {
        hessenberg.blocking_threshold = b;
    }
    let mut evd_from_schur = <EvdFromSchurParams as Auto<T>>::auto();
    if let Some(r) = t.evd_recursion {
        evd_from_schur.recursion_threshold = r;
    }
    EvdParams {
        hessenberg,
        schur: crate::schur_tuning::schur_params::<T>().config,
        evd_from_schur,
        ..<EvdParams as Auto<T>>::auto()
    }
    .into()
}
pub(crate) fn self_adjoint_params<T: ComplexField>() -> Spec<SelfAdjointEvdParams, T> {
    let t = current();
    let mut tridiag = <TridiagParams as Auto<T>>::auto();
    if let Some(p) = t.tridiagonal_par {
        tridiag.par_threshold = ParThreshold::from(p);
    }
    let mut p = <SelfAdjointEvdParams as Auto<T>>::auto();
    p.tridiag = tridiag;
    if let Some(r) = t.self_adjoint_recursion {
        p.recursion_threshold = r;
    }
    p.into()
}

/// `values` holds the five settings and `set` marks which are overridden, in
/// the order Hessenberg par, Hessenberg blocking, EVD recursion, self-adjoint
/// recursion and tridiagonal par.
#[julia]
pub fn set_evd_tuning(values: &[usize], set: &[u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != 5 || set.len() != 5 {
            return Err("EVD tuning needs five values and flags".into());
        }
        let pick = |i: usize| (set[i] != 0).then_some(values[i]);
        for i in [2, 3] {
            if pick(i) == Some(0) {
                return Err("recursion thresholds must be positive".into());
            }
        }
        *TUNING.lock().unwrap_or_else(|e| e.into_inner()) = Tuning {
            hessenberg_par: pick(0),
            hessenberg_blocking: pick(1),
            evd_recursion: pick(2),
            self_adjoint_recursion: pick(3),
            tridiagonal_par: pick(4),
        };
        Ok(())
    })
}
/// Writes the active settings and their flags in `set_evd_tuning` order.
#[julia]
pub fn evd_tuning_state(values: &mut [usize], set: &mut [u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != 5 || set.len() != 5 {
            return Err("EVD tuning needs five values and flags".into());
        }
        let t = current();
        let all = [
            t.hessenberg_par,
            t.hessenberg_blocking,
            t.evd_recursion,
            t.self_adjoint_recursion,
            t.tridiagonal_par,
        ];
        for (i, v) in all.into_iter().enumerate() {
            values[i] = v.unwrap_or(0);
            set[i] = v.is_some() as u8;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overrides_map_to_params() {
        set_evd_tuning(&[0, 64, 8, 16, usize::MAX], &[0, 1, 1, 1, 1]).unwrap();
        let p = evd_params::<f64>();
        assert_eq!(p.hessenberg.blocking_threshold, 64);
        assert_eq!(p.evd_from_schur.recursion_threshold, 8);
        let s = self_adjoint_params::<f64>();
        assert_eq!(s.recursion_threshold, 16);
        let (mut v, mut f) = ([0usize; 5], [0u8; 5]);
        evd_tuning_state(&mut v, &mut f).unwrap();
        assert_eq!((v, f), ([0, 64, 8, 16, usize::MAX], [0, 1, 1, 1, 1]));
        assert!(set_evd_tuning(&[0, 0, 0, 0, 0], &[0, 0, 1, 0, 0]).is_err());
        set_evd_tuning(&[0; 5], &[0; 5]).unwrap();
        assert_eq!(current(), Tuning::default());
    }
}
