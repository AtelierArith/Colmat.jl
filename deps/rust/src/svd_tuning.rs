//! Caller tuning for faer's SVD driver (`SvdParams`), read during one Julia
//! `with_svd_tuning` scope that holds the native-call lock. `recursion_threshold`
//! selects between the bidiagonal implicit QR algorithm (sizes below it) and
//! divide and conquer, so `usize::MAX` gives Julia's `QRIteration()`.
//! `qr_blocking` is the `QrParams.blocking_threshold` of the QR factorization
//! that precedes bidiagonalization for tall inputs; faer's QR scratch does not
//! depend on it, so it changes execution only.
use crate::guarded;
use faer::linalg::qr::no_pivoting::factor::QrParams;
use faer::linalg::svd::bidiag::BidiagParams;
use faer::linalg::svd::SvdParams;
use faer::traits::ComplexField;
use faer::utils::cost::ParThreshold;
use faer::{Auto, Spec};
use rustcall_julia_macros::julia;
use std::sync::Mutex;

#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Tuning {
    recursion: Option<usize>,
    qr_ratio: Option<f64>,
    bidiag_par: Option<usize>,
    qr_blocking: Option<usize>,
}
static TUNING: Mutex<Tuning> = Mutex::new(Tuning {
    recursion: None,
    qr_ratio: None,
    bidiag_par: None,
    qr_blocking: None,
});
fn current() -> Tuning {
    *TUNING.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn svd_params<T: ComplexField>() -> Spec<SvdParams, T> {
    let t = current();
    let mut p = <SvdParams as Auto<T>>::auto();
    if let Some(r) = t.recursion {
        p.recursion_threshold = r;
    }
    if let Some(q) = t.qr_ratio {
        p.qr_ratio_threshold = q;
    }
    if let Some(b) = t.bidiag_par {
        let mut bidiag = <BidiagParams as Auto<T>>::auto();
        bidiag.par_threshold = ParThreshold::from(b);
        p.bidiag = bidiag;
    }
    if let Some(b) = t.qr_blocking {
        let mut qr = <QrParams as Auto<T>>::auto();
        qr.blocking_threshold = b;
        p.qr = qr;
    }
    p.into()
}

/// Flags mark overridden entries: recursion threshold, QR ratio threshold,
/// bidiagonalization parallel floor (`FaerParThreshold` value), QR blocking
/// threshold.
#[julia]
pub fn set_svd_tuning(
    recursion: usize,
    qr_ratio: f64,
    bidiag_par: usize,
    qr_blocking: usize,
    set: &[u8],
) -> Result<(), String> {
    guarded(|| {
        if set.len() != 4 {
            return Err("SVD tuning needs four flags".into());
        }
        if set[0] != 0 && recursion < 2 {
            return Err("recursion_threshold must be at least two".into());
        }
        if set[1] != 0 && !(qr_ratio.is_finite() && qr_ratio >= 1.0) {
            return Err("qr_ratio_threshold must be finite and at least one".into());
        }
        *TUNING.lock().unwrap_or_else(|e| e.into_inner()) = Tuning {
            recursion: (set[0] != 0).then_some(recursion),
            qr_ratio: (set[1] != 0).then_some(qr_ratio),
            bidiag_par: (set[2] != 0).then_some(bidiag_par),
            qr_blocking: (set[3] != 0).then_some(qr_blocking),
        };
        Ok(())
    })
}
/// Writes `[recursion, bidiag_par, qr_blocking]`, the QR ratio and the four
/// flags.
#[julia]
pub fn svd_tuning_state(
    values: &mut [usize],
    ratio: &mut [f64],
    set: &mut [u8],
) -> Result<(), String> {
    guarded(|| {
        if values.len() != 3 || ratio.len() != 1 || set.len() != 4 {
            return Err("SVD tuning state has three values, one ratio and four flags".into());
        }
        let t = current();
        values[0] = t.recursion.unwrap_or(0);
        values[1] = t.bidiag_par.unwrap_or(0);
        values[2] = t.qr_blocking.unwrap_or(0);
        ratio[0] = t.qr_ratio.unwrap_or(0.0);
        set[0] = t.recursion.is_some() as u8;
        set[1] = t.qr_ratio.is_some() as u8;
        set[2] = t.bidiag_par.is_some() as u8;
        set[3] = t.qr_blocking.is_some() as u8;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overrides_map_to_params() {
        set_svd_tuning(usize::MAX, 3.0, 0, 16, &[1, 1, 0, 1]).unwrap();
        let p = svd_params::<f64>();
        assert_eq!(
            (p.recursion_threshold, p.qr_ratio_threshold),
            (usize::MAX, 3.0)
        );
        assert_eq!(p.qr.blocking_threshold, 16);
        let (mut v, mut r, mut f) = ([0usize; 3], [0.0; 1], [0u8; 4]);
        svd_tuning_state(&mut v, &mut r, &mut f).unwrap();
        assert_eq!((v, f), ([usize::MAX, 0, 16], [1, 1, 0, 1]));
        assert!(set_svd_tuning(1, 0.0, 0, 0, &[1, 0, 0, 0]).is_err());
        assert!(set_svd_tuning(8, 0.5, 0, 0, &[0, 1, 0, 0]).is_err());
        assert!(set_svd_tuning(8, 2.0, 0, 0, &[0, 0, 0]).is_err());
        set_svd_tuning(0, 0.0, 0, 0, &[0, 0, 0, 0]).unwrap();
        assert_eq!(current(), Tuning::default());
    }
}
