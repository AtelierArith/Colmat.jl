//! Saved GEEVX expert backends and LP64-to-ILP64 integer conversion.
//! ABI/workspace contracts validated against Reference LAPACK v3.12.1
//! SRC/{s,d,c,z}geevx.f; no condition-estimation algorithm is substituted.
use core::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

static BACKENDS: [AtomicPtr<c_void>; 8] = [const { AtomicPtr::new(core::ptr::null_mut()) }; 8];

pub(crate) fn pointer(id: i32, bits: i32) -> *mut c_void {
    if !(0..4).contains(&id) || !matches!(bits, 32 | 64) {
        return core::ptr::null_mut();
    }
    BACKENDS[id as usize * 2 + usize::from(bits == 64)].load(Ordering::Acquire)
}

#[no_mangle]
pub extern "C" fn colmat_lapack_set_expert_fallback(id: i32, bits: i32, p: *mut c_void) {
    if !(0..4).contains(&id) || !matches!(bits, 32 | 64) {
        return;
    }
    let own = [
        crate::lapack::sgeevx_ as *const () as *mut c_void,
        crate::lapack::dgeevx_ as *const () as *mut c_void,
        crate::lapack::cgeevx_ as *const () as *mut c_void,
        crate::lapack::zgeevx_ as *const () as *mut c_void,
        crate::lapack::sgeevx_64_ as *const () as *mut c_void,
        crate::lapack::dgeevx_64_ as *const () as *mut c_void,
        crate::lapack::cgeevx_64_ as *const () as *mut c_void,
        crate::lapack::zgeevx_64_ as *const () as *mut c_void,
    ];
    if own.contains(&p) {
        return;
    }
    BACKENDS[id as usize * 2 + usize::from(bits == 64)].store(p, Ordering::Release);
}

/// Effective expert availability: LP64 may use the saved ILP64 implementation.
#[no_mangle]
pub extern "C" fn colmat_lapack_has_expert_fallback(id: i32, bits: i32) -> i32 {
    i32::from(!pointer(id, bits).is_null() || (bits == 32 && !pointer(id, 64).is_null()))
}

pub(crate) fn minimum(
    n: isize,
    complex: bool,
    left: bool,
    right: bool,
    sense: u8,
) -> Option<isize> {
    if n == 0 {
        return Some(1);
    }
    let base = n.checked_mul(if complex || !(left || right) { 2 } else { 3 })?;
    if matches!(sense, b'V' | b'B') {
        Some(
            base.max(n.checked_mul(n)?.checked_add(n.checked_mul(if complex {
                2
            } else {
                6
            })?)?),
        )
    } else {
        Some(base.max(1))
    }
}

pub(crate) trait Auxiliary: Copy {
    type Wide;
    type State;
    unsafe fn widen(
        self,
        n: usize,
        query: bool,
        condition_vectors: bool,
    ) -> (Self::Wide, Self::State);
    unsafe fn commit(self, state: Self::State) -> bool;
}
impl Auxiliary for *mut i32 {
    type Wide = *mut i64;
    type State = Vec<i64>;
    unsafe fn widen(
        self,
        n: usize,
        query: bool,
        condition_vectors: bool,
    ) -> (Self::Wide, Self::State) {
        let length = if query || !condition_vectors {
            1
        } else {
            n.saturating_mul(2).saturating_sub(2).max(1)
        };
        let mut values = vec![i64::MIN; length];
        (values.as_mut_ptr(), values)
    }
    unsafe fn commit(self, values: Self::State) -> bool {
        if values
            .iter()
            .any(|&v| v != i64::MIN && i32::try_from(v).is_err())
        {
            return false;
        }
        for (i, v) in values.into_iter().enumerate() {
            if v != i64::MIN {
                *self.add(i) = v as i32;
            }
        }
        true
    }
}
macro_rules! unchanged_aux {
    ($t:ty) => {
        impl Auxiliary for *mut $t {
            type Wide = *mut $t;
            type State = ();
            unsafe fn widen(
                self,
                _n: usize,
                _query: bool,
                _vectors: bool,
            ) -> (Self::Wide, Self::State) {
                (self, ())
            }
            unsafe fn commit(self, _state: Self::State) -> bool {
                true
            }
        }
    };
}
unchanged_aux!(i64);
unchanged_aux!(f32);
unchanged_aux!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lapack::dgeevx_;
    use std::sync::Mutex;
    static REGISTRY: Mutex<()> = Mutex::new(());
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    struct Restore(*mut c_void, *mut c_void);
    impl Drop for Restore {
        fn drop(&mut self) {
            BACKENDS[2].store(self.0, Ordering::Release);
            BACKENDS[3].store(self.1, Ordering::Release);
        }
    }
    unsafe extern "C" fn fake(
        _bal: *const u8,
        _jl: *const u8,
        _jr: *const u8,
        _sense: *const u8,
        n: *const i64,
        _a: *mut f64,
        _la: *const i64,
        _wr: *mut f64,
        _wi: *mut f64,
        _vl: *mut f64,
        _ll: *const i64,
        _vr: *mut f64,
        _lr: *const i64,
        lo: *mut i64,
        hi: *mut i64,
        _scale: *mut f64,
        _abnrm: *mut f64,
        _rconde: *mut f64,
        _rcondv: *mut f64,
        work: *mut f64,
        lwork: *const i64,
        iwork: *mut i64,
        info: *mut i64,
        _l1: usize,
        _l2: usize,
        _l3: usize,
        _l4: usize,
    ) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        if *n != 3 {
            *info = -5;
            return;
        }
        *work = 97.0;
        *info = 0;
        if *lwork != -1 {
            *lo = 1;
            *hi = 3;
            *iwork = 2;
            *iwork.add(3) = -2;
            if *lwork == 28 {
                *iwork = i64::MAX;
            }
            if *lwork == 29 {
                *lo = i64::MAX;
            }
        }
    }
    struct Probe {
        info: i32,
        lo: i32,
        hi: i32,
        work: [f64; 30],
        iw: [i32; 6],
    }
    unsafe fn call(sense: u8, lwork: i32) -> Probe {
        let mut result = Probe {
            info: -777,
            lo: -777,
            hi: -777,
            work: [777.0; 30],
            iw: [777; 6],
        };
        let mut a = [0.0; 9];
        let mut wr = [777.0; 3];
        let mut wi = wr;
        let mut vl = a;
        let mut vr = a;
        let mut scale = wr;
        let mut abnrm = 777.0;
        let mut re = wr;
        let mut rv = wr;
        dgeevx_(
            b"N".as_ptr(),
            b"V".as_ptr(),
            b"V".as_ptr(),
            &sense,
            &3,
            a.as_mut_ptr(),
            &3,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vl.as_mut_ptr(),
            &3,
            vr.as_mut_ptr(),
            &3,
            &mut result.lo,
            &mut result.hi,
            scale.as_mut_ptr(),
            &mut abnrm,
            re.as_mut_ptr(),
            rv.as_mut_ptr(),
            result.work.as_mut_ptr(),
            &lwork,
            result.iw.as_mut_ptr().add(1),
            &mut result.info,
        );
        result
    }
    #[test]
    fn effective_capabilities_dispatch_and_configuration_diagnostics() {
        let _lock = REGISTRY.lock().unwrap();
        let _restore = Restore(pointer(1, 32), pointer(1, 64));
        colmat_lapack_set_expert_fallback(1, 32, core::ptr::null_mut());
        colmat_lapack_set_expert_fallback(1, 64, core::ptr::null_mut());
        crate::colmat_blas_last_error();
        assert_eq!(colmat_lapack_has_expert_fallback(1, 32), 0);
        unsafe {
            let normal = call(b'N', -1);
            assert_eq!(normal.info, 0);
            assert_eq!(normal.work[0], 12.0);
            assert_eq!(crate::colmat_blas_last_error(), 0);
            let missing = call(b'E', -1);
            assert_eq!(missing.info, -777);
            assert_eq!(missing.work, [777.0; 30]);
            assert_eq!(missing.iw, [777; 6]);
            assert_eq!(crate::colmat_blas_last_error(), -1);
            let invalid = call(b'?', -1);
            assert_eq!(invalid.info, -4);
            assert_eq!(crate::colmat_blas_last_error(), 4);
        }
        let p = fake as *const () as *mut c_void;
        colmat_lapack_set_expert_fallback(1, 64, p);
        assert_eq!(colmat_lapack_has_expert_fallback(1, 32), 1);
        assert_eq!(colmat_lapack_has_expert_fallback(1, 64), 1);
        colmat_lapack_set_expert_fallback(
            1,
            64,
            crate::lapack::dgeevx_64_ as *const () as *mut c_void,
        );
        colmat_lapack_set_expert_fallback(
            1,
            64,
            crate::lapack::dgeevx_ as *const () as *mut c_void,
        );
        assert_eq!(pointer(1, 64), p);
        CALLS.store(0, Ordering::Relaxed);
        unsafe {
            let query = call(b'E', -1);
            assert_eq!((query.info, query.lo, query.hi), (0, -777, -777));
            assert_eq!(query.work[0], 97.0);
            assert_eq!(query.iw, [777; 6]);
            let calc = call(b'B', 27);
            assert_eq!((calc.info, calc.lo, calc.hi), (0, 1, 3));
            assert_eq!(calc.iw, [777, 2, 777, 777, -2, 777]);
            assert_eq!(crate::colmat_blas_last_error(), 0);
            let short = call(b'B', 26);
            assert_eq!(short.info, -21);
            assert_eq!(short.work[0], 27.0);
            assert_eq!(crate::colmat_blas_last_error(), 21);
        }
        unsafe {
            for workspace in [28, 29] {
                let overflow = call(b'B', workspace);
                assert_eq!(
                    (overflow.info, overflow.lo, overflow.hi),
                    (-777, -777, -777)
                );
                assert_eq!(overflow.iw, [777; 6]);
                assert_eq!(crate::colmat_blas_last_error(), -2);
            }
        }
        assert_eq!(CALLS.load(Ordering::Relaxed), 4);
    }
    #[test]
    fn auxiliary_copy_preserves_unwritten_slots_and_rejects_overflow() {
        unsafe {
            let mut out = [777_i32; 6];
            let ptr = out.as_mut_ptr().add(1);
            let (wide, values) = ptr.widen(3, false, true);
            *wide = 5;
            *wide.add(3) = -2;
            assert!(ptr.commit(values));
            assert_eq!(out, [777, 5, 777, 777, -2, 777]);
            let (wide, values) = ptr.widen(3, false, true);
            *wide = i64::MAX;
            assert!(!ptr.commit(values));
            assert_eq!(out, [777, 5, 777, 777, -2, 777]);
            let (wide, values) = core::ptr::null_mut::<i32>().widen(usize::MAX, true, true);
            assert!(!wide.is_null());
            assert_eq!(values.len(), 1);
            assert!(core::ptr::null_mut::<i32>().commit(values));
        }
    }
    #[test]
    fn expert_workspaces_and_registry_keys_are_explicit() {
        assert_eq!(minimum(0, false, true, true, b'B'), Some(1));
        assert_eq!(minimum(3, false, true, true, b'E'), Some(9));
        assert_eq!(minimum(3, false, false, false, b'V'), Some(27));
        assert_eq!(minimum(3, true, true, true, b'B'), Some(15));
        assert_eq!(minimum(isize::MAX, false, true, true, b'B'), None);
        assert_eq!(colmat_lapack_has_expert_fallback(-1, 64), 0);
        assert_eq!(colmat_lapack_has_expert_fallback(0, 16), 0);
    }
}
