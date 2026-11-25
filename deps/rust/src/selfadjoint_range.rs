//! Saved SYEVR/HEEVR drivers for absolute tolerances, native failures and
//! value intervals whose endpoints cannot be classified from rounded EVD values.
use crate::kernels::Scalar;
use crate::lapack::LapackInt;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

pub(crate) static FALLBACKS: [AtomicPtr<c_void>; 4] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; 4];

pub(crate) fn uncertain_endpoint<T: Scalar>(values: &[f64], vl: f64, vu: f64) -> bool {
    let single = core::mem::size_of::<T>() / if T::COMPLEX { 2 } else { 1 } == 4;
    let (eps, tiny) = if single {
        (f32::EPSILON as f64, f32::MIN_POSITIVE as f64)
    } else {
        (f64::EPSILON, f64::MIN_POSITIVE)
    };
    let norm = values.iter().map(|x| x.abs()).fold(tiny, f64::max);
    let guard = 64.0 * eps * norm;
    values
        .iter()
        .any(|&x| !x.is_finite() || (x - vl).abs() <= guard || (x - vu).abs() <= guard)
}

fn fits<I: LapackInt>(value: i64) -> bool {
    core::mem::size_of::<I>() == 8 || i32::try_from(value).is_ok()
}
fn buffer(count: usize, initial: i64) -> Option<Vec<i64>> {
    if count.checked_mul(8)? > isize::MAX as usize {
        return None;
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).ok()?;
    values.resize(count, initial);
    Some(values)
}

macro_rules! saved {
    ($name:ident,$t:ty,$r:ty,$id:expr,($($extra:ident:$extra_ty:ty,)*)) => {
        pub(crate) unsafe fn $name<I: LapackInt>(
            job: *const u8, range: *const u8, uplo: *const u8, n: *const I,
            a: *mut $t, lda: *const I, vl: *const $r, vu: *const $r,
            il: *const I, iu: *const I, abstol: *const $r, m: *mut I, w: *mut $r,
            z: *mut $t, ldz: *const I, support: *mut I, work: *mut $t, lwork: *const I,
            _rwork: *mut $r, _lrwork: *const I, iwork: *mut I, liwork: *const I, info: *mut I,
        ) {
            let pointer = FALLBACKS[$id].load(Ordering::Acquire);
            if pointer.is_null() { *m=I::from_i64(0);*info=I::from_i64(1);return; }
            let (nn,ld,lz,lw,li)=((*n).val() as i64,(*lda).val() as i64,
                (*ldz).val() as i64,(*lwork).val() as i64,(*liwork).val() as i64);
            let (lo,hi) = if crate::kernels::flag(range)==b'I' {((*il).val() as i64,(*iu).val() as i64)} else {(1,nn)};
            let _lr = if <$t>::COMPLEX {(*_lrwork).val() as i64} else {1};
            let Some(mut integer) = buffer(li.max(1) as usize,0) else {*info=I::from_i64(1);return;};
            let Some(mut supported) = buffer((nn.max(1) as usize).saturating_mul(2),i64::MIN) else {*info=I::from_i64(1);return;};
            let mut count=i64::MIN;let mut status=0i64;
            $(let $extra=_rwork;)*
            let call: unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*mut $t,
                *const i64,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,
                *mut $t,*const i64,*mut i64,*mut $t,*const i64,$($extra_ty,*const i64,)*
                *mut i64,*const i64,*mut i64,usize,usize,usize)=core::mem::transmute(pointer);
            call(job,range,uplo,&nn,a,&ld,vl,vu,&lo,&hi,abstol,&mut count,w,z,&lz,
                supported.as_mut_ptr(),work,&lw,$($extra,&_lr,)*integer.as_mut_ptr(),&li,&mut status,1,1,1);
            let vectors=crate::kernels::flag(job)==b'V';
            if !fits::<I>(status) || (count!=i64::MIN && (!(0..=nn).contains(&count) || !fits::<I>(count)))
                || integer.iter().any(|&x| !fits::<I>(x))
                || (vectors && supported.iter().any(|&x| x!=i64::MIN && !fits::<I>(x))) {
                *info=I::from_i64(1);return;
            }
            if count!=i64::MIN {*m=I::from_i64(count);}
            *info=I::from_i64(status);
            for (j,&x) in integer.iter().enumerate() {*iwork.add(j)=I::from_i64(x);}
            if vectors {for (j,&x) in supported.iter().enumerate() {
                if x!=i64::MIN {*support.add(j)=I::from_i64(x);}
            }}
        }
    };
}
saved!(s_saved, f32, f32, 0, ());
saved!(d_saved, f64, f64, 1, ());
saved!(c_saved,colmat::c32,f32,2,(rw:*mut f32,));
saved!(z_saved,colmat::c64,f64,3,(rw:*mut f64,));

#[cfg(test)]
mod tests {
    use super::*;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn endpoint_guard_uses_scalar_precision_and_spectral_scale() {
        assert!(uncertain_endpoint::<f64>(&[0.0, 2.0, 4.0], 2.0, 10.0));
        assert!(uncertain_endpoint::<f32>(&[2.0 + 1e-7, 4.0], 2.0, 10.0));
        assert!(!uncertain_endpoint::<f64>(&[2.0 + 1e-7, 4.0], 2.0, 10.0));
        assert!(!uncertain_endpoint::<f64>(&[1.0, 3.0], 0.0, 1e20));
        assert!(!uncertain_endpoint::<f64>(&[1e-200], 0.0, 3e-200));
        assert!(uncertain_endpoint::<f64>(&[f64::NAN], 0.0, 3.0));
    }

    unsafe extern "C" fn fake_driver(
        job: *const u8,
        range: *const u8,
        uplo: *const u8,
        n: *const i64,
        _a: *mut f64,
        lda: *const i64,
        _vl: *const f64,
        _vu: *const f64,
        il: *const i64,
        iu: *const i64,
        _abstol: *const f64,
        m: *mut i64,
        w: *mut f64,
        z: *mut f64,
        ldz: *const i64,
        support: *mut i64,
        work: *mut f64,
        lwork: *const i64,
        iw: *mut i64,
        li: *const i64,
        info: *mut i64,
        jlen: usize,
        rlen: usize,
        ulen: usize,
    ) {
        assert_eq!((*job, *range, *uplo), (b'V', b'V', b'L'));
        assert_eq!(
            (*n, *lda, *ldz, *il, *iu, *lwork, *li),
            (2, 2, 2, 1, 2, 52, 20)
        );
        assert_eq!((jlen, rlen, ulen), (1, 1, 1));
        *m = 1;
        *info = 0;
        *w = 7.0;
        *z = 1.0;
        *z.add(1) = 0.0;
        *support = 1;
        *support.add(1) = 2;
        *iw = 20;
        *work = 52.0;
    }

    unsafe fn bridge<I: LapackInt>() {
        let (n, ld, lw, li) = (
            I::from_i64(2),
            I::from_i64(2),
            I::from_i64(52),
            I::from_i64(20),
        );
        let (mut m, mut info) = (I::from_i64(-91), I::from_i64(-91));
        let mut a = [0.0; 4];
        let mut w = [-91.0; 2];
        let mut z = [-91.0; 4];
        let mut work = [0.0; 52];
        let mut iw = vec![I::from_i64(-91); 22];
        let mut support = vec![I::from_i64(-91); 6];
        d_saved(
            &b'V',
            &b'V',
            &b'L',
            &n,
            a.as_mut_ptr(),
            &ld,
            &0.0,
            &10.0,
            core::ptr::null(),
            core::ptr::null(),
            &0.0,
            &mut m,
            w.as_mut_ptr(),
            z.as_mut_ptr(),
            &ld,
            support.as_mut_ptr().add(1),
            work.as_mut_ptr(),
            &lw,
            core::ptr::null_mut(),
            core::ptr::null(),
            iw.as_mut_ptr().add(1),
            &li,
            &mut info,
        );
        assert_eq!((m.val(), info.val()), (1, 0));
        assert_eq!(w, [7.0, -91.0]);
        assert_eq!(z, [1.0, 0.0, -91.0, -91.0]);
        assert_eq!(iw[0].val(), -91);
        assert_eq!(iw[1].val(), 20);
        assert_eq!(iw[21].val(), -91);
        assert_eq!(
            support.iter().map(|x| x.val()).collect::<Vec<_>>(),
            [-91, 1, 2, -91, -91, -91]
        );
    }

    #[test]
    fn saved_driver_converts_integer_outputs_without_touching_guards() {
        let _lock = LOCK.lock().unwrap();
        struct Restore(*mut c_void);
        impl Drop for Restore {
            fn drop(&mut self) {
                FALLBACKS[1].store(self.0, Ordering::Release);
            }
        }
        let _restore = Restore(FALLBACKS[1].swap(fake_driver as *mut c_void, Ordering::AcqRel));
        unsafe {
            bridge::<i32>();
            bridge::<i64>();
        }
    }
}
