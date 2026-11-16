//! Saved complex LAUUM/LAUU2 implementations for nonreal diagonal compatibility.
use crate::kernels::Scalar;
use core::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static SAVED: [AtomicPtr<c_void>; 4] = [const { AtomicPtr::new(core::ptr::null_mut()) }; 4];

#[no_mangle]
pub extern "C" fn colmat_triangular_product_set_fallback(id: i32, p: *mut c_void) {
    if !(0..4).contains(&id) {
        return;
    }
    use crate::triangular_lapack::*;
    let own = [
        clauum_ as *const () as *mut c_void,
        zlauum_ as *const () as *mut c_void,
        clauum_64_ as *const () as *mut c_void,
        zlauum_64_ as *const () as *mut c_void,
        clauu2_ as *const () as *mut c_void,
        zlauu2_ as *const () as *mut c_void,
        clauu2_64_ as *const () as *mut c_void,
        zlauu2_64_ as *const () as *mut c_void,
    ];
    if own.contains(&p) {
        return;
    }
    SAVED[id as usize].store(p, Ordering::Release);
}
#[no_mangle]
pub extern "C" fn colmat_triangular_product_has_fallback(id: i32) -> i32 {
    i32::from((0..4).contains(&id) && !SAVED[id as usize].load(Ordering::Acquire).is_null())
}

pub(crate) unsafe fn maybe_saved<T: Scalar>(
    family: usize,
    u: u8,
    n: i64,
    a: *mut T,
    ld: i64,
) -> Option<i64> {
    if !T::COMPLEX || n == 0 {
        return None;
    }
    let mut needed = false;
    for i in 0..n {
        let offset = i.checked_mul(ld).and_then(|v| v.checked_add(i));
        let Some(offset) = offset.filter(|&v| v <= isize::MAX as i64) else {
            return Some(1);
        };
        let value = *a.offset(offset as isize);
        if value != T::real(value.re()) {
            needed = true;
            break;
        }
    }
    if !needed {
        return None;
    }
    let id = family * 2 + usize::from(core::mem::size_of::<T>() == 16);
    let p = SAVED[id].load(Ordering::Acquire);
    if p.is_null() {
        return Some(1);
    }
    let f: unsafe extern "C" fn(*const u8, *const i64, *mut T, *const i64, *mut i64, usize) =
        core::mem::transmute(p);
    let mut info = i64::MIN;
    f(&u, &n, a, &ld, &mut info, 1);
    if info == i64::MIN {
        Some(1)
    } else {
        Some(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use colmat::c64;
    unsafe extern "C" fn fake(
        u: *const u8,
        n: *const i64,
        a: *mut c64,
        ld: *const i64,
        info: *mut i64,
        len: usize,
    ) {
        assert_eq!((*u, *n, *ld, len), (b'U', 1, 4294967296, 1));
        *a = c64::new(17., 3.);
        *info = 0;
    }
    unsafe extern "C" fn huge(
        _: *const u8,
        _: *const i64,
        _: *mut c64,
        _: *const i64,
        info: *mut i64,
        _: usize,
    ) {
        *info = i64::MAX;
    }
    #[test]
    fn real_diagonal_needs_no_saved_driver() {
        unsafe {
            let mut a = [c64::new(2., 0.)];
            assert_eq!(maybe_saved(0, b'U', 1, a.as_mut_ptr(), 1), None);
            assert_eq!(
                maybe_saved::<c64>(0, b'U', 0, core::ptr::null_mut(), 1),
                None
            );
        }
    }
    #[test]
    fn saved_wide_arguments_and_checked_info() {
        unsafe {
            struct Restore([*mut c_void; 2]);
            impl Drop for Restore {
                fn drop(&mut self) {
                    SAVED[1].store(self.0[0], Ordering::Release);
                    SAVED[3].store(self.0[1], Ordering::Release);
                }
            }
            let _restore = Restore([
                SAVED[1].load(Ordering::Acquire),
                SAVED[3].load(Ordering::Acquire),
            ]);
            for family in 0..2 {
                let id = (family * 2 + 1) as i32;
                let pointer = fake as *const () as *mut c_void;
                colmat_triangular_product_set_fallback(id, pointer);
                colmat_triangular_product_set_fallback(
                    id,
                    crate::triangular_lapack::zlauum_64_ as *const () as *mut c_void,
                );
                assert_eq!(SAVED[id as usize].load(Ordering::Acquire), pointer);
                let mut a = [c64::new(2., 3.)];
                assert_eq!(
                    maybe_saved(family, b'U', 1, a.as_mut_ptr(), 4294967296),
                    Some(0)
                );
                assert_eq!(a[0], c64::new(17., 3.));
                colmat_triangular_product_set_fallback(id, core::ptr::null_mut());
                a[0] = c64::new(2., 3.);
                assert_eq!(maybe_saved(family, b'U', 1, a.as_mut_ptr(), 1), Some(1));
                assert_eq!(a[0], c64::new(2., 3.));
            }
            colmat_triangular_product_set_fallback(1, huge as *const () as *mut c_void);
            let mut a = [c64::new(2., 3.)];
            let mut info = [-777i32; 3];
            crate::triangular_lapack::zlauum_(
                &b'U',
                &1,
                a.as_mut_ptr(),
                &1,
                info.as_mut_ptr().add(1),
            );
            assert_eq!(info, [-777, 1, -777]);
            assert_eq!(colmat_triangular_product_has_fallback(-1), 0);
            assert_eq!(colmat_triangular_product_has_fallback(4), 0);
        }
    }
}
