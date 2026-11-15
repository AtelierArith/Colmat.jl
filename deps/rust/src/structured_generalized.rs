//! Structured storage around native generalized self-adjoint congruences.
use crate::generalized_selfadjoint::{self as reduction, GeneralizedScalar};
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::storage_layout::TriangleStorage;
use colmat::Mat;

unsafe fn unpack<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    a: *const T,
    hermitian: bool,
) -> Mat<T> {
    let mut result = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (offset, stride, _) = TriangleStorage::Packed.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { n - j } else { j + 1 };
        for k in 0..count {
            let i = first + k;
            let p = a.offset(offset as isize + k as isize * stride);
            let value = if i == j { T::diagonal(p) } else { *p };
            result[(i, j)] = value;
            if hermitian && i != j {
                result[(j, i)] = value.cj();
            }
        }
    }
    result
}
pub(crate) unsafe fn pack<T: GeneralizedScalar>(n: usize, lower: bool, a: *mut T, matrix: &Mat<T>) {
    for j in 0..n {
        let (offset, stride, _) = TriangleStorage::Packed.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { n - j } else { j + 1 };
        for k in 0..count {
            *a.offset(offset as isize + k as isize * stride) = matrix[(first + k, j)];
        }
    }
}
// Stage packed input so native EVD refusal can dispatch to the original packed
// LAPACK driver without exposing a partially standardized caller matrix.
pub(crate) unsafe fn packed_eigen<T: GeneralizedScalar>(
    kind: u8,
    job: u8,
    lower: bool,
    n: usize,
    ap: *const T,
    bp: *mut T,
) -> Result<
    crate::generalized_selfadjoint_drivers::NativeResult<T>,
    crate::generalized_selfadjoint_drivers::Failure,
> {
    let a = unpack(n, lower, ap, true);
    let mut b = unpack(n, lower, bp, true);
    let u = if lower { b'L' } else { b'U' };
    let result = crate::generalized_selfadjoint_drivers::native(
        kind,
        job,
        u,
        n,
        a.as_ref().as_ptr(),
        a.as_ref().col_stride(),
        b.as_mut().as_ptr_mut(),
        b.as_ref().col_stride(),
    );
    if !matches!(
        result,
        Err(crate::generalized_selfadjoint_drivers::Failure::Native)
    ) {
        pack(n, lower, bp, &b);
    }
    result
}
unsafe fn packed_standardize<T: GeneralizedScalar>(
    kind: u8,
    lower: bool,
    n: usize,
    a: *mut T,
    b: *const T,
) {
    if n == 0 {
        return;
    }
    let matrix = unpack(n, lower, a, true);
    let factor = unpack(n, lower, b, false);
    let u = if lower { b'L' } else { b'U' };
    let result = reduction::with_lower(
        u,
        n,
        factor.as_ref().as_ptr(),
        factor.as_ref().col_stride(),
        |l| reduction::standardize(kind, matrix, l),
    );
    for j in 0..n {
        let (offset, stride, _) = TriangleStorage::Packed.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { n - j } else { j + 1 };
        for k in 0..count {
            *a.offset(offset as isize + k as isize * stride) = result[(first + k, j)];
        }
    }
}
unsafe fn packed_entry<T: GeneralizedScalar, I: LapackInt>(
    kind: isize,
    u: u8,
    n: isize,
    a: *mut T,
    b: *const T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let addressable = n >= 0
        && (n as usize)
            .checked_mul(n as usize)
            .and_then(|x| x.checked_mul(std::mem::size_of::<T>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
    for (arg, ok) in [
        (1, (1..=3).contains(&kind)),
        (2, matches!(u, b'U' | b'L')),
        (3, addressable),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    packed_standardize(kind as u8, u == b'L', n as usize, a, b);
}
unsafe fn packed_gv<T: GeneralizedScalar, R: Scalar, I: LapackInt>(
    kind: isize,
    job: u8,
    u: u8,
    n: isize,
    ap: *mut T,
    bp: *mut T,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, (1..=3).contains(&kind)),
        (2, matches!(job, b'N' | b'V')),
        (3, matches!(u, b'U' | b'L')),
        (4, n >= 0),
        (9, ldz >= 1 && (job == b'N' || ldz >= n)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    if n == 0 {
        return true;
    }
    for (arg, ok) in [
        (4, matrix_addressable::<T>(n, n, n)),
        (9, job == b'N' || matrix_addressable::<T>(n, n, ldz)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    match packed_eigen(kind as u8, job, u == b'L', n as usize, ap, bp) {
        Ok(result) => {
            for (i, value) in result.values.into_iter().enumerate() {
                *w.add(i) = R::real(value);
            }
            pack(n as usize, u == b'L', ap, &result.reduced);
            if job == b'V' {
                crate::lapack::output(&result.vectors, z, ldz);
            }
            true
        }
        Err(crate::generalized_selfadjoint_drivers::Failure::NonPositive(index)) => {
            *info = I::from_i64(n as i64 + index as i64);
            true
        }
        Err(crate::generalized_selfadjoint_drivers::Failure::Native) => false,
    }
}
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static FALLBACKS: [AtomicPtr<c_void>; 28] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 28];
#[no_mangle]
pub extern "C" fn colmat_structured_gv_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_structured_gv_has_fallback(id: i32) -> i32 {
    FALLBACKS
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
pub(crate) fn fallback(id: usize) -> *mut c_void {
    let p = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !p.is_null(),
        "initialize structured generalized eigenvalue fallback before use"
    );
    p
}
// Packed GVD exposes its LAPACK workspace contract while sharing the native
// standardization/EVD path. It does not claim a native divide-and-conquer driver.
fn packed_dc_workspace(n: isize, vectors: bool, complex: bool) -> Option<(isize, isize, isize)> {
    if n <= 1 {
        return Some((1, 1, 1));
    }
    if vectors {
        let square = n.checked_mul(n)?.checked_mul(2)?;
        let iw = n.checked_mul(5)?.checked_add(3)?;
        if complex {
            Some((
                n.checked_mul(2)?,
                n.checked_mul(5)?.checked_add(square)?.checked_add(1)?,
                iw,
            ))
        } else {
            Some((
                n.checked_mul(6)?.checked_add(square)?.checked_add(1)?,
                1,
                iw,
            ))
        }
    } else if complex {
        Some((n, n, 1))
    } else {
        Some((n.checked_mul(2)?, 1, 1))
    }
}
unsafe fn packed_dc<T: GeneralizedScalar, R: Scalar, I: LapackInt>(
    kind: isize,
    job: u8,
    u: u8,
    n: isize,
    ap: *mut T,
    bp: *mut T,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    work: *mut T,
    lwork: isize,
    rwork: Option<(*mut R, isize)>,
    iwork: *mut I,
    liwork: isize,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, (1..=3).contains(&kind)),
        (2, matches!(job, b'N' | b'V')),
        (3, matches!(u, b'U' | b'L')),
        (4, n >= 0),
        (9, ldz >= 1 && (job == b'N' || ldz >= n)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    let Some((nw, nr, ni)) = packed_dc_workspace(n, job == b'V', T::COMPLEX) else {
        *info = I::from_i64(-11);
        crate::invalid(11);
        return true;
    };
    let max = if std::mem::size_of::<I>() == 4 {
        i32::MAX as isize
    } else {
        isize::MAX
    };
    for (arg, size) in [(11, nw), (13, nr), (if T::COMPLEX { 15 } else { 13 }, ni)] {
        if size > max {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    *work = crate::lapack::workspace::<T>(nw);
    *iwork = I::from_i64(ni as i64);
    if let Some((p, _)) = rwork {
        *p = crate::lapack::workspace::<R>(nr);
    }
    let query = lwork == -1 || liwork == -1 || rwork.is_some_and(|(_, len)| len == -1);
    for (arg, ok) in [
        (11, query || lwork >= nw),
        (13, query || rwork.is_none_or(|(_, len)| len >= nr)),
        (if T::COMPLEX { 15 } else { 13 }, query || liwork >= ni),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return true;
        }
    }
    if query {
        return true;
    }
    packed_gv(kind, job, u, n, ap, bp, w, z, ldz, info)
}
macro_rules! packed_gvd_real {($p:ident,$t:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p spgvd $s>](kind:*const $i,job:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,w:*mut $t,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
        if packed_dc((*kind).val(),flag(job),flag(u),(*n).val(),ap,bp,w,z,(*ldz).val(),work,(*lwork).val(),None,iwork,(*liwork).val(),info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut i64,*const i64,*mut i64,usize,usize)=std::mem::transmute(fallback(4+$id));
        let (_,_,need)=packed_dc_workspace((*n).val(),flag(job)==b'V',false).unwrap();
        let mut scratch=Vec::new();let (iw,len)=if std::mem::size_of::<$i>()==8 {(iwork.cast::<i64>(),(*liwork).val() as i64)} else {scratch.resize(need as usize,i64::MIN);(scratch.as_mut_ptr(),need as i64)};
        let mut status=0i64;f(&((*kind).val() as i64),job,u,&((*n).val() as i64),ap,bp,w,z,&((*ldz).val() as i64),work,&((*lwork).val() as i64),iw,&len,&mut status,1,1);*info=status as $i;
        for(k,value)in scratch.into_iter().enumerate(){if value!=i64::MIN{*iwork.add(k)=value as $i;}}
    }
}};}
macro_rules! packed_gvd_complex {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p hpgvd $s>](kind:*const $i,job:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,lrwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
        if packed_dc((*kind).val(),flag(job),flag(u),(*n).val(),ap,bp,w,z,(*ldz).val(),work,(*lwork).val(),Some((rwork,(*lrwork).val())),iwork,(*liwork).val(),info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*mut $t,*mut $r,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64,usize,usize)=std::mem::transmute(fallback(4+$id));
        let (_,_,need)=packed_dc_workspace((*n).val(),flag(job)==b'V',true).unwrap();
        let mut scratch=Vec::new();let (iw,len)=if std::mem::size_of::<$i>()==8 {(iwork.cast::<i64>(),(*liwork).val() as i64)} else {scratch.resize(need as usize,i64::MIN);(scratch.as_mut_ptr(),need as i64)};
        let mut status=0i64;f(&((*kind).val() as i64),job,u,&((*n).val() as i64),ap,bp,w,z,&((*ldz).val() as i64),work,&((*lwork).val() as i64),rwork,&((*lrwork).val() as i64),iw,&len,&mut status,1,1);*info=status as $i;
        for(k,value)in scratch.into_iter().enumerate(){if value!=i64::MIN{*iwork.add(k)=value as $i;}}
    }
}};}
packed_gvd_real!(s, f32, i32, _, 0);
packed_gvd_real!(d, f64, i32, _, 1);
packed_gvd_real!(s, f32, i64, _64_, 0);
packed_gvd_real!(d, f64, i64, _64_, 1);
packed_gvd_complex!(c, colmat::c32, f32, i32, _, 2);
packed_gvd_complex!(z, colmat::c64, f64, i32, _, 3);
packed_gvd_complex!(c, colmat::c32, f32, i64, _64_, 2);
packed_gvd_complex!(z, colmat::c64, f64, i64, _64_, 3);

macro_rules! packed_gv_real {($p:ident,$t:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p spgv $s>](kind:*const $i,job:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,w:*mut $t,z:*mut $t,ldz:*const $i,work:*mut $t,info:*mut $i) {
        if packed_gv((*kind).val(),flag(job),flag(u),(*n).val(),ap,bp,w,z,(*ldz).val(),info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*mut i64,usize,usize)=std::mem::transmute(fallback($id));
        let mut status=0i64;f(&((*kind).val() as i64),job,u,&((*n).val() as i64),ap,bp,w,z,&((*ldz).val() as i64),work,&mut status,1,1);*info=status as $i;
    }
}};}
macro_rules! packed_gv_complex {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p hpgv $s>](kind:*const $i,job:*const u8,u:*const u8,n:*const $i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,rwork:*mut $r,info:*mut $i) {
        if packed_gv((*kind).val(),flag(job),flag(u),(*n).val(),ap,bp,w,z,(*ldz).val(),info) {return;}
        let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*mut $t,*mut $r,*mut $t,*const i64,*mut $t,*mut $r,*mut i64,usize,usize)=std::mem::transmute(fallback($id));
        let mut status=0i64;f(&((*kind).val() as i64),job,u,&((*n).val() as i64),ap,bp,w,z,&((*ldz).val() as i64),work,rwork,&mut status,1,1);*info=status as $i;
    }
}};}
packed_gv_real!(s, f32, i32, _, 0);
packed_gv_real!(d, f64, i32, _, 1);
packed_gv_real!(s, f32, i64, _64_, 0);
packed_gv_real!(d, f64, i64, _64_, 1);
packed_gv_complex!(c, colmat::c32, f32, i32, _, 2);
packed_gv_complex!(z, colmat::c64, f64, i32, _, 3);
packed_gv_complex!(c, colmat::c32, f32, i64, _64_, 2);
packed_gv_complex!(z, colmat::c64, f64, i64, _64_, 3);

macro_rules! packed_exports {($p:ident,$name:ident,$t:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $name $s>](kind:*const $i,u:*const u8,n:*const $i,a:*mut $t,b:*const $t,info:*mut $i) {
        packed_entry((*kind).val(),flag(u),(*n).val(),a,b,info);
    }
}};}
packed_exports!(s, spgst, f32, i32, _);
packed_exports!(d, spgst, f64, i32, _);
packed_exports!(c, hpgst, colmat::c32, i32, _);
packed_exports!(z, hpgst, colmat::c64, i32, _);
packed_exports!(s, spgst, f32, i64, _64_);
packed_exports!(d, spgst, f64, i64, _64_);
packed_exports!(c, hpgst, colmat::c32, i64, _64_);
packed_exports!(z, hpgst, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn dc_abi<T: GeneralizedScalar, R: Scalar, I: LapackInt>(tol: f64) {
        let (nw, nr, ni) = if T::COMPLEX { (6, 34, 18) } else { (37, 1, 18) };
        for which in 0..if T::COMPLEX { 3 } else { 2 } {
            let mut work = [T::real(777.0); 3];
            let mut rw = [R::real(777.0); 3];
            let mut iw = [I::from_i64(-777); 3];
            let mut info = I::from_i64(-777);
            assert!(unsafe {
                packed_dc(
                    1,
                    b'V',
                    b'L',
                    3,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    3,
                    &mut work[1],
                    if which == 0 { -1 } else { 0 },
                    if T::COMPLEX {
                        Some((&mut rw[1], if which == 2 { -1 } else { 0 }))
                    } else {
                        None
                    },
                    &mut iw[1],
                    if which == 1 { -1 } else { 0 },
                    &mut info,
                )
            });
            assert_eq!(info.val(), 0);
            assert_eq!(crate::colmat_blas_last_error(), 0);
            assert_eq!(work[1].re(), nw as f64);
            assert_eq!(iw[1].val(), ni);
            if T::COMPLEX {
                assert_eq!(rw[1].re(), nr as f64);
            }
            for k in [0usize, 2] {
                assert_eq!(work[k].re(), 777.0);
                assert_eq!(rw[k].re(), 777.0);
                assert_eq!(iw[k].val(), -777);
            }
        }
        let mut work = T::real(777.0);
        let mut rw = R::real(777.0);
        let mut iw = I::from_i64(-777);
        let mut info = I::from_i64(-777);
        for (lw, lr, li, arg) in [
            (nw - 1, nr, ni, 11),
            (nw, nr, ni - 1, if T::COMPLEX { 15 } else { 13 }),
        ] {
            assert!(unsafe {
                packed_dc(
                    1,
                    b'V',
                    b'L',
                    3,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    3,
                    &mut work,
                    lw,
                    if T::COMPLEX {
                        Some((&mut rw, lr))
                    } else {
                        None
                    },
                    &mut iw,
                    li,
                    &mut info,
                )
            });
            assert_eq!(info.val(), -arg);
            assert_eq!(crate::colmat_blas_last_error(), arg as i32);
        }
        if T::COMPLEX {
            assert!(unsafe {
                packed_dc(
                    1,
                    b'V',
                    b'L',
                    3,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    3,
                    &mut work,
                    nw,
                    Some((&mut rw, nr - 1)),
                    &mut iw,
                    ni,
                    &mut info,
                )
            });
            assert_eq!(info.val(), -13);
            assert_eq!(crate::colmat_blas_last_error(), 13);
        }
        let mut a = T::real(3.0);
        let mut b = T::real(2.0);
        let mut w = R::real(777.0);
        let mut z = T::real(777.0);
        assert!(unsafe {
            packed_dc(
                1,
                b'V',
                b'L',
                1,
                &mut a,
                &mut b,
                &mut w,
                &mut z,
                isize::MAX,
                &mut work,
                1,
                if T::COMPLEX { Some((&mut rw, 1)) } else { None },
                &mut iw,
                1,
                &mut info,
            )
        });
        assert_eq!(info.val(), 0);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        assert!((w.re() - 1.5).abs() < tol);
        assert!((z.magnitude() - 1.0 / 2.0f64.sqrt()).abs() < tol);
        assert_eq!(work.re(), 1.0);
        assert_eq!(iw.val(), 1);
    }
    #[test]
    fn packed_dc_query_and_minimum_work_integer_abis() {
        dc_abi::<f32, f32, i32>(1e-4);
        dc_abi::<f32, f32, i64>(1e-4);
        dc_abi::<f64, f64, i32>(1e-11);
        dc_abi::<f64, f64, i64>(1e-11);
        dc_abi::<colmat::c32, f32, i32>(1e-4);
        dc_abi::<colmat::c32, f32, i64>(1e-4);
        dc_abi::<colmat::c64, f64, i32>(1e-11);
        dc_abi::<colmat::c64, f64, i64>(1e-11);
    }
    fn driver_abi<T: GeneralizedScalar, R: Scalar, I: LapackInt>(tol: f64) {
        for kind in 1..=3 {
            for u in [b'U', b'L'] {
                let mut a = T::real(3.0);
                let mut b = T::real(2.0);
                let mut w = R::real(777.0);
                let mut z = T::real(777.0);
                let mut info = [I::from_i64(-777); 3];
                assert!(unsafe {
                    packed_gv(
                        kind,
                        b'V',
                        u,
                        1,
                        &mut a,
                        &mut b,
                        &mut w,
                        &mut z,
                        isize::MAX,
                        &mut info[1],
                    )
                });
                assert_eq!(info[1].val(), 0);
                assert_eq!(info[0].val(), -777);
                assert_eq!(info[2].val(), -777);
                assert_eq!(crate::colmat_blas_last_error(), 0);
                assert!((w.re() - if kind == 1 { 1.5 } else { 6.0 }).abs() < tol);
                let expected = if kind == 3 {
                    2.0f64.sqrt()
                } else {
                    1.0 / 2.0f64.sqrt()
                };
                assert!((z.magnitude() - expected).abs() < tol);
                assert!((b.re() - 2.0f64.sqrt()).abs() < tol);
            }
        }
        for u in [b'U', b'L'] {
            let mut a = [T::real(3.0), T::z(), T::real(4.0)];
            let olda = a;
            let mut b = [T::real(2.0), T::z(), T::real(-1.0)];
            let mut w = [R::real(777.0); 2];
            let mut z = [T::real(777.0); 4];
            let mut info = I::from_i64(-777);
            assert!(unsafe {
                packed_gv(
                    1,
                    b'V',
                    u,
                    2,
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    w.as_mut_ptr(),
                    z.as_mut_ptr(),
                    2,
                    &mut info,
                )
            });
            assert_eq!(info.val(), 4);
            assert_eq!(a, olda);
            assert!(w.iter().all(|x| x.re() == 777.0));
            assert!(z.iter().all(|x| x.re() == 777.0));
            assert!((b[0].re() - 2.0f64.sqrt()).abs() < tol);
            assert_eq!(b[2].re(), -1.0);
        }
    }
    #[test]
    fn packed_driver_integer_abis_and_nonpositive_b() {
        driver_abi::<f32, f32, i32>(1e-4);
        driver_abi::<f32, f32, i64>(1e-4);
        driver_abi::<f64, f64, i32>(1e-11);
        driver_abi::<f64, f64, i64>(1e-11);
        driver_abi::<colmat::c32, f32, i32>(1e-4);
        driver_abi::<colmat::c32, f32, i64>(1e-4);
        driver_abi::<colmat::c64, f64, i32>(1e-11);
        driver_abi::<colmat::c64, f64, i64>(1e-11);
    }
    fn eigen<T: GeneralizedScalar>(phase: T, tol: f64) {
        for n in [1usize, 4] {
            for lower in [false, true] {
                for kind in 1..=3 {
                    let l = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(2.0 + i as f64 * 0.1)
                        } else if i > j {
                            phase
                        } else {
                            T::z()
                        }
                    });
                    let a = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(3.0 + i as f64)
                        } else if i > j {
                            phase
                        } else {
                            phase.cj()
                        }
                    });
                    let b = Mat::<T>::from_fn(n, n, |i, j| {
                        let mut value = T::z();
                        for k in 0..n {
                            value = value + l[(i, k)] * l[(j, k)].cj();
                        }
                        value
                    });
                    let mut ap = vec![T::real(777.0); n * (n + 1) / 2 + 2];
                    let mut bp = ap.clone();
                    unsafe {
                        pack(n, lower, ap.as_mut_ptr().add(1), &a);
                        pack(n, lower, bp.as_mut_ptr().add(1), &b);
                    }
                    let result = unsafe {
                        packed_eigen(
                            kind,
                            b'V',
                            lower,
                            n,
                            ap.as_ptr().add(1),
                            bp.as_mut_ptr().add(1),
                        )
                    };
                    let result = match result {
                        Ok(x) => x,
                        Err(_) => panic!("positive definite packed eigensystem failed"),
                    };
                    assert!(result.values.windows(2).all(|x| x[0] <= x[1]));
                    for k in 0..n {
                        let az: Vec<T> = (0..n)
                            .map(|i| {
                                let mut x = T::z();
                                for j in 0..n {
                                    x = x + a[(i, j)] * result.vectors[(j, k)];
                                }
                                x
                            })
                            .collect();
                        let bz: Vec<T> = (0..n)
                            .map(|i| {
                                let mut x = T::z();
                                for j in 0..n {
                                    x = x + b[(i, j)] * result.vectors[(j, k)];
                                }
                                x
                            })
                            .collect();
                        for i in 0..n {
                            let (lhs, rhs) = if kind == 1 {
                                (az[i], bz[i] * T::real(result.values[k]))
                            } else {
                                let mut x = T::z();
                                for j in 0..n {
                                    x = x + if kind == 2 {
                                        a[(i, j)] * bz[j]
                                    } else {
                                        b[(i, j)] * az[j]
                                    };
                                }
                                (x, result.vectors[(i, k)] * T::real(result.values[k]))
                            };
                            assert!((lhs - rhs).magnitude() < tol);
                        }
                    }
                    let factor = unsafe { unpack(n, lower, bp.as_ptr().add(1), false) };
                    for j in 0..n {
                        for i in 0..n {
                            let mut value = T::z();
                            for k in 0..n {
                                value = value
                                    + if lower {
                                        factor[(i, k)] * factor[(j, k)].cj()
                                    } else {
                                        factor[(k, i)].cj() * factor[(k, j)]
                                    };
                            }
                            assert!((value - b[(i, j)]).magnitude() < tol);
                        }
                    }
                    assert_eq!(ap[0].re(), 777.0);
                    assert_eq!(ap[ap.len() - 1].re(), 777.0);
                    assert_eq!(bp[0].re(), 777.0);
                    assert_eq!(bp[bp.len() - 1].re(), 777.0);
                }
            }
        }
    }
    #[test]
    fn packed_generalized_eigen_residuals() {
        eigen::<f32>(0.1, 2e-3);
        eigen::<f64>(0.1, 1e-10);
        eigen::<colmat::c32>(colmat::c32::new(0.1, 0.15), 2e-3);
        eigen::<colmat::c64>(colmat::c64::new(0.1, 0.15), 1e-10);
    }
    fn check<T: GeneralizedScalar, I: LapackInt>(phase: T, tol: f64) {
        for n in [0usize, 1, 4] {
            for lower in [false, true] {
                for kind in 1..=3 {
                    let l = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(2.0 + i as f64 * 0.2)
                        } else if i > j {
                            phase
                        } else {
                            T::z()
                        }
                    });
                    let a = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(3.0 + i as f64)
                        } else if i > j {
                            phase * T::real(0.3)
                        } else {
                            phase.cj() * T::real(0.3)
                        }
                    });
                    let mut ap = vec![T::real(777.0); n * (n + 1) / 2 + 2];
                    let mut bp = ap.clone();
                    for j in 0..n {
                        let (offset, _, _) = TriangleStorage::Packed.segment(n, lower, j);
                        let first = if lower { j } else { 0 };
                        let count = if lower { n - j } else { j + 1 };
                        for k in 0..count {
                            let i = first + k;
                            let imaginary = (phase - phase.cj()) * T::real(777.0);
                            ap[1 + offset + k] =
                                a[(i, j)] + if i == j { imaginary } else { T::z() };
                            bp[1 + offset + k] = (if lower { l[(i, j)] } else { l[(j, i)].cj() })
                                + if i == j { imaginary } else { T::z() };
                        }
                    }
                    let oldb = bp.clone();
                    let mut info = I::from_i64(-777);
                    unsafe {
                        packed_entry(
                            kind,
                            if lower { b'L' } else { b'U' },
                            n as isize,
                            ap.as_mut_ptr().add(1),
                            bp.as_ptr().add(1),
                            &mut info,
                        );
                    }
                    assert_eq!(info.val(), 0);
                    assert_eq!(crate::colmat_blas_last_error(), 0);
                    assert_eq!(bp, oldb);
                    assert_eq!(ap[0].re(), 777.0);
                    assert_eq!(ap[ap.len() - 1].re(), 777.0);
                    let c = unsafe { unpack(n, lower, ap.as_ptr().add(1), true) };
                    for j in 0..n {
                        for i in 0..n {
                            let mut value = T::z();
                            for k in 0..n {
                                for m in 0..n {
                                    value = value
                                        + if kind == 1 {
                                            l[(i, k)] * c[(k, m)] * l[(j, m)].cj()
                                        } else {
                                            l[(k, i)].cj() * a[(k, m)] * l[(m, j)]
                                        };
                                }
                            }
                            let expected = if kind == 1 { a[(i, j)] } else { c[(i, j)] };
                            assert!((value - expected).magnitude() < tol);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn packed_congruences_all_scalar_integer_types() {
        check::<f32, i32>(0.2, 1e-4);
        check::<f32, i64>(0.2, 1e-4);
        check::<f64, i32>(0.2, 1e-11);
        check::<f64, i64>(0.2, 1e-11);
        check::<colmat::c32, i32>(colmat::c32::new(0.2, 0.3), 1e-4);
        check::<colmat::c32, i64>(colmat::c32::new(0.2, 0.3), 1e-4);
        check::<colmat::c64, i32>(colmat::c64::new(0.2, 0.3), 1e-11);
        check::<colmat::c64, i64>(colmat::c64::new(0.2, 0.3), 1e-11);
    }
}
