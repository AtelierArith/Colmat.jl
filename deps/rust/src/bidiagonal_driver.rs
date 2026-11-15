//! Checked bidiagonal SVD ABI and accumulation contracts.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;

unsafe fn bdsqr<R: Scalar, T: Scalar, I: LapackInt>(
    uplo: u8,
    n: isize,
    ncvt: isize,
    nru: isize,
    ncc: isize,
    d: *mut R,
    e: *mut R,
    vt: *mut T,
    ldvt: isize,
    u: *mut T,
    ldu: isize,
    c: *mut T,
    ldc: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(uplo, b'U' | b'L')),
        (2, n >= 0),
        (3, ncvt >= 0),
        (4, nru >= 0),
        (5, ncc >= 0),
        (9, ldvt >= if ncvt == 0 { 1 } else { n.max(1) }),
        (11, ldu >= nru.max(1)),
        (13, ldc >= if ncc == 0 { 1 } else { n.max(1) }),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if n == 0 {
        return;
    }
    // The reference singleton path reads no U or C, and reads VT only
    // when correcting a negative diagonal. Their unused strides need not
    // describe an addressable full matrix.
    if n == 1 {
        if (*d).re() < 0.0 {
            if !matrix_addressable::<T>(1, ncvt, ldvt) {
                *info = I::from_i64(-9);
                crate::invalid(9);
                return;
            }
            for j in 0..ncvt as usize {
                *vt.offset(j as isize * ldvt) = -*vt.offset(j as isize * ldvt);
            }
        }
        *d = R::real((*d).re().abs());
        return;
    }
    for (arg, ok) in [
        (2, matrix_addressable::<R>(n, 1, n)),
        (9, matrix_addressable::<T>(n, ncvt, ldvt)),
        (11, matrix_addressable::<T>(nru, n, ldu)),
        (13, matrix_addressable::<T>(n, ncc, ldc)),
        (2, matrix_addressable::<R>(n, n, n)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let n = n as usize;
    let diagonal = std::slice::from_raw_parts(d, n).to_vec();
    let offdiagonal = std::slice::from_raw_parts(e, n - 1).to_vec();
    let Ok((values, left, right)) =
        crate::bidiagonal_svd::decompose(&diagonal, &offdiagonal, uplo == b'U', false)
    else {
        // No external factor has been committed. Report the number of
        // off-diagonals still present in the retained input bidiagonal.
        let remaining = offdiagonal.iter().filter(|x| **x != R::z()).count();
        *info = I::from_i64(remaining.max(1) as i64);
        return;
    };
    let mut out_vt = vec![T::z(); n * ncvt as usize];
    let mut out_u = vec![T::z(); nru as usize * n];
    let mut out_c = vec![T::z(); n * ncc as usize];
    for j in 0..ncvt as usize {
        for i in 0..n {
            for k in 0..n {
                out_vt[i + j * n] = out_vt[i + j * n]
                    + T::real(right[(k, i)].re()) * *vt.offset(k as isize + j as isize * ldvt);
            }
        }
    }
    for j in 0..n {
        for i in 0..nru as usize {
            for k in 0..n {
                out_u[i + j * nru as usize] = out_u[i + j * nru as usize]
                    + *u.offset(i as isize + k as isize * ldu) * T::real(left[(k, j)].re());
            }
        }
    }
    for j in 0..ncc as usize {
        for i in 0..n {
            for k in 0..n {
                out_c[i + j * n] = out_c[i + j * n]
                    + T::real(left[(k, i)].re()) * *c.offset(k as isize + j as isize * ldc);
            }
        }
    }
    for i in 0..n {
        *d.add(i) = values[i];
    }
    for i in 0..n - 1 {
        *e.add(i) = R::z();
    }
    for j in 0..ncvt as usize {
        for i in 0..n {
            *vt.offset(i as isize + j as isize * ldvt) = out_vt[i + j * n];
        }
    }
    for j in 0..n {
        for i in 0..nru as usize {
            *u.offset(i as isize + j as isize * ldu) = out_u[i + j * nru as usize];
        }
    }
    for j in 0..ncc as usize {
        for i in 0..n {
            *c.offset(i as isize + j as isize * ldc) = out_c[i + j * n];
        }
    }
}
macro_rules! exports {($p:ident,$r:ty,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p bdsqr $s>](uplo:*const u8,n:*const $i,ncvt:*const $i,nru:*const $i,ncc:*const $i,d:*mut $r,e:*mut $r,vt:*mut $t,ldvt:*const $i,u:*mut $t,ldu:*const $i,c:*mut $t,ldc:*const $i,_work:*mut $r,info:*mut $i) {
        bdsqr(flag(uplo),*n as isize,*ncvt as isize,*nru as isize,*ncc as isize,d,e,vt,*ldvt as isize,u,*ldu as isize,c,*ldc as isize,info);
    }
}};}
exports!(s, f32, f32, i32, _);
exports!(d, f64, f64, i32, _);
exports!(c, f32, colmat::c32, i32, _);
exports!(z, f64, colmat::c64, i32, _);
exports!(s, f32, f32, i64, _64_);
exports!(d, f64, f64, i64, _64_);
exports!(c, f32, colmat::c32, i64, _64_);
exports!(z, f64, colmat::c64, i64, _64_);

unsafe fn bdsvdx<R: Scalar, I: LapackInt>(
    uplo: u8,
    job: u8,
    range: u8,
    n: isize,
    d: *const R,
    e: *const R,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
    ns: *mut I,
    s: *mut R,
    z: *mut R,
    ldz: isize,
    iwork: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let twice = n.checked_mul(2);
    for (arg, ok) in [
        (1, matches!(uplo, b'U' | b'L')),
        (2, matches!(job, b'N' | b'V')),
        (3, matches!(range, b'A' | b'V' | b'I')),
        (4, n >= 0),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if n > 0 {
        let bad = if range == b'V' {
            if (*vl).re() < 0.0 {
                Some(7)
            } else if (*vu).re() <= (*vl).re() {
                Some(8)
            } else {
                None
            }
        } else if range == b'I' {
            if (*il).val() < 1 || (*il).val() > n {
                Some(9)
            } else if (*iu).val() < (*il).val() || (*iu).val() > n {
                Some(10)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(arg) = bad {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if ldz < 1 || (job == b'V' && twice.is_none_or(|rows| ldz < rows)) {
        *info = I::from_i64(-14);
        crate::invalid(14);
        return;
    }
    *ns = I::from_i64(0);
    if n == 0 {
        return;
    }
    if !matrix_addressable::<R>(n, n, n)
        || (job == b'V' && !matrix_addressable::<R>(twice.unwrap(), n, ldz))
    {
        *info = I::from_i64(-14);
        crate::invalid(14);
        return;
    }
    let n = n as usize;
    let diagonal = std::slice::from_raw_parts(d, n).to_vec();
    let off = if n == 1 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(e, n - 1).to_vec()
    };
    let Ok((values, u, v)) = crate::bidiagonal_svd::decompose(&diagonal, &off, uplo == b'U', false)
    else {
        *info = I::from_i64(2 * n as i64 + 1);
        return;
    };
    let selected: Vec<usize> = (0..n)
        .filter(|&j| match range {
            b'V' => {
                if n == 1 {
                    (*vl).re() < values[j].re() && values[j].re() <= (*vu).re()
                } else {
                    (*vl).re() <= values[j].re() && values[j].re() < (*vu).re()
                }
            }
            b'I' => j + 1 >= (*il).val() as usize && j < (*iu).val() as usize,
            _ => true,
        })
        .collect();
    // Reference sorts in decreasing order. Its N=1 quick path uses
    // (VL,VU], whereas the general path selects [VL,VU).
    for (out, &j) in selected.iter().enumerate() {
        *s.add(out) = values[j];
        if job == b'V' {
            for i in 0..n {
                *z.offset(i as isize + out as isize * ldz) = u[(i, j)];
                *z.offset((i + n) as isize + out as isize * ldz) = v[(i, j)];
            }
            *iwork.add(out) = I::from_i64(0);
        }
    }
    *ns = I::from_i64(selected.len() as i64);
}
macro_rules! selected_exports {($p:ident,$r:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p bdsvdx $s>](uplo:*const u8,job:*const u8,range:*const u8,n:*const $i,d:*const $r,e:*const $r,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,ns:*mut $i,s:*mut $r,z:*mut $r,ldz:*const $i,_work:*mut $r,iwork:*mut $i,info:*mut $i) {
        bdsvdx(flag(uplo),flag(job),flag(range),*n as isize,d,e,vl,vu,il,iu,ns,s,z,*ldz as isize,iwork,info);
    }
}};}
selected_exports!(s, f32, i32, _);
selected_exports!(d, f64, i32, _);
selected_exports!(s, f32, i64, _64_);
selected_exports!(d, f64, i64, _64_);

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static BDSDC_FALLBACKS: [AtomicPtr<c_void>; 2] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; 2];
#[no_mangle]
pub extern "C" fn colmat_bdsdc_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = BDSDC_FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_bdsdc_has_fallback(id: i32) -> i32 {
    BDSDC_FALLBACKS
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
unsafe fn bdsdc<R: Scalar, I: LapackInt>(
    id: usize,
    uplo: u8,
    comp: u8,
    n: isize,
    d: *mut R,
    e: *mut R,
    u: *mut R,
    ldu: isize,
    vt: *mut R,
    ldvt: isize,
    q: *mut R,
    iq: *mut I,
    work: *mut R,
    iwork: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(uplo, b'U' | b'L')),
        (2, matches!(comp, b'N' | b'I' | b'P')),
        (3, n >= 0),
        (7, ldu >= if comp == b'I' { n.max(1) } else { 1 }),
        (9, ldvt >= if comp == b'I' { n.max(1) } else { 1 }),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !matrix_addressable::<R>(n, n, n)
        || (comp == b'I'
            && (!matrix_addressable::<R>(n, n, ldu) || !matrix_addressable::<R>(n, n, ldvt)))
    {
        *info = I::from_i64(-3);
        crate::invalid(3);
        return;
    }
    if comp == b'P' {
        // Preserve LAPACK's implementation-specific compact divide-and-conquer
        // tree. A dense U/V pair cannot stand in for this representation.
        let pointer = BDSDC_FALLBACKS[id].load(Ordering::Acquire);
        assert!(
            !pointer.is_null(),
            "Colmat BDSDC compact fallback must be configured before use"
        );
        let call: unsafe extern "C" fn(
            *const u8,
            *const u8,
            *const i64,
            *mut R,
            *mut R,
            *mut R,
            *const i64,
            *mut R,
            *const i64,
            *mut R,
            *mut i64,
            *mut R,
            *mut i64,
            *mut i64,
        ) = std::mem::transmute(pointer);
        let (ni, lu, lv) = (n as i64, ldu as i64, ldvt as i64);
        let mut result = 0i64;
        if std::mem::size_of::<I>() == 8 {
            call(
                &uplo,
                &comp,
                &ni,
                d,
                e,
                u,
                &lu,
                vt,
                &lv,
                q,
                iq.cast(),
                work,
                iwork.cast(),
                &mut result,
            );
        } else {
            let n = n as usize;
            // This is an allocation upper bound, not a claim about SMLSIZ.
            // Copy back only entries actually written by the reference, so
            // callers need only its documented compact integer extent.
            let levels = usize::BITS as usize - n.leading_zeros() as usize;
            let mut integers = vec![i64::MIN; n * 3 * (levels + 2)];
            let mut scratch = vec![i64::MIN; n * 8];
            call(
                &uplo,
                &comp,
                &ni,
                d,
                e,
                u,
                &lu,
                vt,
                &lv,
                q,
                integers.as_mut_ptr(),
                work,
                scratch.as_mut_ptr(),
                &mut result,
            );
            for (j, &value) in integers.iter().enumerate() {
                if value != i64::MIN {
                    *iq.add(j) = I::from_i64(value);
                }
            }
            for (j, &value) in scratch.iter().enumerate() {
                if value != i64::MIN {
                    *iwork.add(j) = I::from_i64(value);
                }
            }
        }
        *info = I::from_i64(result);
        return;
    }
    let n = n as usize;
    let diagonal = std::slice::from_raw_parts(d, n).to_vec();
    let off = if n == 1 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(e, n - 1).to_vec()
    };
    let Ok((values, left, right)) =
        crate::bidiagonal_svd::decompose(&diagonal, &off, uplo == b'U', true)
    else {
        *info = I::from_i64(1);
        return;
    };
    for i in 0..n {
        *d.add(i) = values[i];
    }
    for i in 0..n - 1 {
        *e.add(i) = R::z();
    }
    if comp == b'I' {
        for j in 0..n {
            for i in 0..n {
                *u.offset(i as isize + j as isize * ldu) = left[(i, j)];
                *vt.offset(i as isize + j as isize * ldvt) = right[(j, i)];
            }
        }
    }
}
macro_rules! dc_exports {($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p bdsdc $s>](uplo:*const u8,comp:*const u8,n:*const $i,d:*mut $r,e:*mut $r,u:*mut $r,ldu:*const $i,vt:*mut $r,ldvt:*const $i,q:*mut $r,iq:*mut $i,work:*mut $r,iwork:*mut $i,info:*mut $i) {
        bdsdc($id,flag(uplo),flag(comp),*n as isize,d,e,u,*ldu as isize,vt,*ldvt as isize,q,iq,work,iwork,info);
    }
}};}
dc_exports!(s, f32, i32, _, 0);
dc_exports!(d, f64, i32, _, 1);
dc_exports!(s, f32, i64, _64_, 0);
dc_exports!(d, f64, i64, _64_, 1);

static LASQ_FALLBACKS: [AtomicPtr<c_void>; 2] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 2];
#[no_mangle]
pub extern "C" fn colmat_lasq_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = LASQ_FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_lasq_has_fallback(id: i32) -> i32 {
    LASQ_FALLBACKS
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
unsafe fn lasq1<R: Scalar + colmat::traits::RealField, I: LapackInt>(
    id: usize,
    n: isize,
    d: *mut R,
    e: *mut R,
    work: *mut R,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n < 0 || !matrix_addressable::<R>(n, 6, n.max(1)) {
        *info = I::from_i64(-1);
        crate::invalid(1);
        return;
    }
    if n == 0 {
        return;
    }
    let count = n as usize;
    let diagonal = std::slice::from_raw_parts(d, count);
    let off = if count == 1 {
        &[]
    } else {
        std::slice::from_raw_parts(e, count - 1)
    };
    if let Ok(values) = crate::bidiagonal_svd::dqds_values(diagonal, off) {
        for (i, value) in values.into_iter().enumerate() {
            *d.add(i) = value;
        }
        return;
    }
    // Rejected native results (including successful but underresolved values,
    // colmat-rs#209) have not changed caller D/E. Preserve reference LASQ1
    // recovery status, intermediate factors and WORK diagnostics.
    let pointer = LASQ_FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat LASQ1 recovery fallback must be configured before use"
    );
    let call: unsafe extern "C" fn(*const i64, *mut R, *mut R, *mut R, *mut i64) =
        std::mem::transmute(pointer);
    let mut result = 0i64;
    let ni = n as i64;
    call(&ni, d, e, work, &mut result);
    *info = I::from_i64(result);
}
macro_rules! lasq_exports {($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p lasq1 $s>](n:*const $i,d:*mut $r,e:*mut $r,work:*mut $r,info:*mut $i) {
        lasq1($id,*n as isize,d,e,work,info);
    }
}};}
lasq_exports!(s, f32, i32, _, 0);
lasq_exports!(d, f64, i32, _, 1);
lasq_exports!(s, f32, i64, _64_, 0);
lasq_exports!(d, f64, i64, _64_, 1);

#[cfg(test)]
mod tests {
    use super::*;
    fn accumulation<R: Scalar, T: Scalar, I: LapackInt>(imaginary: T, tol: f64) {
        for n in [1usize, 3, 7] {
            for upper in [false, true] {
                let ld = n + 2;
                let diagonal: Vec<R> = (0..n)
                    .map(|i| {
                        R::real(if i % 2 == 0 {
                            -(i as f64 + 1.0)
                        } else {
                            i as f64 + 1.0
                        })
                    })
                    .collect();
                let off: Vec<R> = (0..n.saturating_sub(1))
                    .map(|i| R::real(if i == 2 { 0.0 } else { 0.3 }))
                    .collect();
                let mut d = diagonal.clone();
                let mut e = off.clone();
                let mut u = vec![T::real(777.0); ld * n];
                let mut vt = u.clone();
                let mut c = vec![T::real(777.0); ld * 2];
                for j in 0..n {
                    for i in 0..n {
                        u[i + j * ld] = if i == j { T::o() } else { T::z() };
                        vt[i + j * ld] = u[i + j * ld];
                    }
                }
                for j in 0..2 {
                    for i in 0..n {
                        c[i + j * ld] = T::real((i + j + 1) as f64) / T::real(7.0) + imaginary;
                    }
                }
                let before = c.clone();
                let mut info = I::from_i64(-777);
                unsafe {
                    bdsqr(
                        if upper { b'U' } else { b'L' },
                        n as isize,
                        n as isize,
                        n as isize,
                        2,
                        d.as_mut_ptr(),
                        e.as_mut_ptr(),
                        vt.as_mut_ptr(),
                        ld as isize,
                        u.as_mut_ptr(),
                        ld as isize,
                        c.as_mut_ptr(),
                        ld as isize,
                        &mut info,
                    );
                }
                assert_eq!(info.val(), 0);
                assert_eq!(crate::colmat_blas_last_error(), 0);
                for j in 0..n {
                    for i in 0..n {
                        let mut product = T::z();
                        let mut dot = T::z();
                        for k in 0..n {
                            product = product + u[i + k * ld] * T::real(d[k].re()) * vt[k + j * ld];
                            dot = dot + u[k + i * ld].cj() * u[k + j * ld];
                        }
                        let expected = if i == j {
                            diagonal[i]
                        } else if (upper && j == i + 1) || (!upper && i == j + 1) {
                            off[i.min(j)]
                        } else {
                            R::z()
                        };
                        assert!((product - T::real(expected.re())).magnitude() < tol * n as f64);
                        assert!((dot - if i == j { T::o() } else { T::z() }).magnitude() < tol);
                    }
                }
                for j in 0..2 {
                    for i in 0..n {
                        let mut expected = T::z();
                        for k in 0..n {
                            expected = expected + u[k + i * ld].cj() * before[k + j * ld];
                        }
                        assert!((c[i + j * ld] - expected).magnitude() < tol);
                    }
                }
                for j in 0..n {
                    for i in n..ld {
                        assert_eq!(u[i + j * ld].re(), 777.0);
                        assert_eq!(vt[i + j * ld].re(), 777.0);
                    }
                }
                for j in 0..2 {
                    for i in n..ld {
                        assert_eq!(c[i + j * ld].re(), 777.0);
                    }
                }
                assert!(e.iter().all(|x| x.re() == 0.0));
            }
        }
    }
    #[test]
    fn accumulation_all_scalar_and_integer_abis() {
        accumulation::<f32, f32, i32>(0.0, 1e-4);
        accumulation::<f32, f32, i64>(0.0, 1e-4);
        accumulation::<f64, f64, i32>(0.0, 1e-11);
        accumulation::<f64, f64, i64>(0.0, 1e-11);
        accumulation::<f32, colmat::c32, i32>(colmat::c32::new(0.0, 0.25), 1e-4);
        accumulation::<f32, colmat::c32, i64>(colmat::c32::new(0.0, 0.25), 1e-4);
        accumulation::<f64, colmat::c64, i32>(colmat::c64::new(0.0, 0.25), 1e-11);
        accumulation::<f64, colmat::c64, i64>(colmat::c64::new(0.0, 0.25), 1e-11);
    }
    fn selected_contract<R: Scalar, I: LapackInt>(tol: f64) {
        for n in [1usize, 3] {
            for upper in [false, true] {
                for job in [b'N', b'V'] {
                    for range in [b'A', b'V', b'I'] {
                        let d: Vec<R> = if n == 1 {
                            vec![R::real(-4.0)]
                        } else {
                            vec![R::real(4.0), R::real(-2.0), R::z()]
                        };
                        let e = vec![R::z(); n.saturating_sub(1)];
                        let lo = R::real(2.0);
                        let hi = R::real(4.0);
                        let il = I::from_i64(1);
                        let iu = I::from_i64(n.min(2) as i64);
                        let mut ns = [I::from_i64(-777); 3];
                        let mut info = I::from_i64(-777);
                        let mut s = vec![R::real(777.0); n + 2];
                        let ld = 2 * n + 2;
                        let mut z = vec![R::real(777.0); ld * (n + 1)];
                        let mut iw = vec![I::from_i64(-777); 12 * n];
                        unsafe {
                            bdsvdx(
                                if upper { b'U' } else { b'L' },
                                job,
                                range,
                                n as isize,
                                d.as_ptr(),
                                e.as_ptr(),
                                &lo,
                                &hi,
                                &il,
                                &iu,
                                &mut ns[1],
                                s.as_mut_ptr().add(1),
                                z.as_mut_ptr(),
                                ld as isize,
                                iw.as_mut_ptr(),
                                &mut info,
                            );
                        }
                        assert_eq!(info.val(), 0);
                        assert_eq!(ns[0].val(), -777);
                        assert_eq!(ns[2].val(), -777);
                        let expected: Vec<f64> = if n == 1 {
                            vec![4.0]
                        } else {
                            match range {
                                b'V' => vec![2.0],
                                b'I' => vec![4.0, 2.0],
                                _ => vec![4.0, 2.0, 0.0],
                            }
                        };
                        assert_eq!(ns[1].val(), expected.len() as isize);
                        for (j, &value) in expected.iter().enumerate() {
                            assert!((s[j + 1].re() - value).abs() < tol);
                            if job == b'V' {
                                assert_eq!(iw[j].val(), 0);
                                for i in 0..n {
                                    assert!(
                                        (d[i] * z[i + n + j * ld] - s[j + 1] * z[i + j * ld])
                                            .magnitude()
                                            < tol
                                    );
                                }
                                let normu: f64 = (0..n).map(|i| z[i + j * ld].re().powi(2)).sum();
                                let normv: f64 =
                                    (0..n).map(|i| z[i + n + j * ld].re().powi(2)).sum();
                                assert!((normu - 1.0).abs() < tol);
                                assert!((normv - 1.0).abs() < tol);
                            }
                        }
                        assert_eq!(s[0].re(), 777.0);
                        assert_eq!(s[n + 1].re(), 777.0);
                        if job == b'N' {
                            assert!(z.iter().all(|x| x.re() == 777.0));
                        }
                        for j in 0..n + 1 {
                            for i in 2 * n..ld {
                                assert_eq!(z[i + j * ld].re(), 777.0);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn selected_values_vectors_and_boundary_contracts() {
        selected_contract::<f32, i32>(1e-4);
        selected_contract::<f32, i64>(1e-4);
        selected_contract::<f64, i32>(1e-11);
        selected_contract::<f64, i64>(1e-11);
    }
    fn dc_contract<R: Scalar, I: LapackInt>(tol: f64) {
        for n in [1usize, 7, 40] {
            for upper in [false, true] {
                for comp in [b'N', b'I'] {
                    let diagonal: Vec<R> = (0..n)
                        .map(|i| {
                            R::real(if i % 2 == 0 {
                                -(i as f64 + 1.0)
                            } else {
                                i as f64 + 1.0
                            })
                        })
                        .collect();
                    let off = vec![R::real(0.25); n.saturating_sub(1)];
                    let mut d = diagonal.clone();
                    let mut e = off.clone();
                    let ld = n + 2;
                    let mut u = vec![R::real(777.0); ld * n];
                    let mut vt = u.clone();
                    let mut q = vec![R::real(777.0); 1];
                    let mut iq = vec![I::from_i64(-777); 1];
                    let mut work = vec![R::real(777.0); 3 * n * n + 4 * n];
                    let mut iw = vec![I::from_i64(-777); 8 * n];
                    let mut info = I::from_i64(-777);
                    unsafe {
                        bdsdc(
                            0,
                            if upper { b'U' } else { b'L' },
                            comp,
                            n as isize,
                            d.as_mut_ptr(),
                            e.as_mut_ptr(),
                            u.as_mut_ptr(),
                            ld as isize,
                            vt.as_mut_ptr(),
                            ld as isize,
                            q.as_mut_ptr(),
                            iq.as_mut_ptr(),
                            work.as_mut_ptr(),
                            iw.as_mut_ptr(),
                            &mut info,
                        );
                    }
                    assert_eq!(info.val(), 0);
                    assert_eq!(crate::colmat_blas_last_error(), 0);
                    assert_eq!(q[0].re(), 777.0);
                    assert_eq!(iq[0].val(), -777);
                    assert!(work.iter().all(|x| x.re() == 777.0));
                    assert!(iw.iter().all(|x| x.val() == -777));
                    assert!(e.iter().all(|x| x.re() == 0.0));
                    if comp == b'N' {
                        assert!(u.iter().chain(&vt).all(|x| x.re() == 777.0));
                        continue;
                    }
                    for j in 0..n {
                        for i in 0..n {
                            let mut product = R::z();
                            for k in 0..n {
                                product = product + u[i + k * ld] * d[k] * vt[k + j * ld];
                            }
                            let expected = if i == j {
                                diagonal[i]
                            } else if (upper && j == i + 1) || (!upper && i == j + 1) {
                                off[i.min(j)]
                            } else {
                                R::z()
                            };
                            assert!((product - expected).magnitude() < tol * n as f64);
                        }
                    }
                    for j in 0..n {
                        for i in n..ld {
                            assert_eq!(u[i + j * ld].re(), 777.0);
                            assert_eq!(vt[i + j * ld].re(), 777.0);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn divide_and_conquer_explicit_and_values_contracts() {
        dc_contract::<f32, i32>(1e-4);
        dc_contract::<f32, i64>(1e-4);
        dc_contract::<f64, i32>(1e-11);
        dc_contract::<f64, i64>(1e-11);
    }
    fn failure_count<R: Scalar, T: Scalar, I: LapackInt>(large: R) {
        let mut d = vec![large; 3];
        let mut e = vec![large; 2];
        let mut u = vec![T::real(777.0); 15];
        let mut vt = u.clone();
        let mut c = u.clone();
        let mut info = [I::from_i64(-777); 3];
        unsafe {
            bdsqr(
                b'U',
                3,
                3,
                3,
                2,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                vt.as_mut_ptr(),
                5,
                u.as_mut_ptr(),
                5,
                c.as_mut_ptr(),
                5,
                &mut info[1],
            );
        }
        // Finite entries have an unrepresentable largest singular value.
        // The owned native stage fails without committing any external output.
        // BDSQR INFO counts the still nonzero off-diagonal entries.
        assert_eq!(info[1].val(), 2);
        assert_eq!(info[0].val(), -777);
        assert_eq!(info[2].val(), -777);
        assert!(d.iter().all(|x| *x == large));
        assert!(e.iter().all(|x| *x == large));
        assert!(u.iter().chain(&vt).chain(&c).all(|x| x.re() == 777.0));
        assert_eq!(crate::colmat_blas_last_error(), 0);
    }
    #[test]
    fn bdsqr_failure_counts_unconverged_offdiagonals() {
        failure_count::<f32, f32, i32>(f32::MAX);
        failure_count::<f32, f32, i64>(f32::MAX);
        failure_count::<f64, f64, i32>(f64::MAX);
        failure_count::<f64, f64, i64>(f64::MAX);
        failure_count::<f32, colmat::c32, i32>(f32::MAX);
        failure_count::<f32, colmat::c32, i64>(f32::MAX);
        failure_count::<f64, colmat::c64, i32>(f64::MAX);
        failure_count::<f64, colmat::c64, i64>(f64::MAX);
    }
}
