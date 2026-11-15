//! Packed self-adjoint reduction and LAPACK-compatible reflector interfaces.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::structured_evd::Storage;
use colmat::Mat;
use paste::paste;

unsafe fn fail<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn reduce<T: ReflectorScalar + GeneralizedScalar, R: Scalar, I: LapackInt>(
    u: u8,
    n: isize,
    ap: *mut T,
    d: *mut R,
    e: *mut R,
    tau: *mut T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if !matches!(u, b'L' | b'U') {
        fail(info, 1);
        return;
    }
    if n < 0 || !matrix_addressable::<T>(n, n, n.max(1)) {
        fail(info, 2);
        return;
    }
    if n == 0 {
        return;
    }
    let size = n as usize;
    let lower = u == b'L';
    let mut a = Mat::<T>::zeros(size, size);
    for j in 0..size {
        let rows = if lower { j..size } else { 0..j + 1 };
        for i in rows {
            let p = ap.add(Storage::Packed.index(size, lower, i, j));
            a[(i, j)] = if i == j { T::diagonal(p) } else { *p };
        }
    }
    crate::selfadjoint_reduction::ordinary(
        lower,
        size,
        a.as_mut().as_ptr_mut(),
        a.col_stride(),
        d,
        e,
        tau,
    );
    for j in 0..size {
        let rows = if lower { j..size } else { 0..j + 1 };
        for i in rows {
            *ap.add(Storage::Packed.index(size, lower, i, j)) = a[(i, j)];
        }
    }
}
unsafe fn generate<T: Scalar, I: LapackInt>(
    u: u8,
    n: isize,
    ap: *const T,
    tau: *const T,
    q: *mut T,
    ldq: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(u, b'L' | b'U')),
        (2, n >= 0),
        (6, ldq >= n.max(1)),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !matrix_addressable::<T>(n, n, ldq) || !matrix_addressable::<T>(n, n, n) {
        fail(info, 6);
        return;
    }
    for j in 0..n {
        for i in 0..n {
            *q.offset(i + j * ldq) = if i == j { T::o() } else { T::z() };
        }
    }
    crate::selfadjoint_vectors::packed_execute(
        b'L',
        u == b'L',
        b'N',
        n as usize,
        n as usize,
        ap,
        tau,
        q,
        ldq,
    );
}
unsafe fn apply<T: Scalar, I: LapackInt>(
    side: u8,
    u: u8,
    tr: u8,
    m: isize,
    n: isize,
    ap: *const T,
    tau: *const T,
    c: *mut T,
    ldc: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(side, b'L' | b'R')),
        (2, matches!(u, b'L' | b'U')),
        (3, tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }),
        (4, m >= 0),
        (5, n >= 0),
        (9, ldc >= m.max(1)),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    let nq = if side == b'L' { m } else { n };
    if m == 0 || n == 0 || nq <= 1 {
        return;
    }
    if !matrix_addressable::<T>(nq, nq, nq) {
        fail(info, 6);
        return;
    }
    if (0..nq - 1).all(|i| *tau.offset(i) == T::z()) {
        return;
    }
    if !matrix_addressable::<T>(m, n, ldc) {
        fail(info, 9);
        return;
    }
    crate::selfadjoint_vectors::packed_execute(
        side,
        u == b'L',
        tr,
        m as usize,
        n as usize,
        ap,
        tau,
        c,
        ldc,
    );
}
// Dense native reduction adapts the storage contract; it is not a band
// bulge-chasing implementation. Diagonal/tridiagonal inputs retain their
// direct gauges (complex off-diagonal phases are accumulated in Q).
unsafe fn band<T: ReflectorScalar + GeneralizedScalar, R: Scalar, I: LapackInt>(
    vect: u8,
    u: u8,
    n: isize,
    kd: isize,
    ab: *mut T,
    ldab: isize,
    d: *mut R,
    e: *mut R,
    q: *mut T,
    ldq: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let want = vect != b'N';
    for (arg, ok) in [
        (1, matches!(vect, b'N' | b'V' | b'U')),
        (2, matches!(u, b'L' | b'U')),
        (3, n >= 0),
        (4, kd >= 0),
        (6, kd.checked_add(1).is_some_and(|v| ldab >= v)),
        (10, ldq >= 1 && (!want || ldq >= n)),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !matrix_addressable::<T>(kd + 1, n, ldab) {
        fail(info, 6);
        return;
    }
    if want && !matrix_addressable::<T>(n, n, ldq) {
        fail(info, 10);
        return;
    }
    if !matrix_addressable::<T>(n, n, n) {
        fail(info, 3);
        return;
    }
    let size = n as usize;
    let lower = u == b'L';
    let storage = Storage::Band {
        width: kd as usize,
        ld: ldab as usize,
    };
    if kd <= 1 {
        let mut phase = T::o();
        for j in 0..size {
            let diagonal = ab.add(storage.index(size, lower, j, j));
            *d.add(j) = R::real(T::diagonal(diagonal).re());
            *diagonal = T::real((*d.add(j)).re());
            if vect == b'V' {
                for i in 0..size {
                    *q.offset(i as isize + j as isize * ldq) = if i == j { phase } else { T::z() };
                }
            } else if vect == b'U' && T::COMPLEX {
                for i in 0..size {
                    let p = q.offset(i as isize + j as isize * ldq);
                    *p = *p * phase;
                }
            }
            if j + 1 < size {
                if kd == 0 {
                    *e.add(j) = R::z();
                } else {
                    let (row, col) = if lower { (j + 1, j) } else { (j, j + 1) };
                    let p = ab.add(storage.index(size, lower, row, col));
                    if T::COMPLEX {
                        let value = if lower { *p } else { (*p).cj() };
                        let absolute = value.re().hypot(value.im());
                        *e.add(j) = R::real(absolute);
                        *p = T::real(absolute);
                        if absolute != 0.0 {
                            phase = phase * value / T::real(absolute);
                        }
                    } else {
                        *e.add(j) = R::real((*p).re());
                    }
                }
            }
        }
        return;
    }
    let mut a = Mat::<T>::zeros(size, size);
    for j in 0..size {
        for i in if lower {
            j..size.min(j + kd as usize + 1)
        } else {
            j.saturating_sub(kd as usize)..j + 1
        } {
            let p = ab.add(storage.index(size, lower, i, j));
            a[(i, j)] = if i == j { T::diagonal(p) } else { *p };
        }
    }
    let mut tau = vec![T::z(); size.saturating_sub(1)];
    crate::selfadjoint_reduction::ordinary(
        lower,
        size,
        a.as_mut().as_ptr_mut(),
        a.col_stride(),
        d,
        e,
        tau.as_mut_ptr(),
    );
    // Retain a real nonnegative off-diagonal in the Hermitian band interface.
    let mut signs = vec![T::o(); size];
    if T::COMPLEX {
        for j in 0..size - 1 {
            let old = (*e.add(j)).re();
            signs[j + 1] = signs[j] * T::real(if old < 0.0 { -1.0 } else { 1.0 });
            *e.add(j) = R::real(old.abs());
        }
    }
    for j in 0..size {
        for i in if lower {
            j..size.min(j + kd as usize + 1)
        } else {
            j.saturating_sub(kd as usize)..j + 1
        } {
            *ab.add(storage.index(size, lower, i, j)) = if i == j {
                T::real((*d.add(j)).re())
            } else if i.abs_diff(j) == 1 {
                T::real((*e.add(i.min(j))).re())
            } else {
                T::z()
            };
        }
    }
    if want {
        let mut work = vec![T::z(); size.max(1)];
        crate::selfadjoint_vectors::generate(
            u,
            n,
            a.as_mut().as_ptr_mut(),
            a.col_stride(),
            tau.as_ptr(),
            work.as_mut_ptr(),
            n.max(1),
            info,
        );
        for j in 0..size {
            for i in 0..size {
                a[(i, j)] = a[(i, j)] * signs[j];
            }
        }
        if vect == b'V' {
            for j in 0..size {
                for i in 0..size {
                    *q.offset(i as isize + j as isize * ldq) = a[(i, j)];
                }
            }
        } else {
            let mut result = Mat::<T>::zeros(size, size);
            crate::kernels::gemm(
                b'N',
                b'N',
                n,
                n,
                n,
                T::o(),
                q,
                ldq,
                a.as_ref().as_ptr(),
                a.col_stride(),
                T::z(),
                result.as_mut().as_ptr_mut(),
                result.col_stride(),
            );
            for j in 0..size {
                for i in 0..size {
                    *q.offset(i as isize + j as isize * ldq) = result[(i, j)];
                }
            }
        }
    }
}
macro_rules! band_exports {($p:ident,$trd:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $trd $s>](v:*const u8,u:*const u8,n:*const $i,kd:*const $i,ab:*mut $t,ldab:*const $i,d:*mut $r,e:*mut $r,q:*mut $t,ldq:*const $i,_work:*mut $t,info:*mut $i) {band(flag(v),flag(u),*n as isize,*kd as isize,ab,*ldab as isize,d,e,q,*ldq as isize,info);}
}};}
macro_rules! exports {($p:ident,$trd:ident,$gtr:ident,$mtr:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $trd $s>](u:*const u8,n:*const $i,ap:*mut $t,d:*mut $r,e:*mut $r,tau:*mut $t,info:*mut $i) {reduce(flag(u),*n as isize,ap,d,e,tau,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $gtr $s>](u:*const u8,n:*const $i,ap:*const $t,tau:*const $t,q:*mut $t,ldq:*const $i,_work:*mut $t,info:*mut $i) {generate(flag(u),*n as isize,ap,tau,q,*ldq as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $mtr $s>](side:*const u8,u:*const u8,tr:*const u8,m:*const $i,n:*const $i,ap:*const $t,tau:*const $t,c:*mut $t,ldc:*const $i,_work:*mut $t,info:*mut $i) {apply(flag(side),flag(u),flag(tr),*m as isize,*n as isize,ap,tau,c,*ldc as isize,info);}
}};}
macro_rules! abi {
    ($i:ty,$s:tt) => {
        band_exports!(s, sbtrd, f32, f32, $i, $s);
        band_exports!(d, sbtrd, f64, f64, $i, $s);
        band_exports!(c, hbtrd, colmat::c32, f32, $i, $s);
        band_exports!(z, hbtrd, colmat::c64, f64, $i, $s);
        exports!(s, sptrd, opgtr, opmtr, f32, f32, $i, $s);
        exports!(d, sptrd, opgtr, opmtr, f64, f64, $i, $s);
        exports!(c, hptrd, upgtr, upmtr, colmat::c32, f32, $i, $s);
        exports!(z, hptrd, upgtr, upmtr, colmat::c64, f64, $i, $s);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: ReflectorScalar + GeneralizedScalar, I: LapackInt>(imag: T, tol: f64) {
        unsafe {
            for lower in [false, true] {
                for n in [0usize, 1, 2, 5] {
                    for degenerate in [false, true] {
                        let u = if lower { b'L' } else { b'U' };
                        let a = Mat::from_fn(n, n, |i, j| {
                            if degenerate {
                                return if i == j && i == 0 {
                                    T::real(3.0)
                                } else {
                                    T::z()
                                };
                            }
                            if i == j {
                                T::real((i + 2) as f64)
                            } else {
                                T::real((i + j + 1) as f64 / 11.0)
                                    + imag * T::real((i as f64 - j as f64) / 17.0)
                            }
                        });
                        let mut ap = vec![T::real(777.0); n * (n + 1) / 2 + 2];
                        for j in 0..n {
                            for i in if lower { j..n } else { 0..j + 1 } {
                                ap[1 + Storage::Packed.index(n, lower, i, j)] = a[(i, j)];
                            }
                        }
                        let mut d = vec![T::real(777.0); n + 2];
                        let mut e = vec![T::real(777.0); n.max(1) + 1];
                        let mut tau = vec![T::real(777.0); n.max(1) + 1];
                        let mut info = I::from_i64(-777);
                        reduce(
                            u,
                            n as isize,
                            ap.as_mut_ptr().add(1),
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            tau.as_mut_ptr().add(1),
                            &mut info,
                        );
                        assert_eq!(info.val(), 0);
                        assert_eq!(ap[0], T::real(777.0));
                        assert_eq!(*ap.last().unwrap(), T::real(777.0));
                        assert_eq!(d[0], T::real(777.0));
                        assert_eq!(d[n + 1], T::real(777.0));
                        assert_eq!(e[n.max(1)], T::real(777.0));
                        assert_eq!(tau[n.max(1)], T::real(777.0));
                        let ld = n + 2;
                        let mut q = vec![T::real(777.0); ld * n];
                        let old = ap.clone();
                        generate(
                            u,
                            n as isize,
                            ap.as_ptr().add(1),
                            tau.as_ptr().add(1),
                            q.as_mut_ptr(),
                            ld as isize,
                            &mut info,
                        );
                        assert_eq!(info.val(), 0);
                        assert_eq!(ap, old);
                        for j in 0..n {
                            for i in 0..n {
                                let mut rebuilt = T::z();
                                let mut gram = T::z();
                                for k in 0..n {
                                    gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                    rebuilt =
                                        rebuilt + q[i + k * ld] * d[k + 1] * q[j + k * ld].cj();
                                    if k + 1 < n {
                                        rebuilt = rebuilt
                                            + q[i + k * ld] * e[k + 1] * q[j + (k + 1) * ld].cj()
                                            + q[i + (k + 1) * ld] * e[k + 1] * q[j + k * ld].cj();
                                    }
                                }
                                assert!(
                                    (rebuilt - a[(i, j)]).abs1() < tol,
                                    "reconstruction n={n} lower={lower}"
                                );
                                assert!((gram - if i == j { T::o() } else { T::z() }).abs1() < tol);
                            }
                            for i in n..ld {
                                assert_eq!(q[i + j * ld], T::real(777.0));
                            }
                        }
                        for side in [b'L', b'R'] {
                            for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                                let (m, cols) = if side == b'L' { (n, 3) } else { (3, n) };
                                let ldc = m + 2;
                                let mut c = vec![T::real(777.0); ldc * cols];
                                for j in 0..cols {
                                    for i in 0..m {
                                        c[i + j * ldc] = T::real((i + j + 1) as f64 / 9.0)
                                            + imag * T::real((i + 2 * j) as f64 / 13.0);
                                    }
                                }
                                let before = c.clone();
                                apply(
                                    side,
                                    u,
                                    tr,
                                    m as isize,
                                    cols as isize,
                                    ap.as_ptr().add(1),
                                    tau.as_ptr().add(1),
                                    c.as_mut_ptr(),
                                    ldc as isize,
                                    &mut info,
                                );
                                assert_eq!(info.val(), 0);
                                for j in 0..cols {
                                    for i in 0..m {
                                        let mut expected = T::z();
                                        for k in 0..n {
                                            let value = if side == b'L' {
                                                if tr == b'N' {
                                                    q[i + k * ld]
                                                } else {
                                                    q[k + i * ld].cj()
                                                }
                                            } else if tr == b'N' {
                                                q[k + j * ld]
                                            } else {
                                                q[j + k * ld].cj()
                                            };
                                            expected = expected
                                                + value
                                                    * if side == b'L' {
                                                        before[k + j * ldc]
                                                    } else {
                                                        before[i + k * ldc]
                                                    };
                                        }
                                        assert!((c[i + j * ldc] - expected).abs1() < tol);
                                    }
                                    for i in m..ldc {
                                        assert_eq!(c[i + j * ldc], T::real(777.0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fn band_check<T: ReflectorScalar + GeneralizedScalar, I: LapackInt>(imag: T, tol: f64) {
        unsafe {
            for n in [0usize, 1, 2, 6] {
                for kd in [0usize, 1, 2, 8] {
                    for lower in [false, true] {
                        let u = if lower { b'L' } else { b'U' };
                        let ld = kd + 3;
                        let storage = Storage::Band { width: kd, ld };
                        let a = Mat::from_fn(n, n, |i, j| {
                            if i.abs_diff(j) > kd {
                                T::z()
                            } else if i == j {
                                T::real((i + 2) as f64)
                            } else {
                                T::real((i + j + 1) as f64 / 13.0)
                                    + imag * T::real((i as f64 - j as f64) / 19.0)
                            }
                        });
                        let mut original = vec![T::real(777.0); ld * n];
                        for j in 0..n {
                            for i in if lower {
                                j..n.min(j + kd + 1)
                            } else {
                                j.saturating_sub(kd)..j + 1
                            } {
                                original[storage.index(n, lower, i, j)] = a[(i, j)];
                            }
                        }
                        let mut ab = original.clone();
                        let mut d = vec![T::real(777.0); n + 2];
                        let mut e = vec![T::real(777.0); n.max(1) + 1];
                        let ldq = n + 2;
                        let mut q = vec![T::real(777.0); ldq * n];
                        let mut info = I::from_i64(-777);
                        band(
                            b'V',
                            u,
                            n as isize,
                            kd as isize,
                            ab.as_mut_ptr(),
                            ld as isize,
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            q.as_mut_ptr(),
                            ldq as isize,
                            &mut info,
                        );
                        assert_eq!(info.val(), 0);
                        assert_eq!(d[n + 1], T::real(777.0));
                        assert_eq!(e[n.max(1)], T::real(777.0));
                        for j in 0..n {
                            for i in 0..n {
                                let mut rebuilt = T::z();
                                let mut gram = T::z();
                                for k in 0..n {
                                    gram = gram + q[k + i * ldq].cj() * q[k + j * ldq];
                                    rebuilt =
                                        rebuilt + q[i + k * ldq] * d[k + 1] * q[j + k * ldq].cj();
                                    if k + 1 < n {
                                        rebuilt = rebuilt
                                            + q[i + k * ldq] * e[k + 1] * q[j + (k + 1) * ldq].cj()
                                            + q[i + (k + 1) * ldq] * e[k + 1] * q[j + k * ldq].cj();
                                    }
                                }
                                assert!(
                                    (rebuilt - a[(i, j)]).abs1() < tol,
                                    "band reconstruction n={n} kd={kd} lower={lower}"
                                );
                                assert!((gram - if i == j { T::o() } else { T::z() }).abs1() < tol);
                            }
                            for i in n..ldq {
                                assert_eq!(q[i + j * ldq], T::real(777.0));
                            }
                        }
                        for j in 0..n {
                            for r in 0..ld {
                                let selected = if lower {
                                    r <= kd && j + r < n
                                } else {
                                    r <= kd && j + r >= kd
                                };
                                if !selected {
                                    assert_eq!(ab[r + j * ld], original[r + j * ld]);
                                }
                            }
                        }
                        let x = Mat::from_fn(n, n, |i, j| {
                            T::real((i + j + 1) as f64 / 7.0)
                                + imag * T::real((i + 2 * j) as f64 / 17.0)
                        });
                        let mut updated = vec![T::real(777.0); ldq * n];
                        for j in 0..n {
                            for i in 0..n {
                                updated[i + j * ldq] = x[(i, j)];
                            }
                        }
                        let mut ab = original.clone();
                        band(
                            b'U',
                            u,
                            n as isize,
                            kd as isize,
                            ab.as_mut_ptr(),
                            ld as isize,
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            updated.as_mut_ptr(),
                            ldq as isize,
                            &mut info,
                        );
                        assert_eq!(info.val(), 0);
                        for j in 0..n {
                            for i in 0..n {
                                let expected =
                                    (0..n).fold(T::z(), |v, k| v + x[(i, k)] * q[k + j * ldq]);
                                assert!((updated[i + j * ldq] - expected).abs1() < tol);
                            }
                        }
                        let mut ab = original.clone();
                        band(
                            b'N',
                            u,
                            n as isize,
                            kd as isize,
                            ab.as_mut_ptr(),
                            ld as isize,
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            core::ptr::null_mut(),
                            1,
                            &mut info,
                        );
                        assert_eq!(info.val(), 0);
                    }
                }
            }
        }
    }
    #[test]
    fn band_reduction_and_existing_q_update_all_types_and_abis() {
        band_check::<f32, i32>(0.0, 3e-4);
        band_check::<f32, i64>(0.0, 3e-4);
        band_check::<f64, i32>(0.0, 3e-11);
        band_check::<f64, i64>(0.0, 3e-11);
        band_check::<colmat::c32, i32>(colmat::c32::new(0.0, 1.0), 3e-4);
        band_check::<colmat::c32, i64>(colmat::c32::new(0.0, 1.0), 3e-4);
        band_check::<colmat::c64, i32>(colmat::c64::new(0.0, 1.0), 3e-11);
        band_check::<colmat::c64, i64>(colmat::c64::new(0.0, 1.0), 3e-11);
    }
    #[test]
    fn packed_headers_quick_returns_and_zero_coefficients() {
        unsafe {
            let nil = core::ptr::null_mut::<f64>();
            let mut info = 777i64;
            reduce::<f64, f64, i64>(b'?', 3, nil, nil, nil, nil, &mut info);
            assert_eq!(info, -1);
            let _ = crate::colmat_blas_last_error();
            reduce::<f64, f64, i64>(b'U', -1, nil, nil, nil, nil, &mut info);
            assert_eq!(info, -2);
            let _ = crate::colmat_blas_last_error();
            reduce::<f64, f64, i64>(b'L', 0, nil, nil, nil, nil, &mut info);
            assert_eq!(info, 0);
            generate::<f64, i64>(b'L', 0, nil, nil, nil, 1, &mut info);
            assert_eq!(info, 0);
            let mut q = [777.0; 16];
            let tau = [0.0; 3];
            generate::<f64, i64>(b'U', 4, nil, tau.as_ptr(), q.as_mut_ptr(), 4, &mut info);
            assert_eq!(info, 0);
            for j in 0..4 {
                for i in 0..4 {
                    assert_eq!(q[i + j * 4], if i == j { 1.0 } else { 0.0 });
                }
            }
            apply::<f64, i64>(b'L', b'U', b'N', 4, 3, nil, tau.as_ptr(), nil, 4, &mut info);
            assert_eq!(info, 0);
            apply::<f64, i64>(b'R', b'L', b'T', 0, 4, nil, nil, nil, 1, &mut info);
            assert_eq!(info, 0);
        }
    }
    #[test]
    fn packed_reduction_q_and_application_both_integer_abis() {
        check::<f32, i32>(0.0, 2e-4);
        check::<f32, i64>(0.0, 2e-4);
        check::<f64, i32>(0.0, 2e-11);
        check::<f64, i64>(0.0, 2e-11);
        check::<colmat::c32, i32>(colmat::c32::new(0.0, 1.0), 2e-4);
        check::<colmat::c32, i64>(colmat::c32::new(0.0, 1.0), 2e-4);
        check::<colmat::c64, i32>(colmat::c64::new(0.0, 1.0), 2e-11);
        check::<colmat::c64, i64>(colmat::c64::new(0.0, 1.0), 2e-11);
    }
}
