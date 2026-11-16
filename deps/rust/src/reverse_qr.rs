//! LAPACK QL/RQ storage from reversed native-backed QR/LQ factorizations.
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;

pub(crate) unsafe fn factor<T: ReflectorScalar>(
    rq: bool,
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
) {
    let k = m.min(n);
    if k == 0 {
        return;
    }
    let mut reversed = vec![T::z(); m * n];
    let mut coefficients = vec![T::z(); k];
    for j in 0..n {
        for i in 0..m {
            reversed[i + j * m] = *a.offset((m - 1 - i) as isize + (n - 1 - j) as isize * lda);
        }
    }
    if rq {
        crate::lq::standard_lq(
            m,
            n,
            reversed.as_mut_ptr(),
            m as isize,
            coefficients.as_mut_ptr(),
        );
    } else {
        crate::qr::standard_qr(
            m,
            n,
            reversed.as_mut_ptr(),
            m as isize,
            coefficients.as_mut_ptr(),
            false,
        );
    }
    for j in 0..n {
        for i in 0..m {
            *a.offset((m - 1 - i) as isize + (n - 1 - j) as isize * lda) = reversed[i + j * m];
        }
    }
    for i in 0..k {
        *tau.add(i) = coefficients[k - 1 - i];
    }
}
struct Factor<T: Scalar> {
    vector: Vec<T>,
    coefficient: T,
}
unsafe fn gather<T: Scalar>(
    rq: bool,
    order: usize,
    k: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
    shift: usize,
) -> Vec<Factor<T>> {
    let mut factors = Vec::with_capacity(k);
    for step in 0..k {
        // QL: H(k)...H(1); RQ: H(1)^H...H(k)^H.
        let i = if rq { step } else { k - 1 - step };
        let head = order - k + i;
        let coefficient = if rq { (*tau.add(i)).cj() } else { *tau.add(i) };
        let mut vector = vec![];
        if coefficient != T::z() {
            vector = vec![T::o(); head + 1];
            for row in 0..head {
                vector[row] = if rq {
                    (*a.offset((shift + i) as isize + row as isize * lda)).cj()
                } else {
                    *a.offset(row as isize + (shift + i) as isize * lda)
                };
            }
        }
        factors.push(Factor {
            vector,
            coefficient,
        });
    }
    factors
}
unsafe fn execute<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    factors: &[Factor<T>],
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let reverse = (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N');
    let mut work = vec![T::z(); m.max(n)];
    for step in 0..factors.len() {
        let f = &factors[if reverse {
            factors.len() - 1 - step
        } else {
            step
        }];
        if f.coefficient == T::z() {
            continue;
        }
        let (rows, cols) = if side == b'L' {
            (f.vector.len(), n)
        } else {
            (m, f.vector.len())
        };
        crate::householder::apply(
            side,
            rows as isize,
            cols as isize,
            f.vector.as_ptr(),
            1,
            if tr == b'N' {
                f.coefficient
            } else {
                f.coefficient.cj()
            },
            c,
            if cols == 1 { 0 } else { ldc },
            work.as_mut_ptr(),
        );
    }
}
// The caller supplies the last k rows of an RQ factor, as in ORMRQ/UNMRQ.
pub(crate) unsafe fn apply_rq<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    let factors = gather(true, order, k, a, lda, tau, 0);
    execute(side, tr, m, n, &factors, c, ldc);
}

unsafe fn generate<T: Scalar>(
    rq: bool,
    m: usize,
    n: usize,
    k: usize,
    a: *mut T,
    lda: isize,
    tau: *const T,
) {
    let factors = gather(
        rq,
        if rq { n } else { m },
        k,
        a,
        lda,
        tau,
        if rq { m - k } else { n - k },
    );
    for j in 0..n {
        for i in 0..m {
            *a.offset(i as isize + j as isize * lda) =
                if if rq { j == n - m + i } else { i == m - n + j } {
                    T::o()
                } else {
                    T::z()
                };
        }
    }
    execute(if rq { b'R' } else { b'L' }, b'N', m, n, &factors, a, lda);
}
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn factor_driver<T: ReflectorScalar, I: LapackInt>(
    rq: bool,
    m: isize,
    n: isize,
    a: *mut T,
    lda: isize,
    tau: *mut T,
    work: *mut T,
    lwork: Option<isize>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [(1, m >= 0), (2, n >= 0), (4, lda >= m.max(1))] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if let Some(lw) = lwork {
        let minimum = if rq {
            if n == 0 {
                1
            } else {
                m.max(1)
            }
        } else if m == 0 {
            1
        } else {
            n.max(1)
        };
        // Reference QL/RQ publishes WORK before validating LWORK.
        *work = crate::lapack::workspace::<T>(if m == 0 || n == 0 { 1 } else { minimum });
        if lw != -1 && lw < minimum {
            error(info, 7);
            return;
        }
        if lw == -1 {
            return;
        }
    }
    if m == 0 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 4);
        return;
    }
    factor(rq, m as usize, n as usize, a, lda, tau);
}
unsafe fn generate_driver<T: Scalar, I: LapackInt>(
    rq: bool,
    m: isize,
    n: isize,
    k: isize,
    a: *mut T,
    lda: isize,
    tau: *const T,
    work: *mut T,
    lwork: Option<isize>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && if rq { n >= m } else { n <= m }),
        (3, k >= 0 && k <= if rq { m } else { n }),
        (5, lda >= m.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if let Some(lw) = lwork {
        let minimum = if rq { m.max(1) } else { n.max(1) };
        if lw != -1 && lw < minimum {
            error(info, 8);
            return;
        }
        *work = crate::lapack::workspace::<T>(minimum);
        if lw == -1 {
            return;
        }
    }
    if m == 0 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 5);
        return;
    }
    generate(rq, m as usize, n as usize, k as usize, a, lda, tau);
}
unsafe fn apply_driver<T: Scalar, I: LapackInt>(
    rq: bool,
    side: u8,
    tr: u8,
    m: isize,
    n: isize,
    k: isize,
    a: *const T,
    lda: isize,
    tau: *const T,
    c: *mut T,
    ldc: isize,
    work: *mut T,
    lwork: Option<isize>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let order = if side == b'L' { m } else { n };
    for (arg, ok) in [
        (1, matches!(side, b'L' | b'R')),
        (2, tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }),
        (3, m >= 0),
        (4, n >= 0),
        (5, k >= 0 && k <= order),
        (7, lda >= if rq { k.max(1) } else { order.max(1) }),
        (10, ldc >= m.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if let Some(lw) = lwork {
        let minimum = if side == b'L' { n.max(1) } else { m.max(1) };
        if lw != -1 && lw < minimum {
            error(info, 12);
            return;
        }
        *work = crate::lapack::workspace::<T>(minimum);
        if lw == -1 {
            return;
        }
    }
    if m == 0 || n == 0 || k == 0 {
        if (m == 0 || n == 0) && lwork.is_some() {
            *work = T::o();
        }
        return;
    }
    if !matrix_addressable::<T>(k, 1, 1) {
        error(info, 5);
        return;
    }
    let mut extent = 0;
    let mut elements = 0isize;
    for i in 0..k {
        if *tau.offset(i) == T::z() {
            continue;
        }
        let length = order - k + i + 1;
        extent = extent.max(length);
        let Some(total) = elements.checked_add(length) else {
            error(info, 7);
            return;
        };
        elements = total;
        if length > 1 {
            let (rows, cols) = if rq {
                (i + 1, length - 1)
            } else {
                (length - 1, i + 1)
            };
            if !matrix_addressable::<T>(rows, cols, lda) {
                error(info, 7);
                return;
            }
        }
    }
    if extent == 0 {
        return;
    }
    if elements
        .checked_mul(core::mem::size_of::<T>() as isize)
        .is_none()
    {
        error(info, 7);
        return;
    }
    let (rows, cols) = if side == b'L' {
        (extent, n)
    } else {
        (m, extent)
    };
    if !matrix_addressable::<T>(rows, cols, ldc) {
        error(info, 10);
        return;
    }
    let factors = gather(rq, order as usize, k as usize, a, lda, tau, 0);
    execute(side, tr, m as usize, n as usize, &factors, c, ldc);
}
macro_rules! exports {($p:ident,$f:ident,$f2:ident,$g:ident,$g2:ident,$apply:ident,$apply2:ident,$rq:expr,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $f $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,work:*mut $t,lwork:*const $i,info:*mut $i) {factor_driver($rq,*m as isize,*n as isize,a,*lda as isize,tau,work,Some(*lwork as isize),info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $f2 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,tau:*mut $t,work:*mut $t,info:*mut $i) {factor_driver($rq,*m as isize,*n as isize,a,*lda as isize,tau,work,None,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $g $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {generate_driver($rq,*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,work,Some(*lwork as isize),info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $g2 $s>](m:*const $i,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,tau:*const $t,work:*mut $t,info:*mut $i) {generate_driver($rq,*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,work,None,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $apply $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {apply_driver($rq,flag(side),flag(tr),*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,c,*ldc as isize,work,Some(*lwork as isize),info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $apply2 $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,info:*mut $i) {apply_driver($rq,flag(side),flag(tr),*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,c,*ldc as isize,work,None,info);}
}};}
macro_rules! abi {
    ($i:ty,$s:tt) => {
        exports!(s, geqlf, geql2, orgql, org2l, ormql, orm2l, false, f32, $i, $s);
        exports!(d, geqlf, geql2, orgql, org2l, ormql, orm2l, false, f64, $i, $s);
        exports!(
            c,
            geqlf,
            geql2,
            ungql,
            ung2l,
            unmql,
            unm2l,
            false,
            colmat::c32,
            $i,
            $s
        );
        exports!(
            z,
            geqlf,
            geql2,
            ungql,
            ung2l,
            unmql,
            unm2l,
            false,
            colmat::c64,
            $i,
            $s
        );
        exports!(s, gerqf, gerq2, orgrq, orgr2, ormrq, ormr2, true, f32, $i, $s);
        exports!(d, gerqf, gerq2, orgrq, orgr2, ormrq, ormr2, true, f64, $i, $s);
        exports!(
            c,
            gerqf,
            gerq2,
            ungrq,
            ungr2,
            unmrq,
            unmr2,
            true,
            colmat::c32,
            $i,
            $s
        );
        exports!(
            z,
            gerqf,
            gerq2,
            ungrq,
            ungr2,
            unmrq,
            unmr2,
            true,
            colmat::c64,
            $i,
            $s
        );
    };
}
abi!(i32, _);
abi!(i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for rq in [false, true] {
                for (m, n) in [
                    (0usize, 4usize),
                    (7, 0),
                    (1, 1),
                    (2, 2),
                    (8, 3),
                    (3, 8),
                    (5, 5),
                ] {
                    for zero in [false, true] {
                        let k = m.min(n);
                        let order = if rq { n } else { m };
                        let lda = m + 2;
                        let mut a = vec![T::real(777.0); lda * n];
                        let mut original = vec![T::z(); m * n];
                        let mut tau = vec![T::real(777.0); k.max(1)];
                        for j in 0..n {
                            for i in 0..m {
                                let value = if zero {
                                    T::z()
                                } else {
                                    T::real(((i * 13 + j * 17 + i * j) % 29) as f64 / 19.0 - 0.7)
                                        + imag * T::real(((i * 7 + j * 11) % 13) as f64 / 23.0)
                                };
                                a[i + j * lda] = value;
                                original[i + j * m] = value;
                            }
                        }
                        factor(rq, m, n, a.as_mut_ptr(), lda as isize, tau.as_mut_ptr());
                        let shifted = if k == 0 {
                            a.as_ptr()
                        } else if rq {
                            a.as_ptr().add(m - k)
                        } else {
                            a.as_ptr().add((n - k) * lda)
                        };
                        let factors = gather(rq, order, k, shifted, lda as isize, tau.as_ptr(), 0);
                        let mut q = vec![T::z(); order * order];
                        for j in 0..order {
                            q[j + j * order] = T::o();
                        }
                        execute(
                            b'L',
                            b'N',
                            order,
                            order,
                            &factors,
                            q.as_mut_ptr(),
                            order as isize,
                        );
                        for j in 0..n {
                            for i in 0..m {
                                let value = if rq {
                                    (0..n).fold(T::z(), |s, l| {
                                        if l as isize >= n as isize - m as isize + i as isize {
                                            s + a[i + l * lda] * q[l + j * order]
                                        } else {
                                            s
                                        }
                                    })
                                } else {
                                    (0..m).fold(T::z(), |s, l| {
                                        if l as isize >= m as isize - n as isize + j as isize {
                                            s + q[i + l * order] * a[l + j * lda]
                                        } else {
                                            s
                                        }
                                    })
                                };
                                assert!((value - original[i + j * m]).magnitude() < tolerance);
                            }
                            for i in m..lda {
                                assert_eq!(a[i + j * lda], T::real(777.0));
                            }
                        }
                        for j in 0..order {
                            for i in 0..order {
                                let dot = (0..order).fold(T::z(), |s, l| {
                                    s + q[l + i * order].cj() * q[l + j * order]
                                });
                                assert!(
                                    (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                        < tolerance
                                );
                            }
                        }
                        let (rows, cols) = if rq { (k, n) } else { (m, k) };
                        let ldout = rows + 2;
                        let mut output = vec![T::real(777.0); ldout * cols];
                        for j in 0..cols {
                            for i in 0..rows {
                                output[i + j * ldout] = if rq {
                                    a[m - k + i + j * lda]
                                } else {
                                    a[i + (n - k + j) * lda]
                                };
                            }
                        }
                        generate(
                            rq,
                            rows,
                            cols,
                            k,
                            output.as_mut_ptr(),
                            ldout as isize,
                            tau.as_ptr(),
                        );
                        for j in 0..cols {
                            for i in 0..rows {
                                let expected = if rq {
                                    q[order - k + i + j * order]
                                } else {
                                    q[i + (order - k + j) * order]
                                };
                                assert!((output[i + j * ldout] - expected).magnitude() < tolerance);
                            }
                            for i in rows..ldout {
                                assert_eq!(output[i + j * ldout], T::real(777.0));
                            }
                        }
                        for side in [b'L', b'R'] {
                            for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                                let (rows, cols) =
                                    if side == b'L' { (order, 3) } else { (3, order) };
                                let ldc = rows + 2;
                                let mut c = vec![T::real(777.0); ldc * cols];
                                for j in 0..cols {
                                    for i in 0..rows {
                                        c[i + j * ldc] = T::real((i + j + 1) as f64 / 17.0)
                                            + imag * T::real((i + 2 * j) as f64 / 29.0);
                                    }
                                }
                                let before = c.clone();
                                let op = |i: usize, j: usize| {
                                    if tr == b'N' {
                                        q[i + j * order]
                                    } else {
                                        q[j + i * order].cj()
                                    }
                                };
                                execute(
                                    side,
                                    tr,
                                    rows,
                                    cols,
                                    &factors,
                                    c.as_mut_ptr(),
                                    ldc as isize,
                                );
                                for j in 0..cols {
                                    for i in 0..rows {
                                        let expected = if side == b'L' {
                                            (0..order).fold(T::z(), |s, l| {
                                                s + op(i, l) * before[l + j * ldc]
                                            })
                                        } else {
                                            (0..order).fold(T::z(), |s, l| {
                                                s + before[i + l * ldc] * op(l, j)
                                            })
                                        };
                                        assert!(
                                            (c[i + j * ldc] - expected).magnitude() < tolerance
                                        );
                                    }
                                    for i in rows..ldc {
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
    #[test]
    fn real32() {
        verify(0.0f32, 1e-4);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 1e-12);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 1e-4);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 1e-12);
    }
}
