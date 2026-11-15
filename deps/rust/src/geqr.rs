//! GEQR metadata, minimal/optimal workspace and TSQR-aware GEMQR.
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;
fn slots(m: isize, k: isize, mb: isize) -> isize {
    if mb > k && mb < m {
        (m - k) / (mb - k) + isize::from((m - k) % (mb - k) != 0)
    } else {
        1
    }
}
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn factor<T: ReflectorScalar, I: LapackInt>(
    m: isize,
    n: isize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    tsize: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [(1, m >= 0), (2, n >= 0), (4, lda >= m.max(1))] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let query = matches!(tsize, -1 | -2) || matches!(lwork, -1 | -2);
    let mint = (tsize == -2 || lwork == -2) && tsize != -1;
    let minw = (tsize == -2 || lwork == -2) && lwork != -1;
    let mut mb = if m > n {
        n.saturating_add(32).min(m)
    } else {
        m
    };
    let mut nb = m.min(n).min(32).max(1);
    let Some(mintsz) = n.checked_add(5) else {
        error(info, 6);
        return;
    };
    let optimal_t = nb
        .checked_mul(n)
        .and_then(|x| x.checked_mul(slots(m, n, mb)))
        .and_then(|x| x.checked_add(5));
    let Some(optimal_t) = optimal_t else {
        error(info, 6);
        return;
    };
    let Some(optimal_w) = n.checked_mul(nb).map(|x| x.max(1)) else {
        error(info, 8);
        return;
    };
    if !query {
        if tsize < mintsz {
            error(info, 6);
            return;
        }
        if lwork < n.max(1) {
            error(info, 8);
            return;
        }
        if tsize < optimal_t {
            nb = 1;
            mb = m;
        }
        if lwork < optimal_w {
            nb = 1;
        }
    }
    let actual_t = nb * n * slots(m, n, mb) + 5;
    let actual_w = (nb * n).max(1);
    *t = crate::lapack::workspace::<T>(if mint { mintsz } else { actual_t });
    *t.add(1) = T::real(mb as f64);
    *t.add(2) = T::real(nb as f64);
    *work = crate::lapack::workspace::<T>(if minw { n.max(1) } else { actual_w });
    if query || m.min(n) == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 4);
        return;
    }
    if !matrix_addressable::<T>(nb, n * slots(m, n, mb), nb) {
        error(info, 6);
        return;
    }
    if m <= n || mb <= n || mb >= m {
        // Own the input panel so native QR never sees an unused huge
        // leading dimension in a one-column caller matrix.
        let mut panel = vec![T::z(); (m * n) as usize];
        for j in 0..n {
            for i in 0..m {
                panel[(i + j * m) as usize] = *a.offset(i + j * lda);
            }
        }
        crate::qr::compact_qr(
            m as usize,
            n as usize,
            nb as usize,
            panel.as_mut_ptr(),
            m,
            t.add(5),
            nb,
        );
        for j in 0..n {
            for i in 0..m {
                *a.offset(i + j * lda) = panel[(i + j * m) as usize];
            }
        }
    } else {
        crate::tsqr::factor(
            m as usize,
            n as usize,
            mb as usize,
            nb as usize,
            a,
            lda,
            t.add(5),
            nb,
        );
    }
}
unsafe fn apply<T: Scalar, I: LapackInt>(
    side: u8,
    tr: u8,
    m: isize,
    n: isize,
    k: isize,
    a: *const T,
    lda: isize,
    t: *const T,
    tsize: isize,
    c: *mut T,
    ldc: isize,
    work: *mut T,
    lwork: isize,
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
        (7, lda >= order.max(1)),
        (9, tsize >= 5),
        (11, ldc >= m.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if m.min(n).min(k) == 0 {
        if lwork != -1 && lwork < 1 {
            error(info, 13);
            return;
        }
        *work = T::o();
        return;
    }
    let mb = (*t.add(1)).re() as isize;
    let nb = (*t.add(2)).re() as isize;
    if mb < 1 || nb < 1 || nb > k {
        error(info, 8);
        return;
    }
    let Some(minimum) = (if side == b'L' { n } else { mb })
        .checked_mul(nb)
        .map(|x| x.max(1))
    else {
        error(info, 13);
        return;
    };
    if lwork != -1 && lwork < minimum {
        error(info, 13);
        return;
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork == -1 {
        return;
    }
    let Some(columns) = k.checked_mul(slots(order, k, mb)) else {
        error(info, 9);
        return;
    };
    let Some(required) = nb.checked_mul(columns).and_then(|x| x.checked_add(5)) else {
        error(info, 9);
        return;
    };
    if tsize < required {
        error(info, 9);
        return;
    }
    if !matrix_addressable::<T>(order, k, lda) {
        error(info, 7);
        return;
    }
    if !matrix_addressable::<T>(m, n, ldc) {
        error(info, 11);
        return;
    }
    if !matrix_addressable::<T>(nb, columns, nb) {
        error(info, 9);
        return;
    }
    crate::tsqr::apply(
        side,
        tr,
        m as usize,
        n as usize,
        k as usize,
        mb as usize,
        nb as usize,
        a,
        lda,
        t.add(5),
        nb,
        c,
        ldc,
    );
}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p geqr $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,tsize:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {factor(*m as isize,*n as isize,a,*lda as isize,t,*tsize as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p gemqr $s>](side:*const u8,tr:*const u8,m:*const $i,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,t:*const $t,tsize:*const $i,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {apply(flag(side),flag(tr),*m as isize,*n as isize,*k as isize,a,*lda as isize,t,*tsize as isize,c,*ldc as isize,work,*lwork as isize,info);}
}};}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for (m, n) in [(0usize, 4usize), (7, 0), (1, 1), (5, 8), (9, 3), (70, 3)] {
                for minimal in [false, true] {
                    let lda = m + 2;
                    let k = m.min(n);
                    let mut a = vec![T::real(777.0); lda * n];
                    let mut original = vec![T::z(); m * n];
                    for j in 0..n {
                        for i in 0..m {
                            let x = T::real(((i * 13 + j * 17 + i * j) % 29) as f64 / 19.0 - 0.7)
                                + imag * T::real(((i * 7 + j * 11) % 13) as f64 / 23.0);
                            a[i + j * lda] = x;
                            original[i + j * m] = x;
                        }
                    }
                    let mut t = vec![T::real(777.0); 65536];
                    let mut work = vec![T::z(); 4096];
                    let mut info = 99i64;
                    factor(
                        m as isize,
                        n as isize,
                        a.as_mut_ptr(),
                        lda as isize,
                        t.as_mut_ptr(),
                        if minimal { (n + 5) as isize } else { 65536 },
                        work.as_mut_ptr(),
                        if minimal { n.max(1) as isize } else { 4096 },
                        &mut info,
                    );
                    assert_eq!(info, 0);
                    assert_eq!(t[3], T::real(777.0));
                    assert_eq!(t[4], T::real(777.0));
                    let mut q = vec![T::z(); m * m];
                    for i in 0..m {
                        q[i + i * m] = T::o();
                    }
                    apply(
                        b'L',
                        b'N',
                        m as isize,
                        m as isize,
                        k as isize,
                        a.as_ptr(),
                        lda as isize,
                        t.as_ptr(),
                        65536,
                        q.as_mut_ptr(),
                        m.max(1) as isize,
                        work.as_mut_ptr(),
                        4096,
                        &mut info,
                    );
                    assert_eq!(info, 0);
                    for j in 0..n {
                        for i in 0..m {
                            let qr = (0..k.min(j + 1))
                                .fold(T::z(), |s, l| s + q[i + l * m] * a[l + j * lda]);
                            assert!((qr - original[i + j * m]).magnitude() < tolerance);
                        }
                        for i in m..lda {
                            assert_eq!(a[i + j * lda], T::real(777.0));
                        }
                    }
                    for j in 0..m {
                        for i in 0..m {
                            let dot =
                                (0..m).fold(T::z(), |s, l| s + q[l + i * m].cj() * q[l + j * m]);
                            assert!(
                                (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                    < tolerance
                            );
                        }
                    }
                    for side in [b'L', b'R'] {
                        for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                            let (rows, cols) = if side == b'L' { (m, 3) } else { (3, m) };
                            let ldc = rows + 2;
                            let mut c = vec![T::real(777.0); ldc * cols];
                            for j in 0..cols {
                                for i in 0..rows {
                                    c[i + j * ldc] = T::real((i + j + 1) as f64 / 29.0)
                                        + imag * T::real((i + 2 * j) as f64 / 47.0);
                                }
                            }
                            let before = c.clone();
                            let op = |i: usize, j: usize| {
                                if tr == b'N' {
                                    q[i + j * m]
                                } else {
                                    q[j + i * m].cj()
                                }
                            };
                            apply(
                                side,
                                tr,
                                rows as isize,
                                cols as isize,
                                k as isize,
                                a.as_ptr(),
                                lda as isize,
                                t.as_ptr(),
                                65536,
                                c.as_mut_ptr(),
                                ldc as isize,
                                work.as_mut_ptr(),
                                4096,
                                &mut info,
                            );
                            assert_eq!(info, 0);
                            for j in 0..cols {
                                for i in 0..rows {
                                    let expected = if side == b'L' {
                                        (0..m)
                                            .fold(T::z(), |s, l| s + op(i, l) * before[l + j * ldc])
                                    } else {
                                        (0..m)
                                            .fold(T::z(), |s, l| s + before[i + l * ldc] * op(l, j))
                                    };
                                    assert!(
                                        (c[i + j * ldc] - expected).magnitude() < tolerance * 10.0
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
    #[test]
    fn real32() {
        verify(0.0f32, 5e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 5e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 5e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 5e-13);
    }
}
