//! Generate/apply the LAPACK Q/P sequences stored by bidiagonal reduction.
use crate::kernels::{flag, Scalar};
use crate::lapack::LapackInt;
use paste::paste;

struct Factor<T: Scalar> {
    start: usize,
    vector: Vec<T>,
    tau: T,
}
fn geometry(q: bool, order: usize, k: usize) -> (usize, usize) {
    let shifted = if q { order < k } else { order <= k };
    (
        if shifted { order.saturating_sub(1) } else { k },
        usize::from(shifted),
    )
}
unsafe fn gather<T: Scalar>(
    q: bool,
    order: usize,
    k: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
) -> Vec<Factor<T>> {
    let (count, shift) = geometry(q, order, k);
    (0..count)
        .map(|j| {
            let coefficient = *tau.add(j);
            let start = j + shift;
            let mut vector = vec![];
            if coefficient != T::z() {
                vector = vec![T::o(); order - start];
                for i in start + 1..order {
                    vector[i - start] = if q {
                        *a.offset(i as isize + j as isize * lda)
                    } else {
                        (*a.offset(j as isize + i as isize * lda)).cj()
                    };
                }
            }
            Factor {
                start,
                vector,
                tau: coefficient,
            }
        })
        .collect()
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
    if m == 0 || n == 0 || factors.is_empty() {
        return;
    }
    let reverse = (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N');
    let mut work = vec![T::z(); m.max(n)];
    for step in 0..factors.len() {
        let factor = &factors[if reverse {
            factors.len() - 1 - step
        } else {
            step
        }];
        if factor.tau == T::z() {
            continue;
        }
        let tau = if tr == b'N' {
            factor.tau
        } else {
            factor.tau.cj()
        };
        let start = factor.start;
        let (rows, cols, offset) = if side == b'L' {
            (m - start, n, start as isize)
        } else {
            (m, n - start, start as isize * ldc)
        };
        crate::householder::apply(
            side,
            rows as isize,
            cols as isize,
            factor.vector.as_ptr(),
            1,
            tau,
            c.offset(offset),
            if cols == 1 { 0 } else { ldc },
            work.as_mut_ptr(),
        );
    }
}
fn fits<T>(rows: isize, cols: isize) -> bool {
    rows.checked_mul(cols)
        .and_then(|x| x.checked_mul(std::mem::size_of::<T>() as isize))
        .is_some()
}
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}

unsafe fn apply<T: Scalar, I: LapackInt>(
    vect: u8,
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
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let q = vect == b'Q';
    let order = if side == b'L' { m } else { n };
    let minimum = if side == b'L' { n.max(1) } else { m.max(1) };
    let minlda = if q { order.max(1) } else { order.min(k).max(1) };
    for (arg, valid) in [
        (1, matches!(vect, b'Q' | b'P')),
        (2, matches!(side, b'L' | b'R')),
        (3, tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }),
        (4, m >= 0),
        (5, n >= 0),
        (6, k >= 0),
        (8, lda >= minlda),
        (11, ldc >= m.max(1)),
    ] {
        if !valid {
            error(info, arg);
            return;
        }
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork != -1 && lwork < minimum {
        error(info, 13);
        return;
    }
    if lwork == -1 {
        return;
    }
    if m == 0 || n == 0 {
        *work = T::o();
        return;
    }
    let (count, shift) = geometry(q, order as usize, k as usize);
    if count == 0 {
        return;
    }
    if !fits::<T>(count as isize, 1) {
        error(info, 6);
        return;
    }
    if (0..count).all(|j| *tau.add(j) == T::z()) {
        return;
    }
    if !crate::kernels::matrix_addressable::<T>(m, n, ldc) {
        error(info, 11);
        return;
    }
    if !fits::<T>(order, count as isize) {
        error(info, 8);
        return;
    }
    for j in 0..count {
        if *tau.add(j) != T::z() && j + shift + 1 < order as usize {
            let (rows, cols) = if q {
                (order, j as isize + 1)
            } else {
                (j as isize + 1, order)
            };
            if !crate::kernels::matrix_addressable::<T>(rows, cols, lda) {
                error(info, 8);
                return;
            }
        }
    }
    let factors = gather(q, order as usize, k as usize, a, lda, tau);
    execute(side, tr, m as usize, n as usize, &factors, c, ldc);
}

unsafe fn generate<T: Scalar, I: LapackInt>(
    vect: u8,
    m: isize,
    n: isize,
    k: isize,
    a: *mut T,
    lda: isize,
    tau: *const T,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let q = vect == b'Q';
    for (arg, valid) in [
        (1, matches!(vect, b'Q' | b'P')),
        (2, m >= 0),
        (
            3,
            n >= 0
                && if q {
                    n <= m && n >= m.min(k)
                } else {
                    m <= n && m >= n.min(k)
                },
        ),
        (4, k >= 0),
        (6, lda >= m.max(1)),
    ] {
        if !valid {
            error(info, arg);
            return;
        }
    }
    let minimum = m.min(n).max(1);
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork != -1 && lwork < minimum {
        error(info, 9);
        return;
    }
    if lwork == -1 {
        return;
    }
    if m == 0 || n == 0 {
        *work = T::o();
        return;
    }
    if !crate::kernels::matrix_addressable::<T>(m, n, lda) {
        error(info, 6);
        return;
    }
    let order = if q { m } else { n };
    let (count, _) = geometry(q, order as usize, k as usize);
    if !fits::<T>(order, count as isize) {
        error(info, 6);
        return;
    }
    // Snapshot only stored, nonzero reflectors before replacing A by an
    // identity slice. Reading an unstored triangle is unnecessary even here.
    let factors = gather(q, order as usize, k as usize, a, lda, tau);
    for j in 0..n {
        for i in 0..m {
            *a.offset(i + j * lda) = if i == j { T::o() } else { T::z() };
        }
    }
    execute(
        if q { b'L' } else { b'R' },
        if q {
            b'N'
        } else if T::COMPLEX {
            b'C'
        } else {
            b'T'
        },
        m as usize,
        n as usize,
        &factors,
        a,
        lda,
    );
}

macro_rules! exports {
    ($p:ident,$org:ident,$orm:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $org $suffix>](vect:*const u8,m:*const $i,n:*const $i,k:*const $i,
            a:*mut $t,lda:*const $i,tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {
            generate(flag(vect),*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,work,*lwork as isize,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $orm $suffix>](vect:*const u8,side:*const u8,tr:*const u8,
            m:*const $i,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,
            ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            apply(flag(vect),flag(side),flag(tr),*m as isize,*n as isize,*k as isize,a,*lda as isize,tau,
                c,*ldc as isize,work,*lwork as isize,info);
        }
    }};
}
macro_rules! abi {
    ($i:ty,$suffix:tt) => {
        exports!(s, orgbr, ormbr, f32, $i, $suffix);
        exports!(d, orgbr, ormbr, f64, $i, $suffix);
        exports!(c, ungbr, unmbr, colmat::c32, $i, $suffix);
        exports!(z, ungbr, unmbr, colmat::c64, $i, $suffix);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;
    fn product<T: Scalar>(matrix: &mut [T], order: usize, v: &[T], tau: T) {
        for i in 0..order {
            let dot = (0..order).fold(T::z(), |s, j| s + matrix[i + j * order] * v[j]);
            for j in 0..order {
                matrix[i + j * order] = matrix[i + j * order] - tau * dot * v[j].cj();
            }
        }
    }
    fn verify<T: ReflectorScalar>(imaginary: T, tolerance: f64) {
        unsafe {
            for m in [0, 1, 2, 5] {
                for n in [0, 1, 3, 6] {
                    let k = m.min(n);
                    let lda = m + 2;
                    let mut a = vec![T::real(777.0); lda * n];
                    for j in 0..n {
                        for i in 0..m {
                            a[i + j * lda] = T::real(((i + 2 * j + i * j) % 17) as f64 / 5.0 - 0.9)
                                + imaginary * T::real(((3 * i + j + 1) % 11) as f64 / 7.0 - 0.3);
                        }
                    }
                    let mut d = vec![0.0f64; k.max(1)];
                    let mut e = d.clone();
                    let mut tq = vec![T::z(); k];
                    let mut tp = tq.clone();
                    crate::bidiagonal::reduce(
                        m,
                        n,
                        a.as_mut_ptr(),
                        lda as isize,
                        d.as_mut_ptr(),
                        e.as_mut_ptr(),
                        tq.as_mut_ptr(),
                        tp.as_mut_ptr(),
                    );
                    for q in [false, true] {
                        let order = if q { m } else { n };
                        let param = if q { n } else { m };
                        let count = if (q && m >= n) || (!q && m < n) {
                            k
                        } else {
                            k.saturating_sub(1)
                        };
                        let shift = if (q && m < n) || (!q && m >= n) { 1 } else { 0 };
                        let tau = if q { &tq } else { &tp };
                        let mut expected = vec![T::z(); order * order];
                        let outlda = order + 2;
                        let mut stored = vec![T::real(777.0); outlda * order];
                        for j in 0..order {
                            expected[j + j * order] = T::o();
                            for i in 0..order {
                                stored[i + j * outlda] = T::real(f64::NAN);
                            }
                        }
                        for j in 0..count {
                            let start = j + shift;
                            let mut v = vec![T::z(); order];
                            v[start] = T::o();
                            for i in start + 1..order {
                                let value = if q {
                                    a[i + j * lda]
                                } else {
                                    a[j + i * lda].cj()
                                };
                                v[i] = value;
                                if q {
                                    stored[i + j * outlda] = value;
                                } else {
                                    stored[j + i * outlda] = value.cj();
                                }
                            }
                            product(&mut expected, order, &v, tau[j]);
                        }
                        let before = stored.clone();
                        let mut work = vec![T::real(777.0); order.max(1) + 2];
                        let mut info = -99i64;
                        generate(
                            if q { b'Q' } else { b'P' },
                            order as isize,
                            order as isize,
                            param as isize,
                            stored.as_mut_ptr(),
                            outlda as isize,
                            tau.as_ptr(),
                            work.as_mut_ptr().add(1),
                            order.max(1) as isize,
                            &mut info,
                        );
                        assert_eq!(info, 0);
                        for j in 0..order {
                            for i in 0..order {
                                let value = if q {
                                    expected[i + j * order]
                                } else {
                                    expected[j + i * order].cj()
                                };
                                assert!(
                                    (stored[i + j * outlda] - value).magnitude() < tolerance,
                                    "generate m={m} n={n} q={q} i={i} j={j}"
                                );
                            }
                            for i in order..outlda {
                                assert_eq!(stored[i + j * outlda], T::real(777.0));
                            }
                        }
                        assert_eq!(work[0], T::real(777.0));
                        assert_eq!(*work.last().unwrap(), T::real(777.0));
                        for side in [b'L', b'R'] {
                            for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                                let (rows, cols) =
                                    if side == b'L' { (order, 3) } else { (3, order) };
                                let ldc = rows + 2;
                                let mut c = vec![T::real(777.0); ldc * cols];
                                for j in 0..cols {
                                    for i in 0..rows {
                                        c[i + j * ldc] = T::real((2 * i + j + 1) as f64 / 7.0)
                                            + imaginary * T::real((i + 3 * j) as f64 / 11.0);
                                    }
                                }
                                let original = c.clone();
                                let mut info = -99i32;
                                let mut work = vec![T::real(777.0); rows.max(cols).max(1) + 2];
                                apply(
                                    if q { b'Q' } else { b'P' },
                                    side,
                                    tr,
                                    rows as isize,
                                    cols as isize,
                                    param as isize,
                                    before.as_ptr(),
                                    outlda as isize,
                                    tau.as_ptr(),
                                    c.as_mut_ptr(),
                                    ldc as isize,
                                    work.as_mut_ptr().add(1),
                                    rows.max(cols).max(1) as isize,
                                    &mut info,
                                );
                                assert_eq!(info, 0);
                                let op = |i: usize, j: usize| {
                                    if tr == b'N' {
                                        expected[i + j * order]
                                    } else {
                                        expected[j + i * order].cj()
                                    }
                                };
                                for j in 0..cols {
                                    for i in 0..rows {
                                        let value = (0..order).fold(T::z(), |sum, s| {
                                            sum + if side == b'L' {
                                                op(i, s) * original[s + j * ldc]
                                            } else {
                                                original[i + s * ldc] * op(s, j)
                                            }
                                        });
                                        assert!((c[i+j*ldc]-value).magnitude()<tolerance,"apply m={m} n={n} q={q} side={side} tr={tr} i={i} j={j}");
                                    }
                                    for i in rows..ldc {
                                        assert_eq!(c[i + j * ldc], T::real(777.0));
                                    }
                                }
                                assert_eq!(work[0], T::real(777.0));
                                assert_eq!(*work.last().unwrap(), T::real(777.0));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32_sequences() {
        verify(0.0f32, 1e-4);
    }
    #[test]
    fn real64_sequences() {
        verify(0.0f64, 1e-11);
    }
    #[test]
    fn complex32_sequences() {
        verify(colmat::c32::new(0.0, 1.0), 2e-4);
    }
    #[test]
    fn complex64_sequences() {
        verify(colmat::c64::new(0.0, 1.0), 1e-11);
    }
}
