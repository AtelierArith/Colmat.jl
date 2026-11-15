//! LAPACK Q generation/application for self-adjoint tridiagonal factors.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;

struct Factor<T: Scalar> {
    start: usize,
    vector: Vec<T>,
    tau: T,
}
unsafe fn gather<T: Scalar>(
    lower: bool,
    n: usize,
    a: *const T,
    lda: isize,
    tau: *const T,
) -> Vec<Factor<T>> {
    (0..n.saturating_sub(1))
        .map(|step| {
            let j = if lower { step } else { n - 2 - step };
            let coefficient = *tau.add(j);
            let start = if lower { j + 1 } else { 0 };
            let len = if lower { n - start } else { j + 1 };
            let mut vector = Vec::new();
            if coefficient != T::z() {
                vector = vec![T::o(); len];
                if lower {
                    for k in 1..len {
                        vector[k] = *a.offset((start + k) as isize + j as isize * lda);
                    }
                } else {
                    for k in 0..len - 1 {
                        vector[k] = *a.offset(k as isize + (j + 1) as isize * lda);
                    }
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
    let reverse = (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N');
    let mut work = vec![T::z(); m.max(n)];
    for step in 0..factors.len() {
        let f = &factors[if reverse {
            factors.len() - 1 - step
        } else {
            step
        }];
        if f.tau == T::z() {
            continue;
        }
        let (rows, cols, offset) = if side == b'L' {
            (f.vector.len(), n, f.start as isize)
        } else {
            (m, f.vector.len(), f.start as isize * ldc)
        };
        crate::householder::apply(
            side,
            rows as isize,
            cols as isize,
            f.vector.as_ptr(),
            1,
            if tr == b'N' { f.tau } else { f.tau.cj() },
            c.offset(offset),
            if cols == 1 { 0 } else { ldc },
            work.as_mut_ptr(),
        );
    }
}
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn factors_addressable<T: Scalar>(lower: bool, n: isize, lda: isize, tau: *const T) -> bool {
    if !matrix_addressable::<T>(n.saturating_sub(1), 1, 1) {
        return false;
    }
    for j in 0..n.saturating_sub(1) {
        if *tau.offset(j) != T::z() {
            let (rows, cols) = if lower {
                if j + 2 >= n {
                    continue;
                }
                (n, j + 1)
            } else {
                if j == 0 {
                    continue;
                }
                (j, j + 2)
            };
            if !matrix_addressable::<T>(rows, cols, lda) {
                return false;
            }
        }
    }
    true
}
unsafe fn generate<T: Scalar, I: LapackInt>(
    u: u8,
    n: isize,
    a: *mut T,
    lda: isize,
    tau: *const T,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(u, b'L' | b'U')),
        (2, n >= 0),
        (4, lda >= n.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let minimum = (n - 1).max(1);
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork != -1 && lwork < minimum {
        error(info, 7);
        return;
    }
    if lwork == -1 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(n, n, lda) {
        error(info, 4);
        return;
    }
    let factors = gather(u == b'L', n as usize, a, lda, tau);
    for j in 0..n {
        for i in 0..n {
            *a.offset(i + j * lda) = if i == j { T::o() } else { T::z() };
        }
    }
    execute(b'L', b'N', n as usize, n as usize, &factors, a, lda);
}
unsafe fn apply<T: Scalar, I: LapackInt>(
    side: u8,
    u: u8,
    tr: u8,
    m: isize,
    n: isize,
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
    let nq = if side == b'L' { m } else { n };
    let minimum = if side == b'L' { n.max(1) } else { m.max(1) };
    for (arg, ok) in [
        (1, matches!(side, b'L' | b'R')),
        (2, matches!(u, b'L' | b'U')),
        (3, tr == b'N' || tr == if T::COMPLEX { b'C' } else { b'T' }),
        (4, m >= 0),
        (5, n >= 0),
        (7, lda >= nq.max(1)),
        (10, ldc >= m.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork != -1 && lwork < minimum {
        error(info, 12);
        return;
    }
    if lwork == -1 || m == 0 || n == 0 || nq <= 1 {
        return;
    }
    if !factors_addressable(u == b'L', nq, lda, tau) {
        error(info, 7);
        return;
    }
    if (0..nq - 1).all(|j| *tau.offset(j) == T::z()) {
        return;
    }
    if !matrix_addressable::<T>(m, n, ldc) {
        error(info, 10);
        return;
    }
    let factors = gather(u == b'L', nq as usize, a, lda, tau);
    execute(side, tr, m as usize, n as usize, &factors, c, ldc);
}
macro_rules! exports { ($p:ident,$org:ident,$orm:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $org $suffix>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,tau:*const $t,work:*mut $t,lwork:*const $i,info:*mut $i) {generate(flag(u),*n as isize,a,*lda as isize,tau,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $orm $suffix>](side:*const u8,u:*const u8,tr:*const u8,m:*const $i,n:*const $i,a:*const $t,lda:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {apply(flag(side),flag(u),flag(tr),*m as isize,*n as isize,a,*lda as isize,tau,c,*ldc as isize,work,*lwork as isize,info);}
}}; }
macro_rules! abi {
    ($i:ty,$s:tt) => {
        exports!(s, orgtr, ormtr, f32, $i, $s);
        exports!(d, orgtr, ormtr, f64, $i, $s);
        exports!(c, ungtr, unmtr, colmat::c32, $i, $s);
        exports!(z, ungtr, unmtr, colmat::c64, $i, $s);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: Scalar>(imag: T, tolerance: f64) {
        unsafe {
            for lower in [false, true] {
                for n in [0usize, 1, 2, 5, 8] {
                    let lda = n + 2;
                    let mut stored = vec![T::real(777.0); lda * n];
                    for j in 0..n {
                        for i in 0..n {
                            stored[i + j * lda] = T::real(f64::NAN);
                        }
                    }
                    let mut tau = vec![T::z(); n.saturating_sub(1)];
                    let mut expected = vec![T::z(); n * n];
                    for j in 0..n {
                        expected[j + j * n] = T::o();
                    }
                    for step in 0..n.saturating_sub(1) {
                        let j = if lower { step } else { n - 2 - step };
                        let mut v = vec![T::z(); n];
                        let (start, end, head) = if lower {
                            (j + 1, n, j + 1)
                        } else {
                            (0, j + 1, j)
                        };
                        v[head] = T::o();
                        for k in start..end {
                            if k != head {
                                v[k] = T::real((k + 1) as f64 / 11.0)
                                    + imag * T::real((j + k + 1) as f64 / 19.0);
                                stored[k + if lower { j * lda } else { (j + 1) * lda }] = v[k];
                            }
                        }
                        let norm = (start..end).fold(T::z(), |a, k| a + v[k].cj() * v[k]);
                        // A phase in the coefficient tests adjoints even for a
                        // length-one complex reflector. A zero coefficient must
                        // leave poisoned stored tails unread.
                        tau[j] = if j % 3 == 1 {
                            T::z()
                        } else {
                            (T::real(2.0) + imag * T::real(0.2)) / norm
                        };
                        for i in 0..n {
                            let dot = (0..n).fold(T::z(), |a, k| a + expected[i + k * n] * v[k]);
                            for k in 0..n {
                                expected[i + k * n] =
                                    expected[i + k * n] - tau[j] * dot * v[k].cj();
                            }
                        }
                        if tau[j] == T::z() {
                            for k in start..end {
                                if k != head {
                                    stored[k + if lower { j * lda } else { (j + 1) * lda }] =
                                        T::real(f64::NAN);
                                }
                            }
                        }
                    }
                    let u = if lower { b'L' } else { b'U' };
                    let mut generated = stored.clone();
                    let mut work = vec![T::z(); n.max(1)];
                    let mut info = 99i64;
                    generate(
                        u,
                        n as isize,
                        generated.as_mut_ptr(),
                        lda as isize,
                        tau.as_ptr(),
                        work.as_mut_ptr(),
                        n.max(1) as isize,
                        &mut info,
                    );
                    assert_eq!(info, 0);
                    for j in 0..n {
                        for i in 0..n {
                            assert!(
                                (generated[i + j * lda] - expected[i + j * n]).magnitude()
                                    < tolerance
                            );
                        }
                        for i in n..lda {
                            assert_eq!(generated[i + j * lda], T::real(777.0));
                        }
                    }
                    for side in [b'L', b'R'] {
                        for tr in [b'N', if T::COMPLEX { b'C' } else { b'T' }] {
                            let (m, cols) = if side == b'L' { (n, 3) } else { (3, n) };
                            let ldc = m + 2;
                            let mut c = vec![T::real(777.0); ldc * cols];
                            for j in 0..cols {
                                for i in 0..m {
                                    c[i + j * ldc] = T::real((i + 2 * j + 1) as f64 / 7.0)
                                        + imag * T::real((2 * i + j) as f64 / 13.0);
                                }
                            }
                            let original = c.clone();
                            let mut want = vec![T::z(); m * cols];
                            let q = |i: usize, j: usize| {
                                if tr == b'N' {
                                    expected[i + j * n]
                                } else {
                                    expected[j + i * n].cj()
                                }
                            };
                            for j in 0..cols {
                                for i in 0..m {
                                    want[i + j * m] = if side == b'L' {
                                        (0..n).fold(T::z(), |a, k| {
                                            a + q(i, k) * original[k + j * ldc]
                                        })
                                    } else {
                                        (0..n).fold(T::z(), |a, k| {
                                            a + original[i + k * ldc] * q(k, j)
                                        })
                                    };
                                }
                            }
                            let mut work = vec![T::z(); m.max(cols).max(1)];
                            apply(
                                side,
                                u,
                                tr,
                                m as isize,
                                cols as isize,
                                stored.as_ptr(),
                                lda as isize,
                                tau.as_ptr(),
                                c.as_mut_ptr(),
                                ldc as isize,
                                work.as_mut_ptr(),
                                work.len() as isize,
                                &mut info,
                            );
                            assert_eq!(info, 0);
                            for j in 0..cols {
                                for i in 0..m {
                                    assert!(
                                        (c[i + j * ldc] - want[i + j * m]).magnitude() < tolerance
                                    );
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
    #[test]
    fn real32() {
        verify::<f32>(0.0, 2e-5);
    }
    #[test]
    fn real64() {
        verify::<f64>(0.0, 2e-13);
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
