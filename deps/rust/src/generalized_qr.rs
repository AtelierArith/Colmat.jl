//! Generalized QR/RQ compositions using the existing native-backed QR and RQ.
//! GGQRF: A = Q R, B = Q T Z. GGRQF: A = R Q, B = Z T Q.
use crate::householder::ReflectorScalar;
use crate::kernels::matrix_addressable;
use crate::lapack::LapackInt;
use paste::paste;

unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}

pub(crate) unsafe fn pair<T: ReflectorScalar>(
    rq: bool,
    first: usize,
    second: usize,
    third: usize,
    a: *mut T,
    lda: isize,
    taua: *mut T,
    b: *mut T,
    ldb: isize,
    taub: *mut T,
) {
    let adjoint = if T::COMPLEX { b'C' } else { b'T' };
    if rq {
        // M-by-N A and P-by-N B; A's Q acts on B from the right.
        let (m, p, n) = (first, second, third);
        let k = m.min(n);
        crate::reverse_qr::factor(true, m, n, a, lda, taua);
        if k != 0 && p != 0 {
            crate::reverse_qr::apply_rq(b'R', adjoint, p, n, k, a.add(m - k), lda, taua, b, ldb);
        }
        crate::qr::standard_qr(p, n, b, ldb, taub, false);
    } else {
        // N-by-M A and N-by-P B; A's Q acts on B from the left.
        let (n, m, p) = (first, second, third);
        let k = n.min(m);
        crate::qr::standard_qr(n, m, a, lda, taua, false);
        if k != 0 && p != 0 {
            crate::qr::apply_q(b'L', adjoint, n, p, k, a, lda, taua, b, ldb);
        }
        crate::reverse_qr::factor(true, n, p, b, ldb, taub);
    }
}

unsafe fn pair_driver<T: ReflectorScalar, I: LapackInt>(
    rq: bool,
    first: isize,
    second: isize,
    third: isize,
    a: *mut T,
    lda: isize,
    taua: *mut T,
    b: *mut T,
    ldb: isize,
    taub: *mut T,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let minimum = first.max(second).max(third).max(1);
    *work = crate::lapack::workspace::<T>(minimum);
    let (ar, ac, br, bc) = if rq {
        (first, third, second, third)
    } else {
        (first, second, first, third)
    };
    for (arg, ok) in [
        (1, first >= 0),
        (2, second >= 0),
        (3, third >= 0),
        (5, lda >= ar.max(1)),
        (8, ldb >= br.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if lwork != -1 && lwork < minimum {
        error(info, 11);
        return;
    }
    if lwork == -1 {
        return;
    }
    if !matrix_addressable::<T>(ar, ac, lda) {
        error(info, 5);
        return;
    }
    if !matrix_addressable::<T>(br, bc, ldb) {
        error(info, 8);
        return;
    }
    pair(
        rq,
        first as usize,
        second as usize,
        third as usize,
        a,
        lda,
        taua,
        b,
        ldb,
        taub,
    );
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {
        paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p ggqrf $s>](n:*const $i,m:*const $i,p:*const $i,
                a:*mut $t,lda:*const $i,taua:*mut $t,b:*mut $t,ldb:*const $i,taub:*mut $t,
                work:*mut $t,lwork:*const $i,info:*mut $i) {
                pair_driver(false,*n as isize,*m as isize,*p as isize,a,*lda as isize,taua,
                    b,*ldb as isize,taub,work,*lwork as isize,info);
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p ggrqf $s>](m:*const $i,p:*const $i,n:*const $i,
                a:*mut $t,lda:*const $i,taua:*mut $t,b:*mut $t,ldb:*const $i,taub:*mut $t,
                work:*mut $t,lwork:*const $i,info:*mut $i) {
                pair_driver(true,*m as isize,*p as isize,*n as isize,a,*lda as isize,taua,
                    b,*ldb as isize,taub,work,*lwork as isize,info);
            }
        }
    };
}
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
    use crate::kernels::Scalar;
    fn identity<T: Scalar>(n: usize) -> Vec<T> {
        (0..n * n)
            .map(|i| if i / n == i % n { T::o() } else { T::z() })
            .collect()
    }
    fn multiply<T: Scalar>(m: usize, k: usize, n: usize, a: &[T], b: &[T]) -> Vec<T> {
        let mut c = vec![T::z(); m * n];
        for j in 0..n {
            for l in 0..k {
                for i in 0..m {
                    c[i + j * m] = c[i + j * m] + a[i + l * m] * b[l + j * k];
                }
            }
        }
        c
    }
    unsafe fn extract<T: Scalar>(
        rq: bool,
        m: usize,
        n: usize,
        a: &[T],
        ld: usize,
        tau: &[T],
    ) -> (Vec<T>, Vec<T>) {
        let order = if rq { n } else { m };
        let k = m.min(n);
        let mut q = identity::<T>(order);
        if rq {
            if k > 0 {
                crate::reverse_qr::apply_rq(
                    b'R',
                    b'N',
                    n,
                    n,
                    k,
                    a.as_ptr().add(m - k),
                    ld as isize,
                    tau.as_ptr(),
                    q.as_mut_ptr(),
                    n.max(1) as isize,
                );
            }
        } else {
            crate::qr::apply_q(
                b'L',
                b'N',
                m,
                m,
                k,
                a.as_ptr(),
                ld as isize,
                tau.as_ptr(),
                q.as_mut_ptr(),
                m.max(1) as isize,
            );
        }
        let mut r = vec![T::z(); m * n];
        for j in 0..n {
            for i in 0..m {
                if if rq {
                    i as isize <= j as isize + m as isize - n as isize
                } else {
                    i <= j
                } {
                    r[i + j * m] = a[i + j * ld];
                }
            }
        }
        (q, r)
    }
    fn close<T: Scalar>(a: &[T], b: &[T], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(
                (*x - *y).abs1() <= tol * (1.0 + y.abs1()),
                "reconstruction residual {}",
                (*x - *y).abs1()
            );
        }
    }
    fn check<T: ReflectorScalar>(tol: f64) {
        unsafe {
            for rq in [false, true] {
                for first in 0..5 {
                    for second in 0..5 {
                        for third in 0..5 {
                            let (ar, ac, br, bc) = if rq {
                                (first, third, second, third)
                            } else {
                                (first, second, first, third)
                            };
                            let (la, lb) = (ar + 2, br + 3);
                            let guard = T::parts(71.0, -9.0);
                            let mut a = vec![guard; la * ac + 1];
                            let mut b = vec![guard; lb * bc + 1];
                            let mut ao = vec![T::z(); ar * ac];
                            let mut bo = vec![T::z(); br * bc];
                            for j in 0..ac {
                                for i in 0..ar {
                                    let z = T::parts(
                                        (1 + i + 3 * j) as f64 + if i == j { 2.0 } else { 0.0 },
                                        (i as f64 - j as f64) * 0.31,
                                    );
                                    a[i + j * la] = z;
                                    ao[i + j * ar] = z;
                                }
                            }
                            for j in 0..bc {
                                for i in 0..br {
                                    let z = T::parts(
                                        (2 + 2 * i + j) as f64 - if i == j { 3.0 } else { 0.0 },
                                        (i as f64 + j as f64) * 0.23,
                                    );
                                    b[i + j * lb] = z;
                                    bo[i + j * br] = z;
                                }
                            }
                            let mut ta = vec![guard; ar.min(ac) + 1];
                            let mut tb = vec![guard; br.min(bc) + 1];
                            pair(
                                rq,
                                first,
                                second,
                                third,
                                a.as_mut_ptr(),
                                la as isize,
                                ta.as_mut_ptr(),
                                b.as_mut_ptr(),
                                lb as isize,
                                tb.as_mut_ptr(),
                            );
                            let (qa, ra) = extract(rq, ar, ac, &a, la, &ta);
                            let (qb, rb) = extract(!rq, br, bc, &b, lb, &tb);
                            if rq {
                                close(&multiply(ar, ac, ac, &ra, &qa), &ao, tol);
                                close(
                                    &multiply(br, ac, ac, &multiply(br, br, ac, &qb, &rb), &qa),
                                    &bo,
                                    tol,
                                );
                            } else {
                                close(&multiply(ar, ar, ac, &qa, &ra), &ao, tol);
                                close(
                                    &multiply(br, br, bc, &qa, &multiply(br, bc, bc, &rb, &qb)),
                                    &bo,
                                    tol,
                                );
                            }
                            assert_eq!(ta[ar.min(ac)], guard);
                            assert_eq!(tb[br.min(bc)], guard);
                            for j in 0..ac {
                                for i in ar..la {
                                    assert_eq!(a[i + j * la], guard);
                                }
                            }
                            for j in 0..bc {
                                for i in br..lb {
                                    assert_eq!(b[i + j * lb], guard);
                                }
                            }
                            assert_eq!(a[la * ac], guard);
                            assert_eq!(b[lb * bc], guard);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn pair_real32() {
        check::<f32>(5e-5);
    }
    #[test]
    fn pair_real64() {
        check::<f64>(1e-12);
    }
    #[test]
    fn pair_complex32() {
        check::<colmat::c32>(8e-5);
    }
    #[test]
    fn pair_complex64() {
        check::<colmat::c64>(1e-12);
    }
}
