//! LAPACK-gauge bidiagonalization from native bidiagonal reflector panels.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;

fn divide<T: ReflectorScalar>(a: T, b: T) -> T {
    let scale = b.re().abs().max(b.im().abs());
    let (r, i) = (b.re() / scale, b.im() / scale);
    let (x, y) = (a.re() / scale, a.im() / scale);
    let norm = r * r + i * i;
    T::parts((x * r + y * i) / norm, (y * r - x * i) / norm)
}

/// Native H uses a real reciprocal coefficient and may leave complex beta.
/// LAPACK instead requires real beta and a complex tau. Both vectors are
/// proportional to x-tail, so rescale the native tail by
/// (alpha-beta_native)/(alpha-beta_lapack), then convert the scalar coefficient.
unsafe fn reflector<T: ReflectorScalar>(n: usize, a: *mut T, inc: isize) -> T {
    if n == 0 {
        return T::z();
    }
    let original = *a;
    let mut scale = original.re().abs().max(original.im().abs());
    let mut tail_nonzero = false;
    let mut finite = original.re().is_finite() && original.im().is_finite();
    for i in 1..n {
        let value = *a.offset(i as isize * inc);
        scale = scale.max(value.re().abs()).max(value.im().abs());
        tail_nonzero |= value != T::z();
        finite &= value.re().is_finite() && value.im().is_finite();
    }
    if !tail_nonzero && original.im() == 0.0 {
        return T::z();
    }
    if !finite {
        let mut tau = T::z();
        crate::householder::generate(n as isize, a, a.offset(inc), inc, &mut tau, false);
        return tau;
    }
    let mut panel = Mat::<T>::from_fn(n, 1, |i, _| {
        let v = *a.offset(i as isize * inc);
        T::parts(v.re() / scale, v.im() / scale)
    });
    let alpha = panel[(0, 0)];
    let mut left = Mat::<T>::zeros(1, 1);
    let mut right = Mat::<T>::zeros(1, 0);
    use colmat::linalg::svd::bidiag::{bidiag_in_place, bidiag_in_place_scratch};
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(bidiag_in_place_scratch::<T>(
            n,
            1,
            par.shape(),
            Default::default(),
        ));
        bidiag_in_place(
            panel.as_mut(),
            left.as_mut(),
            right.as_mut(),
            par,
            MemStack::new(&mut memory),
            Default::default(),
        );
    });
    let beta_native = panel[(0, 0)];
    let beta = -beta_native.magnitude().copysign(alpha.re());
    let target = T::parts(alpha.re() - beta, alpha.im());
    let tau = T::parts(1.0 - alpha.re() / beta, -alpha.im() / beta);
    // The native panel intentionally suppresses tails below min-positive.
    // Keep LAPACK's representable tails and real-beta gauge in that case.
    let native_zero = left[(0, 0)].re().is_infinite();
    let minimum = if T::EPS > 1e-10 {
        f32::MIN_POSITIVE as f64
    } else {
        f64::MIN_POSITIVE
    };
    let native_alpha = if alpha.magnitude() < minimum {
        T::z()
    } else {
        alpha
    };
    let ratio = if native_zero {
        T::z()
    } else {
        divide(native_alpha - beta_native, target)
    };
    for i in 1..n {
        let value = panel[(i, 0)];
        *a.offset(i as isize * inc) = if native_zero {
            divide(value, target)
        } else {
            value * ratio
        };
    }
    *a = T::real(beta * scale);
    tau
}

unsafe fn left<T: ReflectorScalar>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    tau: T,
    work: *mut T,
) {
    if m == 0 || n == 0 {
        return;
    }
    let beta = *a;
    *a = T::o();
    crate::householder::apply(
        b'L',
        m as isize,
        n as isize,
        a,
        1,
        tau.cj(),
        a.offset(lda),
        lda,
        work,
    );
    *a = beta;
}
unsafe fn right<T: ReflectorScalar>(m: usize, n: usize, a: *mut T, lda: isize, work: *mut T) -> T {
    for i in 0..n {
        *a.offset(i as isize * lda) = (*a.offset(i as isize * lda)).cj();
    }
    let tau = reflector(n, a, lda);
    let beta = *a;
    *a = T::o();
    if m > 0 {
        crate::householder::apply(
            b'R',
            m as isize,
            n as isize,
            a,
            lda,
            tau,
            a.add(1),
            lda,
            work,
        );
    }
    for i in 0..n {
        *a.offset(i as isize * lda) = (*a.offset(i as isize * lda)).cj();
    }
    *a = beta;
    tau
}

pub(crate) unsafe fn reduce<T: ReflectorScalar, R: Scalar>(
    m: usize,
    n: usize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tauq: *mut T,
    taup: *mut T,
) {
    if m == 0 || n == 0 {
        return;
    }
    let mut work = vec![T::z(); m.max(n)];
    if m >= n {
        for j in 0..n {
            let diagonal = a.offset(j as isize * (lda + 1));
            let tq = reflector(m - j, diagonal, 1);
            *tauq.add(j) = tq;
            *d.add(j) = R::real((*diagonal).re());
            left(m - j, n - j - 1, diagonal, lda, tq, work.as_mut_ptr());
            if j + 1 < n {
                let offdiagonal = diagonal.offset(lda);
                *taup.add(j) = right(m - j - 1, n - j - 1, offdiagonal, lda, work.as_mut_ptr());
                *e.add(j) = R::real((*offdiagonal).re());
            } else {
                *taup.add(j) = T::z();
            }
        }
    } else {
        for j in 0..m {
            let diagonal = a.offset(j as isize * (lda + 1));
            *taup.add(j) = right(m - j - 1, n - j, diagonal, lda, work.as_mut_ptr());
            *d.add(j) = R::real((*diagonal).re());
            if j + 1 < m {
                let offdiagonal = diagonal.add(1);
                let tq = reflector(m - j - 1, offdiagonal, 1);
                *tauq.add(j) = tq;
                *e.add(j) = R::real((*offdiagonal).re());
                left(
                    m - j - 1,
                    n - j - 1,
                    offdiagonal,
                    lda,
                    tq,
                    work.as_mut_ptr(),
                );
            } else {
                *tauq.add(j) = T::z();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn multiply_reflector<T: Scalar>(matrix: &mut [T], n: usize, v: &[T], tau: T) {
        for i in 0..n {
            let dot = (0..n).fold(T::z(), |sum, j| sum + matrix[i + j * n] * v[j]);
            for j in 0..n {
                matrix[i + j * n] = matrix[i + j * n] - tau * dot * v[j].cj();
            }
        }
    }
    fn reconstruction<T: ReflectorScalar>(imaginary: T, tol: f64) {
        unsafe {
            for m in [0, 1, 2, 4, 7] {
                for n in [0, 1, 3, 6] {
                    for mode in 0..3 {
                        let k = m.min(n);
                        let lda = m + 2;
                        let mut stored = vec![T::real(777.0); lda * n + 2];
                        let mut original = vec![T::z(); m * n];
                        for j in 0..n {
                            for i in 0..m {
                                let value = match mode {
                                    0 => {
                                        T::real(
                                            ((13 * i + 17 * j + 11 * i * j) % 31) as f64 / 17.0
                                                - 0.8,
                                        ) + imaginary
                                            * T::real(
                                                ((7 * i + 3 * j + 1) % 19) as f64 / 13.0 - 0.5,
                                            )
                                    }
                                    1 => {
                                        (T::real((i + 1) as f64) + imaginary * T::real(i as f64))
                                            * (T::real((j + 1) as f64)
                                                - imaginary * T::real(j as f64))
                                    }
                                    _ => {
                                        if i == j {
                                            T::real((i + 1) as f64) + imaginary
                                        } else {
                                            T::z()
                                        }
                                    }
                                };
                                stored[1 + i + j * lda] = value;
                                original[i + j * m] = value;
                            }
                        }
                        let mut d = vec![777.0f64; k + 2];
                        let mut e = vec![777.0f64; k + 2];
                        let mut tq = vec![T::real(777.0); k + 2];
                        let mut tp = tq.clone();
                        reduce(
                            m,
                            n,
                            stored.as_mut_ptr().add(1),
                            lda as isize,
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            tq.as_mut_ptr().add(1),
                            tp.as_mut_ptr().add(1),
                        );
                        let mut q = vec![T::z(); m * m];
                        let mut p = vec![T::z(); n * n];
                        for i in 0..m {
                            q[i + i * m] = T::o();
                        }
                        for i in 0..n {
                            p[i + i * n] = T::o();
                        }
                        for j in 0..k {
                            let start = if m >= n { j } else { j + 1 };
                            if start < m {
                                let mut v = vec![T::z(); m];
                                v[start] = T::o();
                                for i in start + 1..m {
                                    v[i] = stored[1 + i + j * lda];
                                }
                                multiply_reflector(&mut q, m, &v, tq[j + 1]);
                            }
                            let start = if m >= n { j + 1 } else { j };
                            if start < n {
                                let mut v = vec![T::z(); n];
                                v[start] = T::o();
                                for i in start + 1..n {
                                    v[i] = stored[1 + j + i * lda].cj();
                                }
                                multiply_reflector(&mut p, n, &v, tp[j + 1]);
                            }
                        }
                        for j in 0..n {
                            for i in 0..m {
                                let mut value = T::z();
                                for s in 0..k {
                                    value = value
                                        + q[i + s * m] * T::real(d[s + 1]) * p[j + s * n].cj();
                                }
                                for s in 0..k.saturating_sub(1) {
                                    value = value
                                        + if m >= n {
                                            q[i + s * m]
                                                * T::real(e[s + 1])
                                                * p[j + (s + 1) * n].cj()
                                        } else {
                                            q[i + (s + 1) * m]
                                                * T::real(e[s + 1])
                                                * p[j + s * n].cj()
                                        };
                                }
                                assert!(
                                    (value - original[i + j * m]).magnitude() < tol,
                                    "m={m} n={n} mode={mode} row={i} col={j}"
                                );
                            }
                        }
                        for (matrix, size) in [(&q, m), (&p, n)] {
                            for j in 0..size {
                                for i in 0..size {
                                    let dot = (0..size).fold(T::z(), |sum, s| {
                                        sum + matrix[s + i * size].cj() * matrix[s + j * size]
                                    });
                                    assert!(
                                        (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                            < tol
                                    );
                                }
                            }
                        }
                        for j in 0..n {
                            for i in m..lda {
                                assert_eq!(stored[1 + i + j * lda], T::real(777.0));
                            }
                        }
                        assert_eq!(stored[0], T::real(777.0));
                        assert_eq!(*stored.last().unwrap(), T::real(777.0));
                        for v in [&d, &e] {
                            assert_eq!(v[0], 777.0);
                            assert_eq!(*v.last().unwrap(), 777.0);
                        }
                        for v in [&tq, &tp] {
                            assert_eq!(v[0], T::real(777.0));
                            assert_eq!(*v.last().unwrap(), T::real(777.0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32_upper_lower_reconstruction() {
        reconstruction(0.0f32, 2e-4);
    }
    #[test]
    fn real64_upper_lower_reconstruction() {
        reconstruction(0.0f64, 1e-11);
    }
    #[test]
    fn complex32_upper_lower_reconstruction() {
        reconstruction(colmat::c32::new(0.0, 1.0), 5e-4);
    }
    #[test]
    fn complex64_upper_lower_reconstruction() {
        reconstruction(colmat::c64::new(0.0, 1.0), 1e-11);
    }
    fn check<T: ReflectorScalar>(imaginary: T, tol: f64) {
        unsafe {
            for alpha in [
                T::real(2.0),
                T::real(-2.0),
                imaginary,
                T::real(1.0) + imaginary,
                T::z(),
            ] {
                for tail in [0.0, 0.5, 3.0] {
                    let original = [
                        alpha,
                        T::real(tail) + imaginary * T::real(tail),
                        T::real(-tail),
                    ];
                    let mut data = original;
                    let tau = reflector(3, data.as_mut_ptr(), 1);
                    let v = [T::o(), data[1], data[2]];
                    let dot = v
                        .iter()
                        .zip(original)
                        .fold(T::z(), |sum, (&x, y)| sum + x.cj() * y);
                    for i in 0..3 {
                        let result = original[i] - tau.cj() * v[i] * dot;
                        assert!((result - if i == 0 { data[0] } else { T::z() }).magnitude() < tol);
                    }
                    let vn = v.iter().map(|x| x.magnitude().powi(2)).sum::<f64>();
                    assert!((tau + tau.cj() - tau * tau.cj() * T::real(vn)).magnitude() < tol);
                    assert_eq!(data[0].im(), 0.0);
                }
            }
        }
    }
    #[test]
    fn real32_reflector_gauge() {
        check(0.0f32, 2e-5);
    }
    #[test]
    fn real64_reflector_gauge() {
        check(0.0f64, 1e-12);
    }
    #[test]
    fn complex32_reflector_gauge() {
        check(colmat::c32::new(0.0, 1.0), 5e-5);
    }
    #[test]
    fn complex64_reflector_gauge() {
        check(colmat::c64::new(0.0, 1.0), 1e-12);
    }
}
