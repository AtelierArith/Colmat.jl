//! Adapt native self-adjoint tridiagonalization to LAPACK triangle storage.
use crate::generalized_selfadjoint::{hermitian, GeneralizedScalar};
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;

// Native real reflectors have LAPACK's real-beta gauge. Complex native
// reflectors need a separate real-E gauge adaptation before being exposed.
pub(crate) unsafe fn native_real<T: ReflectorScalar + GeneralizedScalar, R: Scalar>(
    lower: bool,
    n: usize,
    a: *mut T,
    lda: isize,
    d: *mut R,
    e: *mut R,
    tau: *mut T,
) {
    assert!(!T::COMPLEX);
    if n == 0 {
        return;
    }
    let mut matrix = hermitian(if lower { b'L' } else { b'U' }, n, a, lda);
    if !lower {
        // Reversal maps LAPACK's descending upper reflectors to the native
        // ascending lower sequence without changing their real gauge.
        for j in 0..n {
            for i in 0..n {
                let (r, c) = (n - 1 - i, n - 1 - j);
                if i + j * n < r + c * n {
                    let old = matrix[(i, j)];
                    matrix[(i, j)] = matrix[(r, c)];
                    matrix[(r, c)] = old;
                }
            }
        }
    }
    let mut coefficients = Mat::<T>::zeros(1, n.saturating_sub(1));
    use colmat::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(tridiag_in_place_scratch::<T>(
            n,
            par.shape(),
            Default::default(),
        ));
        tridiag_in_place(
            matrix.as_mut(),
            coefficients.as_mut(),
            par,
            MemStack::new(&mut memory),
            Default::default(),
        );
    });
    for j in 0..n {
        let column = if lower { j } else { n - 1 - j };
        *d.add(j) = R::real(matrix[(column, column)].re());
        if j + 1 < n {
            let stage = if lower { j } else { n - 2 - j };
            *e.add(j) = R::real(matrix[(stage + 1, stage)].re());
            *tau.add(j) = T::real(1.0 / coefficients[(0, stage)].re());
        }
        let rows = if lower { j..n } else { 0..j + 1 };
        for i in rows {
            *a.offset(i as isize + j as isize * lda) = if lower {
                matrix[(i, j)]
            } else {
                matrix[(n - 1 - i, n - 1 - j)]
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar + GeneralizedScalar>(tolerance: f64) {
        unsafe {
            for lower in [false, true] {
                for n in [0, 1, 2, 5, 8] {
                    for mode in 0..3 {
                        let lda = n + 3;
                        let mut a = vec![T::real(777.0); lda * n + 2];
                        let mut original = vec![T::z(); n * n];
                        for j in 0..n {
                            for i in 0..n {
                                let value = match mode {
                                    0 => T::real(
                                        ((i.min(j) * 17 + i.max(j) * 31 + i * j) % 29) as f64
                                            / 13.0
                                            - 0.8,
                                    ),
                                    1 => {
                                        if i == j {
                                            T::real((i + 1) as f64)
                                        } else {
                                            T::z()
                                        }
                                    }
                                    _ => T::z(),
                                };
                                original[i + j * n] = value;
                                a[1 + i + j * lda] = if (lower && i >= j) || (!lower && i <= j) {
                                    value
                                } else {
                                    T::real(f64::NAN)
                                };
                            }
                        }
                        let before = a.clone();
                        let mut d = vec![777.0f64; n + 2];
                        let mut e = d.clone();
                        let mut tau = vec![T::real(777.0); n + 2];
                        native_real(
                            lower,
                            n,
                            a.as_mut_ptr().add(1),
                            lda as isize,
                            d.as_mut_ptr().add(1),
                            e.as_mut_ptr().add(1),
                            tau.as_mut_ptr().add(1),
                        );
                        let mut q = vec![T::z(); n * n];
                        for i in 0..n {
                            q[i + i * n] = T::o();
                        }
                        for step in 0..n.saturating_sub(1) {
                            let stage = if lower { step } else { n - 2 - step };
                            let mut v = vec![T::z(); n];
                            if lower {
                                v[stage + 1] = T::o();
                                for i in stage + 2..n {
                                    v[i] = a[1 + i + stage * lda];
                                }
                            } else {
                                v[stage] = T::o();
                                for i in 0..stage {
                                    v[i] = a[1 + i + (stage + 1) * lda];
                                }
                            }
                            for i in 0..n {
                                let dot = (0..n).fold(T::z(), |sum, j| sum + q[i + j * n] * v[j]);
                                for j in 0..n {
                                    q[i + j * n] = q[i + j * n] - tau[stage + 1] * dot * v[j].cj();
                                }
                            }
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let mut value = T::z();
                                let mut dot = T::z();
                                for s in 0..n {
                                    value = value
                                        + q[i + s * n] * T::real(d[s + 1]) * q[j + s * n].cj();
                                    dot = dot + q[s + i * n].cj() * q[s + j * n];
                                }
                                for s in 0..n.saturating_sub(1) {
                                    value = value
                                        + T::real(e[s + 1])
                                            * (q[i + s * n] * q[j + (s + 1) * n].cj()
                                                + q[i + (s + 1) * n] * q[j + s * n].cj());
                                }
                                assert!(
                                    (value - original[i + j * n]).magnitude() < tolerance,
                                    "n={n} lower={lower} mode={mode} i={i} j={j}"
                                );
                                assert!(
                                    (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                        < tolerance
                                );
                                if (lower && i < j) || (!lower && i > j) {
                                    assert!(a[1 + i + j * lda].re().is_nan());
                                }
                            }
                            for i in n..lda {
                                assert_eq!(a[1 + i + j * lda], before[1 + i + j * lda]);
                            }
                        }
                        assert_eq!(a[0], T::real(777.0));
                        assert_eq!(*a.last().unwrap(), T::real(777.0));
                    }
                }
            }
        }
    }
    #[test]
    fn native_real32_triangles() {
        verify::<f32>(5e-4);
    }
    #[test]
    fn native_real64_triangles() {
        verify::<f64>(1e-11);
    }
}
