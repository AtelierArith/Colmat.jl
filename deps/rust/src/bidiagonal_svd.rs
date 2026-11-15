//! Bidiagonal SVD composition shared by the LAPACK bidiagonal drivers.
//! The native API returns V, while LAPACK accumulates V^H, U and U^H C.
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{linalg::svd::*, Mat};

pub(crate) fn decompose<R: Scalar>(
    diagonal: &[R],
    offdiagonal: &[R],
    upper: bool,
    divide: bool,
) -> Result<(Vec<R>, Mat<R>, Mat<R>), ()> {
    assert!(!R::COMPLEX);
    let n = diagonal.len();
    assert_eq!(offdiagonal.len(), n.saturating_sub(1));
    let mut values = diagonal.to_vec();
    let mut u = Mat::<R>::zeros(n, n);
    let mut v = Mat::<R>::zeros(n, n);
    let mut params: colmat::Spec<SvdParams, R> = Default::default();
    if !divide {
        params.config.recursion_threshold = usize::MAX;
    }
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(bidiagonal_svd_scratch::<R>(n, par.shape(), params));
        bidiagonal_svd(
            &mut values,
            offdiagonal,
            upper,
            u.as_mut(),
            v.as_mut(),
            par,
            MemStack::new(&mut memory),
            params,
        )
        .map_err(|_| ())
    })?;
    Ok((values, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reconstruction<R: Scalar>(tol: f64) {
        for n in [0usize, 1, 2, 7, 40] {
            for upper in [false, true] {
                for divide in [false, true] {
                    for rank_deficient in [false, true] {
                        let d: Vec<R> = (0..n)
                            .map(|i| {
                                R::real(if rank_deficient && i % 3 == 0 {
                                    0.0
                                } else {
                                    (i as f64 + 1.0) * if i % 2 == 0 { -1.0 } else { 1.0 }
                                })
                            })
                            .collect();
                        let e: Vec<R> = (0..n.saturating_sub(1))
                            .map(|i| {
                                R::real(if rank_deficient {
                                    0.0
                                } else {
                                    0.3 + i as f64 / 17.0
                                })
                            })
                            .collect();
                        let (s, u, v) = decompose(&d, &e, upper, divide).unwrap();
                        for j in 0..n {
                            assert!(s[j].re() >= 0.0);
                            if j > 0 {
                                assert!(s[j - 1].re() >= s[j].re());
                            }
                            for i in 0..n {
                                let mut actual = R::z();
                                let mut udot = R::z();
                                let mut vdot = R::z();
                                for k in 0..n {
                                    actual = actual + u[(i, k)] * s[k] * v[(j, k)];
                                    udot = udot + u[(k, i)] * u[(k, j)];
                                    vdot = vdot + v[(k, i)] * v[(k, j)];
                                }
                                let expected = if i == j {
                                    d[i]
                                } else if (upper && j == i + 1) || (!upper && i == j + 1) {
                                    e[i.min(j)]
                                } else {
                                    R::z()
                                };
                                assert!(
                                    (actual - expected).magnitude() < tol * n.max(1) as f64,
                                    "n={n} upper={upper} divide={divide} at {i},{j}"
                                );
                                let identity = if i == j { R::o() } else { R::z() };
                                assert!((udot - identity).magnitude() < tol);
                                assert!((vdot - identity).magnitude() < tol);
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn bidiagonal_native_f32() {
        reconstruction::<f32>(1e-4);
    }
    #[test]
    fn bidiagonal_native_f64() {
        reconstruction::<f64>(1e-11);
    }
}
