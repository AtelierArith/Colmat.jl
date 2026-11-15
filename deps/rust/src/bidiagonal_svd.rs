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

// Work around tensor4all/colmat-rs#209 at the LAPACK boundary until a fixed
// native revision is selected. Keep original inputs for exact LASQ1 recovery.
pub(crate) fn dqds_values<R: Scalar + colmat::traits::RealField>(
    diagonal: &[R],
    offdiagonal: &[R],
) -> Result<Vec<R>, ()> {
    let n = diagonal.len();
    let mut values = diagonal.to_vec();
    let mut memory = MemBuffer::new(bidiagonal_dqds_scratch::<R>(n));
    bidiagonal_dqds(&mut values, offdiagonal, MemStack::new(&mut memory)).map_err(|_| ())?;
    let scale = diagonal
        .iter()
        .chain(offdiagonal)
        .fold(0.0f64, |s, x| s.max(x.magnitude()));
    // Real Scalar ABI instantiations are f32 and f64. Evaluate the range
    // criterion in f64 but enforce the source precision's normal minimum.
    let minimum = if std::mem::size_of::<R>() == 4 {
        f32::MIN_POSITIVE as f64
    } else {
        f64::MIN_POSITIVE
    };
    if !values.iter().all(|x| {
        if !x.re().is_finite() {
            return false;
        }
        if n <= 1 {
            return true;
        }
        let normalized = x.re() / scale;
        let square = normalized * normalized;
        x.re() > 0.0 && square.is_finite() && square >= minimum
    }) {
        return Err(());
    }
    Ok(values)
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
    fn scaling<R: Scalar>(scales: &[R], epsilon: f64, minimum: f64) {
        let n = 7;
        for &scale in scales {
            for upper in [false, true] {
                let d: Vec<R> = (0..n)
                    .map(|i| {
                        R::real((1.0 + (i + 1) as f64 / 10.0) * if i % 2 == 0 { -1.0 } else { 1.0 })
                            * scale
                    })
                    .collect();
                let e = vec![R::real(0.125) * scale; n - 1];
                let (values, u, v) = decompose(&d, &e, upper, false).unwrap();
                let tolerance = 3000.0 * epsilon + 20.0 * n as f64 * minimum / scale.re();
                for j in 0..n {
                    for i in 0..n {
                        let mut actual = 0.0;
                        for k in 0..n {
                            actual +=
                                u[(i, k)].re() * (values[k].re() / scale.re()) * v[(j, k)].re();
                        }
                        let expected = if i == j {
                            d[i].re() / scale.re()
                        } else if (upper && j == i + 1) || (!upper && i == j + 1) {
                            e[i.min(j)].re() / scale.re()
                        } else {
                            0.0
                        };
                        assert!((actual-expected).abs()<tolerance,"scale={} upper={upper} at {i},{j}: actual={actual}, expected={expected}, tolerance={tolerance}",scale.re());
                    }
                }
            }
        }
    }
    #[test]
    fn scale_real32() {
        scaling(
            &[1e-30f32, 1e30, 1e-40],
            f32::EPSILON as f64,
            f32::from_bits(1) as f64,
        );
    }
    #[test]
    fn scale_real64() {
        scaling(&[1e-300f64, 1e300, 1e-320], f64::EPSILON, f64::from_bits(1));
    }
    #[test]
    fn dqds_boundary_rejects_underresolved_success() {
        assert!(dqds_values(&[1e-22f32, 1.0], &[0.25]).is_err());
        assert!(dqds_values(&[1e-161f64, 1.0], &[0.25]).is_err());
        let s = dqds_values(&[3.0f64, 4.0], &[1.0]).unwrap();
        assert!((s[0] - 18.0f64.sqrt()).abs() < 1e-12);
        assert!((s[1] - 8.0f64.sqrt()).abs() < 1e-12);
    }
}
