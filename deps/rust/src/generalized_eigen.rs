//! LAPACK homogeneous generalized eigenproblem representation over native GEVD.
use crate::householder::ReflectorScalar;
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Mat};

pub(crate) struct EigenResult<T: ReflectorScalar> {
    pub a: Mat<T>,
    pub b: Mat<T>,
    pub alpha: Vec<T>,
    pub imaginary: Vec<f64>,
    pub beta: Vec<T>,
    pub left: Mat<T>,
    pub right: Mat<T>,
}
pub(crate) trait EigenScalar: ReflectorScalar {
    fn eigen(a: Mat<Self>, b: Mat<Self>, left: bool, right: bool) -> Result<EigenResult<Self>, ()>;
}
macro_rules! real_eigen {
    ($t:ty) => {
        impl EigenScalar for $t {
            fn eigen(
                mut a: Mat<Self>,
                mut b: Mat<Self>,
                left: bool,
                right: bool,
            ) -> Result<EigenResult<Self>, ()> {
                use colmat::linalg::evd::ComputeEigenvectors;
                use colmat::linalg::gevd::{gevd_real, gevd_scratch};
                let n = a.nrows();
                let mut ar = Diag::<Self>::zeros(n);
                let mut ai = Diag::<Self>::zeros(n);
                let mut beta = Diag::<Self>::zeros(n);
                let mut vl = Mat::<Self>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<Self>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let flag = |yes| {
                        if yes {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(gevd_scratch::<Self>(
                        n,
                        flag(left),
                        flag(right),
                        par.shape(),
                        Default::default(),
                    ));
                    gevd_real(
                        a.as_mut(),
                        b.as_mut(),
                        ar.as_mut(),
                        ai.as_mut(),
                        beta.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(EigenResult {
                    a,
                    b,
                    alpha: (0..n).map(|i| ar[i]).collect(),
                    imaginary: (0..n).map(|i| ai[i] as f64).collect(),
                    beta: (0..n).map(|i| beta[i]).collect(),
                    left: vl,
                    right: vr,
                })
            }
        }
    };
}
macro_rules! complex_eigen {
    ($t:ty) => {
        impl EigenScalar for $t {
            fn eigen(
                mut a: Mat<Self>,
                mut b: Mat<Self>,
                left: bool,
                right: bool,
            ) -> Result<EigenResult<Self>, ()> {
                use colmat::linalg::evd::ComputeEigenvectors;
                use colmat::linalg::gevd::{gevd_cplx, gevd_scratch};
                let n = a.nrows();
                let mut alpha = Diag::<Self>::zeros(n);
                let mut beta = Diag::<Self>::zeros(n);
                let mut vl = Mat::<Self>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<Self>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let flag = |yes| {
                        if yes {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(gevd_scratch::<Self>(
                        n,
                        flag(left),
                        flag(right),
                        par.shape(),
                        Default::default(),
                    ));
                    gevd_cplx(
                        a.as_mut(),
                        b.as_mut(),
                        alpha.as_mut(),
                        beta.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(EigenResult {
                    a,
                    b,
                    alpha: (0..n).map(|i| alpha[i]).collect(),
                    imaginary: vec![0.0; n],
                    beta: (0..n).map(|i| beta[i]).collect(),
                    left: vl,
                    right: vr,
                })
            }
        }
    };
}
real_eigen!(f32);
real_eigen!(f64);
complex_eigen!(c32);
complex_eigen!(c64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Scalar;
    fn check<T: EigenScalar>() {
        let n = 3;
        let a = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real((i + 2) as f64)
            } else if i < j {
                T::real((i + j + 1) as f64)
            } else {
                T::z()
            }
        });
        let b = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real((i + 1) as f64)
            } else {
                T::z()
            }
        });
        let result = T::eigen(a.clone(), b.clone(), true, true).unwrap();
        for j in 0..n {
            assert_eq!(result.imaginary[j], 0.0);
            let alpha = result.alpha[j];
            let beta = result.beta[j];
            for i in 0..n {
                let mut right = T::z();
                let mut left = T::z();
                for k in 0..n {
                    right = right + (beta * a[(i, k)] - alpha * b[(i, k)]) * result.right[(k, j)];
                    left = left + result.left[(k, j)].cj() * (beta * a[(k, i)] - alpha * b[(k, i)]);
                }
                assert!(right.magnitude() < 1e-10);
                assert!(left.magnitude() < 1e-10);
            }
        }
    }
    #[test]
    fn homogeneous_left_and_right_residuals() {
        check::<f64>();
        check::<c64>();
    }
}
