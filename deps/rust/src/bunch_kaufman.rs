//! LAPACK representation adapters for native real-symmetric/Hermitian LDL.
use crate::householder::ReflectorScalar;
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;

struct Factors<T: ReflectorScalar> {
    matrix: Mat<T>,
    subdiagonal: Diag<T>,
    permutation: Vec<usize>,
    inverse: Vec<usize>,
}
fn native_factor<T: ReflectorScalar>(mut matrix: Mat<T>, unblocked: bool) -> Factors<T> {
    use colmat::linalg::cholesky::lblt::factor::{
        cholesky_in_place, cholesky_in_place_scratch, LbltParams, PivotingStrategy,
    };
    let n = matrix.nrows();
    let mut subdiagonal = Diag::<T>::zeros(n);
    let mut permutation = vec![0_usize; n];
    let mut inverse = vec![0_usize; n];
    let mut params: colmat::Spec<LbltParams, T> = Default::default();
    params.config.pivoting = PivotingStrategy::Partial;
    if unblocked {
        params.config.block_size = 1;
    }
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, T>(
            n,
            par.shape(),
            params,
        ));
        cholesky_in_place(
            matrix.as_mut(),
            subdiagonal.as_mut(),
            &mut permutation,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            params,
        );
    });
    Factors {
        matrix,
        subdiagonal,
        permutation,
        inverse,
    }
}
fn native_solve<T: ReflectorScalar>(factors: &Factors<T>, rhs: &mut Mat<T>) {
    use colmat::linalg::cholesky::lblt::solve::{solve_in_place, solve_in_place_scratch};
    let n = factors.matrix.nrows();
    let columns = rhs.ncols();
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(solve_in_place_scratch::<usize, T>(n, columns, par.shape()));
        let permutation = unsafe {
            colmat::perm::PermRef::new_unchecked(&factors.permutation, &factors.inverse, n)
        };
        solve_in_place(
            factors.matrix.as_ref(),
            factors.matrix.as_ref().diagonal(),
            factors.subdiagonal.as_ref(),
            permutation,
            rhs.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schur::multiply;
    use colmat::c64;
    fn check<T: ReflectorScalar>() {
        let n = 5;
        let diagonal = [0.0, 0.0, 3.0, 0.0, 2.0];
        let original = Mat::<T>::from_fn(n, n, |i, j| {
            if i == j {
                T::real(diagonal[i])
            } else if i > j {
                T::parts(
                    if i == 1 && j == 0 {
                        2.0
                    } else {
                        0.03 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        0.01 * (i - j) as f64
                    } else {
                        0.0
                    },
                )
            } else {
                T::parts(
                    if j == 1 && i == 0 {
                        2.0
                    } else {
                        0.03 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        -0.01 * (j - i) as f64
                    } else {
                        0.0
                    },
                )
            }
        });
        for unblocked in [false, true] {
            let factors = native_factor(original.clone(), unblocked);
            let l = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    T::o()
                } else if i > j {
                    factors.matrix[(i, j)]
                } else {
                    T::z()
                }
            });
            let d = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    factors.matrix[(i, i)]
                } else if i == j + 1 {
                    factors.subdiagonal[j]
                } else if j == i + 1 {
                    factors.subdiagonal[i].cj()
                } else {
                    T::z()
                }
            });
            let reconstructed = multiply(&multiply(&l, b'N', &d, b'N'), b'N', &l, b'C');
            for j in 0..n {
                for i in 0..n {
                    assert!(
                        (reconstructed[(i, j)]
                            - original[(factors.permutation[i], factors.permutation[j])])
                            .magnitude()
                            < 1e-10
                    );
                }
            }
            let rhs = Mat::from_fn(n, 2, |i, j| T::real((i + j + 1) as f64));
            let mut solution = rhs.clone();
            native_solve(&factors, &mut solution);
            let reconstructed = multiply(&original, b'N', &solution, b'N');
            for j in 0..2 {
                for i in 0..n {
                    assert!((reconstructed[(i, j)] - rhs[(i, j)]).magnitude() < 1e-10);
                }
            }
        }
    }
    #[test]
    fn real_and_complex_factor_and_solve() {
        check::<f64>();
        check::<c64>();
    }
}
