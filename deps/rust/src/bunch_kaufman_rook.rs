//! ROOK uses two independently encoded swaps for each two-column pivot.
use super::*;

fn pack_lower_rook<T: ReflectorScalar>(factors: &Factors<T>) -> (Mat<T>, Vec<isize>) {
    let n = factors.matrix.nrows();
    let mut packed = Mat::<T>::zeros(n, n);
    let mut pivots = vec![0; n];
    let mut order: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = j + 1 < n && factors.subdiagonal[j] != T::z();
        let size = if pair { 2 } else { 1 };
        for k in j..j + size {
            let pivot = (k..n)
                .find(|&i| order[i] == factors.permutation[k])
                .unwrap();
            order.swap(k, pivot);
            pivots[k] = (pivot + 1) as isize * if pair { -1 } else { 1 };
        }
        for k in j..j + size {
            packed[(k, k)] = factors.matrix[(k, k)];
            for i in j + size..n {
                packed[(i, k)] = factors.matrix[(factors.inverse[order[i]], k)];
            }
        }
        if pair {
            packed[(j + 1, j)] = factors.subdiagonal[j];
        }
        j += size;
    }
    (packed, pivots)
}

fn unpack_lower_rook<T: ReflectorScalar>(packed: &Mat<T>, pivots: &[isize]) -> Factors<T> {
    let n = packed.nrows();
    let mut matrix = Mat::<T>::zeros(n, n);
    let mut subdiagonal = Diag::<T>::zeros(n);
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = pivots[j] < 0;
        let size = if pair { 2 } else { 1 };
        for k in j..j + size {
            let pivot = pivots[k].unsigned_abs() - 1;
            permutation.swap(k, pivot);
            for column in 0..j {
                let value = matrix[(k, column)];
                matrix[(k, column)] = matrix[(pivot, column)];
                matrix[(pivot, column)] = value;
            }
        }
        for k in j..j + size {
            matrix[(k, k)] = packed[(k, k)];
            for i in j + size..n {
                matrix[(i, k)] = packed[(i, k)];
            }
        }
        if pair {
            subdiagonal[j] = packed[(j + 1, j)];
        }
        j += size;
    }
    let mut inverse = vec![0; n];
    for i in 0..n {
        inverse[permutation[i]] = i;
    }
    Factors {
        matrix,
        subdiagonal,
        permutation,
        inverse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use colmat::linalg::cholesky::lblt::factor::PivotingStrategy;

    #[test]
    fn rook_two_swap_indices_and_later_column_permutations() {
        let matrix = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                (i + 1) as f64
            } else if i > j && !(i == 1 && j == 0) {
                (10 * i + j) as f64
            } else {
                0.0
            }
        });
        let mut subdiagonal = Diag::<f64>::zeros(4);
        subdiagonal[0] = 2.0;
        let factors = Factors {
            matrix,
            subdiagonal,
            permutation: vec![2, 3, 1, 0],
            inverse: vec![3, 2, 0, 1],
        };
        let (packed, pivots) = pack_lower_rook(&factors);
        assert_eq!(pivots, [-3, -4, 4, 4]);
        assert_eq!(packed[(2, 0)], factors.matrix[(3, 0)]);
        assert_eq!(packed[(3, 0)], factors.matrix[(2, 0)]);
        let restored = unpack_lower_rook(&packed, &pivots);
        assert_eq!(restored.permutation, factors.permutation);
        assert_eq!(restored.inverse, factors.inverse);
        for j in 0..4 {
            for i in j..4 {
                assert_eq!(restored.matrix[(i, j)], factors.matrix[(i, j)]);
            }
        }
        assert_eq!(restored.subdiagonal[0], 2.0);
    }

    fn native_roundtrip<T: ReflectorScalar>() {
        let n = 6;
        let original = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real(if i < 2 { 0.0 } else { (i + 1) as f64 })
            } else if i > j {
                T::parts(
                    if i == 1 && j == 0 {
                        3.0
                    } else {
                        0.1 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        0.03 * (i - j) as f64
                    } else {
                        0.0
                    },
                )
            } else {
                T::parts(
                    if j == 1 && i == 0 {
                        3.0
                    } else {
                        0.1 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        -0.03 * (j - i) as f64
                    } else {
                        0.0
                    },
                )
            }
        });
        for strategy in [PivotingStrategy::Rook, PivotingStrategy::RookDiag] {
            for unblocked in [false, true] {
                let factors = native_factor_with_pivoting(original.clone(), unblocked, strategy);
                let (packed, pivots) = pack_lower_rook(&factors);
                assert!(pivots.iter().any(|&p| p < 0));
                let restored = unpack_lower_rook(&packed, &pivots);
                assert_eq!(restored.permutation, factors.permutation);
                let mut rhs = original.clone();
                native_solve(&restored, &mut rhs);
                for j in 0..n {
                    for i in 0..n {
                        assert!(
                            (rhs[(i, j)] - if i == j { T::o() } else { T::z() }).magnitude()
                                < 200.0 * T::EPS
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn rook_native_four_scalar_types() {
        native_roundtrip::<f32>();
        native_roundtrip::<f64>();
        native_roundtrip::<colmat::c32>();
        native_roundtrip::<colmat::c64>();
    }
}
