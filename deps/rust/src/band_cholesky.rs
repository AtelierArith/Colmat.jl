//! Band and split Cholesky from native diagonal LLT, triangular solves,
//! and selected-triangle rank-one products over the stored band.
use crate::generalized_selfadjoint::GeneralizedScalar;
use colmat::{Accum, Conj, Mat, MatMut, MatRef};

unsafe fn step<T: GeneralizedScalar>(
    lower: bool,
    width: usize,
    j: usize,
    first: usize,
    count: usize,
    backwards: bool,
    split: bool,
    ab: *mut T,
    ld: usize,
) -> bool {
    let row = if lower { 0 } else { width };
    let diagonal = ab.add(row + j * ld);
    // Hermitian diagonal imaginary components are not input data.
    *diagonal = T::diagonal(diagonal);
    let nan = (*diagonal).re().is_nan();
    let status = crate::lapack::cholesky(MatMut::from_raw_parts_mut(diagonal, 1, 1, 1, 1));
    // Reference PBSTF does not classify a NaN as a non-positive pivot.
    if status != 0 && !(split && nan) {
        return false;
    }
    if count == 0 {
        return true;
    }
    let (vector, increment) = match (lower, backwards) {
        (true, false) => (ab.add(1 + j * ld), 1),
        (false, false) => (ab.add(width - 1 + (j + 1) * ld), ld - 1),
        (false, true) => (ab.add(width - count + j * ld), 1),
        (true, true) => (ab.add(count + first * ld), ld - 1),
    };
    crate::threading::with_par(|par| {
        colmat::linalg::triangular_solve::solve_lower_triangular_in_place(
            MatRef::from_raw_parts(diagonal, 1, 1, 1, 1),
            MatMut::from_raw_parts_mut(vector, 1, count, 0, increment as isize),
            par,
        );
    });
    // The unused half of a skew-strided target view can overlap the stored
    // vector. Snapshot it before constructing the mutable target, keeping
    // the native product's input/output views disjoint even there.
    let vector = Mat::from_fn(count, 1, |i, _| *vector.add(i * increment));
    let target =
        MatMut::from_raw_parts_mut(ab.add(row + first * ld), count, count, 1, (ld - 1) as isize);
    let conjugate_left = lower == backwards;
    use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
    crate::threading::with_par(|par| {
        matmul_with_conj(
            target,
            if lower {
                TriangularLower
            } else {
                TriangularUpper
            },
            Accum::Add,
            vector.as_ref(),
            Rectangular,
            if conjugate_left { Conj::Yes } else { Conj::No },
            vector.as_ref().transpose(),
            Rectangular,
            if conjugate_left { Conj::No } else { Conj::Yes },
            -T::o(),
            par,
        );
    });
    true
}
pub(crate) unsafe fn factor<T: GeneralizedScalar>(
    lower: bool,
    n: usize,
    width: usize,
    ab: *mut T,
    ld: usize,
    split: bool,
) -> usize {
    if n == 0 {
        return 0;
    }
    // With a redundant declared width the mathematical split cannot exceed
    // n. In particular, never index a nonexistent column when width>n.
    let middle = if split {
        (n / 2 + width / 2 + (n % 2 + width % 2) / 2).min(n)
    } else {
        n
    };
    if split {
        for j in (middle..n).rev() {
            let count = width.min(j);
            if !step(lower, width, j, j - count, count, true, true, ab, ld) {
                return j + 1;
            }
        }
    }
    for j in 0..middle {
        let count = width.min(middle - j - 1);
        if !step(lower, width, j, j + 1, count, false, split, ab, ld) {
            return j + 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::householder::ReflectorScalar;

    fn reconstruction<T: GeneralizedScalar + ReflectorScalar>(tolerance: f64) {
        for n in 0usize..9 {
            for width in 0usize..10 {
                for split in [false, true] {
                    let middle = if split { ((n + width) / 2).min(n) } else { n };
                    let mut source = vec![T::z(); n * n];
                    for i in 0..n {
                        let (first, end) = if split && i >= middle {
                            (i.saturating_sub(width), i + 1)
                        } else {
                            (i, middle.min(i + width + 1))
                        };
                        for j in first..end {
                            source[i + j * n] = if i == j {
                                T::real(2.0 + i as f64 / 8.0)
                            } else {
                                T::parts(
                                    (i + j + 1) as f64 / 16.0,
                                    if T::COMPLEX {
                                        (i as f64 - j as f64) / 32.0
                                    } else {
                                        0.0
                                    },
                                )
                            };
                        }
                    }
                    let mut original = vec![T::z(); n * n];
                    for j in 0..n {
                        for i in 0..n {
                            for k in 0..n {
                                original[i + j * n] = original[i + j * n]
                                    + source[k + i * n].cj() * source[k + j * n];
                            }
                        }
                    }
                    for lower in [false, true] {
                        let ld = width + 3;
                        let mut ab = vec![T::parts(f64::NAN, f64::NAN); ld * n];
                        let mut selected = vec![false; ld * n];
                        for j in 0..n {
                            ab[ld - 2 + j * ld] = T::real(777.0);
                            ab[ld - 1 + j * ld] = T::real(777.0);
                            let (first, end) = if lower {
                                (j, n.min(j + width + 1))
                            } else {
                                (j.saturating_sub(width), j + 1)
                            };
                            for i in first..end {
                                let row = if lower { i - j } else { width + i - j };
                                ab[row + j * ld] = if T::COMPLEX && i == j {
                                    T::parts(original[i + j * n].re(), f64::NAN)
                                } else {
                                    original[i + j * n]
                                };
                                selected[row + j * ld] = true;
                            }
                        }
                        assert_eq!(
                            unsafe { factor(lower, n, width, ab.as_mut_ptr(), ld, split) },
                            0
                        );
                        let mut actual = vec![T::z(); n * n];
                        for j in 0..n {
                            for row in 0..width + 1 {
                                if !selected[row + j * ld] {
                                    assert!(ab[row + j * ld].re().is_nan());
                                    continue;
                                }
                                let i = if lower { j + row } else { j + row - width };
                                let value = ab[row + j * ld];
                                if split {
                                    if lower {
                                        if i >= middle {
                                            actual[i + j * n] = value;
                                        } else {
                                            actual[j + i * n] = value.cj();
                                        }
                                    } else if j >= middle {
                                        actual[j + i * n] = value.cj();
                                    } else {
                                        actual[i + j * n] = value;
                                    }
                                } else {
                                    actual[i + j * n] = value;
                                }
                            }
                            assert_eq!(ab[ld - 2 + j * ld], T::real(777.0));
                            assert_eq!(ab[ld - 1 + j * ld], T::real(777.0));
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let mut value = T::z();
                                for k in 0..n {
                                    value = value
                                        + if split || !lower {
                                            actual[k + i * n].cj() * actual[k + j * n]
                                        } else {
                                            actual[i + k * n] * actual[j + k * n].cj()
                                        };
                                }
                                assert!(
                                    (value - original[i + j * n]).magnitude() < tolerance,
                                    "n={n} width={width} lower={lower} split={split}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32_band_and_split_reconstruct() {
        reconstruction::<f32>(5e-4);
    }
    #[test]
    fn real64_band_and_split_reconstruct() {
        reconstruction::<f64>(1e-10);
    }
    #[test]
    fn complex32_band_and_split_reconstruct() {
        reconstruction::<colmat::c32>(5e-4);
    }
    #[test]
    fn complex64_band_and_split_reconstruct() {
        reconstruction::<colmat::c64>(1e-10);
    }
    #[test]
    fn diagonal_only_and_empty_storage() {
        unsafe {
            assert_eq!(factor::<f64>(true, 0, 7, std::ptr::null_mut(), 8, true), 0);
            let mut ab = [4.0, 9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, false), 0);
            assert_eq!(ab, [2.0, 3.0, 4.0]);
            let mut ab = [4.0, 9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, true), 0);
            assert_eq!(ab, [2.0, 3.0, 4.0]);
        }
    }
    #[test]
    fn split_failure_uses_reverse_pivot_order() {
        unsafe {
            let mut ab = [4.0, -9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, true), 2);
            assert_eq!(ab, [4.0, -9.0, 4.0]);
            let mut ab = [4.0, -9.0, 16.0];
            assert_eq!(factor(true, 3, 0, ab.as_mut_ptr(), 1, false), 2);
            assert_eq!(ab, [2.0, -9.0, 16.0]);
        }
    }
    #[test]
    fn hermitian_diagonal_ignores_nan_imaginary_component() {
        unsafe {
            let mut ab = [colmat::c64::new(4.0, f64::NAN)];
            assert_eq!(factor(true, 1, 0, ab.as_mut_ptr(), 1, false), 0);
            assert_eq!(ab, [colmat::c64::new(2.0, 0.0)]);
        }
    }
}
