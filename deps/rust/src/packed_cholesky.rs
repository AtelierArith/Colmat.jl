//! Packed/RFP representations around native LLT and Cholesky solves.
//! All dense work is owned by the adapter; native LLT's opposite-triangle
//! scratch never aliases a caller's packed storage.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::storage_layout::TriangleStorage;
use colmat::{Mat, MatMut};

unsafe fn read<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *const T,
) -> Mat<T> {
    let mut work = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            let p = a.offset(offset as isize + k as isize * stride);
            work[(i, j)] = if i == j {
                T::diagonal(p)
            } else if conjugated {
                (*p).cj()
            } else {
                *p
            };
        }
    }
    work
}
unsafe fn write<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
    work: &Mat<T>,
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            let value = if i == j {
                T::real(work[(i, j)].re())
            } else if conjugated {
                work[(i, j)].cj()
            } else {
                work[(i, j)]
            };
            *a.offset(offset as isize + k as isize * stride) = value;
        }
    }
}
pub(crate) unsafe fn factor<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut work = read(n, lower, layout, a);
    let info = if lower {
        crate::lapack::cholesky(work.as_mut())
    } else {
        crate::lapack::cholesky(work.as_mut().transpose_mut())
    };
    write(n, lower, layout, a, &work);
    info
}
pub(crate) unsafe fn solve<T: GeneralizedScalar>(
    n: usize,
    nrhs: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *const T,
    b: *mut T,
    ldb: usize,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let work = read(n, lower, layout, a);
    crate::lapack::solve_cholesky(
        work.as_ref(),
        MatMut::from_raw_parts_mut(b, n, nrhs, 1, ldb as isize),
        if lower { b'L' } else { b'U' },
    );
}
pub(crate) unsafe fn inverse<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    // Check all diagonal entries before overwriting any factor, as TPTRI does.
    for j in 0..n {
        let (offset, stride, _) = layout.segment(n, lower, j);
        let p = a.offset(offset as isize + if lower { 0 } else { j as isize * stride });
        if T::diagonal(p) == T::z() {
            return j + 1;
        }
    }
    let work = read(n, lower, layout, a);
    let mut result = Mat::<T>::identity(n, n);
    crate::lapack::solve_cholesky(
        work.as_ref(),
        result.as_mut(),
        if lower { b'L' } else { b'U' },
    );
    write(n, lower, layout, a, &result);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Scalar;
    #[test]
    fn upper_packed_factor_solve_and_inverse_are_native() {
        unsafe {
            let mut ap = [4.0f64, 1.0, 3.0];
            assert_eq!(
                factor(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            assert_eq!(ap[0], 2.0);
            assert_eq!(ap[1], 0.5);
            assert!((ap[2] - (2.75f64).sqrt()).abs() < 1e-15);
            let mut b = [6.0, 7.0];
            solve(
                2,
                1,
                false,
                TriangleStorage::Packed,
                ap.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert!((b[0] - 1.0).abs() < 1e-14);
            assert!((b[1] - 2.0).abs() < 1e-14);
            assert_eq!(
                inverse(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            for (actual, expected) in ap.iter().zip([3.0 / 11.0, -1.0 / 11.0, 4.0 / 11.0]) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }
    #[test]
    fn complex_packed_ignores_imaginary_diagonals_and_conjugates_solves() {
        use colmat::c64;
        unsafe {
            let mut ap = [
                c64::new(4.0, f64::NAN),
                c64::new(1.0, 2.0),
                c64::new(3.0, f64::NAN),
            ];
            assert_eq!(
                factor(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                0
            );
            assert_eq!(ap[0], c64::new(2.0, 0.0));
            assert_eq!(ap[2].im, 0.0);
            let x = [c64::new(1.0, 1.0), c64::new(2.0, -1.0)];
            let mut b = [
                c64::new(4.0, 0.0) * x[0] + c64::new(1.0, 2.0) * x[1],
                c64::new(1.0, -2.0) * x[0] + c64::new(3.0, 0.0) * x[1],
            ];
            solve(
                2,
                1,
                false,
                TriangleStorage::Packed,
                ap.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            for i in 0..2 {
                assert!((b[i] - x[i]).magnitude() < 1e-13);
            }
        }
    }
    #[test]
    fn zero_diagonal_inverse_preserves_the_entire_factor() {
        unsafe {
            let mut ap = [2.0f64, 0.5, 0.0];
            let before = ap;
            assert_eq!(
                inverse(2, false, TriangleStorage::Packed, ap.as_mut_ptr()),
                2
            );
            assert_eq!(ap, before);
            assert_eq!(
                factor::<f64>(0, true, TriangleStorage::Packed, std::ptr::null_mut()),
                0
            );
            solve::<f64>(
                9,
                0,
                true,
                TriangleStorage::Packed,
                std::ptr::null(),
                std::ptr::null_mut(),
                9,
            );
        }
    }
}

#[cfg(test)]
mod rfp_tests {
    use super::*;
    #[test]
    fn two_by_two_rfp_layouts_factor_solve_and_invert() {
        unsafe {
            for lower in [false, true] {
                for transposed in [false, true] {
                    let layout = TriangleStorage::Rfp { transposed };
                    let mut a = if lower {
                        [3.0f64, 4.0, 1.0]
                    } else {
                        [1.0, 3.0, 4.0]
                    };
                    assert_eq!(factor(2, lower, layout, a.as_mut_ptr()), 0);
                    let mut b = [6.0, 7.0];
                    solve(2, 1, lower, layout, a.as_ptr(), b.as_mut_ptr(), 2);
                    assert!((b[0] - 1.0).abs() < 1e-14);
                    assert!((b[1] - 2.0).abs() < 1e-14);
                    assert_eq!(inverse(2, lower, layout, a.as_mut_ptr()), 0);
                    let expected = if lower {
                        [4.0 / 11.0, 3.0 / 11.0, -1.0 / 11.0]
                    } else {
                        [-1.0 / 11.0, 4.0 / 11.0, 3.0 / 11.0]
                    };
                    for i in 0..3 {
                        assert!((a[i] - expected[i]).abs() < 1e-14);
                    }
                }
            }
        }
    }
}
