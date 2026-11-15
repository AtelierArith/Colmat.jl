//! Owned triangle conversion for packed/RFP LAPACK triangular operations.
use crate::kernels::Scalar;
use crate::storage_layout::TriangleStorage;

unsafe fn read<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *const T,
) -> Vec<T> {
    let mut dense = vec![T::z(); n * n];
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            dense[i + j * n] = if unit && i == j {
                T::o()
            } else {
                let value = *a.offset(offset as isize + k as isize * stride);
                if conjugated {
                    value.cj()
                } else {
                    value
                }
            };
        }
    }
    dense
}
unsafe fn write<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *mut T,
    dense: &[T],
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            if unit && i == j {
                continue;
            }
            let value = dense[i + j * n];
            *a.offset(offset as isize + k as isize * stride) =
                if conjugated { value.cj() } else { value };
        }
    }
}
unsafe fn inverse<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut dense = read(n, lower, unit, layout, a);
    if !unit {
        for j in 0..n {
            if dense[j + j * n] == T::z() {
                return j + 1;
            }
        }
    }
    crate::triangular_lapack::inverse(
        if lower { b'L' } else { b'U' },
        unit,
        n,
        dense.as_mut_ptr(),
        n as isize,
    );
    write(n, lower, unit, layout, a, &dense);
    0
}
#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: Scalar>(imag: T, tolerance: f64) {
        unsafe {
            for n in [0usize, 1, 2, 5, 6] {
                for lower in [false, true] {
                    for unit in [false, true] {
                        for layout in [
                            TriangleStorage::Packed,
                            TriangleStorage::Rfp { transposed: false },
                            TriangleStorage::Rfp { transposed: true },
                        ] {
                            let len = n * (n + 1) / 2;
                            let mut data = vec![T::real(777.0); len + 2];
                            let a = data.as_mut_ptr().add(1);
                            let mut original = vec![T::z(); n * n];
                            for j in 0..n {
                                let (offset, stride, conjugated) = layout.segment(n, lower, j);
                                let first = if lower { j } else { 0 };
                                let length = if lower { n - j } else { j + 1 };
                                for k in 0..length {
                                    let i = first + k;
                                    let value = if i == j {
                                        if unit {
                                            T::o()
                                        } else {
                                            T::real((i + 2) as f64) + imag * T::real(0.2)
                                        }
                                    } else {
                                        T::real((i + j + 1) as f64 / 19.0)
                                            + imag * T::real((i + 1) as f64 / 29.0)
                                    };
                                    original[i + j * n] = value;
                                    *a.offset(offset as isize + k as isize * stride) =
                                        if unit && i == j {
                                            T::real(f64::NAN)
                                        } else if conjugated {
                                            value.cj()
                                        } else {
                                            value
                                        };
                                }
                            }
                            assert_eq!(inverse(n, lower, unit, layout, a), 0);
                            let result = read(n, lower, unit, layout, a);
                            for j in 0..n {
                                for i in 0..n {
                                    let product = (0..n).fold(T::z(), |s, k| {
                                        s + original[i + k * n] * result[k + j * n]
                                    });
                                    assert!(
                                        (product - if i == j { T::o() } else { T::z() })
                                            .magnitude()
                                            < tolerance
                                    );
                                }
                            }
                            assert_eq!(data[0], T::real(777.0));
                            assert_eq!(data[len + 1], T::real(777.0));
                            if unit {
                                for j in 0..n {
                                    let (offset, stride, _) = layout.segment(n, lower, j);
                                    let k = if lower { 0 } else { j };
                                    assert!((*a.offset(offset as isize + k as isize * stride))
                                        .re()
                                        .is_nan());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        verify(0.0f32, 1e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 1e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 1e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 1e-13);
    }
}
