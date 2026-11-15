//! Band/packed self-adjoint matrices adapted to native full-spectrum EVD.
//! This is a dense expansion, not a dedicated band/packed eigensolver.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::kernels::Scalar;
use crate::storage_layout::TriangleStorage;
use colmat::Mat;

#[derive(Clone, Copy)]
pub(crate) enum Storage {
    Band { width: usize, ld: usize },
    Packed,
}
impl Storage {
    fn rows(self, n: usize, lower: bool, j: usize) -> (usize, usize) {
        match self {
            Self::Band { width, .. } => {
                if lower {
                    (j, n.min(j + width + 1))
                } else {
                    (j.saturating_sub(width), j + 1)
                }
            }
            Self::Packed => {
                if lower {
                    (j, n)
                } else {
                    (0, j + 1)
                }
            }
        }
    }
    fn index(self, n: usize, lower: bool, i: usize, j: usize) -> usize {
        match self {
            Self::Band { width, ld } => j * ld + if lower { i - j } else { width + i - j },
            Self::Packed => {
                TriangleStorage::Packed.segment(n, lower, j).0 + if lower { i - j } else { i }
            }
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Selection {
    All,
    Value(f64, f64),
    Index(usize, usize),
}
pub(crate) struct Spectrum<T: Scalar> {
    pub(crate) values: Vec<f64>,
    pub(crate) vectors: Mat<T>,
}
impl<T: Scalar> Spectrum<T> {
    pub(crate) fn selected(&self, range: Selection) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let selected = match range {
                    Selection::All => true,
                    Selection::Value(lo, hi) => lo < value && value <= hi,
                    Selection::Index(lo, hi) => lo <= i && i < hi,
                };
                selected.then_some(i)
            })
            .collect()
    }
    /// A diagonal matrix is a valid tridiagonal reduction. With eigenvectors
    /// as the full reduction basis Q, A=Q*diag(values)*Q^H holds, including
    /// selected drivers whose final Z contains only a subset of Q.
    pub(crate) unsafe fn write_diagonal_reduction(&self, lower: bool, storage: Storage, a: *mut T) {
        let n = self.values.len();
        for j in 0..n {
            let (first, end) = storage.rows(n, lower, j);
            for i in first..end {
                *a.add(storage.index(n, lower, i, j)) = if i == j {
                    T::real(self.values[j])
                } else {
                    T::z()
                };
            }
        }
    }
}
pub(crate) unsafe fn compute<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    storage: Storage,
    a: *const T,
    vectors: bool,
) -> Result<Spectrum<T>, ()> {
    if n == 0 {
        return Ok(Spectrum {
            values: vec![],
            vectors: Mat::zeros(0, 0),
        });
    }
    let mut matrix = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (first, end) = storage.rows(n, lower, j);
        for i in first..end {
            let p = a.add(storage.index(n, lower, i, j));
            let value = if i == j { T::diagonal(p) } else { *p };
            matrix[(i, j)] = value;
            if i != j {
                matrix[(j, i)] = value.cj();
            }
        }
    }
    if !matrix.as_ref().is_all_finite() {
        return Err(());
    }
    let (values, vectors) = crate::lapack::selfadj(matrix.as_ref(), vectors)?;
    if !values.iter().all(|x| x.is_finite()) || !vectors.as_ref().is_all_finite() {
        return Err(());
    }
    Ok(Spectrum { values, vectors })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn residual<T: Scalar>(a: &[T], s: &Spectrum<T>, tol: f64) {
        let n = s.values.len();
        for j in 0..n {
            for i in 0..n {
                let mut value = T::z();
                let mut dot = T::z();
                for k in 0..n {
                    value = value + a[i + k * n] * s.vectors[(k, j)];
                    dot = dot + s.vectors[(k, i)].cj() * s.vectors[(k, j)];
                }
                assert!((value - s.vectors[(i, j)] * T::real(s.values[j])).magnitude() < tol);
                assert!((dot - if i == j { T::o() } else { T::z() }).magnitude() < tol);
            }
        }
    }
    #[test]
    fn band_triangles_ignore_padding_and_produce_a_reduction_basis() {
        unsafe {
            let original = [2.0f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
            for lower in [false, true] {
                let mut ab = if lower {
                    [2.0, 1.0, 777.0, 2.0, 1.0, 777.0, 2.0, f64::NAN, 777.0]
                } else {
                    [f64::NAN, 2.0, 777.0, 1.0, 2.0, 777.0, 1.0, 2.0, 777.0]
                };
                let before = ab;
                let storage = Storage::Band { width: 1, ld: 3 };
                let s = compute(3, lower, storage, ab.as_ptr(), true).unwrap();
                residual(&original, &s, 1e-12);
                assert!((s.values[0] - (2.0 - 2.0f64.sqrt())).abs() < 1e-12);
                assert_eq!(s.selected(Selection::All), [0, 1, 2]);
                assert_eq!(s.selected(Selection::Index(1, 2)), [1]);
                assert_eq!(s.selected(Selection::Value(s.values[1], s.values[2])), [2]);
                for i in 0..9 {
                    assert!(ab[i] == before[i] || (ab[i].is_nan() && before[i].is_nan()));
                }
                s.write_diagonal_reduction(lower, storage, ab.as_mut_ptr());
                for j in 0..3 {
                    assert_eq!(ab[2 + 3 * j], 777.0);
                    assert_eq!(ab[(if lower { 0 } else { 1 }) + 3 * j], s.values[j]);
                }
            }
        }
    }
    #[test]
    fn packed_hermitian_ignores_imaginary_diagonals() {
        use colmat::c64;
        unsafe {
            let a = [
                c64::new(2.0, f64::NAN),
                c64::new(0.0, 1.0),
                c64::new(2.0, f64::NAN),
            ];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
            residual(
                &[
                    c64::new(2.0, 0.0),
                    c64::new(0.0, -1.0),
                    c64::new(0.0, 1.0),
                    c64::new(2.0, 0.0),
                ],
                &s,
                1e-12,
            );
            assert!((s.values[0] - 1.0).abs() < 1e-12);
            assert!((s.values[1] - 3.0).abs() < 1e-12);
        }
    }
    #[test]
    fn single_precision_and_values_only_paths() {
        unsafe {
            let a = [2.0f32, 0.0, 3.0];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), false).unwrap();
            assert_eq!(s.values, [2.0, 3.0]);
            assert_eq!(s.vectors.ncols(), 0);
            let a = [
                colmat::c32::new(2.0, f32::NAN),
                colmat::c32::new(0.0, 1.0),
                colmat::c32::new(2.0, f32::NAN),
            ];
            let s = compute(2, false, Storage::Packed, a.as_ptr(), true).unwrap();
            assert!((s.values[0] - 1.0).abs() < 1e-5);
            assert!((s.values[1] - 3.0).abs() < 1e-5);
        }
    }
    #[test]
    fn empty_and_nonfinite_problems_do_not_mutate_input() {
        unsafe {
            let s = compute::<f64>(0, false, Storage::Packed, std::ptr::null(), true).unwrap();
            assert!(s.values.is_empty());
            let a = [f64::NAN, 0.0, 2.0];
            assert!(compute(2, false, Storage::Packed, a.as_ptr(), true).is_err());
            assert!(a[0].is_nan());
        }
    }
}
