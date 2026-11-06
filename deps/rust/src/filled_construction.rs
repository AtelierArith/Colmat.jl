//! Direct constant constructors; only one scalar crosses the checked boundary.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::diag::Diag;
use faer::traits::ComplexField;
use faer::{c32, c64, Col, Mat, Row};
use rustcall_julia_macros::julia;

fn full_collection<T: ComplexField + Copy>(
    n: usize,
    kind: i32,
    value: T,
) -> Result<VectorStore<T>, String> {
    Ok(match kind {
        0 => VectorStore::Column(Col::full(n, value)),
        1 => VectorStore::Row(Row::full(n, value)),
        2 => VectorStore::Diagonal(Diag::full(n, value)),
        _ => return Err("unknown collection kind".into()),
    })
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn filled_matrix_f64(
        &mut self,
        rows: usize,
        cols: usize,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            let (1, [value]) = (tag, values) else {
                return Err("scalar tag or value buffer length mismatch".into());
            };
            self.matrix = Mat::full(rows, cols, *value);
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn filled_matrix_f32(
        &mut self,
        rows: usize,
        cols: usize,
        tag: i32,
        values: &[f32],
    ) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            let data = match (tag, values) {
                (0, [value]) => MatrixData::F32(Mat::full(rows, cols, *value)),
                (2, [re, im]) => MatrixData::C32(Mat::full(rows, cols, c32::new(*re, *im))),
                _ => return Err("scalar tag or value buffer length mismatch".into()),
            };
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn filled_matrix_f64(
        &mut self,
        rows: usize,
        cols: usize,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            let data = match (tag, values) {
                (1, [value]) => MatrixData::F64(Mat::full(rows, cols, *value)),
                (3, [re, im]) => MatrixData::C64(Mat::full(rows, cols, c64::new(*re, *im))),
                _ => return Err("scalar tag or value buffer length mismatch".into()),
            };
            self.data = Some(data);
            Ok(())
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn filled_collection_f32(
        &mut self,
        n: usize,
        kind: i32,
        tag: i32,
        values: &[f32],
    ) -> Result<(), String> {
        guarded(|| {
            let data = match (tag, values) {
                (0, [value]) => VectorData::F32(full_collection(n, kind, *value)?),
                (2, [re, im]) => VectorData::C32(full_collection(n, kind, c32::new(*re, *im))?),
                _ => return Err("scalar tag or value buffer length mismatch".into()),
            };
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn filled_collection_f64(
        &mut self,
        n: usize,
        kind: i32,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(|| {
            let data = match (tag, values) {
                (1, [value]) => VectorData::F64(full_collection(n, kind, *value)?),
                (3, [re, im]) => VectorData::C64(full_collection(n, kind, c64::new(*re, *im))?),
                _ => return Err("scalar tag or value buffer length mismatch".into()),
            };
            self.data = Some(data);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_constants_f32_and_c32() {
        let mut a = TypedMatrix { data: None };
        a.filled_matrix_f32(2, 3, 0, &[1.5]).unwrap();
        assert!(a.filled_matrix_f32(2, 3, 0, &[]).is_err());
        match &a.data {
            Some(MatrixData::F32(x)) => {
                assert_eq!(x.shape(), (2, 3));
                assert_eq!(x[(1, 2)], 1.5);
            }
            _ => panic!("wrong dtype"),
        }
        a.filled_matrix_f32(2, 3, 2, &[1.0, -2.0]).unwrap();
        assert!(a.filled_matrix_f32(2, 3, 2, &[1.0]).is_err());
        assert!(a.filled_matrix_f32(usize::MAX, 2, 2, &[1.0, 2.0]).is_err());
        match &a.data {
            Some(MatrixData::C32(x)) => {
                assert_eq!(x.shape(), (2, 3));
                assert_eq!(x[(1, 2)], c32::new(1.0, -2.0));
            }
            _ => panic!("wrong dtype"),
        }
    }
    #[test]
    fn matrix_constants_f64_and_c64() {
        let mut a = TypedMatrix { data: None };
        a.filled_matrix_f64(2, 3, 1, &[1.5]).unwrap();
        assert!(a.filled_matrix_f64(2, 3, 1, &[]).is_err());
        match &a.data {
            Some(MatrixData::F64(x)) => assert_eq!(x[(1, 2)], 1.5),
            _ => panic!("wrong dtype"),
        }
        a.filled_matrix_f64(2, 3, 3, &[1.0, -2.0]).unwrap();
        assert!(a.filled_matrix_f64(2, 3, 3, &[1.0]).is_err());
        match &a.data {
            Some(MatrixData::C64(x)) => assert_eq!(x[(1, 2)], c64::new(1.0, -2.0)),
            _ => panic!("wrong dtype"),
        }
        let mut a = OwnedMatrix::new();
        a.filled_matrix_f64(2, 3, 1, &[1.5]).unwrap();
        let before = a.matrix.clone();
        assert!(a.filled_matrix_f64(2, 3, 3, &[1.0, 2.0]).is_err());
        assert_eq!(a.matrix, before);
    }
    #[test]
    fn collection_constants_f32_and_c32() {
        let mut a = VectorOwner { data: None };
        for kind in 0..3 {
            a.filled_collection_f32(4, kind, 0, &[1.5]).unwrap();
            match &a.data {
                Some(VectorData::F32(x)) => {
                    assert_eq!(x.kind(), kind);
                    assert_eq!(x.len(), 4);
                    assert_eq!(x.get(3), 1.5);
                }
                _ => panic!("wrong dtype"),
            }
            a.filled_collection_f32(4, kind, 2, &[1.0, -2.0]).unwrap();
            assert!(a.filled_collection_f32(4, 99, 2, &[1.0, -2.0]).is_err());
            assert!(a.filled_collection_f32(4, kind, 2, &[1.0]).is_err());
            match &a.data {
                Some(VectorData::C32(x)) => {
                    assert_eq!(x.kind(), kind);
                    assert_eq!(x.get(3), c32::new(1.0, -2.0));
                }
                _ => panic!("wrong dtype"),
            }
        }
    }
    #[test]
    fn collection_constants_f64_and_c64() {
        let mut a = VectorOwner { data: None };
        for kind in 0..3 {
            a.filled_collection_f64(4, kind, 1, &[1.5]).unwrap();
            match &a.data {
                Some(VectorData::F64(x)) => {
                    assert_eq!(x.kind(), kind);
                    assert_eq!(x.get(3), 1.5);
                }
                _ => panic!("wrong dtype"),
            }
            a.filled_collection_f64(4, kind, 3, &[1.0, -2.0]).unwrap();
            assert!(a.filled_collection_f64(4, 99, 3, &[1.0, -2.0]).is_err());
            assert!(a.filled_collection_f64(4, kind, 3, &[1.0]).is_err());
            match &a.data {
                Some(VectorData::C64(x)) => {
                    assert_eq!(x.kind(), kind);
                    assert_eq!(x.get(3), c64::new(1.0, -2.0));
                }
                _ => panic!("wrong dtype"),
            }
        }
    }
}
