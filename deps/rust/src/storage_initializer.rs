//! Bulk initialized values for safe Julia resize callbacks; no callback crosses FFI.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::traits::ComplexField;
use faer::{c32, c64, Mat};
use rustcall_julia_macros::julia;

fn count(rows: usize, cols: usize, old_rows: usize, old_cols: usize) -> Result<usize, String> {
    let total = rows.checked_mul(cols).ok_or("resize dimensions overflow")?;
    let overlap = rows
        .min(old_rows)
        .checked_mul(cols.min(old_cols))
        .ok_or("resize dimensions overflow")?;
    Ok(total - overlap)
}
fn resize_matrix<T: ComplexField>(
    a: &mut Mat<T>,
    rows: usize,
    cols: usize,
    old_rows: usize,
    old_cols: usize,
    values_len: usize,
    components: usize,
    get: impl Fn(usize) -> T,
) -> Result<(), String> {
    if a.shape() != (old_rows, old_cols) {
        return Err("matrix shape changed before initialization".into());
    }
    let expected = count(rows, cols, old_rows, old_cols)?
        .checked_mul(components)
        .ok_or("initializer length overflow")?;
    if values_len != expected {
        return Err("incorrect initializer buffer length".into());
    }
    // Reserve before faer's resize_with can truncate either existing dimension.
    a.try_reserve(rows, cols).map_err(|e| format!("{e:?}"))?;
    let mut index = 0;
    a.resize_with(rows, cols, |_, _| {
        let v = get(index);
        index += 1;
        v
    });
    Ok(())
}
fn resize_collection<T: ComplexField>(
    a: &mut VectorStore<T>,
    n: usize,
    old: usize,
    values_len: usize,
    components: usize,
    get: impl Fn(usize) -> T,
) -> Result<(), String> {
    if a.len() != old {
        return Err("collection length changed before initialization".into());
    }
    let expected = n
        .saturating_sub(old)
        .checked_mul(components)
        .ok_or("initializer length overflow")?;
    if values_len != expected {
        return Err("incorrect initializer buffer length".into());
    }
    let mut index = 0;
    let mut next = |_| {
        let v = get(index);
        index += 1;
        v
    };
    match a {
        VectorStore::Column(x) => {
            x.try_reserve(n).map_err(|e| format!("{e:?}"))?;
            x.resize_with(n, &mut next);
        }
        VectorStore::Row(x) => {
            x.try_reserve(n).map_err(|e| format!("{e:?}"))?;
            x.resize_with(n, &mut next);
        }
        VectorStore::Diagonal(_) => return Err("diagonal dimensions are fixed".into()),
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn resize_matrix_values_f64(
        &mut self,
        rows: usize,
        cols: usize,
        old_rows: usize,
        old_cols: usize,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("initializer scalar type mismatch".into());
            }
            resize_matrix(
                &mut self.matrix,
                rows,
                cols,
                old_rows,
                old_cols,
                values.len(),
                1,
                |i| values[i],
            )
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn resize_matrix_values_f32(
        &mut self,
        rows: usize,
        cols: usize,
        old_rows: usize,
        old_cols: usize,
        tag: i32,
        values: &[f32],
    ) -> Result<(), String> {
        guarded(
            || match self.data.as_mut().ok_or("owner is not initialized")? {
                MatrixData::F32(a) if tag == 0 => {
                    resize_matrix(a, rows, cols, old_rows, old_cols, values.len(), 1, |i| {
                        values[i]
                    })
                }
                MatrixData::C32(a) if tag == 2 => {
                    resize_matrix(a, rows, cols, old_rows, old_cols, values.len(), 2, |i| {
                        c32::new(values[2 * i], values[2 * i + 1])
                    })
                }
                _ => Err("initializer scalar type mismatch".into()),
            },
        )
    }
    #[julia]
    pub fn resize_matrix_values_f64(
        &mut self,
        rows: usize,
        cols: usize,
        old_rows: usize,
        old_cols: usize,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(
            || match self.data.as_mut().ok_or("owner is not initialized")? {
                MatrixData::F64(a) if tag == 1 => {
                    resize_matrix(a, rows, cols, old_rows, old_cols, values.len(), 1, |i| {
                        values[i]
                    })
                }
                MatrixData::C64(a) if tag == 3 => {
                    resize_matrix(a, rows, cols, old_rows, old_cols, values.len(), 2, |i| {
                        c64::new(values[2 * i], values[2 * i + 1])
                    })
                }
                _ => Err("initializer scalar type mismatch".into()),
            },
        )
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn resize_collection_values_f32(
        &mut self,
        n: usize,
        old: usize,
        kind: i32,
        tag: i32,
        values: &[f32],
    ) -> Result<(), String> {
        guarded(
            || match self.data.as_mut().ok_or("owner is not initialized")? {
                VectorData::F32(a) if tag == 0 && a.kind() == kind => {
                    resize_collection(a, n, old, values.len(), 1, |i| values[i])
                }
                VectorData::C32(a) if tag == 2 && a.kind() == kind => {
                    resize_collection(a, n, old, values.len(), 2, |i| {
                        c32::new(values[2 * i], values[2 * i + 1])
                    })
                }
                _ => Err("initializer scalar type mismatch".into()),
            },
        )
    }
    #[julia]
    pub fn resize_collection_values_f64(
        &mut self,
        n: usize,
        old: usize,
        kind: i32,
        tag: i32,
        values: &[f64],
    ) -> Result<(), String> {
        guarded(
            || match self.data.as_mut().ok_or("owner is not initialized")? {
                VectorData::F64(a) if tag == 1 && a.kind() == kind => {
                    resize_collection(a, n, old, values.len(), 1, |i| values[i])
                }
                VectorData::C64(a) if tag == 3 && a.kind() == kind => {
                    resize_collection(a, n, old, values.len(), 2, |i| {
                        c64::new(values[2 * i], values[2 * i + 1])
                    })
                }
                _ => Err("initializer scalar type mismatch".into()),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::traits::math_utils::from_f64;

    fn contracts<T: ComplexField + Copy + std::fmt::Debug>() {
        let mut a = Mat::<T>::from_fn(2, 3, |i, j| from_f64((10 * i + j) as f64));
        let before = a.clone();
        assert!(resize_matrix(&mut a, 3, 4, 2, 3, 5, 1, |_| from_f64(0.0)).is_err());
        assert_eq!(a, before);
        assert!(resize_matrix(&mut a, 3, 4, 1, 3, 6, 1, |_| from_f64(0.0)).is_err());
        assert_eq!(a, before);
        assert!(resize_matrix(&mut a, usize::MAX, 2, 2, 3, 0, 1, |_| from_f64(0.0)).is_err());
        assert_eq!(a, before);
        a.try_reserve(3, 4).unwrap();
        let pointer = a.as_ptr();
        resize_matrix(&mut a, 3, 4, 2, 3, 6, 1, |i| from_f64((100 + i) as f64)).unwrap();
        assert_eq!(a.as_ptr(), pointer);
        for j in 0..4 {
            for i in 0..3 {
                let expected = if j < 3 && i < 2 {
                    before[(i, j)]
                } else if j < 3 {
                    from_f64((100 + j) as f64)
                } else {
                    from_f64((103 + i) as f64)
                };
                assert_eq!(a[(i, j)], expected);
            }
        }
        for kind in [0, 1, 2] {
            let mut v = VectorStore::<T>::new(kind, 3, |i| from_f64(i as f64)).unwrap();
            assert!(resize_collection(&mut v, 6, 3, 2, 1, |_| from_f64(0.0)).is_err());
            assert_eq!(v.len(), 3);
            for i in 0..3 {
                assert_eq!(v.get(i), from_f64(i as f64));
            }
            if kind == 2 {
                assert!(resize_collection(&mut v, 2, 3, 0, 1, |_| from_f64(0.0)).is_err());
                assert_eq!(v.len(), 3);
            } else {
                resize_collection(&mut v, 6, 3, 3, 1, |i| from_f64((200 + i) as f64)).unwrap();
                for i in 0..6 {
                    assert_eq!(
                        v.get(i),
                        from_f64(if i < 3 { i as f64 } else { (197 + i) as f64 })
                    );
                }
            }
        }
    }
    #[test]
    fn initializer_contract_f32() {
        contracts::<f32>();
    }
    #[test]
    fn initializer_contract_f64() {
        contracts::<f64>();
    }
    #[test]
    fn initializer_contract_c32() {
        contracts::<c32>();
    }
    #[test]
    fn initializer_contract_c64() {
        contracts::<c64>();
    }
}
