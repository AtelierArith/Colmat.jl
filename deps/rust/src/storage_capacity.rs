//! Read allocation metadata from checked owned handles.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::Mat;
use rustcall_julia_macros::julia;

fn matrix_capacity<T>(matrix: &Mat<T>, out: &mut [usize]) -> Result<(), String> {
    let [rows, cols] = out else {
        return Err("capacity output must have two entries".into());
    };
    *rows = matrix.row_capacity();
    *cols = matrix.col_capacity();
    Ok(())
}
fn vector_capacity<T>(vector: &VectorStore<T>) -> usize {
    match vector {
        VectorStore::Column(v) => v.capacity(),
        VectorStore::Row(v) => v.capacity(),
        VectorStore::Diagonal(v) => v.capacity(),
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn storage_capacities(&self, out: &mut [usize]) -> Result<(), String> {
        guarded(|| matrix_capacity(&self.matrix, out))
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn storage_capacities(&self, out: &mut [usize]) -> Result<(), String> {
        guarded(
            || match self.data.as_ref().ok_or("matrix is not initialized")? {
                MatrixData::F32(m) => matrix_capacity(m, out),
                MatrixData::F64(m) => matrix_capacity(m, out),
                MatrixData::C32(m) => matrix_capacity(m, out),
                MatrixData::C64(m) => matrix_capacity(m, out),
            },
        )
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn collection_capacity(&self) -> Result<usize, String> {
        guarded(|| {
            Ok(match self.data()? {
                VectorData::F32(v) => vector_capacity(v),
                VectorData::F64(v) => vector_capacity(v),
                VectorData::C32(v) => vector_capacity(v),
                VectorData::C64(v) => vector_capacity(v),
            })
        })
    }
}
