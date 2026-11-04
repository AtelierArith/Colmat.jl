//! Dense-solve collection owner validation before Julia buffer conversion.
use crate::{
    collections::{VectorData, VectorOwner},
    guarded,
};
use rustcall_julia_macros::julia;
#[julia]
impl VectorOwner {
    #[julia]
    pub fn factor_rhs_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        layout: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let (len, kind, t) = match self.data.as_ref().ok_or("collection is not initialized")? {
                VectorData::F32(a) => (a.len(), a.kind(), 0),
                VectorData::F64(a) => (a.len(), a.kind(), 1),
                VectorData::C32(a) => (a.len(), a.kind(), 2),
                VectorData::C64(a) => (a.len(), a.kind(), 3),
            };
            let (rows, cols) = match kind {
                0 => (len, 1),
                1 => (1, len),
                2 => (len, len),
                _ => unreachable!(),
            };
            if (rows, cols, t, kind) == (m, n, tag, layout) {
                Ok(())
            } else {
                Err("dense solve collection native metadata mismatch".into())
            }
        })
    }
}
