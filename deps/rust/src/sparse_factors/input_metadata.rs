//! Validate sparse construction metadata before Julia reads value buffers.
use super::{Analysis, SparseSymbolic};
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner, Storage};
use faer::traits::ComplexField;
use rustcall_julia_macros::julia;
fn describe<T: ComplexField>(a: &Storage<T>) -> (usize, usize, bool, usize) {
    let (m, n) = a.shape();
    match a {
        Storage::Csc(x) => (m, n, false, x.val().len()),
        Storage::Csr(x) => (m, n, true, x.val().len()),
    }
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_factor_matrix_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        csr: bool,
        stored: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let (actual, scalar) = match self.data()? {
                SparseData::F32(x) => (describe(x), 0),
                SparseData::F64(x) => (describe(x), 1),
                SparseData::C32(x) => (describe(x), 2),
                SparseData::C64(x) => (describe(x), 3),
            };
            if actual != (m, n, csr, stored) || scalar != tag {
                return Err("sparse factor input native metadata mismatch".into());
            }
            Ok(())
        })
    }
}
#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn sparse_factor_plan_validate(
        &self,
        m: usize,
        n: usize,
        kind: i32,
        lower: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let p = self.plan()?;
            let actual = match p.analysis {
                Analysis::Lu(_) | Analysis::SimplicialLu(_) => 0,
                Analysis::Qr(_) | Analysis::SimplicialQr(_) | Analysis::WideQr(_) => 1,
                Analysis::Llt(_) => 2,
            };
            if (p.pattern.nrows(), p.pattern.ncols(), actual, p.lower) != (m, n, kind, lower) {
                return Err("sparse factor plan native metadata mismatch".into());
            }
            Ok(())
        })
    }
}
