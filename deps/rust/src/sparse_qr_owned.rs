//! Public low-level sparse QR with adapter-owned numeric buffers.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::{
    linalg::qr::{QrRef, SymbolicQr},
    SparseColMatRef,
};
use faer::traits::{math_utils::zero, ComplexField};
use faer::{Conj, Mat, MatRef, Par};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Factors<T> {
    symbolic: Arc<SymbolicQr<usize>>,
    indices: Vec<usize>,
    values: Vec<T>,
}
impl<T: ComplexField> Factors<T> {
    pub(crate) fn new(
        symbolic: Arc<SymbolicQr<usize>>,
        a: SparseColMatRef<'_, usize, T>,
    ) -> Result<Self, String> {
        let mut mem = MemBuffer::try_new(
            symbolic.factorize_numeric_qr_scratch::<T>(Par::Seq.shape(), Default::default()),
        )
        .map_err(|e| format!("{e:?}"))?;
        Self::new_in(symbolic, a, MemStack::new(&mut mem))
    }
    pub(crate) fn new_in(
        symbolic: Arc<SymbolicQr<usize>>,
        a: SparseColMatRef<'_, usize, T>,
        stack: &mut MemStack,
    ) -> Result<Self, String> {
        let mut indices = vec![0; symbolic.len_idx()];
        let mut values = vec![zero::<T>(); symbolic.len_val()];
        symbolic.factorize_numeric_qr(
            &mut indices,
            &mut values,
            a,
            Par::Seq,
            stack,
            Default::default(),
        );
        Ok(Self {
            symbolic,
            indices,
            values,
        })
    }
    pub(crate) fn permutation(&self) -> faer::perm::Perm<usize> {
        let (p, inv) = self.symbolic.col_perm().arrays();
        faer::perm::Perm::new_checked(
            p.to_vec().into_boxed_slice(),
            inv.to_vec().into_boxed_slice(),
            self.symbolic.ncols(),
        )
    }
    pub(crate) fn solve_requirements(&self, k: usize) -> faer::dyn_stack::StackReq {
        self.symbolic
            .solve_in_place_scratch::<T>(k, Par::Seq.shape())
    }
    pub(crate) fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, String> {
        let mut mem =
            MemBuffer::try_new(self.solve_requirements(b.ncols())).map_err(|e| format!("{e:?}"))?;
        self.solve_in(b, MemStack::new(&mut mem))
    }
    pub(crate) fn solve_in(
        &self,
        b: MatRef<'_, T>,
        stack: &mut MemStack,
    ) -> Result<Mat<T>, String> {
        if b.nrows() != self.symbolic.nrows() {
            return Err("sparse QR RHS row mismatch".into());
        }
        let mut rhs = b.to_owned();
        // The buffers were initialized by this symbolic owner's numeric API
        // and remain immutable for the lifetime of the borrowed QrRef.
        let qr = unsafe { QrRef::new_unchecked(&self.symbolic, &self.indices, &self.values) };
        qr.solve_in_place_with_conj(Conj::No, rhs.as_mut(), Par::Seq, stack);
        Ok(rhs.as_ref().subrows(0, self.symbolic.ncols()).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_qr_solves_with_external_workspace() {
        use faer::sparse::{SparseColMat, Triplet};
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(
            3,
            2,
            &[
                Triplet {
                    row: 0,
                    col: 0,
                    val: 2.0,
                },
                Triplet {
                    row: 1,
                    col: 1,
                    val: 3.0,
                },
                Triplet {
                    row: 2,
                    col: 0,
                    val: 1.0,
                },
            ],
        )
        .unwrap();
        let sym = Arc::new(
            faer::sparse::linalg::qr::factorize_symbolic_qr(a.symbolic(), Default::default())
                .unwrap(),
        );
        let f = Factors::new(sym, a.as_ref()).unwrap();
        let b = faer::mat![[4.0], [9.0], [2.0]];
        let req = f.solve_requirements(1);
        let mut scratch = crate::workspace::OwnedScratch::new();
        assert!(scratch.stack(req).is_err());
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        for _ in 0..3 {
            let x = f.solve_in(b.as_ref(), scratch.stack(req).unwrap()).unwrap();
            assert!((x[(0, 0)] - 2.0).abs() < 1e-12);
            assert!((x[(1, 0)] - 3.0).abs() < 1e-12);
        }
    }
}
