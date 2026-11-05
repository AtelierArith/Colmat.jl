//! Owned low-level LU with public symbolic/numeric buffers and caller scratch.
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::LuError;
use faer::sparse::{
    linalg::lu::{LuRef, NumericLu, SymbolicLu},
    SparseColMatRef,
};
use faer::traits::ComplexField;
use faer::{Conj, Mat, MatRef, Par};
use std::sync::Arc;
#[derive(Clone)]
pub(crate) struct Factors<T> {
    symbolic: Arc<SymbolicLu<usize>>,
    numeric: NumericLu<usize, T>,
}
impl<T: ComplexField> Factors<T> {
    pub(crate) fn new(
        symbolic: Arc<SymbolicLu<usize>>,
        a: SparseColMatRef<'_, usize, T>,
        par: Par<'_>,
    ) -> Result<Self, LuError> {
        let mut memory = MemBuffer::try_new(
            symbolic.factorize_numeric_lu_scratch::<T>(par.shape(), Default::default()),
        )?;
        Self::new_in(symbolic, a, par, MemStack::new(&mut memory))
    }
    pub(crate) fn new_in(
        symbolic: Arc<SymbolicLu<usize>>,
        a: SparseColMatRef<'_, usize, T>,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Self, LuError> {
        let mut numeric = NumericLu::new();
        symbolic.factorize_numeric_lu(&mut numeric, a, par, stack, Default::default())?;
        Ok(Self { symbolic, numeric })
    }
    pub(crate) fn as_ref(&self) -> LuRef<'_, usize, T> {
        // Both owners are the exact public outputs of factorize_numeric_lu for this symbolic plan.
        LuRef::new_unchecked(&self.symbolic, &self.numeric)
    }
    pub(crate) fn solve(&self, b: MatRef<'_, T>, trans: bool, adj: bool) -> Result<Mat<T>, String> {
        let mut out = b.to_owned();
        let par = Par::Seq;
        let req = if trans {
            self.symbolic
                .solve_transpose_in_place_scratch::<T>(b.ncols(), par.shape())
        } else {
            self.symbolic
                .solve_in_place_scratch::<T>(b.ncols(), par.shape())
        };
        let mut memory = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        if trans {
            self.as_ref().solve_transpose_in_place_with_conj(
                if adj { Conj::Yes } else { Conj::No },
                out.as_mut(),
                par,
                MemStack::new(&mut memory),
            );
        } else {
            self.as_ref().solve_in_place_with_conj(
                Conj::No,
                out.as_mut(),
                par,
                MemStack::new(&mut memory),
            );
        }
        Ok(out)
    }
}
