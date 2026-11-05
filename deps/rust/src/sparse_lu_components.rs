//! Owned public simplicial LU components and their coherent solve permutations.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::{Perm, PermRef};
use faer::sparse::linalg::{lu::simplicial, LuError};
use faer::sparse::SparseColMatRef;
use faer::traits::ComplexField;
use faer::{Conj, Mat, MatRef, Par};
#[derive(Clone)]
pub(crate) struct SimplicialFactors<T> {
    pub(crate) numeric: simplicial::SimplicialLu<usize, T>,
    pub(crate) rows: Perm<usize>,
    pub(crate) cols: Perm<usize>,
}
impl<T: ComplexField> SimplicialFactors<T> {
    pub(crate) fn new(
        a: SparseColMatRef<'_, usize, T>,
        cols: PermRef<'_, usize>,
    ) -> Result<Self, LuError> {
        let mut memory = MemBuffer::try_new(Self::scratch(a.nrows()))?;
        Self::new_in(a, cols, MemStack::new(&mut memory))
    }
    pub(crate) fn scratch(n: usize) -> StackReq {
        simplicial::factorize_simplicial_numeric_lu_scratch::<usize, T>(n, n)
    }
    pub(crate) fn new_in(
        a: SparseColMatRef<'_, usize, T>,
        cols: PermRef<'_, usize>,
        stack: &mut MemStack,
    ) -> Result<Self, LuError> {
        let n = a.nrows();
        assert_eq!(n, a.ncols());
        let mut p = vec![0; n];
        let mut pinv = vec![0; n];
        let mut numeric = simplicial::SimplicialLu::new();
        simplicial::factorize_simplicial_numeric_lu(
            &mut p,
            &mut pinv,
            &mut numeric,
            a,
            cols,
            stack,
        )?;
        let (q, qinv) = cols.arrays();
        // The factor kernel initializes both row arrays on success.
        let rows = Perm::new_checked(p.into_boxed_slice(), pinv.into_boxed_slice(), n);
        let cols = Perm::new_checked(
            q.to_vec().into_boxed_slice(),
            qinv.to_vec().into_boxed_slice(),
            n,
        );
        Ok(Self {
            numeric,
            rows,
            cols,
        })
    }
    pub(crate) fn solve(&self, b: MatRef<'_, T>, transposed: bool, conjugated: bool) -> Mat<T> {
        let n = self.numeric.nrows();
        let mut out = b.to_owned();
        let mut mem = MemBuffer::new(if transposed {
            simplicial::solve_transpose_in_place_scratch::<usize, T>(n, b.ncols(), Par::Seq.shape())
        } else {
            simplicial::solve_in_place_scratch::<usize, T>(n, b.ncols(), Par::Seq.shape())
        });
        let conj = if conjugated { Conj::Yes } else { Conj::No };
        if transposed {
            self.numeric.solve_transpose_in_place_with_conj(
                self.rows.as_ref(),
                self.cols.as_ref(),
                conj,
                out.as_mut(),
                Par::Seq,
                MemStack::new(&mut mem),
            );
        } else {
            self.numeric.solve_in_place_with_conj(
                self.rows.as_ref(),
                self.cols.as_ref(),
                conj,
                out.as_mut(),
                Par::Seq,
                MemStack::new(&mut mem),
            );
        }
        out
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::traits::math_utils::*;
    fn check<T: ComplexField>() {
        let entries = [
            (0, 0, 0.0),
            (0, 1, 2.0),
            (0, 2, 1.0),
            (1, 0, 3.0),
            (1, 1, 1.0),
            (2, 1, 1.0),
            (2, 2, 4.0),
        ];
        let entries: Vec<_> = entries
            .into_iter()
            .map(|(row, col, x)| Triplet {
                row,
                col,
                val: from_f64::<T>(x),
            })
            .collect();
        let a = SparseColMat::<usize, T>::try_new_from_triplets(3, 3, &entries).unwrap();
        let q = Perm::new_checked(
            vec![2, 0, 1].into_boxed_slice(),
            vec![1, 2, 0].into_boxed_slice(),
            3,
        );
        let f = SimplicialFactors::new(a.as_ref(), q.as_ref()).unwrap();
        let l = f.numeric.l_factor_unsorted().to_dense();
        let u = f.numeric.u_factor_unsorted().to_dense();
        let product = &l * &u;
        let dense = a.to_dense();
        let p = f.rows.as_ref().arrays().0;
        let q = f.cols.as_ref().arrays().0;
        let expected = Mat::<T>::from_fn(3, 3, |i, j| copy(&dense[(p[i], q[j])]));
        let tolerance = from_f64::<T::Real>(1e-4);
        assert!((&product - &expected).norm_l2() < tolerance);
        let b = Mat::<T>::from_fn(3, 2, |i, j| from_f64::<T>((i + j + 1) as f64));
        for trans in [false, true] {
            for conj in [false, true] {
                let x = f.solve(b.as_ref(), trans, conj);
                let transformed = if trans {
                    dense.as_ref().transpose().to_owned()
                } else {
                    dense.clone()
                };
                let transformed = if conj {
                    transformed.as_ref().conjugate().to_owned()
                } else {
                    transformed
                };
                assert!((&transformed * &x - &b).norm_l2() < tolerance);
            }
        }
    }
    #[test]
    fn f32_components() {
        check::<f32>();
    }
    #[test]
    fn f64_components() {
        check::<f64>();
    }
    #[test]
    fn c32_components() {
        check::<faer::c32>();
    }
    #[test]
    fn c64_components() {
        check::<faer::c64>();
    }
}
