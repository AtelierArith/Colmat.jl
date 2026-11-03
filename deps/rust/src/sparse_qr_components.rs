//! Public simplicial QR kernels with independently owned numeric components.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::{Perm, PermRef};
use faer::sparse::linalg::qr::{self, simplicial};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::traits::{
    math_utils::{copy, one, zero},
    ComplexField,
};
use faer::{Conj, Mat, MatRef, Par};
use std::sync::Arc;
#[derive(Clone)]
pub(crate) struct Plan {
    pub symbolic: Arc<simplicial::SymbolicSimplicialQr<usize>>,
    pub cols: Perm<usize>,
}
impl Plan {
    pub fn new(
        a: SymbolicSparseColMatRef<'_, usize>,
        cols: PermRef<'_, usize>,
    ) -> Result<Self, String> {
        let (m, n) = (a.nrows(), a.ncols());
        let req = StackReq::any_of(&[
            qr::col_etree_scratch::<usize>(m, n),
            qr::postorder_scratch::<usize>(n),
            qr::column_counts_aat_scratch::<usize>(n, m),
            simplicial::factorize_simplicial_symbolic_qr_scratch::<usize>(m, n),
        ]);
        let mut mem = MemBuffer::new(req);
        let stack = MemStack::new(&mut mem);
        let mut tree = vec![0; n];
        let etree = qr::col_etree(a, Some(cols), &mut tree, stack);
        let mut post = vec![0; n];
        qr::postorder(&mut post, etree, stack);
        let at = a.transpose().to_col_major().map_err(|e| format!("{e:?}"))?;
        let mut counts = vec![0; n];
        let mut min_col = vec![0; m];
        qr::column_counts_ata(
            &mut counts,
            &mut min_col,
            at.as_ref(),
            Some(cols),
            etree,
            &post,
            stack,
        );
        let symbolic =
            simplicial::factorize_simplicial_symbolic_qr(&min_col, etree, &counts, stack)
                .map_err(|e| format!("{e:?}"))?;
        let (p, inv) = cols.arrays();
        Ok(Self {
            symbolic: Arc::new(symbolic),
            cols: Perm::new_checked(
                p.to_vec().into_boxed_slice(),
                inv.to_vec().into_boxed_slice(),
                n,
            ),
        })
    }
}
#[derive(Clone)]
pub(crate) struct Factors<T> {
    pub plan: Plan,
    pub r: SparseColMat<usize, T>,
    h: SparseColMat<usize, T>,
    tau: Vec<T>,
}
impl<T: ComplexField> Factors<T> {
    pub fn new(plan: &Plan, a: SparseColMatRef<'_, usize, T>) -> Result<Self, String> {
        let s = &plan.symbolic;
        let (m, n) = (a.nrows(), a.ncols());
        let mut rp = vec![0; n + 1];
        let mut ri = vec![0; s.len_r()];
        let mut rv = vec![zero::<T>(); s.len_r()];
        let mut hp = vec![0; n + 1];
        let mut hi = vec![0; s.len_householder()];
        let mut hv = vec![zero::<T>(); s.len_householder()];
        let mut tau = vec![zero::<T>(); n];
        let mut mem = MemBuffer::new(simplicial::factorize_simplicial_numeric_qr_scratch::<
            usize,
            T,
        >(s));
        simplicial::factorize_simplicial_numeric_qr_unsorted(
            &mut rp,
            &mut ri,
            &mut rv,
            &mut hp,
            &mut hi,
            &mut hv,
            &mut tau,
            a,
            Some(plan.cols.as_ref()),
            s,
            MemStack::new(&mut mem),
        );
        // These unsorted CSC structures come directly from faer's numeric kernel;
        // preserve the leading Householder pivot and diagonal-last R ordering.
        let r = SparseColMat::new(
            unsafe { SymbolicSparseColMat::new_unchecked(n, n, rp, None, ri) },
            rv,
        );
        let h = SparseColMat::new(
            unsafe { SymbolicSparseColMat::new_unchecked(m, n, hp, None, hi) },
            hv,
        );
        Ok(Self {
            plan: plan.clone(),
            r,
            h,
            tau,
        })
    }
    fn as_ref(&self) -> simplicial::SimplicialQrRef<'_, usize, T> {
        simplicial::SimplicialQrRef::new(
            &self.plan.symbolic,
            self.r.as_ref(),
            self.h.as_ref(),
            &self.tau,
        )
    }
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, String> {
        let s = &self.plan.symbolic;
        if b.nrows() != s.nrows() {
            return Err("sparse QR RHS shape mismatch".into());
        }
        let mut out = b.to_owned();
        let mut mem = MemBuffer::new(s.solve_in_place_scratch::<T>(b.ncols(), Par::Seq.shape()));
        self.as_ref().solve_in_place_with_conj(
            Conj::No,
            out.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );
        let (_, inv) = self.plan.cols.as_ref().arrays();
        Ok(Mat::from_fn(s.ncols(), b.ncols(), |i, j| {
            copy(&out[(inv[i], j)])
        }))
    }
    pub fn q(&self) -> Mat<T> {
        use faer::traits::math_utils::{add, conj, mul, mul_real, real, recip, sub};
        let s = &self.plan.symbolic;
        let (m, n) = (s.nrows(), s.ncols());
        let h = self.h.as_ref();
        let mut q = Mat::zeros(m, n);
        for j in 0..n {
            if let Some(&pivot) = h.row_idx_of_col_raw(j).first() {
                q[(pivot, j)] = one::<T>();
            }
        }
        // Reverse the same Hermitian reflectors used by faer's reduced Q^H
        // application, starting at its pivot-row embedding. Storage is m*n.
        for j in (0..n).rev() {
            let rows = h.row_idx_of_col_raw(j);
            let vals = h.val_of_col(j);
            if rows.is_empty() {
                continue;
            }
            let scale = recip(&real(&self.tau[j]));
            for k in 0..n {
                let mut dot = zero::<T>();
                for (&i, v) in rows.iter().zip(vals) {
                    dot = add(&dot, &mul(&conj(v), &q[(i, k)]));
                }
                dot = mul_real(&dot, &scale);
                for (&i, v) in rows.iter().zip(vals) {
                    q[(i, k)] = sub(&q[(i, k)], &mul(v, &dot));
                }
            }
        }
        q
    }
    pub fn reconstruct(&self) -> Mat<T> {
        use faer::traits::math_utils::{add, mul};
        let q = self.q();
        let (m, n) = q.shape();
        let mut a = Mat::zeros(m, n);
        let (p, _) = self.plan.cols.as_ref().arrays();
        for col in 0..n {
            let r = self.r.as_ref();
            for (&row, value) in r.row_idx_of_col_raw(col).iter().zip(r.val_of_col(col)) {
                for i in 0..m {
                    a[(i, p[col])] = add(&a[(i, p[col])], &mul(&q[(i, row)], value));
                }
            }
        }
        a
    }
}
