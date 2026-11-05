//! Public simplicial QR kernels with independently owned numeric components.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::{Perm, PermRef};
use faer::sparse::linalg::qr::{self, simplicial};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::traits::{
    math_utils::{copy, one, zero},
    ComplexField,
};
use faer::{Conj, Mat, MatMut, MatRef, Par};
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
    pub(crate) h: SparseColMat<usize, T>,
    pub(crate) tau: Vec<T>,
}
impl<T: ComplexField> Factors<T> {
    pub fn new(plan: &Plan, a: SparseColMatRef<'_, usize, T>) -> Result<Self, String> {
        let mut memory = MemBuffer::try_new(Self::scratch(plan)).map_err(|e| format!("{e:?}"))?;
        Self::new_in(plan, a, MemStack::new(&mut memory))
    }
    pub(crate) fn scratch(plan: &Plan) -> StackReq {
        simplicial::factorize_simplicial_numeric_qr_scratch::<usize, T>(&plan.symbolic)
    }
    pub(crate) fn new_in(
        plan: &Plan,
        a: SparseColMatRef<'_, usize, T>,
        stack: &mut MemStack,
    ) -> Result<Self, String> {
        let s = &plan.symbolic;
        let (m, n) = (a.nrows(), a.ncols());
        let mut rp = vec![0; n + 1];
        let mut ri = vec![0; s.len_r()];
        let mut rv = vec![zero::<T>(); s.len_r()];
        let mut hp = vec![0; n + 1];
        let mut hi = vec![0; s.len_householder()];
        let mut hv = vec![zero::<T>(); s.len_householder()];
        let mut tau = vec![zero::<T>(); n];
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
            stack,
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
    pub(crate) fn as_ref(&self) -> simplicial::SimplicialQrRef<'_, usize, T> {
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
    fn apply_q(&self, b: MatRef<'_, T>, conjugated: bool) -> Mat<T> {
        let mut out = Mat::zeros(self.plan.symbolic.nrows(), b.ncols());
        self.apply_q_into(b, out.as_mut(), conjugated);
        out
    }
    pub(crate) fn q_full(&self) -> Mat<T> {
        use faer::traits::math_utils::one;
        let (m, n) = (self.plan.symbolic.nrows(), self.plan.symbolic.ncols());
        let seed = Mat::from_fn(n, m, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        let mut out = Mat::zeros(m, m);
        self.apply_q_basis(seed.as_ref(), out.as_mut(), false, true);
        out
    }
    pub(crate) fn apply_q_into(&self, b: MatRef<'_, T>, q: MatMut<'_, T>, conjugated: bool) {
        self.apply_q_basis(b, q, conjugated, false);
    }
    fn apply_q_basis(&self, b: MatRef<'_, T>, mut q: MatMut<'_, T>, conjugated: bool, full: bool) {
        use faer::traits::math_utils::{add, conj, mul, mul_real, real, recip, sub};
        let s = &self.plan.symbolic;
        let (m, n) = (s.nrows(), s.ncols());
        assert_eq!(b.nrows(), n);
        let h = self.h.as_ref();
        assert_eq!(q.shape(), (m, b.ncols()));
        q.fill(zero::<T>());
        let complete = full || (0..n).any(|j| h.row_idx_of_col_raw(j).is_empty());
        let mut used = if complete { vec![false; m] } else { Vec::new() };
        let mut assigned = if complete { vec![false; n] } else { Vec::new() };
        for j in 0..n {
            if let Some(&pivot) = h.row_idx_of_col_raw(j).first() {
                if complete {
                    used[pivot] = true;
                    assigned[j] = true;
                }
                for k in 0..b.ncols() {
                    q[(pivot, k)] = copy(&b[(j, k)]);
                }
            }
        }
        if complete {
            crate::sparse_qr_owned::complete_q_seed(q.as_mut(), b, &mut used, &assigned, full);
        }
        for j in (0..n).rev() {
            let rows = h.row_idx_of_col_raw(j);
            let vals = h.val_of_col(j);
            if rows.is_empty() {
                continue;
            }
            let scale = recip(&real(&self.tau[j]));
            for k in 0..b.ncols() {
                let mut dot = zero::<T>();
                for (&i, v) in rows.iter().zip(vals) {
                    let v = if conjugated { copy(v) } else { conj(v) };
                    dot = add(&dot, &mul(&v, &q[(i, k)]));
                }
                dot = mul_real(&dot, &scale);
                for (&i, v) in rows.iter().zip(vals) {
                    let v = if conjugated { conj(v) } else { copy(v) };
                    q[(i, k)] = sub(&q[(i, k)], &mul(&v, &dot));
                }
            }
        }
    }
    pub fn q(&self) -> Mat<T> {
        let n = self.plan.symbolic.ncols();
        let identity = Mat::from_fn(n, n, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        self.apply_q(identity.as_ref(), false)
    }
    pub fn solve_transpose(&self, b: MatRef<'_, T>, adjoint: bool) -> Result<Mat<T>, String> {
        crate::typed::finite(b)?;
        let n = self.plan.symbolic.ncols();
        if b.nrows() != n {
            return Err("transposed sparse QR RHS shape mismatch".into());
        }
        let (p, _) = self.plan.cols.as_ref().arrays();
        let mut z = Mat::from_fn(n, b.ncols(), |i, j| copy(&b[(p[i], j)]));
        let rt = self
            .r
            .transpose()
            .to_col_major()
            .map_err(|e| format!("{e:?}"))?;
        for j in 0..n {
            let r = rt.as_ref();
            let diagonal = r
                .row_idx_of_col_raw(j)
                .iter()
                .zip(r.val_of_col(j))
                .find(|(&row, _)| row == j)
                .map(|(_, v)| v);
            if diagonal.is_none_or(|v| *v == zero::<T>()) {
                return Err("transposed sparse QR solve requires full column rank".into());
            }
        }
        faer::sparse::linalg::triangular_solve::solve_lower_triangular_in_place(
            rt.as_ref(),
            if adjoint { Conj::Yes } else { Conj::No },
            z.as_mut(),
            Par::Seq,
        );
        Ok(self.apply_q(z.as_ref(), !adjoint))
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
