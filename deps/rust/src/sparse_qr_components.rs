//! Public simplicial QR kernels with independently owned numeric components.
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::{Perm, PermRef};
use faer::prelude::{Reborrow, ReborrowMut};
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
    /// Kernel stack for `new_in` on an `m`×`n` pattern with `nnz` entries: the
    /// etree, postorder, column counts, `min_col` and the transposed pattern,
    /// then the largest of faer's public kernel requirements.
    pub(crate) fn scratch(m: usize, n: usize, nnz: usize) -> StackReq {
        StackReq::all_of(&[
            StackReq::new::<usize>(n).array(3),
            StackReq::new::<usize>(m),
            StackReq::new::<usize>(m + 1),
            StackReq::new::<usize>(nnz),
            StackReq::any_of(&[
                faer::sparse::utils::transpose_scratch::<usize>(m, n),
                qr::col_etree_scratch::<usize>(m, n),
                qr::postorder_scratch::<usize>(n),
                qr::column_counts_aat_scratch::<usize>(n, m),
                simplicial::factorize_simplicial_symbolic_qr_scratch::<usize>(m, n),
            ]),
        ])
    }
    pub fn new(
        a: SymbolicSparseColMatRef<'_, usize>,
        cols: PermRef<'_, usize>,
    ) -> Result<Self, String> {
        let req = Self::scratch(a.nrows(), a.ncols(), a.compute_nnz());
        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        Self::new_in(a, cols, MemStack::new(&mut mem))
    }
    /// Simplicial symbolic analysis with every temporary in `stack`; only the
    /// returned symbolic structure and the copied permutation allocate.
    pub(crate) fn new_in(
        a: SymbolicSparseColMatRef<'_, usize>,
        cols: PermRef<'_, usize>,
        stack: &mut MemStack,
    ) -> Result<Self, String> {
        use faer::traits::Symbolic;
        let (m, n) = (a.nrows(), a.ncols());
        let nnz = a.compute_nnz();
        let (mut tree, stack) = stack.make_with::<usize>(n, |_| 0);
        let (mut post, stack) = stack.make_with::<usize>(n, |_| 0);
        let (mut counts, stack) = stack.make_with::<usize>(n, |_| 0);
        let (mut min_col, stack) = stack.make_with::<usize>(m, |_| 0);
        let (mut at_ptr, stack) = stack.make_with::<usize>(m + 1, |_| 0);
        let (mut at_idx, stack) = stack.make_with::<usize>(nnz, |_| 0);
        // The column-count kernel reads only the transposed pattern.
        let at = faer::sparse::utils::transpose(
            Symbolic::materialize(nnz),
            &mut at_ptr,
            &mut at_idx,
            SparseColMatRef::new(a, Symbolic::materialize(a.row_idx().len())),
            stack,
        )
        .symbolic();
        let etree = qr::col_etree(a, Some(cols), &mut tree, stack);
        qr::postorder(&mut post, etree, stack);
        qr::column_counts_ata(
            &mut counts,
            &mut min_col,
            at,
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
    /// Converts values while preserving faer's unsorted numeric CSC layout
    /// (leading Householder pivot, diagonal-last R).
    pub(crate) fn map_values<U: ComplexField>(&self, cast: impl Fn(&T) -> U) -> Factors<U> {
        let map = |a: &SparseColMat<usize, T>| {
            let s = a.symbolic();
            // SAFETY: copies an existing valid (unsorted) faer CSC structure.
            let symbolic = unsafe {
                SymbolicSparseColMat::new_unchecked(
                    s.nrows(),
                    s.ncols(),
                    s.col_ptr().to_vec(),
                    None,
                    s.row_idx().to_vec(),
                )
            };
            SparseColMat::new(symbolic, a.val().iter().map(&cast).collect())
        };
        Factors {
            plan: self.plan.clone(),
            r: map(&self.r),
            h: map(&self.h),
            tau: self.tau.iter().map(&cast).collect(),
        }
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
    /// Stack for Q extraction/application: the pivot bookkeeping flags.
    pub(crate) fn q_scratch(m: usize, n: usize) -> StackReq {
        StackReq::new::<bool>(m).and(StackReq::new::<bool>(n))
    }
    fn with_q_stack<R>(&self, f: impl FnOnce(&mut MemStack) -> R) -> Result<R, String> {
        let s = &self.plan.symbolic;
        let mut mem = MemBuffer::try_new(Self::q_scratch(s.nrows(), s.ncols()))
            .map_err(|e| format!("{e:?}"))?;
        Ok(f(MemStack::new(&mut mem)))
    }
    /// Thin (`m`×`n`) or full (`m`×`m`) Q written into `q`; the identity seed
    /// is implicit, so only `q_scratch` bytes of `stack` are used.
    pub(crate) fn q_into(&self, q: MatMut<'_, T>, full: bool, stack: &mut MemStack) {
        let n = self.plan.symbolic.ncols();
        let seed = |j: usize, k: usize| if j == k { one::<T>() } else { zero::<T>() };
        self.apply_q_basis(seed, q, false, full, n, stack);
    }
    pub(crate) fn q_full(&self) -> Mat<T> {
        let m = self.plan.symbolic.nrows();
        let mut out = Mat::zeros(m, m);
        self.with_q_stack(|stack| self.q_into(out.as_mut(), true, stack))
            .expect("sparse QR Q workspace allocation");
        out
    }
    pub(crate) fn apply_q_into(
        &self,
        b: MatRef<'_, T>,
        q: MatMut<'_, T>,
        conjugated: bool,
        stack: &mut MemStack,
    ) {
        assert_eq!(b.nrows(), self.plan.symbolic.ncols());
        self.apply_q_basis(
            |j, k| copy(&b[(j, k)]),
            q,
            conjugated,
            false,
            b.nrows(),
            stack,
        );
    }
    // `seed(j, k)` is the n×q.ncols() basis entry placed at reflector j's pivot.
    fn apply_q_basis(
        &self,
        seed: impl Fn(usize, usize) -> T,
        mut q: MatMut<'_, T>,
        conjugated: bool,
        full: bool,
        seed_rows: usize,
        stack: &mut MemStack,
    ) {
        use faer::traits::math_utils::{add, conj, mul, mul_real, real, recip, sub};
        let s = &self.plan.symbolic;
        let (m, n) = (s.nrows(), s.ncols());
        assert_eq!(seed_rows, n);
        let h = self.h.as_ref();
        assert_eq!(q.nrows(), m);
        let cols = q.ncols();
        q.fill(zero::<T>());
        let (mut used, stack) = stack.make_with::<bool>(m, |_| false);
        let (mut assigned, _) = stack.make_with::<bool>(n, |_| false);
        for j in 0..n {
            if let Some(&pivot) = h.row_idx_of_col_raw(j).first() {
                used[pivot] = true;
                assigned[j] = true;
                for k in 0..cols {
                    q[(pivot, k)] = seed(j, k);
                }
            }
        }
        // Rank-deficient columns take the first unused rows, in column order;
        // the full basis then appends unit vectors on the remaining rows.
        let mut row = 0;
        for (j, &present) in assigned.iter().enumerate() {
            if !present {
                while used[row] {
                    row += 1;
                }
                used[row] = true;
                for k in 0..cols {
                    q[(row, k)] = seed(j, k);
                }
            }
        }
        if full {
            let mut col = n;
            for (row, &present) in used.iter().enumerate() {
                if !present {
                    q[(row, col)] = one::<T>();
                    col += 1;
                }
            }
        }
        for j in (0..n).rev() {
            let rows = h.row_idx_of_col_raw(j);
            let vals = h.val_of_col(j);
            if rows.is_empty() {
                continue;
            }
            let scale = recip(&real(&self.tau[j]));
            for k in 0..cols {
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
        let (m, n) = (self.plan.symbolic.nrows(), self.plan.symbolic.ncols());
        let mut out = Mat::zeros(m, n);
        self.with_q_stack(|stack| self.q_into(out.as_mut(), false, stack))
            .expect("sparse QR Q workspace allocation");
        out
    }
    /// Stack for `solve_transpose_in` with `k` right-hand sides.
    pub(crate) fn solve_transpose_scratch(m: usize, n: usize, k: usize) -> StackReq {
        faer::linalg::temp_mat_scratch::<T>(n, k).and(Self::q_scratch(m, n))
    }
    pub fn solve_transpose(&self, b: MatRef<'_, T>, adjoint: bool) -> Result<Mat<T>, String> {
        let s = &self.plan.symbolic;
        let req = Self::solve_transpose_scratch(s.nrows(), s.ncols(), b.ncols());
        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        self.solve_transpose_in(b, adjoint, MemStack::new(&mut mem))
    }
    /// Minimum-norm `Aᵀx = b` / `Aᴴx = b`: the triangular solve runs on the
    /// saved diagonal-last R in `stack`; only the returned m×k result allocates.
    pub(crate) fn solve_transpose_in(
        &self,
        b: MatRef<'_, T>,
        adjoint: bool,
        stack: &mut MemStack,
    ) -> Result<Mat<T>, String> {
        crate::typed::finite(b)?;
        let (m, n) = (self.plan.symbolic.nrows(), self.plan.symbolic.ncols());
        if b.nrows() != n {
            return Err("transposed sparse QR RHS shape mismatch".into());
        }
        let r = self.r.as_ref();
        for j in 0..n {
            // faer's simplicial kernel stores each R diagonal last.
            let last = r.row_idx_of_col_raw(j).last().zip(r.val_of_col(j).last());
            if last.is_none_or(|(&i, v)| i != j || *v == zero::<T>()) {
                return Err("transposed sparse QR solve requires full column rank".into());
            }
        }
        let (p, _) = self.plan.cols.as_ref().arrays();
        let (mut z, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, b.ncols(), stack);
        let mut z = faer::mat::AsMatMut::as_mat_mut(&mut z);
        for j in 0..b.ncols() {
            for i in 0..n {
                z[(i, j)] = copy(&b[(p[i], j)]);
            }
        }
        // The kernel reads each column's last stored entry as its diagonal.
        faer::sparse::linalg::triangular_solve::solve_upper_triangular_transpose_in_place(
            r,
            if adjoint { Conj::Yes } else { Conj::No },
            z.rb_mut(),
            Par::Seq,
        );
        let mut out = Mat::zeros(m, b.ncols());
        self.apply_q_into(z.rb(), out.as_mut(), !adjoint, stack);
        Ok(out)
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
#[cfg(test)]
mod allocation_tests {
    use super::*;
    use crate::allocation_probe::measure;
    use faer::sparse::Triplet;
    fn matrix(m: usize, n: usize) -> SparseColMat<usize, f64> {
        let e: Vec<_> =
            (0..n)
                .flat_map(|j| {
                    [(j, 2.0 + j as f64), (j + n, 0.5), ((j * 7) % m, 0.25)]
                        .map(|(row, val)| Triplet { row, col: j, val })
                })
                .collect();
        SparseColMat::try_new_from_triplets(m, n, &e).unwrap()
    }
    // Baseline before caller-stack routing (m=400, n=200, f64):
    // plan 14 allocations / largest 11200 B; q 2 / 640000 B (n×n identity
    // seed); q_full 4 / 1280000 B (n×m seed); solve_transpose 6 / 3544 B.
    #[test]
    fn simplicial_plan_q_and_transpose_solve_use_caller_stack() {
        let (m, n) = (400, 200);
        let a = matrix(m, n);
        let id: Vec<usize> = (0..n).collect();
        let cols = Perm::new_checked(id.clone().into_boxed_slice(), id.into_boxed_slice(), n);
        let reference = Plan::new(a.symbolic(), cols.as_ref()).unwrap();
        let mut work = MemBuffer::new(Plan::scratch(m, n, a.compute_nnz()));
        let mut plan = None;
        let p = measure(|| {
            plan =
                Some(Plan::new_in(a.symbolic(), cols.as_ref(), MemStack::new(&mut work)).unwrap())
        });
        let plan = plan.unwrap();
        assert_eq!(plan.symbolic.len_r(), reference.symbolic.len_r());
        assert_eq!(
            plan.symbolic.len_householder(),
            reference.symbolic.len_householder()
        );
        let f = Factors::new(&plan, a.as_ref()).unwrap();
        let b = Mat::from_fn(n, 2, |i, j| 1.0 + (i + j) as f64);
        let mut thin = Mat::zeros(m, n);
        let mut full = Mat::zeros(m, m);
        let mut q_work = MemBuffer::new(Factors::<f64>::q_scratch(m, n));
        let q = measure(|| {
            let stack = MemStack::new(&mut q_work);
            f.q_into(thin.as_mut(), false, stack);
            f.q_into(full.as_mut(), true, stack);
        });
        let mut t_work = MemBuffer::new(Factors::<f64>::solve_transpose_scratch(m, n, 2));
        let mut x = None;
        let t = measure(|| {
            x = Some(
                f.solve_transpose_in(b.as_ref(), true, MemStack::new(&mut t_work))
                    .unwrap(),
            )
        });
        eprintln!("caller stack: plan={p:?} q+q_full={q:?} solve_transpose={t:?}");
        // Only the returned symbolic structure and permutation copy remain.
        // (Measured: 6 allocations, largest 1600 B = one m-length index array.)
        assert!(p.0 <= 6 && p.1 <= 8 * m, "{p:?}");
        assert_eq!(q.0, 0);
        assert_eq!(t.0, 1); // the returned m×k solution
                            // Q has orthonormal columns, the full basis extends the thin one, and
                            // the minimum-norm solution satisfies Aᴴx = b.
        let eye = Mat::<f64>::identity(m, m);
        assert!((full.adjoint() * &full - &eye).norm_max() < 1e-12);
        assert!((full.get(.., ..n) - &thin).norm_max() < 1e-12);
        let dense = a.to_dense();
        assert!((dense.adjoint() * x.unwrap() - &b).norm_max() < 1e-10);
        let owned = measure(|| drop(f.q()));
        assert_eq!(owned.0, 2); // output + flag buffer, no identity seed
    }
}
