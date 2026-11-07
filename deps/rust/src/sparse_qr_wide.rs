//! Wide QR assembled from native prefix QR and saved-reflector projections.
use super::sparse_qr_owned;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::perm::Perm;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::linalg::qr::{self, QrSymbolicParams, SymbolicQr};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMatRef, Triplet};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par, ParShape};
use std::sync::Arc;
#[derive(Clone)]
pub(crate) struct Plan {
    head: Arc<SymbolicQr<usize>>,
    columns: Perm<usize>,
    m: usize,
    n: usize,
    /// Dense `QrParams.blocking_threshold` override for the head numeric QR.
    pub(crate) blocking: Option<usize>,
}
impl Plan {
    /// Symbolic QR of the square leading-pivot head.
    pub(crate) fn head(&self) -> Arc<SymbolicQr<usize>> {
        self.head.clone()
    }
    pub(crate) fn new(
        a: SymbolicSparseColMatRef<'_, usize>,
        columns: Perm<usize>,
        params: QrSymbolicParams<'_>,
    ) -> Result<Self, String> {
        let (m, n) = (a.nrows(), a.ncols());
        if m >= n || columns.len() != n {
            return Err("wide QR shape/permutation mismatch".into());
        }
        let p = columns.as_ref().arrays().0;
        let entries: Vec<_> = (0..m)
            .flat_map(|j| {
                a.row_idx_of_col(p[j]).map(move |i| Triplet {
                    row: i,
                    col: j,
                    val: 1.0,
                })
            })
            .collect();
        let head = SparseColMat::<usize, f64>::try_new_from_triplets(m, m, &entries)
            .map_err(|e| format!("{e:?}"))?;
        let identity: Vec<_> = (0..m).collect();
        let identity = Perm::new_checked(
            identity.clone().into_boxed_slice(),
            identity.into_boxed_slice(),
            m,
        );
        let head = Arc::new(
            qr::factorize_symbolic_qr_with_column_ordering(
                head.symbolic(),
                params,
                Some(identity.as_ref()),
            )
            .map_err(|e| format!("{e:?}"))?,
        );
        Ok(Self {
            head,
            columns,
            m,
            n,
            blocking: None,
        })
    }
}
#[derive(Clone)]
pub(crate) struct Factors<T> {
    head: sparse_qr_owned::Factors<T>,
    r: SparseColMat<usize, T>,
    columns: Perm<usize>,
    m: usize,
    n: usize,
}
impl<T: ComplexField> Factors<T> {
    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }
    pub(crate) fn new(plan: &Plan, a: SparseColMatRef<'_, usize, T>) -> Result<Self, String> {
        let mut memory =
            MemBuffer::try_new(Self::scratch(plan, &Par::Seq)).map_err(|e| format!("{e:?}"))?;
        Self::new_in(plan, a, Par::Seq, MemStack::new(&mut memory))
    }
    pub(crate) fn scratch(plan: &Plan, par: &Par<'_>) -> StackReq {
        let projection = faer::linalg::temp_mat_scratch::<T>(plan.m, 1)
            .and(faer::linalg::temp_mat_scratch::<T>(plan.m, 1))
            .and(sparse_qr_owned::Factors::<T>::full_q_adjoint_scratch(
                &plan.head, 1,
            ));
        sparse_qr_owned::Factors::<T>::numeric_scratch(&plan.head, par.shape(), plan.blocking)
            .or(projection)
    }
    pub(crate) fn new_in(
        plan: &Plan,
        a: SparseColMatRef<'_, usize, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Self, String> {
        if (a.nrows(), a.ncols()) != (plan.m, plan.n) {
            return Err("wide QR numeric shape mismatch".into());
        }
        let (m, n) = (plan.m, plan.n);
        let p = plan.columns.as_ref().arrays().0;
        let entries: Vec<_> = (0..m)
            .flat_map(|j| {
                a.row_idx_of_col(p[j])
                    .zip(a.val_of_col(p[j]))
                    .map(move |(i, v)| Triplet {
                        row: i,
                        col: j,
                        val: copy(v),
                    })
            })
            .collect();
        let prefix = SparseColMat::<usize, T>::try_new_from_triplets(m, m, &entries)
            .map_err(|e| format!("{e:?}"))?;
        let head = sparse_qr_owned::Factors::new_tuned(
            plan.head.clone(),
            prefix.as_ref(),
            stack,
            par.reborrow(),
            plan.blocking,
        )?;
        let mut entries = Vec::new();
        head.visit_r(|row, col, val| {
            entries.push(Triplet {
                row,
                col,
                val: copy(val),
            })
        });
        let (mut column, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, 1, stack);
        let mut column = column.as_mat_mut();
        let (mut projected, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, 1, stack);
        let mut projected = projected.as_mat_mut();
        // Project one column at a time: no dense Q or m-by-(n-m) temporary.
        for j in m..n {
            column.fill(zero::<T>());
            for (i, v) in a.row_idx_of_col(p[j]).zip(a.val_of_col(p[j])) {
                column[(i, 0)] = copy(v);
            }
            head.apply_full_q_adjoint_in(
                projected.rb_mut(),
                column.as_ref(),
                par.reborrow(),
                stack,
            );
            for i in 0..m {
                if projected[(i, 0)] != zero::<T>() {
                    entries.push(Triplet {
                        row: i,
                        col: j,
                        val: copy(&projected[(i, 0)]),
                    });
                }
            }
        }
        let r =
            SparseColMat::try_new_from_triplets(m, n, &entries).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            head,
            r,
            columns: plan.columns.clone(),
            m,
            n,
        })
    }
    pub(crate) fn head_ref(&self) -> &sparse_qr_owned::Factors<T> {
        &self.head
    }
    pub(crate) fn r_ref(&self) -> SparseColMatRef<'_, usize, T> {
        self.r.as_ref()
    }
    pub(crate) fn permutation_ref(&self) -> faer::perm::PermRef<'_, usize> {
        self.columns.as_ref()
    }
    pub(crate) fn r(&self) -> SparseColMat<usize, T> {
        self.r.clone()
    }
    pub(crate) fn q(&self) -> Result<Mat<T>, String> {
        self.head.q_full()
    }
    pub(crate) fn permutation(&self) -> Perm<usize> {
        self.columns.clone()
    }
    pub(crate) fn reconstruct(&self) -> Result<Mat<T>, String> {
        let product = &self.q()? * &self.r.to_dense();
        let inverse = self.columns.as_ref().arrays().1;
        Ok(Mat::from_fn(self.m, self.n, |i, j| {
            copy(&product[(i, inverse[j])])
        }))
    }
    /// Construction temporaries of `pinv(R)`: dense R, its thin SVD and
    /// faer's SVD/reconstruction workspace. Only the `n x m` result is kept.
    pub(crate) fn inverse_r_scratch(m: usize, n: usize, par: ParShape) -> StackReq {
        if m == 0 {
            return StackReq::EMPTY;
        }
        let thin = faer::linalg::svd::ComputeSvdVectors::Thin;
        StackReq::all_of(&[
            faer::linalg::temp_mat_scratch::<T>(m, n),
            faer::linalg::temp_mat_scratch::<T>(m, 1),
            faer::linalg::temp_mat_scratch::<T>(m, m),
            faer::linalg::temp_mat_scratch::<T>(n, m),
            StackReq::any_of(&[
                faer::linalg::svd::svd_scratch::<T>(m, n, thin, thin, par, Default::default()),
                faer::linalg::svd::pseudoinverse_from_svd_scratch::<T>(m, n, par),
            ]),
        ])
    }
    fn inverse_r(&self) -> Result<Mat<T>, String> {
        let req = Self::inverse_r_scratch(self.m, self.n, Par::Seq.shape());
        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        self.inverse_r_in(Par::Seq, MemStack::new(&mut mem))
    }
    /// `pinv(R)` with every temporary in `stack`; the result is the only
    /// allocation besides faer's matrix-product internals.
    pub(crate) fn inverse_r_in(
        &self,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Mat<T>, String> {
        let (m, n) = (self.m, self.n);
        let mut out = Mat::zeros(n, m);
        if m == 0 {
            return Ok(out);
        }
        let (mut dense, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, n, stack);
        let (mut s, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, 1, stack);
        let (mut u, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, m, stack);
        let (mut v, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, m, stack);
        let mut dense = dense.as_mat_mut();
        let r = self.r.as_ref();
        for j in 0..n {
            for (i, x) in r.row_idx_of_col(j).zip(r.val_of_col(j)) {
                dense[(i, j)] = copy(x);
            }
        }
        let (mut s, mut u, mut v) = (s.as_mat_mut(), u.as_mat_mut(), v.as_mat_mut());
        faer::linalg::svd::svd(
            dense.rb(),
            s.rb_mut().col_mut(0).as_diagonal_mut(),
            Some(u.rb_mut()),
            Some(v.rb_mut()),
            par.reborrow(),
            stack,
            Default::default(),
        )
        .map_err(|e| format!("{e:?}"))?;
        faer::linalg::svd::pseudoinverse_from_svd_with_tolerance(
            out.as_mut(),
            s.rb().col(0).as_diagonal(),
            u.rb(),
            v.rb(),
            zero::<T::Real>(),
            eps::<T::Real>() * from_f64::<T::Real>(m.min(n) as f64),
            par,
            stack,
        );
        Ok(out)
    }
    /// Inverse-action state: an independent head factor and permutation plus
    /// `pinv(R)` built in caller scratch. The saved sparse R is not cloned.
    pub(crate) fn action_parts_in(
        &self,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<(sparse_qr_owned::Factors<T>, Mat<T>, Perm<usize>), String> {
        let inverse = self.inverse_r_in(par, stack)?;
        Ok((self.head.clone(), inverse, self.columns.clone()))
    }
    pub(crate) fn solve_transpose(
        &self,
        b: MatRef<'_, T>,
        adjoint: bool,
    ) -> Result<Mat<T>, String> {
        if b.nrows() != self.n {
            return Err("transposed wide QR RHS shape mismatch".into());
        }
        let p = self.columns.as_ref().arrays().0;
        let rhs = Mat::from_fn(self.n, b.ncols(), |i, j| copy(&b[(p[i], j)]));
        let inverse = self.inverse_r()?;
        let projected = if adjoint {
            inverse.adjoint() * &rhs
        } else {
            inverse.transpose() * &rhs
        };
        self.head.apply_q(projected.as_ref(), !adjoint)
    }
    pub(crate) fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, String> {
        if b.nrows() != self.m {
            return Err("wide QR RHS shape mismatch".into());
        }
        let projected = self.head.apply_full_q_adjoint(b)?;
        let solution = &self.inverse_r()? * &projected;
        let inverse = self.columns.as_ref().arrays().1;
        Ok(Mat::from_fn(self.n, b.ncols(), |i, j| {
            copy(&solution[(inverse[i], j)])
        }))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::linalg::SupernodalThreshold;
    fn check<T: ComplexField>(tol: f64, value: fn(f64, f64) -> T) {
        for (m, n) in [(0, 3), (1, 3), (3, 5), (5, 12)] {
            for empty_head in [false, true] {
                let a = Mat::from_fn(m, n, |i, j| {
                    if empty_head && j < m {
                        zero::<T>()
                    } else {
                        value(
                            if j % m.max(1) == i {
                                3.0 + j as f64
                            } else {
                                0.1
                            },
                            (i + j) as f64 * 0.05,
                        )
                    }
                });
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| (0..m).map(move |i| (i, j)))
                    .filter(|&(i, j)| a[(i, j)] != zero::<T>())
                    .map(|(i, j)| Triplet {
                        row: i,
                        col: j,
                        val: copy(&a[(i, j)]),
                    })
                    .collect();
                let sparse =
                    SparseColMat::<usize, T>::try_new_from_triplets(m, n, &entries).unwrap();
                for threshold in [
                    SupernodalThreshold::AUTO,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                ] {
                    for reverse in [false, true] {
                        let order: Vec<_> = (0..n)
                            .map(|i| if reverse { n - 1 - i } else { i })
                            .collect();
                        let order = Perm::new_checked(
                            order.clone().into_boxed_slice(),
                            order.into_boxed_slice(),
                            n,
                        );
                        let plan = Plan::new(
                            sparse.symbolic(),
                            order,
                            QrSymbolicParams {
                                supernodal_flop_ratio_threshold: threshold,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        let f = Factors::new(&plan, sparse.as_ref()).unwrap();
                        assert!(
                            (&f.reconstruct().unwrap() - &a).norm_max() < from_f64::<T::Real>(tol)
                        );
                        let b = Mat::from_fn(m, 2, |i, j| value((i + j + 1) as f64, 0.1));
                        let expected = &a.thin_svd(Par::Seq).unwrap().pseudoinverse(Par::Seq) * &b;
                        let actual = f.solve(b.as_ref()).unwrap();
                        assert!((&actual - &expected).norm_max() < from_f64::<T::Real>(tol));
                    }
                }
            }
        }
    }
    #[test]
    fn f32_wide() {
        check::<f32>(1e-3, |r, _| r as f32);
    }
    #[test]
    fn f64_wide() {
        check::<f64>(1e-10, |r, _| r);
    }
    #[test]
    fn c32_wide() {
        check::<faer::c32>(1e-3, |r, i| faer::c32::new(r as f32, i as f32));
    }
    #[test]
    fn c64_wide() {
        check::<faer::c64>(1e-10, faer::c64::new);
    }
}

#[cfg(test)]
mod cutoff_tests {
    use super::*;
    #[test]
    fn wide_default_uses_minimum_dimension_tolerance() {
        let tiny = 10.0 * f64::EPSILON;
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            100,
            &[
                Triplet {
                    row: 0,
                    col: 0,
                    val: 1.0,
                },
                Triplet {
                    row: 1,
                    col: 1,
                    val: tiny,
                },
            ],
        )
        .unwrap();
        let p: Vec<_> = (0..100).collect();
        let p = Perm::new_checked(p.clone().into_boxed_slice(), p.into_boxed_slice(), 100);
        let plan = Plan::new(a.symbolic(), p, Default::default()).unwrap();
        let f = Factors::new(&plan, a.as_ref()).unwrap();
        let x = f.solve(Mat::from_fn(2, 1, |_, _| 1.0).as_ref()).unwrap();
        assert!((x[(1, 0)] * tiny - 1.0).abs() < 1e-12);
    }
}
