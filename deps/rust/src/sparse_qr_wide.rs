//! Wide QR assembled from native prefix QR and saved-reflector projections.
use super::sparse_qr_owned;
use faer::perm::Perm;
use faer::sparse::linalg::qr::{self, QrSymbolicParams, SymbolicQr};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMatRef, Triplet};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par};
use std::sync::Arc;
#[derive(Clone)]
pub(crate) struct Plan {
    head: Arc<SymbolicQr<usize>>,
    columns: Perm<usize>,
    m: usize,
    n: usize,
}
impl Plan {
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
    pub(crate) fn new(plan: &Plan, a: SparseColMatRef<'_, usize, T>) -> Result<Self, String> {
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
        let head = sparse_qr_owned::Factors::new(plan.head.clone(), prefix.as_ref())?;
        let left = head.r()?;
        let mut entries: Vec<_> = left
            .as_ref()
            .triplet_iter()
            .map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: copy(e.val),
            })
            .collect();
        let mut column = Mat::<T>::zeros(m, 1);
        // Project one column at a time: no dense Q or m-by-(n-m) temporary.
        for j in m..n {
            column.fill(zero::<T>());
            for (i, v) in a.row_idx_of_col(p[j]).zip(a.val_of_col(p[j])) {
                column[(i, 0)] = copy(v);
            }
            let projected = head.apply_full_q_adjoint(column.as_ref())?;
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
    fn inverse_r(&self) -> Result<Mat<T>, String> {
        let dense = self.r.to_dense();
        let f = dense.thin_svd(Par::Seq).map_err(|e| format!("{e:?}"))?;
        let mut out = Mat::zeros(self.n, self.m);
        let req = faer::linalg::svd::pseudoinverse_from_svd_scratch::<T>(
            self.m,
            self.n,
            Par::Seq.shape(),
        );
        let mut mem = faer::dyn_stack::MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
        faer::linalg::svd::pseudoinverse_from_svd_with_tolerance(
            out.as_mut(),
            f.S(),
            f.U(),
            f.V(),
            zero::<T::Real>(),
            eps::<T::Real>() * from_f64::<T::Real>(self.m.min(self.n) as f64),
            Par::Seq,
            faer::dyn_stack::MemStack::new(&mut mem),
        );
        Ok(out)
    }
    pub(crate) fn into_action_parts(
        self,
    ) -> Result<(sparse_qr_owned::Factors<T>, Mat<T>, Perm<usize>), String> {
        let inverse = self.inverse_r()?;
        Ok((self.head, inverse, self.columns))
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
