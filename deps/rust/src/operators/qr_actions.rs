//! Saved sparse QR inverse actions with prepared projections and caller scratch.
use crate::sparse_qr_owned::projection::Projection;
use faer::dyn_stack::{MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::matrix_free::{BiLinOp, LinOp};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::SparseColMat;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par, ParShape};
#[derive(Clone)]
pub(super) struct OwnedQrInverse<T> {
    projection: Projection<T>,
    r: SparseColMat<usize, T>,
    columns: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for OwnedQrInverse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OwnedSparseQrInverse")
    }
}
impl<T: ComplexField> OwnedQrInverse<T> {
    pub(super) fn new(f: crate::sparse_qr_owned::Factors<T>) -> Result<Self, String> {
        let r = f.r()?;
        for j in 0..r.ncols() {
            if r.as_ref()
                .row_idx_of_col(j)
                .zip(r.as_ref().val_of_col(j))
                .find(|(i, _)| *i == j)
                .is_none_or(|(_, v)| abs(v) == zero::<T::Real>())
            {
                return Err("sparse QR inverse action requires full column rank".into());
            }
        }
        let columns = f.permutation();
        Ok(Self {
            projection: Projection::new(f),
            r,
            columns,
        })
    }
    pub(super) fn map_values<U: ComplexField>(
        &self,
        cast: impl Fn(&T) -> U,
    ) -> Result<OwnedQrInverse<U>, String> {
        Ok(OwnedQrInverse {
            projection: self.projection.map_values(&cast),
            r: SparseColMat::new(
                self.r.symbolic().to_owned().map_err(|e| format!("{e:?}"))?,
                self.r.val().iter().map(cast).collect(),
            ),
            columns: self.columns.clone(),
        })
    }
    fn scratch(&self, k: usize) -> StackReq {
        if self.r.nrows() == 0 || k == 0 {
            return StackReq::EMPTY;
        }
        faer::linalg::temp_mat_scratch::<T>(self.r.nrows(), k).and(self.projection.scratch(k))
    }
    fn run<const TR: bool, const C: bool>(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) {
        let n = self.r.nrows();
        let k = b.ncols();
        if n == 0 || k == 0 {
            out.fill(zero::<T>());
            return;
        }
        // SAFETY: all z entries are initialized by projection or permutation
        // before the sparse triangular solve reads the temporary.
        let (mut z, stack) = unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(n, k, stack) };
        let mut z = z.as_mat_mut();
        let conj = if C { Conj::Yes } else { Conj::No };
        let (p, inv) = self.columns.as_ref().arrays();
        if TR {
            for j in 0..k {
                for i in 0..n {
                    z[(i, j)] = copy(&b[(p[i], j)]);
                }
            }
            faer::sparse::linalg::triangular_solve::solve_upper_triangular_transpose_in_place(
                self.r.as_ref(),
                conj,
                z.rb_mut(),
                par.reborrow(),
            );
            if C {
                self.projection
                    .apply::<false, false, false>(out, z.rb(), par, stack);
            } else {
                self.projection
                    .apply::<false, true, false>(out, z.rb(), par, stack);
            }
        } else {
            self.projection
                .apply::<true, C, false>(z.rb_mut(), b, par.reborrow(), stack);
            faer::sparse::linalg::triangular_solve::solve_upper_triangular_in_place(
                self.r.as_ref(),
                conj,
                z.rb_mut(),
                par,
            );
            for j in 0..k {
                for i in 0..n {
                    out[(i, j)] = copy(&z[(inv[i], j)]);
                }
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for OwnedQrInverse<T> {
    fn nrows(&self) -> usize {
        self.r.ncols()
    }
    fn ncols(&self) -> usize {
        self.projection.dimensions().0
    }
    fn apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, false>(o, b, p, s)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, true>(o, b, p, s)
    }
}
impl<T: ComplexField> BiLinOp<T> for OwnedQrInverse<T> {
    fn transpose_apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, false>(o, b, p, s)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, true>(o, b, p, s)
    }
}
#[derive(Clone)]
pub(super) struct WideQrInverse<T> {
    projection: Projection<T>,
    inverse: Mat<T>,
    columns: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for WideQrInverse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WideSparseQrInverse")
    }
}
impl<T: ComplexField> WideQrInverse<T> {
    /// Construction scratch: dense R, its SVD and the `pinv(R)` reconstruction.
    pub(super) fn construction_scratch(m: usize, n: usize, par: ParShape) -> StackReq {
        crate::sparse_qr_wide::Factors::<T>::inverse_r_scratch(m, n, par)
    }
    /// The retained `pinv(R)` (`n x m`) is materialized. Retaining rank-`r`
    /// `U` (`m x r`) and `V` (`n x r`) instead is cheaper in storage and flops
    /// only when `r * (m + n) < m * n`; at full rank it costs more storage and
    /// an extra product per application.
    pub(super) fn new_in(
        f: &crate::sparse_qr_wide::Factors<T>,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Self, String> {
        let (head, inverse, columns) = f.action_parts_in(par, stack)?;
        Ok(Self {
            projection: Projection::new(head),
            inverse,
            columns,
        })
    }
    pub(super) fn map_values<U: ComplexField>(
        &self,
        cast: impl Fn(&T) -> U,
    ) -> Result<WideQrInverse<U>, String> {
        Ok(WideQrInverse {
            projection: self.projection.map_values(&cast),
            inverse: Mat::from_fn(self.inverse.nrows(), self.inverse.ncols(), |i, j| {
                cast(&self.inverse[(i, j)])
            }),
            columns: self.columns.clone(),
        })
    }
    fn scratch(&self, k: usize) -> StackReq {
        let (n, m) = self.inverse.shape();
        if m == 0 || k == 0 {
            return StackReq::EMPTY;
        }
        StackReq::all_of(&[
            faer::linalg::temp_mat_scratch::<T>(m, k),
            faer::linalg::temp_mat_scratch::<T>(n, k),
            self.projection.scratch(k),
        ])
    }
    fn run<const TR: bool, const C: bool>(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) {
        let (n, m) = self.inverse.shape();
        let k = b.ncols();
        if m == 0 || k == 0 {
            out.fill(zero::<T>());
            return;
        }
        // SAFETY: each temporary is fully initialized by coordinate copying,
        // projection or matmul(Replace) before its values are read.
        let (mut small, stack) = unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(m, k, stack) };
        let (mut large, stack) = unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(n, k, stack) };
        let mut small = small.as_mat_mut();
        let mut large = large.as_mat_mut();
        let (p, inv) = self.columns.as_ref().arrays();
        let conj = if C { Conj::Yes } else { Conj::No };
        if TR {
            for j in 0..k {
                for i in 0..n {
                    large[(i, j)] = copy(&b[(p[i], j)]);
                }
            }
            faer::linalg::matmul::matmul_with_conj(
                small.rb_mut(),
                Accum::Replace,
                self.inverse.as_ref().transpose(),
                conj,
                large.rb(),
                Conj::No,
                one::<T>(),
                par.reborrow(),
            );
            if C {
                self.projection
                    .apply::<false, false, true>(out, small.rb(), par, stack);
            } else {
                self.projection
                    .apply::<false, true, true>(out, small.rb(), par, stack);
            }
        } else {
            self.projection
                .apply::<true, C, true>(small.rb_mut(), b, par.reborrow(), stack);
            faer::linalg::matmul::matmul_with_conj(
                large.rb_mut(),
                Accum::Replace,
                self.inverse.as_ref(),
                conj,
                small.rb(),
                Conj::No,
                one::<T>(),
                par,
            );
            for j in 0..k {
                for i in 0..n {
                    out[(i, j)] = copy(&large[(inv[i], j)]);
                }
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for WideQrInverse<T> {
    fn nrows(&self) -> usize {
        self.inverse.nrows()
    }
    fn ncols(&self) -> usize {
        self.inverse.ncols()
    }
    fn apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, false>(o, b, p, s)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, true>(o, b, p, s)
    }
}
impl<T: ComplexField> BiLinOp<T> for WideQrInverse<T> {
    fn transpose_apply_scratch(&self, k: usize, _p: ParShape) -> StackReq {
        self.scratch(k)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, false>(o, b, p, s)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, true>(o, b, p, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::dyn_stack::MemBuffer;
    use faer::sparse::linalg::{
        qr::{self, QrSymbolicParams},
        SupernodalThreshold,
    };
    use faer::sparse::{SparseColMat, Triplet};
    use std::sync::Arc;
    fn exercise<T: ComplexField, O: BiLinOp<T>>(
        op: &O,
        expected: MatRef<'_, T>,
        rhs: MatRef<'_, T>,
        trans: bool,
        conjugated: bool,
    ) {
        let req = if trans {
            op.transpose_apply_scratch(rhs.ncols(), ParShape::Seq)
        } else {
            op.apply_scratch(rhs.ncols(), ParShape::Seq)
        };
        let mut memory = MemBuffer::new(req);
        let mut out = Mat::zeros(expected.nrows(), expected.ncols());
        let mut run = || {
            let stack = MemStack::new(&mut memory);
            if trans {
                if conjugated {
                    op.adjoint_apply(out.as_mut(), rhs, Par::Seq, stack);
                } else {
                    op.transpose_apply(out.as_mut(), rhs, Par::Seq, stack);
                }
            } else if conjugated {
                op.conj_apply(out.as_mut(), rhs, Par::Seq, stack);
            } else {
                op.apply(out.as_mut(), rhs, Par::Seq, stack);
            }
        };
        run();
        let allocations = crate::allocation_probe::count(|| {
            run();
            run();
        });
        assert_eq!(
            allocations, 0,
            "prepared sparse QR application allocated on the calling thread"
        );
        drop(run);
        let mut team = crate::teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut parallel = || {
            team.execute(|par| {
                let actual = if trans {
                    op.transpose_apply_scratch(rhs.ncols(), par.shape())
                } else {
                    op.apply_scratch(rhs.ncols(), par.shape())
                };
                assert!(actual.size_bytes() <= req.size_bytes());
                let stack = MemStack::new(&mut memory);
                if trans {
                    if conjugated {
                        op.adjoint_apply(out.as_mut(), rhs, par, stack);
                    } else {
                        op.transpose_apply(out.as_mut(), rhs, par, stack);
                    }
                } else if conjugated {
                    op.conj_apply(out.as_mut(), rhs, par, stack);
                } else {
                    op.apply(out.as_mut(), rhs, par, stack);
                }
            })
            .unwrap()
        };
        parallel();
        let allocations = crate::allocation_probe::count(|| {
            parallel();
            parallel();
        });
        assert_eq!(
            allocations, 0,
            "prepared sparse QR team application allocated on the calling thread"
        );
        drop(parallel);
        team.team_close().unwrap();
        let error = (&out - expected).norm_max();
        let bound = from_f64::<T::Real>(500.0 * expected.nrows().max(rhs.nrows()).max(1) as f64)
            * eps::<T::Real>()
            * if expected.norm_max() > one::<T::Real>() {
                expected.norm_max()
            } else {
                one::<T::Real>()
            };
        assert!(
            error < bound,
            "sparse QR action residual {error:?}, bound {bound:?}"
        );
    }
    fn sparse<T: ComplexField>(
        m: usize,
        n: usize,
        empty: bool,
        value: fn(f64, f64) -> T,
    ) -> SparseColMat<usize, T> {
        let mut entries = Vec::new();
        for j in 0..n {
            for i in 0..m {
                let v = if empty && j < m {
                    zero::<T>()
                } else {
                    value(
                        if i == j % m.max(1) {
                            4.0 + j as f64
                        } else {
                            0.1
                        },
                        ((i + 2 * j) % 3) as f64 * 0.05,
                    )
                };
                if v != zero::<T>() {
                    entries.push(Triplet {
                        row: i,
                        col: j,
                        val: v,
                    });
                }
            }
        }
        SparseColMat::try_new_from_triplets(m, n, &entries).unwrap()
    }
    fn owned<T: ComplexField>(value: fn(f64, f64) -> T) {
        for threshold in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            let a = sparse(9, 5, false, value);
            let symbolic = Arc::new(
                qr::factorize_symbolic_qr(
                    a.symbolic(),
                    QrSymbolicParams {
                        supernodal_flop_ratio_threshold: threshold,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let f = crate::sparse_qr_owned::Factors::new(symbolic, a.as_ref()).unwrap();
            let op = OwnedQrInverse::new(f.clone()).unwrap();
            for (t, c) in [(false, false), (false, true), (true, false), (true, true)] {
                let b = Mat::from_fn(if t { 5 } else { 9 }, 3, |i, j| {
                    value((i + j + 1) as f64, 0.1)
                });
                let expected = if t {
                    f.solve_transpose(b.as_ref(), c).unwrap()
                } else if c {
                    f.solve(b.conjugate().to_owned().as_ref())
                        .unwrap()
                        .conjugate()
                        .to_owned()
                } else {
                    f.solve(b.as_ref()).unwrap()
                };
                exercise(&op, expected.as_ref(), b.as_ref(), t, c);
            }
        }
    }
    fn wide_action<T: ComplexField>(f: &crate::sparse_qr_wide::Factors<T>) -> WideQrInverse<T> {
        let (m, n) = f.dimensions();
        let req = WideQrInverse::<T>::construction_scratch(m, n, ParShape::Seq);
        let mut memory = MemBuffer::new(req);
        WideQrInverse::new_in(f, Par::Seq, MemStack::new(&mut memory)).unwrap()
    }
    /// faer's dense SVD plus `pinv` reconstruction alone, with outputs and
    /// workspace preallocated: above small sizes its only allocations are
    /// matrix-product packing buffers (t4a-faer upstream-reports/gemm-001).
    fn pinv_kernel_allocations<T: ComplexField>(r: MatRef<'_, T>) -> usize {
        let (m, n) = r.shape();
        let thin = faer::linalg::svd::ComputeSvdVectors::Thin;
        let req = faer::linalg::svd::svd_scratch::<T>(
            m,
            n,
            thin,
            thin,
            ParShape::Seq,
            Default::default(),
        )
        .or(faer::linalg::svd::pseudoinverse_from_svd_scratch::<T>(
            m,
            n,
            ParShape::Seq,
        ));
        let mut memory = MemBuffer::new(req);
        let (mut s, mut u, mut v) = (Mat::zeros(m, 1), Mat::zeros(m, m), Mat::zeros(n, m));
        let mut out = Mat::zeros(n, m);
        let mut run = || {
            faer::linalg::svd::svd(
                r,
                s.as_mut().col_mut(0).as_diagonal_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                MemStack::new(&mut memory),
                Default::default(),
            )
            .unwrap();
            faer::linalg::svd::pseudoinverse_from_svd_with_tolerance(
                out.as_mut(),
                s.as_ref().col(0).as_diagonal(),
                u.as_ref(),
                v.as_ref(),
                zero::<T::Real>(),
                eps::<T::Real>(),
                Par::Seq,
                MemStack::new(&mut memory),
            );
        };
        run();
        crate::allocation_probe::count(run)
    }
    fn wide_construction<T: ComplexField>(value: fn(f64, f64) -> T) {
        let mut team = crate::teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for (m, n) in [(3, 6), (24, 60)] {
            let a = sparse(m, n, false, value);
            let order: Vec<_> = (0..n).rev().collect();
            let columns = faer::perm::Perm::new_checked(
                order.clone().into_boxed_slice(),
                order.into_boxed_slice(),
                n,
            );
            let plan = crate::sparse_qr_wide::Plan::new(a.symbolic(), columns, Default::default())
                .unwrap();
            let f = crate::sparse_qr_wide::Factors::new(&plan, a.as_ref()).unwrap();
            // Retained state other than pinv(R): the independent head factor,
            // its prepared projection and the column permutation.
            let retained = crate::allocation_probe::count(|| {
                let head = Projection::new(f.head_ref().clone());
                let columns = f.permutation();
                drop((head, columns));
            });
            let kernel = pinv_kernel_allocations(f.r_ref().to_dense().as_ref());
            if m < 8 {
                assert_eq!(kernel, 0);
            }
            let req = WideQrInverse::<T>::construction_scratch(m, n, ParShape::Seq);
            let mut memory = MemBuffer::new(req);
            let mut op = None;
            let count = crate::allocation_probe::count(|| {
                op = Some(WideQrInverse::new_in(&f, Par::Seq, MemStack::new(&mut memory)).unwrap());
            });
            // `+ 1` is the retained n x m pinv(R).
            assert!(
                count <= retained + 1 + kernel,
                "wide QR action construction allocated {count} for {m}x{n} \
                 (retained {retained} + 1, kernel alone {kernel})"
            );
            let op = op.unwrap();
            let parallel = team
                .execute(|par| {
                    let req = WideQrInverse::<T>::construction_scratch(m, n, par.shape());
                    let mut memory = MemBuffer::new(req);
                    WideQrInverse::new_in(&f, par, MemStack::new(&mut memory)).unwrap()
                })
                .unwrap();
            let bound = from_f64::<T::Real>(1e3) * eps::<T::Real>();
            assert!((&op.inverse - &parallel.inverse).norm_max() < bound);
            let b = Mat::from_fn(m, 2, |i, j| value((i + j + 1) as f64, 0.1));
            exercise(
                &op,
                f.solve(b.as_ref()).unwrap().as_ref(),
                b.as_ref(),
                false,
                false,
            );
        }
        team.team_close().unwrap();
    }
    #[test]
    fn f32_wide_construction_uses_caller_scratch() {
        wide_construction::<f32>(|r, _| r as f32)
    }
    #[test]
    fn f64_wide_construction_uses_caller_scratch() {
        wide_construction::<f64>(|r, _| r)
    }
    #[test]
    fn c32_wide_construction_uses_caller_scratch() {
        wide_construction::<faer::c32>(|r, i| faer::c32::new(r as f32, i as f32))
    }
    #[test]
    fn c64_wide_construction_uses_caller_scratch() {
        wide_construction::<faer::c64>(faer::c64::new)
    }
    fn wide<T: ComplexField>(value: fn(f64, f64) -> T) {
        for threshold in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            for empty in [false, true] {
                let a = sparse(3, 6, empty, value);
                let p: Vec<_> = if empty {
                    (0..6).collect()
                } else {
                    vec![2, 0, 1, 4, 5, 3]
                };
                let mut inv = vec![0; 6];
                for (i, &j) in p.iter().enumerate() {
                    inv[j] = i;
                }
                let columns =
                    faer::perm::Perm::new_checked(p.into_boxed_slice(), inv.into_boxed_slice(), 6);
                let plan = crate::sparse_qr_wide::Plan::new(
                    a.symbolic(),
                    columns,
                    QrSymbolicParams {
                        supernodal_flop_ratio_threshold: threshold,
                        ..Default::default()
                    },
                )
                .unwrap();
                let f = crate::sparse_qr_wide::Factors::new(&plan, a.as_ref()).unwrap();
                let op = wide_action(&f);
                for (t, c) in [(false, false), (false, true), (true, false), (true, true)] {
                    let b = Mat::from_fn(if t { 6 } else { 3 }, 3, |i, j| {
                        value((i + j + 1) as f64, 0.1)
                    });
                    let expected = if t {
                        f.solve_transpose(b.as_ref(), c).unwrap()
                    } else if c {
                        f.solve(b.conjugate().to_owned().as_ref())
                            .unwrap()
                            .conjugate()
                            .to_owned()
                    } else {
                        f.solve(b.as_ref()).unwrap()
                    };
                    exercise(&op, expected.as_ref(), b.as_ref(), t, c);
                }
            }
        }
    }
    #[test]
    fn f32_owned_no_application_allocations() {
        owned::<f32>(|r, _| r as f32)
    }
    #[test]
    fn f64_owned_no_application_allocations() {
        owned::<f64>(|r, _| r)
    }
    #[test]
    fn c32_owned_no_application_allocations() {
        owned::<faer::c32>(|r, i| faer::c32::new(r as f32, i as f32))
    }
    #[test]
    fn c64_owned_no_application_allocations() {
        owned::<faer::c64>(faer::c64::new)
    }
    #[test]
    fn f32_wide_no_application_allocations() {
        wide::<f32>(|r, _| r as f32)
    }
    #[test]
    fn f64_wide_no_application_allocations() {
        wide::<f64>(|r, _| r)
    }
    #[test]
    fn c32_wide_no_application_allocations() {
        wide::<faer::c32>(|r, i| faer::c32::new(r as f32, i as f32))
    }
    #[test]
    fn c64_wide_no_application_allocations() {
        wide::<faer::c64>(faer::c64::new)
    }
}
