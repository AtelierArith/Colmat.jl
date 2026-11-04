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
    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (self.symbolic.nrows(), self.symbolic.ncols())
    }

    pub(crate) fn map_values<U: ComplexField>(&self, cast: impl Fn(&T) -> U) -> Factors<U> {
        Factors {
            symbolic: self.symbolic.clone(),
            indices: self.indices.clone(),
            values: self.values.iter().map(cast).collect(),
        }
    }

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
    // These immutable slices are the outputs of factorize_numeric_qr for this
    // exact symbolic owner. Split them using the public symbolic lengths, just
    // as QrRef does, without accessing private fields or guessing Rust layout.
    fn simplicial(
        &self,
    ) -> Option<faer::sparse::linalg::qr::simplicial::SimplicialQrRef<'_, usize, T>> {
        use faer::sparse::{
            linalg::qr::{simplicial::SimplicialQrRef, SymbolicQrRaw},
            SymbolicSparseColMatRef,
        };
        let SymbolicQrRaw::Simplicial(s) = self.symbolic.raw() else {
            return None;
        };
        let (m, n) = (self.symbolic.nrows(), self.symbolic.ncols());
        let (rp, idx) = self.indices.split_at(n + 1);
        let (ri, idx) = idx.split_at(s.len_r());
        let (hp, idx) = idx.split_at(n + 1);
        let (hi, _) = idx.split_at(s.len_householder());
        let (rv, val) = self.values.split_at(s.len_r());
        let (hv, val) = val.split_at(s.len_householder());
        let (tau, _) = val.split_at(n);
        // The numeric kernel initializes these unsorted CSC structures.
        let r = SparseColMatRef::new(
            unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, rp, None, ri) },
            rv,
        );
        let h = SparseColMatRef::new(
            unsafe { SymbolicSparseColMatRef::new_unchecked(m, n, hp, None, hi) },
            hv,
        );
        Some(SimplicialQrRef::new(s, r, h, tau))
    }
    fn supernodal(&self) -> Option<SuperParts<'_, T>> {
        use faer::sparse::linalg::qr::SymbolicQrRaw;
        let SymbolicQrRaw::Supernodal(s) = self.symbolic.raw() else {
            return None;
        };
        let h = s.householder();
        let len = h.len_householder_row_idx();
        let blocks = len + h.n_supernodes();
        let (rows, idx) = self.indices.split_at(len);
        let (bs, idx) = idx.split_at(blocks);
        let (nr, idx) = idx.split_at(blocks);
        let (nc, _) = idx.split_at(blocks);
        let (r, val) = self.values.split_at(s.R_adjoint().len_val());
        let (hv, val) = val.split_at(h.len_householder_val());
        let (tau, _) = val.split_at(h.len_tau_val());
        Some(SuperParts {
            symbolic: s,
            rows,
            bs,
            nr,
            nc,
            r,
            h: hv,
            tau,
        })
    }
    pub(crate) fn r(&self) -> Result<faer::sparse::SparseColMat<usize, T>, String> {
        use faer::sparse::{SparseColMat, Triplet};
        use faer::traits::math_utils::{conj, copy};
        let mut entries = Vec::new();
        if let Some(f) = self.simplicial() {
            entries.extend(f.R().triplet_iter().map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: copy(e.val),
            }));
        } else {
            let f = self.supernodal().unwrap();
            let l = faer::sparse::linalg::cholesky::supernodal::SupernodalLltRef::new(
                f.symbolic.R_adjoint(),
                f.r,
            );
            for s in 0..f.symbolic.R_adjoint().n_supernodes() {
                let node = l.supernode(s);
                let a = node.val();
                let w = a.ncols();
                for j in 0..w {
                    for i in j..w {
                        entries.push(Triplet {
                            row: node.start() + j,
                            col: node.start() + i,
                            val: conj(&a[(i, j)]),
                        });
                    }
                    for (i, &row) in node.pattern().iter().enumerate() {
                        entries.push(Triplet {
                            row: node.start() + j,
                            col: row,
                            val: conj(&a[(w + i, j)]),
                        });
                    }
                }
            }
        }
        SparseColMat::try_new_from_triplets(self.symbolic.ncols(), self.symbolic.ncols(), &entries)
            .map_err(|e| format!("{e:?}"))
    }
    pub(crate) fn apply_q(&self, b: MatRef<'_, T>, conjugated: bool) -> Result<Mat<T>, String> {
        self.apply_q_basis(b, conjugated, false)
    }
    fn apply_q_basis(
        &self,
        b: MatRef<'_, T>,
        conjugated: bool,
        full: bool,
    ) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::{add, conj, copy, mul, mul_real, real, recip, sub};
        let (m, n) = (self.symbolic.nrows(), self.symbolic.ncols());
        if b.nrows() != n {
            return Err("sparse QR Q RHS row mismatch".into());
        }
        let mut out = Mat::zeros(m, b.ncols());

        if let Some(f) = self.simplicial() {
            let h = f.householder();
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
                        out[(pivot, k)] = copy(&b[(j, k)]);
                    }
                }
            }
            if complete {
                complete_q_seed(out.as_mut(), b, &mut used, &assigned, full);
            }
            for j in (0..n).rev() {
                let rows = h.row_idx_of_col_raw(j);
                let vals = h.val_of_col(j);
                if rows.is_empty() {
                    continue;
                }
                let scale = recip(&real(&f.tau_val()[j]));
                for k in 0..b.ncols() {
                    let mut dot = zero::<T>();
                    for (&i, v) in rows.iter().zip(vals) {
                        let v = if conjugated { copy(v) } else { conj(v) };
                        dot = add(&dot, &mul(&v, &out[(i, k)]));
                    }
                    dot = mul_real(&dot, &scale);
                    for (&i, v) in rows.iter().zip(vals) {
                        let v = if conjugated { conj(v) } else { copy(v) };
                        out[(i, k)] = sub(&out[(i, k)], &mul(&v, &dot));
                    }
                }
            }
        } else {
            let f = self.supernodal().unwrap();
            let h = f.symbolic.householder();
            let l = f.symbolic.R_adjoint();
            let rp = h.col_ptr_for_householder_row_idx();
            let complete = full
                || (0..h.n_supernodes())
                    .any(|s| l.supernode_end()[s] - l.supernode_begin()[s] > rp[s + 1] - rp[s]);
            let mut used = if complete { vec![false; m] } else { Vec::new() };
            let mut assigned = if complete { vec![false; n] } else { Vec::new() };
            // The first supernode-width rows are the coordinates retained in R.
            for s in 0..h.n_supernodes() {
                let start = l.supernode_begin()[s];
                let width = l.supernode_end()[s] - start;
                for i in 0..width.min(rp[s + 1] - rp[s]) {
                    if complete {
                        used[f.rows[rp[s] + i]] = true;
                        assigned[start + i] = true;
                    }
                    for k in 0..b.ncols() {
                        out[(f.rows[rp[s] + i], k)] = copy(&b[(start + i, k)]);
                    }
                }
            }
            if complete {
                complete_q_seed(out.as_mut(), b, &mut used, &assigned, full);
            }
            // Record actual numeric block dimensions (not a guessed fixed block
            // size); inverse application reverses nodes and blocks separately.
            let mut nodes = Vec::new();
            let mut block = 0;
            for s in 0..h.n_supernodes() {
                let rows = rp[s + 1] - rp[s];
                let hv = &f.h
                    [h.col_ptr_for_householder_val()[s]..h.col_ptr_for_householder_val()[s + 1]];
                let cols = if rows == 0 { 0 } else { hv.len() / rows };
                let mut blocks = Vec::new();
                let mut start = 0;
                while start < cols {
                    blocks.push((start, f.bs[block], f.nr[block], f.nc[block]));
                    start += f.nc[block];
                    block += 1;
                    if start >= rows {
                        break;
                    }
                }
                nodes.push(blocks);
            }
            for s in (0..h.n_supernodes()).rev() {
                let rows = &f.rows[rp[s]..rp[s + 1]];
                let hv = &f.h
                    [h.col_ptr_for_householder_val()[s]..h.col_ptr_for_householder_val()[s + 1]];
                let cols = if rows.is_empty() {
                    0
                } else {
                    hv.len() / rows.len()
                };
                let basis = MatRef::from_column_major_slice(hv, rows.len(), cols);
                let tv = &f.tau[h.col_ptr_for_tau_val()[s]..h.col_ptr_for_tau_val()[s + 1]];
                let rank = rows.len().min(cols);
                let max_bs = if rank == 0 { 0 } else { tv.len() / rank };
                let tau = MatRef::from_column_major_slice(tv, max_bs, rank);
                let mut work = Mat::from_fn(rows.len(), b.ncols(), |i, j| copy(&out[(rows[i], j)]));
                for &(start, bs, nr, nc) in nodes[s].iter().rev() {
                    if nc == 0 {
                        continue;
                    }
                    let req=faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(nr,bs,b.ncols());
                    let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
                    faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                        basis.submatrix(start,start,nr,nc),tau.submatrix(0,start,bs,nc),
                        if conjugated{Conj::Yes}else{Conj::No},work.as_mut().subrows_mut(start,nr),Par::Seq,MemStack::new(&mut mem));
                }
                for (i, &row) in rows.iter().enumerate() {
                    for k in 0..b.ncols() {
                        out[(row, k)] = copy(&work[(i, k)]);
                    }
                }
            }
        }
        Ok(out)
    }
    pub(crate) fn apply_full_q_adjoint(&self, b: MatRef<'_, T>) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::{add, conj, copy, mul, mul_real, real, recip, sub, zero};
        let (m, n) = (self.symbolic.nrows(), self.symbolic.ncols());
        if b.nrows() != m {
            return Err("full Q adjoint RHS row mismatch".into());
        }
        let mut out = b.to_owned();
        let mut pivots = vec![None; n];
        if let Some(f) = self.simplicial() {
            let h = f.householder();
            for j in 0..n {
                let rows = h.row_idx_of_col_raw(j);
                let vals = h.val_of_col(j);
                pivots[j] = rows.first().copied();
                if rows.is_empty() {
                    continue;
                }
                let scale = recip(&real(&f.tau_val()[j]));
                for k in 0..b.ncols() {
                    let mut dot = zero::<T>();
                    for (&i, v) in rows.iter().zip(vals) {
                        dot = add(&dot, &mul(&conj(v), &out[(i, k)]));
                    }
                    dot = mul_real(&dot, &scale);
                    for (&i, v) in rows.iter().zip(vals) {
                        out[(i, k)] = sub(&out[(i, k)], &mul(v, &dot));
                    }
                }
            }
        } else {
            let f = self.supernodal().unwrap();
            let h = f.symbolic.householder();
            let l = f.symbolic.R_adjoint();
            let rp = h.col_ptr_for_householder_row_idx();
            let mut block = 0;
            for s in 0..h.n_supernodes() {
                let rows = &f.rows[rp[s]..rp[s + 1]];
                let begin = l.supernode_begin()[s];
                let width = l.supernode_end()[s] - begin;
                for i in 0..width.min(rows.len()) {
                    pivots[begin + i] = Some(rows[i]);
                }
                let hv = &f.h
                    [h.col_ptr_for_householder_val()[s]..h.col_ptr_for_householder_val()[s + 1]];
                let cols = if rows.is_empty() {
                    0
                } else {
                    hv.len() / rows.len()
                };
                let basis = MatRef::from_column_major_slice(hv, rows.len(), cols);
                let tv = &f.tau[h.col_ptr_for_tau_val()[s]..h.col_ptr_for_tau_val()[s + 1]];
                let rank = rows.len().min(cols);
                let max_bs = if rank == 0 { 0 } else { tv.len() / rank };
                let tau = MatRef::from_column_major_slice(tv, max_bs, rank);
                let mut work = Mat::from_fn(rows.len(), b.ncols(), |i, j| copy(&out[(rows[i], j)]));
                let mut start = 0;
                while start < cols {
                    let (bs, nr, nc) = (f.bs[block], f.nr[block], f.nc[block]);
                    if nc > 0 {
                        let req=faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(nr,bs,b.ncols());
                        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
                        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(basis.submatrix(start,start,nr,nc),tau.submatrix(0,start,bs,nc),Conj::Yes,work.as_mut().subrows_mut(start,nr),Par::Seq,MemStack::new(&mut mem));
                    }
                    start += nc;
                    block += 1;
                    if start >= rows.len() {
                        break;
                    }
                }
                for (i, &row) in rows.iter().enumerate() {
                    for j in 0..b.ncols() {
                        out[(row, j)] = copy(&work[(i, j)]);
                    }
                }
            }
        }
        let mut used = vec![false; m];
        for row in pivots.iter().flatten() {
            used[*row] = true;
        }
        let mut cursor = 0;
        for p in &mut pivots {
            if p.is_none() {
                while used[cursor] {
                    cursor += 1;
                }
                *p = Some(cursor);
                used[cursor] = true;
            }
        }
        let mut order: Vec<_> = pivots.into_iter().flatten().collect();
        order.extend((0..m).filter(|&i| !used[i]));
        Ok(Mat::from_fn(m, b.ncols(), |i, j| copy(&out[(order[i], j)])))
    }
    pub(crate) fn q_full(&self) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::one;
        let (m, n) = (self.symbolic.nrows(), self.symbolic.ncols());
        let seed = Mat::from_fn(n, m, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        self.apply_q_basis(seed.as_ref(), false, true)
    }
    pub(crate) fn q(&self) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::one;
        let n = self.symbolic.ncols();
        self.apply_q(
            Mat::from_fn(n, n, |i, j| if i == j { one::<T>() } else { zero::<T>() }).as_ref(),
            false,
        )
    }
    pub(crate) fn solve_transpose(
        &self,
        b: MatRef<'_, T>,
        adjoint: bool,
    ) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::copy;
        let n = self.symbolic.ncols();
        if b.nrows() != n {
            return Err("transposed sparse QR RHS row mismatch".into());
        }
        let r = self.r()?;
        for j in 0..n {
            if r.as_ref()
                .row_idx_of_col_raw(j)
                .iter()
                .zip(r.as_ref().val_of_col(j))
                .find(|(&i, _)| i == j)
                .is_none_or(|(_, v)| *v == zero::<T>())
            {
                return Err("transposed sparse QR solve requires full column rank".into());
            }
        }
        let (p, _) = self.symbolic.col_perm().arrays();
        let mut z = Mat::from_fn(n, b.ncols(), |i, j| copy(&b[(p[i], j)]));
        faer::sparse::linalg::triangular_solve::solve_upper_triangular_transpose_in_place(
            r.as_ref(),
            if adjoint { Conj::Yes } else { Conj::No },
            z.as_mut(),
            Par::Seq,
        );
        self.apply_q(z.as_ref(), !adjoint)
    }
    pub(crate) fn reconstruct(&self) -> Result<Mat<T>, String> {
        use faer::traits::math_utils::{copy, one};
        let q = self.q()?;
        let r = self.r()?;
        let mut a = Mat::zeros(q.nrows(), r.ncols());
        faer::sparse::linalg::matmul::dense_sparse_matmul(
            a.as_mut(),
            faer::Accum::Replace,
            q.as_ref(),
            r.as_ref(),
            one::<T>(),
            Par::Seq,
        );
        let (_, inv) = self.symbolic.col_perm().arrays();
        Ok(Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
            copy(&a[(i, inv[j])])
        }))
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

struct SuperParts<'a, T> {
    symbolic: &'a faer::sparse::linalg::qr::supernodal::SymbolicSupernodalQr<usize>,
    rows: &'a [usize],
    bs: &'a [usize],
    nr: &'a [usize],
    nc: &'a [usize],
    r: &'a [T],
    h: &'a [T],
    tau: &'a [T],
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_complex_qr_components_and_adjoint() {
        use faer::sparse::linalg::{qr::QrSymbolicParams, SupernodalThreshold};
        use faer::sparse::{SparseColMat, Triplet};
        use faer::{c64, prelude::*};
        let dense = Mat::from_fn(9, 5, |i, j| {
            if i == j {
                c64::new(5.0, 0.2)
            } else if (i + 2 * j) % 3 == 0 {
                c64::new(0.4, -0.3)
            } else {
                c64::new(0.0, 0.0)
            }
        });
        let entries: Vec<_> = (0..5)
            .flat_map(|j| {
                let dense = &dense;
                (0..9).filter_map(move |i| {
                    let val = dense[(i, j)];
                    (val != c64::new(0.0, 0.0)).then_some(Triplet {
                        row: i,
                        col: j,
                        val,
                    })
                })
            })
            .collect();
        let a = SparseColMat::try_new_from_triplets(9, 5, &entries).unwrap();
        for threshold in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            let symbolic = Arc::new(
                faer::sparse::linalg::qr::factorize_symbolic_qr(
                    a.symbolic(),
                    QrSymbolicParams {
                        supernodal_flop_ratio_threshold: threshold,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let f = Factors::new(symbolic, a.as_ref()).unwrap();
            let q = f.q().unwrap();
            let r = f.r().unwrap().to_dense();
            let p = f.permutation();
            let expected = Mat::from_fn(9, 5, |i, j| dense[(i, p.as_ref().arrays().0[j])]);
            assert!((&q * &r - expected).norm_max() < 1e-12);
            assert!((f.reconstruct().unwrap() - &dense).norm_max() < 1e-12);
            let b = Mat::from_fn(5, 2, |i, j| c64::new((i + j + 1) as f64, 0.3));
            let x = f.solve_transpose(b.as_ref(), true).unwrap();
            // Independent dense QR reference for the minimum-norm solution.
            let gram = dense.adjoint() * &dense;
            let reference = &dense * gram.partial_piv_lu(Par::Seq).solve(&b, Par::Seq);
            assert!((x - reference).norm_max() < 1e-12);
        }
    }
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

pub(crate) fn complete_q_seed<T: ComplexField>(
    mut out: faer::MatMut<'_, T>,
    b: MatRef<'_, T>,
    used: &mut [bool],
    assigned: &[bool],
    full: bool,
) {
    use faer::traits::math_utils::{copy, one};
    let mut row = 0;
    for (j, &present) in assigned.iter().enumerate() {
        if !present {
            while used[row] {
                row += 1;
            }
            used[row] = true;
            for k in 0..b.ncols() {
                out[(row, k)] = copy(&b[(j, k)]);
            }
        }
    }
    if full {
        let mut col = b.nrows();
        for (row, &present) in used.iter().enumerate() {
            if !present {
                out[(row, col)] = one::<T>();
                col += 1;
            }
        }
    }
}
#[cfg(test)]
mod full_adjoint_tests {
    use super::*;
    use faer::sparse::linalg::{
        qr::{factorize_symbolic_qr, QrSymbolicParams},
        SupernodalThreshold,
    };
    use faer::sparse::{SparseColMat, Triplet};
    use faer::traits::math_utils::from_f64;
    fn check<T: ComplexField>(tol: f64, value: fn(f64, f64) -> T) {
        for (m, n) in [(0, 0), (5, 0), (5, 3), (5, 5), (40, 20)] {
            for empty in [false, true] {
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| (0..m).map(move |i| (i, j)))
                    .filter(|&(i, j)| !empty && (i == m - n + j || i == j))
                    .map(|(i, j)| Triplet {
                        row: i,
                        col: j,
                        val: value(if i == m - n + j { 4.0 + j as f64 } else { 0.2 }, 0.1),
                    })
                    .collect();
                let a = SparseColMat::<usize, T>::try_new_from_triplets(m, n, &entries).unwrap();
                for threshold in [
                    SupernodalThreshold::AUTO,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                ] {
                    let symbolic = Arc::new(
                        factorize_symbolic_qr(
                            a.symbolic(),
                            QrSymbolicParams {
                                supernodal_flop_ratio_threshold: threshold,
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                    );
                    let f = Factors::new(symbolic, a.as_ref()).unwrap();
                    let b = Mat::from_fn(m, 2, |i, j| {
                        value((i + j + 1) as f64, (i + 2 * j) as f64 * 0.2)
                    });
                    let expected = f.q_full().unwrap().adjoint() * &b;
                    let actual = f.apply_full_q_adjoint(b.as_ref()).unwrap();
                    assert!((&actual - &expected).norm_max() < from_f64::<T::Real>(tol));
                }
            }
        }
    }
    #[test]
    fn f32_full_adjoint() {
        check::<f32>(1e-3, |r, _| r as f32);
    }
    #[test]
    fn f64_full_adjoint() {
        check::<f64>(1e-10, |r, _| r);
    }
    #[test]
    fn c32_full_adjoint() {
        check::<faer::c32>(1e-3, |r, i| faer::c32::new(r as f32, i as f32));
    }
    #[test]
    fn c64_full_adjoint() {
        check::<faer::c64>(1e-10, faer::c64::new);
    }
}

pub(crate) mod projection;
