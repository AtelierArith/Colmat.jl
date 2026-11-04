//! Borrowed QR component staging and Q application in caller-owned scratch.
use super::Factors;
use faer::dyn_stack::{MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::prelude::ReborrowMut;
use faer::sparse::SparseColMatRef;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, MatMut, MatRef, Par};

pub(crate) fn simplicial_scratch(m: usize, n: usize) -> StackReq {
    StackReq::new::<usize>(m).and(StackReq::new::<usize>(n))
}
pub(crate) fn simplicial_apply<T: ComplexField>(
    h: SparseColMatRef<'_, usize, T>,
    tau: &[T],
    mut out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    stack: &mut MemStack,
) {
    let (m, n) = (h.nrows(), h.ncols());
    assert_eq!(b.nrows(), n);
    assert_eq!(out.shape(), (m, b.ncols()));
    let (mut used, stack) = stack.make_with::<usize>(m, |_| 0);
    let (mut pivots, _) = stack.make_with::<usize>(n, |_| usize::MAX);
    for j in 0..n {
        if let Some(&i) = h.row_idx_of_col_raw(j).first() {
            pivots[j] = i;
            used[i] = 1;
        }
    }
    complete_pivots(&mut used, &mut pivots);
    out.fill(zero::<T>());
    for j in 0..b.ncols() {
        for i in 0..n {
            out[(pivots[i], j)] = copy(&b[(i, j)]);
        }
    }
    for col in (0..n).rev() {
        let rows = h.row_idx_of_col_raw(col);
        if rows.is_empty() {
            continue;
        }
        let val = h.val_of_col(col);
        let scale = recip(&real(&tau[col]));
        for j in 0..b.ncols() {
            let mut dot = zero::<T>();
            for (&row, v) in rows.iter().zip(val) {
                dot = add(&dot, &mul(&conj(v), &out[(row, j)]));
            }
            dot = mul_real(&dot, &scale);
            for (&row, v) in rows.iter().zip(val) {
                out[(row, j)] = sub(&out[(row, j)], &mul(v, &dot));
            }
        }
    }
}
fn complete_pivots(used: &mut [usize], pivots: &mut [usize]) {
    let mut cursor = 0;
    for p in pivots {
        if *p == usize::MAX {
            while used[cursor] != 0 {
                cursor += 1;
            }
            *p = cursor;
            used[cursor] = 1;
        }
    }
}
impl<T: ComplexField> Factors<T> {
    pub(crate) fn permutation_ref(&self) -> faer::perm::PermRef<'_, usize> {
        self.symbolic.col_perm()
    }
    pub(crate) fn write_r(&self, mut out: MatMut<'_, T>) {
        let (_, n) = self.dimensions();
        assert_eq!(out.shape(), (n, n));
        out.fill(zero::<T>());
        if let Some(f) = self.simplicial() {
            for e in f.R().triplet_iter() {
                out[(e.row, e.col)] = add(&out[(e.row, e.col)], e.val);
            }
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
                        out[(node.start() + j, node.start() + i)] = conj(&a[(i, j)]);
                    }
                    for (i, &row) in node.pattern().iter().enumerate() {
                        out[(node.start() + j, row)] = conj(&a[(w + i, j)]);
                    }
                }
            }
        }
    }
    pub(crate) fn apply_q_scratch(&self, k: usize) -> StackReq {
        let (m, n) = self.dimensions();
        let base = simplicial_scratch(m, n);
        let Some(f) = self.supernodal() else {
            return base;
        };
        let h = f.symbolic.householder();
        let rp = h.col_ptr_for_householder_row_idx();
        let mut index = 0;
        let mut local = StackReq::EMPTY;
        for node in 0..h.n_supernodes() {
            let rows = rp[node + 1] - rp[node];
            let len =
                h.col_ptr_for_householder_val()[node + 1] - h.col_ptr_for_householder_val()[node];
            let cols = if rows == 0 { 0 } else { len / rows };
            let mut start = 0;
            let mut inner = StackReq::EMPTY;
            while start < cols {
                let width = f.nc[index];
                if width != 0 {
                    inner=inner.or(faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(f.nr[index],f.bs[index],k));
                }
                start += width;
                index += 1;
                if start >= rows {
                    break;
                }
            }
            local = local.or(faer::linalg::temp_mat_scratch::<T>(rows, k).and(inner));
        }
        base.and(StackReq::new::<usize>(h.n_supernodes() + 1))
            .and(local)
    }
    pub(crate) fn apply_q_in(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par,
        stack: &mut MemStack,
    ) {
        let (m, n) = self.dimensions();
        assert_eq!(out.shape(), (m, b.ncols()));
        assert_eq!(b.nrows(), n);
        if let Some(f) = self.simplicial() {
            simplicial_apply(f.householder(), f.tau_val(), out, b, stack);
            return;
        }
        let f = self.supernodal().unwrap();
        let h = f.symbolic.householder();
        let r = f.symbolic.R_adjoint();
        let rp = h.col_ptr_for_householder_row_idx();
        let (mut used, stack) = stack.make_with::<usize>(m, |_| 0);
        let (mut pivots, stack) = stack.make_with::<usize>(n, |_| usize::MAX);
        let (mut offsets, stack) = stack.make_with::<usize>(h.n_supernodes() + 1, |_| 0);
        let mut index = 0;
        for node in 0..h.n_supernodes() {
            offsets[node] = index;
            let rows = &f.rows[rp[node]..rp[node + 1]];
            let begin = r.supernode_begin()[node];
            let width = r.supernode_end()[node] - begin;
            for i in 0..width.min(rows.len()) {
                pivots[begin + i] = rows[i];
                used[rows[i]] = 1;
            }
            let len =
                h.col_ptr_for_householder_val()[node + 1] - h.col_ptr_for_householder_val()[node];
            let cols = if rows.is_empty() { 0 } else { len / rows.len() };
            let mut start = 0;
            while start < cols {
                start += f.nc[index];
                index += 1;
                if start >= rows.len() {
                    break;
                }
            }
        }
        offsets[h.n_supernodes()] = index;
        complete_pivots(&mut used, &mut pivots);
        out.fill(zero::<T>());
        for j in 0..b.ncols() {
            for i in 0..n {
                out[(pivots[i], j)] = copy(&b[(i, j)]);
            }
        }
        for node in (0..h.n_supernodes()).rev() {
            let rows = &f.rows[rp[node]..rp[node + 1]];
            let hv = &f.h
                [h.col_ptr_for_householder_val()[node]..h.col_ptr_for_householder_val()[node + 1]];
            let cols = if rows.is_empty() {
                0
            } else {
                hv.len() / rows.len()
            };
            let basis = MatRef::from_column_major_slice(hv, rows.len(), cols);
            let tv = &f.tau[h.col_ptr_for_tau_val()[node]..h.col_ptr_for_tau_val()[node + 1]];
            let rank = rows.len().min(cols);
            let max_bs = if rank == 0 { 0 } else { tv.len() / rank };
            let tau = MatRef::from_column_major_slice(tv, max_bs, rank);
            let (mut local, remaining) =
                faer::linalg::temp_mat_zeroed::<T, _, _>(rows.len(), b.ncols(), stack);
            let mut local = local.as_mat_mut();
            for j in 0..b.ncols() {
                for (i, &row) in rows.iter().enumerate() {
                    local[(i, j)] = copy(&out[(row, j)]);
                }
            }
            let begin = offsets[node];
            let end = offsets[node + 1];
            let mut start = (begin..end).map(|i| f.nc[i]).sum::<usize>();
            for i in (begin..end).rev() {
                let cols = f.nc[i];
                start -= cols;
                if cols == 0 {
                    continue;
                }
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
     basis.submatrix(start,start,f.nr[i],cols),tau.submatrix(0,start,f.bs[i],cols),Conj::No,local.rb_mut().subrows_mut(start,f.nr[i]),par.reborrow(),remaining);
            }
            for j in 0..b.ncols() {
                for (i, &row) in rows.iter().enumerate() {
                    out[(row, j)] = copy(&local[(i, j)]);
                }
            }
        }
    }
}
