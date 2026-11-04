//! Prepared sparse Householder projection without per-application allocation.
//! Uses the same public factor layout and reflector order as Factors::apply_q.
use super::Factors;
use faer::dyn_stack::{MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, MatMut, MatRef, Par};
use std::sync::Arc;
#[derive(Clone, Copy)]
struct Block {
    start: usize,
    size: usize,
    rows: usize,
    cols: usize,
}
struct Layout {
    order: Vec<usize>,
    blocks: Vec<Block>,
    ranges: Vec<std::ops::Range<usize>>,
}
#[derive(Clone)]
pub(crate) struct Projection<T> {
    factors: Factors<T>,
    layout: Arc<Layout>,
}
impl<T: ComplexField> Projection<T> {
    pub(crate) fn new(factors: Factors<T>) -> Self {
        let (m, n) = factors.dimensions();
        let mut pivots = vec![None; n];
        let mut blocks = Vec::new();
        let mut ranges = Vec::new();
        if let Some(f) = factors.simplicial() {
            for (j, pivot) in pivots.iter_mut().enumerate() {
                *pivot = f.householder().row_idx_of_col_raw(j).first().copied();
            }
        } else {
            let f = factors.supernodal().unwrap();
            let h = f.symbolic.householder();
            let r = f.symbolic.R_adjoint();
            let rp = h.col_ptr_for_householder_row_idx();
            let mut index = 0;
            for node in 0..h.n_supernodes() {
                let rows = &f.rows[rp[node]..rp[node + 1]];
                let begin = r.supernode_begin()[node];
                let width = r.supernode_end()[node] - begin;
                for i in 0..width.min(rows.len()) {
                    pivots[begin + i] = Some(rows[i]);
                }
                let len = h.col_ptr_for_householder_val()[node + 1]
                    - h.col_ptr_for_householder_val()[node];
                let cols = if rows.is_empty() { 0 } else { len / rows.len() };
                let first = blocks.len();
                let mut start = 0;
                while start < cols {
                    let block = Block {
                        start,
                        size: f.bs[index],
                        rows: f.nr[index],
                        cols: f.nc[index],
                    };
                    blocks.push(block);
                    start += block.cols;
                    index += 1;
                    if start >= rows.len() {
                        break;
                    }
                }
                ranges.push(first..blocks.len());
            }
        }
        // Complete missing reflector pivots once, keeping native (unsorted)
        // sparse Householder pivot order and then the orthogonal complement.
        let mut used = vec![false; m];
        for row in pivots.iter().flatten() {
            used[*row] = true;
        }
        let mut cursor = 0;
        for pivot in &mut pivots {
            if pivot.is_none() {
                while used[cursor] {
                    cursor += 1;
                }
                *pivot = Some(cursor);
                used[cursor] = true;
            }
        }
        let mut order: Vec<_> = pivots.into_iter().flatten().collect();
        order.extend((0..m).filter(|&i| !used[i]));
        Self {
            factors,
            layout: Arc::new(Layout {
                order,
                blocks,
                ranges,
            }),
        }
    }
    pub(crate) fn dimensions(&self) -> (usize, usize) {
        self.factors.dimensions()
    }
    pub(crate) fn map_values<U: ComplexField>(&self, cast: impl Fn(&T) -> U) -> Projection<U> {
        Projection {
            factors: self.factors.map_values(cast),
            layout: self.layout.clone(),
        }
    }
    pub(crate) fn scratch(&self, k: usize) -> StackReq {
        let (m, _) = self.dimensions();
        let mut local = StackReq::EMPTY;
        if let Some(f) = self.factors.supernodal() {
            let h = f.symbolic.householder();
            let rp = h.col_ptr_for_householder_row_idx();
            for (node, range) in self.layout.ranges.iter().enumerate() {
                let mut reflectors = StackReq::EMPTY;
                for i in range.clone() {
                    let b = self.layout.blocks[i];
                    if b.cols > 0 {
                        let normal=faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(b.rows,b.size,k);
                        let adjoint=faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(b.rows,b.size,k);
                        reflectors = StackReq::any_of(&[reflectors, normal, adjoint]);
                    }
                }
                local = StackReq::any_of(&[
                    local,
                    faer::linalg::temp_mat_scratch::<T>(rp[node + 1] - rp[node], k).and(reflectors),
                ]);
            }
        }
        faer::linalg::temp_mat_scratch::<T>(m, k).and(local)
    }
    pub(crate) fn apply<const ADJ: bool, const C: bool, const FULL: bool>(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) {
        let (m, n) = self.dimensions();
        let seed = if FULL { m } else { n };
        let k = b.ncols();
        assert_eq!(b.nrows(), if ADJ { m } else { seed });
        assert_eq!(out.shape(), (if ADJ { seed } else { m }, k));
        // SAFETY: before any reflector or copy reads work, every entry is
        // initialized by copy_from (adjoint) or fill plus coordinate embedding.
        let (mut work, stack) = unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(m, k, stack) };
        let mut work = work.as_mat_mut();
        if ADJ {
            work.copy_from(b);
        } else {
            work.fill(zero::<T>());
            for j in 0..k {
                for i in 0..seed {
                    work[(self.layout.order[i], j)] = copy(&b[(i, j)]);
                }
            }
        }
        if let Some(f) = self.factors.simplicial() {
            let h = f.householder();
            // Sparse indexed reflectors have no public faer reverse-Q driver;
            // retain the existing native sparse-vector traversal and order.
            let mut apply = |column: usize| {
                let rows = h.row_idx_of_col_raw(column);
                let values = h.val_of_col(column);
                if rows.is_empty() {
                    return;
                }
                let scale = recip(&real(&f.tau_val()[column]));
                for j in 0..k {
                    let mut dot = zero::<T>();
                    for (&i, v) in rows.iter().zip(values) {
                        let vc = if C { conj(v) } else { copy(v) };
                        dot = add(&dot, &mul(&conj(&vc), &work[(i, j)]));
                    }
                    dot = mul_real(&dot, &scale);
                    for (&i, v) in rows.iter().zip(values) {
                        let vc = if C { conj(v) } else { copy(v) };
                        work[(i, j)] = sub(&work[(i, j)], &mul(&vc, &dot));
                    }
                }
            };
            if ADJ {
                for j in 0..n {
                    apply(j);
                }
            } else {
                for j in (0..n).rev() {
                    apply(j);
                }
            }
        } else {
            let f = self.factors.supernodal().unwrap();
            let h = f.symbolic.householder();
            let rp = h.col_ptr_for_householder_row_idx();
            let mut apply_node = |node: usize, mut p: Par<'_>| {
                let rows = &f.rows[rp[node]..rp[node + 1]];
                let hv = &f.h[h.col_ptr_for_householder_val()[node]
                    ..h.col_ptr_for_householder_val()[node + 1]];
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
                // SAFETY: the complete gathered node is initialized below
                // before any Householder call reads or updates its values.
                let (mut local, remaining) =
                    unsafe { faer::linalg::temp_mat_uninit::<T, _, _>(rows.len(), k, stack) };
                let mut local = local.as_mat_mut();
                for j in 0..k {
                    for (i, &row) in rows.iter().enumerate() {
                        local[(i, j)] = copy(&work[(row, j)]);
                    }
                }
                let mut apply_block = |index: usize, p: Par<'_>| {
                    let b = self.layout.blocks[index];
                    if b.cols == 0 {
                        return;
                    }
                    let v = basis.submatrix(b.start, b.start, b.rows, b.cols);
                    let t = tau.submatrix(0, b.start, b.size, b.cols);
                    let target = local.rb_mut().subrows_mut(b.start, b.rows);
                    let conj = if ADJ ^ C { Conj::Yes } else { Conj::No };
                    if ADJ {
                        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(v,t,conj,target,p,remaining);
                    } else {
                        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(v,t,conj,target,p,remaining);
                    }
                };
                let range = self.layout.ranges[node].clone();
                if ADJ {
                    for i in range {
                        apply_block(i, p.reborrow());
                    }
                } else {
                    for i in range.rev() {
                        apply_block(i, p.reborrow());
                    }
                }
                for j in 0..k {
                    for (i, &row) in rows.iter().enumerate() {
                        work[(row, j)] = copy(&local[(i, j)]);
                    }
                }
            };
            if ADJ {
                for node in 0..h.n_supernodes() {
                    apply_node(node, par.reborrow());
                }
            } else {
                for node in (0..h.n_supernodes()).rev() {
                    apply_node(node, par.reborrow());
                }
            }
        }
        if ADJ {
            for j in 0..k {
                for i in 0..seed {
                    out[(i, j)] = copy(&work[(self.layout.order[i], j)]);
                }
            }
        } else {
            out.copy_from(work);
        }
    }
}
