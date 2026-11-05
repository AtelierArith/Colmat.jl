//! Full saved Q-adjoint application with symbolic workspace bounds.
use super::Factors;
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
use faer::mat::AsMatMut;
use faer::prelude::ReborrowMut;
use faer::sparse::linalg::qr::{SymbolicQr, SymbolicQrRaw};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, MatMut, MatRef, Par};
impl<T: ComplexField> Factors<T> {
    pub(crate) fn full_q_adjoint_scratch(symbolic: &SymbolicQr<usize>, k: usize) -> StackReq {
        let (m, n) = (symbolic.nrows(), symbolic.ncols());
        let base = StackReq::new::<usize>(m)
            .and(StackReq::new::<usize>(n))
            .and(temp_mat_scratch::<T>(m, k));
        let SymbolicQrRaw::Supernodal(s) = symbolic.raw() else {
            return base;
        };
        let h = s.householder();
        let rp = h.col_ptr_for_householder_row_idx();
        let vp = h.col_ptr_for_householder_val();
        let tp = h.col_ptr_for_tau_val();
        let mut local = StackReq::EMPTY;
        for node in 0..h.n_supernodes() {
            let rows = rp[node + 1] - rp[node];
            let cols = if rows == 0 {
                0
            } else {
                (vp[node + 1] - vp[node]) / rows
            };
            let rank = rows.min(cols);
            let max_bs = if rank == 0 {
                0
            } else {
                (tp[node + 1] - tp[node]) / rank
            };
            local=local.or(temp_mat_scratch::<T>(rows,k).and(
                faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(rows,max_bs,k)));
        }
        base.and(local)
    }
    pub(crate) fn apply_full_q_adjoint_in(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) {
        let (m, n) = self.dimensions();
        assert_eq!(b.nrows(), m);
        assert_eq!(out.shape(), b.shape());
        let (mut used, stack) = stack.make_with::<usize>(m, |_| 0);
        let (mut pivots, stack) = stack.make_with::<usize>(n, |_| usize::MAX);
        let (mut values, stack) = temp_mat_zeroed::<T, _, _>(m, b.ncols(), stack);
        let mut values = values.as_mat_mut();
        values.copy_from(b);
        if let Some(f) = self.simplicial() {
            let h = f.householder();
            for col in 0..n {
                let rows = h.row_idx_of_col_raw(col);
                let vals = h.val_of_col(col);
                if let Some(&row) = rows.first() {
                    pivots[col] = row;
                }
                if rows.is_empty() {
                    continue;
                }
                let scale = recip(&real(&f.tau_val()[col]));
                for j in 0..b.ncols() {
                    let mut dot = zero::<T>();
                    for (&row, v) in rows.iter().zip(vals) {
                        dot = add(&dot, &mul(&conj(v), &values[(row, j)]));
                    }
                    dot = mul_real(&dot, &scale);
                    for (&row, v) in rows.iter().zip(vals) {
                        values[(row, j)] = sub(&values[(row, j)], &mul(v, &dot));
                    }
                }
            }
        } else {
            let f = self.supernodal().unwrap();
            let h = f.symbolic.householder();
            let l = f.symbolic.R_adjoint();
            let rp = h.col_ptr_for_householder_row_idx();
            let mut block = 0;
            for node in 0..h.n_supernodes() {
                let rows = &f.rows[rp[node]..rp[node + 1]];
                let begin = l.supernode_begin()[node];
                let width = l.supernode_end()[node] - begin;
                for i in 0..width.min(rows.len()) {
                    pivots[begin + i] = rows[i];
                }
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
                let (mut local, remaining) =
                    temp_mat_zeroed::<T, _, _>(rows.len(), b.ncols(), stack);
                let mut local = local.as_mat_mut();
                for j in 0..b.ncols() {
                    for (i, &row) in rows.iter().enumerate() {
                        local[(i, j)] = copy(&values[(row, j)]);
                    }
                }
                let mut start = 0;
                while start < cols {
                    let (bs, nr, nc) = (f.bs[block], f.nr[block], f.nc[block]);
                    if nc > 0 {
                        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                            basis.submatrix(start,start,nr,nc),tau.submatrix(0,start,bs,nc),Conj::Yes,
                            local.rb_mut().subrows_mut(start,nr),par.reborrow(),remaining);
                    }
                    start += nc;
                    block += 1;
                    if start >= rows.len() {
                        break;
                    }
                }
                for j in 0..b.ncols() {
                    for (i, &row) in rows.iter().enumerate() {
                        values[(row, j)] = copy(&local[(i, j)]);
                    }
                }
            }
        }
        for &row in pivots.iter().filter(|&&row| row != usize::MAX) {
            used[row] = 1;
        }
        let mut cursor = 0;
        for row in pivots.iter_mut() {
            if *row == usize::MAX {
                while used[cursor] != 0 {
                    cursor += 1;
                }
                *row = cursor;
                used[cursor] = 1;
            }
        }
        for j in 0..b.ncols() {
            for i in 0..n {
                out[(i, j)] = copy(&values[(pivots[i], j)]);
            }
            let mut i = n;
            for row in 0..m {
                if used[row] == 0 {
                    out[(i, j)] = copy(&values[(row, j)]);
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::NativeValue;
    use crate::teams::OwnedTeam;
    use faer::dyn_stack::MemBuffer;
    use faer::sparse::linalg::{
        qr::{factorize_symbolic_qr, QrSymbolicParams},
        SupernodalThreshold,
    };
    use faer::sparse::{SparseColMat, Triplet};
    use std::sync::Arc;
    fn check<T: NativeValue>() {
        for (m, n) in [(0, 0), (5, 0), (7, 4), (64, 64), (96, 64)] {
            for deficient in [false, true] {
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| (0..m).map(move |i| (i, j)))
                    .filter(|&(_, j)| !deficient || j != 0)
                    .map(|(i, j)| Triplet {
                        row: i,
                        col: j,
                        val: T::from_parts(
                            if i == j {
                                5.0
                            } else {
                                0.01 * ((i + 2 * j) as f64).sin()
                            },
                            0.02 * ((2 * i + j) as f64).cos(),
                        ),
                    })
                    .collect();
                let a = SparseColMat::try_new_from_triplets(m, n, &entries).unwrap();
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
                    let f = Factors::new(symbolic.clone(), a.as_ref()).unwrap();
                    let b = faer::Mat::from_fn(m, 2, |i, j| T::from_parts((i + j + 1) as f64, 0.1));
                    let reference = f.apply_full_q_adjoint_reference(b.as_ref()).unwrap();
                    let explicit = f.q_full().unwrap().adjoint() * &b;
                    let mut memory =
                        MemBuffer::new(Factors::<T>::full_q_adjoint_scratch(&symbolic, 2));
                    let mut actual = faer::Mat::zeros(m, 2);
                    for width in [1, 2] {
                        let mut team = OwnedTeam::new();
                        team.team_initialize(width).unwrap();
                        team.execute(|mut par| {
                            f.apply_full_q_adjoint_in(
                                actual.as_mut(),
                                b.as_ref(),
                                par.reborrow(),
                                MemStack::new(&mut memory),
                            );
                            assert_eq!(
                                crate::allocation_probe::count(|| f.apply_full_q_adjoint_in(
                                    actual.as_mut(),
                                    b.as_ref(),
                                    par,
                                    MemStack::new(&mut memory)
                                )),
                                0
                            );
                        })
                        .unwrap();
                        for j in 0..2 {
                            for i in 0..m {
                                let (x, y) = actual[(i, j)].parts();
                                for expected in [&reference, &explicit] {
                                    let (u, v) = expected[(i, j)].parts();
                                    assert!((x - u).hypot(y - v) < 2e-3);
                                }
                            }
                        }
                        team.team_close().unwrap();
                    }
                }
            }
        }
    }
    #[test]
    fn scratch_full_adjoint_f32() {
        check::<f32>();
    }
    #[test]
    fn scratch_full_adjoint_f64() {
        check::<f64>();
    }
    #[test]
    fn scratch_full_adjoint_c32() {
        check::<faer::c32>();
    }
    #[test]
    fn scratch_full_adjoint_c64() {
        check::<faer::c64>();
    }
}
