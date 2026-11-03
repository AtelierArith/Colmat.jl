//! Public AMD/COLAMD ordering, independent of numeric factorization.
use crate::{guarded, permutations::OwnedPermutation, sparsity::SparsityOwner};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    perm::Perm,
    sparse::linalg::{amd, colamd},
};
use rustcall_julia_macros::julia;
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_ordering(
        &mut self,
        pattern: *const u8,
        symmetric: bool,
        dense_row: f64,
        dense_col: f64,
        aggressive: bool,
        stats: *mut f64,
    ) -> Result<(), String> {
        guarded(|| {
            if !dense_row.is_finite() || !dense_col.is_finite() {
                return Err("ordering thresholds must be finite".into());
            }
            let a = unsafe { &*pattern.cast::<SparsityOwner>() }.data()?.csc()?;
            let (m, n) = (a.nrows(), a.ncols());
            if symmetric && m != n {
                return Err("AMD requires a square pattern".into());
            }
            let mut forward = vec![0usize; n];
            let mut inverse = vec![0usize; n];
            let flops = if m == 0 || n == 0 {
                for i in 0..n {
                    forward[i] = i;
                    inverse[i] = i;
                }
                amd::FlopCount::default()
            } else if symmetric {
                let mut mem = MemBuffer::try_new(amd::order_scratch::<usize>(n, a.compute_nnz()))
                    .map_err(|e| format!("{e:?}"))?;
                amd::order(
                    &mut forward,
                    &mut inverse,
                    a.as_ref(),
                    amd::Control {
                        dense: dense_row,
                        aggressive,
                    },
                    MemStack::new(&mut mem),
                )
                .map_err(|e| format!("{e:?}"))?
            } else {
                let mut mem =
                    MemBuffer::try_new(colamd::order_scratch::<usize>(m, n, a.compute_nnz()))
                        .map_err(|e| format!("{e:?}"))?;
                colamd::order(
                    &mut forward,
                    &mut inverse,
                    a.as_ref(),
                    colamd::Control {
                        dense_row,
                        dense_col,
                        aggressive,
                    },
                    MemStack::new(&mut mem),
                )
                .map_err(|e| format!("{e:?}"))?;
                amd::FlopCount::default()
            };
            self.data = Some(Perm::new_checked(
                forward.into_boxed_slice(),
                inverse.into_boxed_slice(),
                n,
            ));
            unsafe {
                stats.write(flops.n_div);
                stats.add(1).write(flops.n_mult_subs_ldl);
                stats.add(2).write(flops.n_mult_subs_lu);
            }
            Ok(())
        })
    }
}
