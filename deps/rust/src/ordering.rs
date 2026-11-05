//! Public AMD/COLAMD ordering, independent of numeric factorization.
use crate::{
    guarded,
    permutations::OwnedPermutation,
    sparsity::{Pattern, SparsityOwner},
    workspace::OwnedScratch,
};
use faer::{
    dyn_stack::StackReq,
    perm::Perm,
    sparse::linalg::{amd, colamd},
};
use rustcall_julia_macros::julia;
fn requirement(
    pattern: &SparsityOwner,
    m: usize,
    n: usize,
    symmetric: bool,
) -> Result<StackReq, String> {
    let (rows, cols, nnz) = match pattern.data()? {
        Pattern::Csc(a) => (a.nrows(), a.ncols(), a.compute_nnz()),
        Pattern::Csr(a) => (a.nrows(), a.ncols(), a.compute_nnz()),
    };
    if (rows, cols) != (m, n) {
        return Err("ordering native shape does not match requested metadata".into());
    }
    if symmetric && m != n {
        return Err("AMD requires a square pattern".into());
    }
    Ok(if m == 0 || n == 0 {
        StackReq::EMPTY
    } else if symmetric {
        amd::order_scratch::<usize>(n, nnz)
    } else {
        colamd::order_scratch::<usize>(m, n, nnz)
    })
}
fn validate_output<T>(out: *mut T, len: usize) -> Result<(), String> {
    if out.is_null()
        || (out as usize) % std::mem::align_of::<T>() != 0
        || (out as usize)
            .checked_add(
                len.checked_mul(std::mem::size_of::<T>())
                    .ok_or("output span overflow")?,
            )
            .is_none()
    {
        return Err("ordering requires an aligned writable output buffer".into());
    }
    Ok(())
}
#[julia]
impl SparsityOwner {
    #[julia]
    pub fn ordering_requirements(
        &self,
        m: usize,
        n: usize,
        symmetric: bool,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let req = requirement(self, m, n, symmetric)?;
            validate_output(out, 2)?;
            // SAFETY: Julia preserves a writable two-entry UInt buffer; alignment and span are checked above.
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
}
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_ordering(
        &mut self,
        pattern: &SparsityOwner,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
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
            validate_output(stats, 3)?;
            let req = requirement(pattern, m, n, symmetric)?;
            let stack = scratch.stack(req)?;
            // CSC cloning/CSR assembly and the owning permutation are outside kernel scratch.
            let a = pattern.data()?.csc()?;
            let mut forward = vec![0usize; n];
            let mut inverse = vec![0usize; n];
            let flops = if m == 0 || n == 0 {
                for i in 0..n {
                    forward[i] = i;
                    inverse[i] = i;
                }
                amd::FlopCount::default()
            } else if symmetric {
                amd::order(
                    &mut forward,
                    &mut inverse,
                    a.as_ref(),
                    amd::Control {
                        dense: dense_row,
                        aggressive,
                    },
                    stack,
                )
                .map_err(|e| format!("{e:?}"))?
            } else {
                colamd::order(
                    &mut forward,
                    &mut inverse,
                    a.as_ref(),
                    colamd::Control {
                        dense_row,
                        dense_col,
                        aggressive,
                    },
                    stack,
                )
                .map_err(|e| format!("{e:?}"))?;
                amd::FlopCount::default()
            };
            self.data = Some(Perm::new_checked(
                forward.into_boxed_slice(),
                inverse.into_boxed_slice(),
                n,
            ));
            // SAFETY: Julia preserves a writable three-entry Float64 buffer; validated before work starts.
            unsafe {
                stats.write(flops.n_div);
                stats.add(1).write(flops.n_mult_subs_ldl);
                stats.add(2).write(flops.n_mult_subs_lu);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caller_workspace_and_preflight_preserve_outputs() {
        let a = faer::sparse::SymbolicSparseColMat::new_checked(
            3,
            3,
            vec![0usize, 2, 4, 5],
            None,
            vec![0usize, 1, 0, 1, 2],
        );
        let source = SparsityOwner {
            data: Some(Pattern::Csc(a)),
        };
        for symmetric in [true, false] {
            let req = requirement(&source, 3, 3, symmetric).unwrap();
            let mut work = OwnedScratch::new();
            work.scratch_reserve(req.size_bytes(), req.align_bytes())
                .unwrap();
            let mut out = OwnedPermutation::new();
            let mut stats = [0.0; 3];
            out.sparse_ordering(
                &source,
                &mut work,
                3,
                3,
                symmetric,
                10.0,
                0.5,
                true,
                stats.as_mut_ptr(),
            )
            .unwrap();
            let saved = out.data().unwrap().as_ref().arrays().0.to_vec();
            assert_eq!(saved.len(), 3);
            for (m, n) in [(2, 3), (3, 2)] {
                stats.fill(-1.0);
                assert!(out
                    .sparse_ordering(
                        &source,
                        &mut work,
                        m,
                        n,
                        symmetric,
                        10.0,
                        0.5,
                        true,
                        stats.as_mut_ptr()
                    )
                    .is_err());
                assert_eq!(out.data().unwrap().as_ref().arrays().0, saved);
                assert_eq!(stats, [-1.0; 3]);
            }
            let mut short = OwnedScratch::new();
            short
                .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                .unwrap();
            assert!(out
                .sparse_ordering(
                    &source,
                    &mut short,
                    3,
                    3,
                    symmetric,
                    10.0,
                    0.5,
                    true,
                    stats.as_mut_ptr()
                )
                .is_err());
            assert_eq!(out.data().unwrap().as_ref().arrays().0, saved);
            assert_eq!(stats, [-1.0; 3]);
            assert!(source
                .ordering_requirements(3, 3, symmetric, std::ptr::null_mut())
                .is_err());
            assert!(out
                .sparse_ordering(
                    &source,
                    &mut work,
                    3,
                    3,
                    symmetric,
                    10.0,
                    0.5,
                    true,
                    std::ptr::null_mut()
                )
                .is_err());
        }
    }
}
