//! Caller-managed fill-reducing ordering workspace for sparse symbolic analysis.
//!
//! faer's `factorize_symbolic_{cholesky,lu,qr}` take no `MemStack` and allocate
//! their own buffer. Cholesky AMD is therefore run here on the caller's stack and
//! passed as `SymmetricOrdering::Custom`, which removes the AMD term from faer's
//! internal requirement. Wide QR runs COLAMD in Colmat and uses the stack
//! directly. Square/tall LU/QR COLAMD stays inside faer: its requirement keeps
//! the COLAMD term even for an explicit ordering, so precomputing would not
//! shrink the internal buffer.
use super::SparseSymbolic;
use crate::{
    guarded,
    sparsity::{Pattern, SparsityOwner},
    workspace::OwnedScratch,
};
use faer::dyn_stack::{MemStack, StackReq};
use faer::perm::PermRef;
use faer::sparse::linalg::{
    amd,
    cholesky::{self, simplicial, SymbolicCholesky, SymmetricOrdering},
    colamd, SupernodalThreshold,
};
use faer::sparse::{utils, SparseColMatRef, SymbolicSparseColMatRef};
use faer::traits::Symbolic;
use faer::Side;
use rustcall_julia_macros::julia;

/// faer's private `CHOLESKY_SUPERNODAL_RATIO_FACTOR` at the pinned revision.
const CHOLESKY_SUPERNODAL_RATIO_FACTOR: f64 = 40.0;

fn cholesky_requirement(n: usize, nnz: usize, ratio: bool) -> StackReq {
    let perm = StackReq::new::<usize>(n);
    let decision = if ratio {
        StackReq::all_of(&[
            StackReq::new::<usize>(n + 1),
            StackReq::new::<usize>(nnz),
            StackReq::new::<isize>(n),
            StackReq::new::<usize>(n),
            utils::permute_self_adjoint_scratch::<usize>(n).or(
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, nnz),
            ),
        ])
    } else {
        StackReq::EMPTY
    };
    StackReq::all_of(&[
        perm,
        perm,
        amd::order_maybe_unsorted_scratch::<usize>(n, nnz).or(decision),
    ])
}

/// Kernel stack bytes for the ordering phase that the caller can manage.
pub(super) fn requirement(
    m: usize,
    n: usize,
    nnz: usize,
    kind: i32,
    ordering: i32,
    supernodal: i32,
) -> StackReq {
    match kind {
        // Only the AUTO threshold needs factor column counts for the decision.
        2 if ordering == 0 => cholesky_requirement(n, nnz, supernodal == 0),
        1 if m < n && ordering == 0 && supernodal != 1 => colamd::order_scratch::<usize>(m, n, nnz),
        _ => StackReq::EMPTY,
    }
}

/// Reproduces faer's AMD-path simplicial/supernodal choice. With `Amd`, faer
/// compares AMD's flop estimate with the exact factor nonzeros; with `Custom`
/// it would recount flops from column counts (including the diagonal), which
/// can select a different algorithm. Returns a forced threshold instead.
fn decided_threshold(
    a: SymbolicSparseColMatRef<'_, usize>,
    perm: PermRef<'_, usize>,
    side: Side,
    flops: amd::FlopCount,
    threshold: SupernodalThreshold,
    stack: &mut MemStack,
) -> Result<SupernodalThreshold, String> {
    let flops = flops.n_div + flops.n_mult_subs_ldl;
    let supernodal = if threshold == SupernodalThreshold::FORCE_SIMPLICIAL {
        false
    } else if threshold == SupernodalThreshold::FORCE_SUPERNODAL {
        // faer's `flops / L_nnz > 0` with L_nnz >= n > 0, or NaN for n == 0.
        flops > 0.0
    } else {
        let n = a.nrows();
        let nnz = a.compute_nnz();
        let (mut col_ptr, stack) = stack.make_with::<usize>(n + 1, |_| 0);
        let (mut row_idx, stack) = stack.make_with::<usize>(nnz, |_| 0);
        let upper = utils::permute_self_adjoint_to_unsorted(
            Symbolic::materialize(nnz),
            &mut col_ptr,
            &mut row_idx,
            SparseColMatRef::new(a, Symbolic::materialize(a.row_idx().len())),
            perm,
            side,
            Side::Upper,
            stack,
        );
        let (mut etree, stack) = stack.make_with::<isize>(n, |_| 0);
        let (mut counts, stack) = stack.make_with::<usize>(n, |_| 0);
        simplicial::prefactorize_symbolic_cholesky::<usize>(
            &mut etree,
            &mut counts,
            upper.symbolic().as_ref(),
            stack,
        );
        let mut l_nnz = 0usize;
        for &c in counts.iter() {
            l_nnz = l_nnz
                .checked_add(c)
                .ok_or("sparse Cholesky factor size overflow")?;
        }
        flops / l_nnz as f64 > threshold.0 * CHOLESKY_SUPERNODAL_RATIO_FACTOR
    };
    Ok(if supernodal {
        SupernodalThreshold::FORCE_SUPERNODAL
    } else {
        SupernodalThreshold::FORCE_SIMPLICIAL
    })
}

/// Symbolic Cholesky with faer's AMD ordering computed in caller scratch.
/// faer still allocates its post-ordering symbolic buffer, the copied
/// permutation and the symbolic structure, which are owned output.
pub(super) fn amd_cholesky(
    a: SymbolicSparseColMatRef<'_, usize>,
    side: Side,
    control: amd::Control,
    threshold: SupernodalThreshold,
    supernodal_params: faer::sparse::linalg::SymbolicSupernodalParams<'_>,
    stack: &mut MemStack,
) -> Result<SymbolicCholesky<usize>, String> {
    let n = a.nrows();
    let (mut fwd, stack) = stack.make_with::<usize>(n, |_| 0);
    let (mut inv, stack) = stack.make_with::<usize>(n, |_| 0);
    let flops = amd::order_maybe_unsorted(&mut fwd, &mut inv, a, control, stack)
        .map_err(|e| format!("{e:?}"))?;
    let perm = PermRef::new_checked(&fwd, &inv, n);
    let forced = decided_threshold(a, perm, side, flops, threshold, stack)?;
    cholesky::factorize_symbolic_cholesky(
        a,
        side,
        SymmetricOrdering::Custom(perm),
        cholesky::CholeskySymbolicParams {
            amd_params: control,
            supernodal_flop_ratio_threshold: forced,
            supernodal_params,
        },
    )
    .map_err(|e| format!("{e:?}"))
}

fn pattern_requirement(
    pattern: &SparsityOwner,
    m: usize,
    n: usize,
    kind: i32,
    ordering: i32,
    supernodal: i32,
) -> Result<StackReq, String> {
    let (rows, cols, nnz) = match pattern.data()? {
        Pattern::Csc(a) => (a.nrows(), a.ncols(), a.compute_nnz()),
        Pattern::Csr(a) => (a.nrows(), a.ncols(), a.compute_nnz()),
    };
    if (rows, cols) != (m, n) {
        return Err("symbolic analysis native shape does not match requested metadata".into());
    }
    if !(0..=2).contains(&kind) || !(0..=2).contains(&ordering) || !(0..=2).contains(&supernodal) {
        return Err("unknown sparse analysis options".into());
    }
    if kind != 1 && m != n {
        return Err("sparse LU/Cholesky requires a square matrix".into());
    }
    Ok(requirement(m, n, nnz, kind, ordering, supernodal))
}

#[julia]
impl SparsityOwner {
    #[julia]
    pub fn sparse_analysis_requirements(
        &self,
        m: usize,
        n: usize,
        kind: i32,
        ordering: i32,
        supernodal: i32,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let req = pattern_requirement(self, m, n, kind, ordering, supernodal)?;
            if out.is_null() || (out as usize) % std::mem::align_of::<usize>() != 0 {
                return Err("analysis requirements need an aligned output buffer".into());
            }
            // SAFETY: Julia preserves a writable two-entry UInt buffer; alignment is checked.
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
}

#[julia]
impl SparseSymbolic {
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_analyze_workspace(
        &mut self,
        pattern: &SparsityOwner,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        kind: i32,
        lower: bool,
        ordering: i32,
        permutation: *const u8,
        dense: f64,
        aggressive: bool,
        supernodal: i32,
        dense_row: f64,
        dense_col: f64,
    ) -> Result<(), String> {
        guarded(|| {
            let req = pattern_requirement(pattern, m, n, kind, ordering, supernodal)?;
            let stack = scratch.stack(req)?;
            self.analyze(
                pattern,
                kind,
                lower,
                ordering,
                permutation,
                dense,
                aggressive,
                supernodal,
                dense_row,
                dense_col,
                Some(stack),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::SymbolicSparseColMat;

    fn owner(n: usize, entries: &[(usize, usize)]) -> SparsityOwner {
        let mut cols = vec![Vec::new(); n];
        for &(i, j) in entries {
            cols[j].push(i);
        }
        let mut ptr = vec![0usize];
        let mut idx = Vec::new();
        for mut c in cols {
            c.sort_unstable();
            c.dedup();
            idx.extend(c);
            ptr.push(idx.len());
        }
        SparsityOwner {
            data: Some(Pattern::Csc(SymbolicSparseColMat::new_checked(
                n, n, ptr, None, idx,
            ))),
        }
    }
    fn laplacian(k: usize) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let v = i * k + j;
                e.push((v, v));
                for (di, dj) in [(1, 0), (0, 1)] {
                    if i + di < k && j + dj < k {
                        let w = (i + di) * k + j + dj;
                        e.push((v, w));
                        e.push((w, v));
                    }
                }
            }
        }
        e
    }
    fn analyze(
        s: &mut SparseSymbolic,
        p: &SparsityOwner,
        work: Option<&mut OwnedScratch>,
        alg: i32,
    ) {
        let n = p.data().unwrap().csc().unwrap().nrows();
        match work {
            Some(w) => s
                .sparse_analyze_workspace(
                    p,
                    w,
                    n,
                    n,
                    2,
                    true,
                    0,
                    std::ptr::null(),
                    10.0,
                    true,
                    alg,
                    0.5,
                    0.5,
                )
                .unwrap(),
            None => s
                .sparse_analyze(p, 2, true, 0, std::ptr::null(), 10.0, true, alg, 0.5, 0.5)
                .unwrap(),
        }
    }
    fn llt(s: &SparseSymbolic) -> &SymbolicCholesky<usize> {
        match &s.plan().unwrap().analysis {
            super::super::Analysis::Llt(sym) => sym,
            _ => unreachable!(),
        }
    }

    #[test]
    fn caller_amd_matches_faer_and_shrinks_largest_allocation() {
        let mut seen = [false; 2];
        for (n, entries) in [
            (400, laplacian(20)),
            (6, (0..6).map(|i| (i, i)).collect::<Vec<_>>()),
            (
                150,
                (0..150)
                    .flat_map(|i| (0..150).map(move |j| (i, j)))
                    .filter(|&(i, j)| (i * 7 + j * 3) % 5 != 0 || i == j)
                    .collect(),
            ),
        ] {
            let p = owner(n, &entries);
            let a = p.data().unwrap().csc().unwrap();
            for alg in 0..=2 {
                let req = requirement(n, n, a.compute_nnz(), 2, 0, alg);
                let mut work = OwnedScratch::new();
                work.scratch_reserve(req.size_bytes(), req.align_bytes())
                    .unwrap();
                let mut reference = SparseSymbolic::new();
                let mut caller = SparseSymbolic::new();
                let largest_ref =
                    crate::allocation_probe::largest(|| analyze(&mut reference, &p, None, alg));
                let largest_caller = crate::allocation_probe::largest(|| {
                    analyze(&mut caller, &p, Some(&mut work), alg)
                });
                let (r, c) = (llt(&reference), llt(&caller));
                assert_eq!(r.perm().unwrap().arrays().0, c.perm().unwrap().arrays().0);
                let kind = |s: &SymbolicCholesky<usize>| {
                    matches!(
                        s.raw(),
                        faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(_)
                    )
                };
                assert_eq!(kind(r), kind(c));
                assert_eq!(r.len_val(), c.len_val());
                if alg == 0 {
                    seen[kind(r) as usize] = true;
                }
                if n == 400 {
                    eprintln!(
                        "n={n} alg={alg} caller scratch={} largest faer-internal ref={largest_ref} caller={largest_caller}",
                        req.size_bytes()
                    );
                    assert!(largest_caller < largest_ref);
                }
            }
        }
        assert_eq!(seen, [true, true]);
    }

    fn tune(s: &mut SparseSymbolic, ratio: f64, relax: Option<&[(usize, f64)]>) {
        let (sizes, fractions): (Vec<usize>, Vec<f64>) =
            relax.unwrap_or(&[]).iter().copied().unzip();
        s.sparse_symbolic_tuning(
            ratio,
            relax.is_some(),
            sizes.as_ptr(),
            fractions.as_ptr(),
            sizes.len(),
        )
        .unwrap();
    }
    #[test]
    fn supernodal_tuning_reaches_cholesky_and_lu_analyses() {
        let p = owner(400, &laplacian(20));
        let a = p.data().unwrap().csc().unwrap();
        let simplicial = |s: &SparseSymbolic| {
            matches!(
                llt(s).raw(),
                faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Simplicial(_)
            )
        };
        let req = requirement(400, 400, a.compute_nnz(), 2, 0, 0);
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let mut s = SparseSymbolic::new();
        analyze(&mut s, &p, None, 0);
        let default = simplicial(&s);
        for (ratio, expected) in [(f64::INFINITY, true), (0.0, false)] {
            for caller in [false, true] {
                let mut s = SparseSymbolic::new();
                tune(&mut s, ratio, None);
                analyze(&mut s, &p, caller.then_some(&mut work), 0);
                assert_eq!(simplicial(&s), expected);
                // Tuning is consumed by one analysis.
                analyze(&mut s, &p, caller.then_some(&mut work), 0);
                assert_eq!(simplicial(&s), default);
            }
        }
        // No relaxation stores no padding; faer's default relaxation does.
        let storage = |relax: Option<&[(usize, f64)]>, caller: bool, work: &mut OwnedScratch| {
            let mut s = SparseSymbolic::new();
            tune(&mut s, f64::NAN, relax);
            analyze(&mut s, &p, caller.then_some(work), 2);
            llt(&s).len_val()
        };
        for caller in [false, true] {
            let default = storage(None, caller, &mut work);
            let none = storage(Some(&[]), caller, &mut work);
            let wide = storage(Some(&[(usize::MAX, 1.0)]), caller, &mut work);
            assert!(none < default && default < wide, "{none} {default} {wide}");
        }
        let lu = |ratio: f64, relax: Option<&[(usize, f64)]>, alg: i32| {
            let mut s = SparseSymbolic::new();
            tune(&mut s, ratio, relax);
            s.sparse_analyze(&p, 0, true, 0, std::ptr::null(), 10.0, true, alg, 0.5, 0.5)
                .map(|()| match &s.plan().unwrap().analysis {
                    super::super::Analysis::Lu(sym) => sym.clone(),
                    _ => unreachable!(),
                })
        };
        // faer exposes no LU variant accessor; numeric scratch identifies it.
        let numeric_req = |sym: &faer::sparse::linalg::lu::SymbolicLu<usize>| {
            sym.factorize_numeric_lu_scratch::<f64>(faer::Par::Seq.shape(), Default::default())
        };
        let forced = |t| {
            faer::sparse::linalg::lu::factorize_symbolic_lu(
                a.as_ref(),
                faer::sparse::linalg::lu::LuSymbolicParams {
                    supernodal_flop_ratio_threshold: t,
                    supernodal_params: faer::sparse::linalg::SymbolicSupernodalParams {
                        relax: Some(&[]),
                    },
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let simplicial_lu = numeric_req(&forced(SupernodalThreshold::FORCE_SIMPLICIAL));
        let supernodal_lu = numeric_req(&forced(SupernodalThreshold::FORCE_SUPERNODAL));
        assert_ne!(simplicial_lu, supernodal_lu);
        assert_eq!(
            numeric_req(&lu(f64::INFINITY, None, 0).unwrap()),
            simplicial_lu
        );
        assert_eq!(numeric_req(&lu(0.0, Some(&[]), 0).unwrap()), supernodal_lu);
        assert!(lu(f64::NAN, Some(&[(8, 0.5)]), 2).is_ok());
        // Rejected combinations.
        assert!(lu(1.0, None, 2).is_err());
        assert!(lu(f64::NAN, Some(&[(8, 0.5)]), 1).is_err());
        let mut s = SparseSymbolic::new();
        tune(&mut s, 1.0, None);
        assert!(s
            .sparse_analyze(&p, 1, true, 0, std::ptr::null(), 10.0, true, 0, 0.5, 0.5)
            .is_err());
        assert!(s
            .sparse_symbolic_tuning(-1.0, false, std::ptr::null(), std::ptr::null(), 0)
            .is_err());
        let (sizes, fractions) = ([4usize], [1.5f64]);
        assert!(s
            .sparse_symbolic_tuning(f64::NAN, true, sizes.as_ptr(), fractions.as_ptr(), 1)
            .is_err());
        assert!(s
            .sparse_symbolic_tuning(f64::NAN, true, std::ptr::null(), std::ptr::null(), 1)
            .is_err());
    }

    #[test]
    fn short_scratch_preserves_existing_plan() {
        let p = owner(3, &[(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)]);
        let mut s = SparseSymbolic::new();
        analyze(&mut s, &p, None, 0);
        let before = llt(&s).perm().unwrap().arrays().0.to_vec();
        let req = requirement(3, 3, 5, 2, 0, 0);
        let mut short = OwnedScratch::new();
        short
            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        let q = owner(3, &[(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)]);
        assert!(s
            .sparse_analyze_workspace(
                &q,
                &mut short,
                3,
                3,
                2,
                true,
                0,
                std::ptr::null(),
                10.0,
                true,
                0,
                0.5,
                0.5
            )
            .is_err());
        assert_eq!(llt(&s).perm().unwrap().arrays().0, before);
        let mut out = [0usize; 2];
        assert!(q
            .sparse_analysis_requirements(4, 4, 2, 0, 0, out.as_mut_ptr())
            .is_err());
        assert!(q
            .sparse_analysis_requirements(3, 3, 2, 0, 0, std::ptr::null_mut())
            .is_err());
    }
}
