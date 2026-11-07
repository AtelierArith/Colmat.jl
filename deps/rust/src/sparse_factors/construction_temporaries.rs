//! Construction temporaries of sparse Cholesky/LU: borrowed or scratch-backed
//! triangle selection and direct sorted-CSC assembly of extracted factors.
use faer::dyn_stack::{MemStack, StackReq};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::traits::{
    math_utils::{conj, copy, zero},
    ComplexField,
};
use std::ops::Range;

// Stored rows of a sorted column `j` that lie in the selected triangle.
fn kept(rows: &[usize], j: usize, lower: bool) -> Range<usize> {
    if lower {
        rows.partition_point(|&i| i < j)..rows.len()
    } else {
        0..rows.partition_point(|&i| i <= j)
    }
}
/// Entry count of the selected triangle, or `None` when every stored entry
/// already lies in it, so the input can be passed to faer unchanged.
pub(super) fn triangle_nnz(p: SymbolicSparseColMatRef<'_, usize>, lower: bool) -> Option<usize> {
    let mut nnz = 0;
    let mut stripped = false;
    for j in 0..p.ncols() {
        let rows = p.row_idx_of_col_raw(j);
        let keep = kept(rows, j, lower).len();
        stripped |= keep != rows.len();
        nnz += keep;
    }
    stripped.then_some(nnz)
}
/// Caller scratch holding the selected triangle when it must be stripped.
pub(super) fn triangle_req<T>(p: SymbolicSparseColMatRef<'_, usize>, lower: bool) -> StackReq {
    match triangle_nnz(p, lower) {
        None => StackReq::EMPTY,
        Some(nnz) => StackReq::all_of(&[
            StackReq::new::<usize>(p.ncols() + 1),
            StackReq::new::<usize>(nnz),
            StackReq::new::<T>(nnz),
        ]),
    }
}
/// Runs `f` on the selected triangle of the sorted CSC input `a`: the input
/// itself when it holds only that triangle, otherwise a copy carved from
/// `stack` (sized by [`triangle_req`]); `f` receives the remaining stack.
pub(super) fn with_triangle<T: ComplexField, R>(
    a: SparseColMatRef<'_, usize, T>,
    lower: bool,
    stack: &mut MemStack,
    f: impl FnOnce(SparseColMatRef<'_, usize, T>, &mut MemStack) -> R,
) -> R {
    let s = a.symbolic();
    let Some(nnz) = triangle_nnz(s, lower) else {
        return f(a, stack);
    };
    let n = s.ncols();
    let (mut ptr, stack) = stack.make_with::<usize>(n + 1, |_| 0);
    let (mut idx, stack) = stack.make_with::<usize>(nnz, |_| 0);
    let (mut val, stack) = stack.make_with::<T>(nnz, |_| zero());
    let mut k = 0;
    for j in 0..n {
        let rows = s.row_idx_of_col_raw(j);
        let range = kept(rows, j, lower);
        let start = s.col_range(j).start;
        for r in range {
            idx[k] = rows[r];
            val[k] = copy(&a.val()[start + r]);
            k += 1;
        }
        ptr[j + 1] = k;
    }
    // SAFETY: `ptr` is nondecreasing from 0 to `nnz`, and each column keeps a
    // sorted, unique, in-bounds subrange of a validated column.
    let triangle = unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, &ptr, None, &idx) };
    f(SparseColMatRef::new(triangle, &val), stack)
}
/// Enumerates stored entries `(row, col, value)` in nondecreasing column order.
pub(super) type Entries<'a, T> = &'a dyn Fn(&mut dyn FnMut(usize, usize, &T));
fn reserved<U>(len: usize) -> Result<Vec<U>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| format!("{e:?}"))?;
    Ok(v)
}
/// Owned sorted CSC from `nnz` entries whose rows increase within each column.
pub(super) fn csc_from_entries<T: ComplexField>(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    entries: Entries<'_, T>,
) -> Result<SparseColMat<usize, T>, String> {
    let mut ptr = reserved(ncols + 1)?;
    let mut idx = reserved(nnz)?;
    let mut val = reserved(nnz)?;
    ptr.push(0);
    let mut overflow = false;
    entries(&mut |i, j, x| {
        if idx.len() == nnz || j >= ncols || j + 1 < ptr.len() {
            overflow = true;
            return;
        }
        while ptr.len() <= j {
            ptr.push(idx.len());
        }
        idx.push(i);
        val.push(copy(x));
    });
    if overflow {
        return Err("sparse factor assembly received out-of-order entries".into());
    }
    while ptr.len() <= ncols {
        ptr.push(idx.len());
    }
    finish(nrows, ncols, ptr, idx, val)
}
/// Owned sorted CSC of the conjugate transpose of `nnz` entries (counting sort).
pub(super) fn adjoint_from_entries<T: ComplexField>(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    entries: Entries<'_, T>,
) -> Result<SparseColMat<usize, T>, String> {
    let mut ptr = reserved(nrows + 1)?;
    ptr.resize(nrows + 1, 0usize);
    let mut seen = 0;
    let mut bad = false;
    entries(&mut |i, _, _| {
        if i < nrows {
            ptr[i + 1] += 1;
            seen += 1;
        } else {
            bad = true;
        }
    });
    if bad || seen != nnz {
        return Err("sparse factor assembly received inconsistent entries".into());
    }
    for i in 0..nrows {
        ptr[i + 1] += ptr[i];
    }
    let mut next = reserved(nrows)?;
    next.extend_from_slice(&ptr[..nrows]);
    let mut idx = reserved(nnz)?;
    let mut val = reserved(nnz)?;
    idx.resize(nnz, 0);
    val.resize_with(nnz, zero::<T>);
    // Visiting source columns in order leaves every output column sorted.
    entries(&mut |i, j, x| {
        idx[next[i]] = j;
        val[next[i]] = conj(x);
        next[i] += 1;
    });
    finish(ncols, nrows, ptr, idx, val)
}
fn finish<T>(
    nrows: usize,
    ncols: usize,
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<T>,
) -> Result<SparseColMat<usize, T>, String> {
    if ptr.len() != ncols + 1 || idx.len() != val.len() || ptr[ncols] != idx.len() {
        return Err("sparse factor assembly produced inconsistent sizes".into());
    }
    for j in 0..ncols {
        let rows = &idx[ptr[j]..ptr[j + 1]];
        if rows.iter().any(|&i| i >= nrows) || rows.windows(2).any(|w| w[0] >= w[1]) {
            return Err("sparse factor assembly produced unsorted rows".into());
        }
    }
    // SAFETY: pointers, bounds and strictly increasing rows were checked above.
    let s = unsafe { SymbolicSparseColMat::new_unchecked(nrows, ncols, ptr, None, idx) };
    Ok(SparseColMat::new(s, val))
}
/// Entries of a CSC column by column, in stored order.
pub(super) fn visit_csc<T>(
    a: SparseColMatRef<'_, usize, T>,
    emit: &mut dyn FnMut(usize, usize, &T),
) {
    let s = a.symbolic();
    for j in 0..s.ncols() {
        for k in s.col_range(j) {
            emit(s.row_idx()[k], j, &a.val()[k]);
        }
    }
}
/// Sorted CSC copy of a CSC whose columns may be unsorted; the per-column
/// ordering buffer is only allocated when an unsorted column is present.
pub(super) fn sorted_csc<T: ComplexField>(
    a: SparseColMatRef<'_, usize, T>,
) -> Result<SparseColMat<usize, T>, String> {
    let s = a.symbolic();
    let sorted = (0..s.ncols()).all(|j| s.row_idx_of_col_raw(j).windows(2).all(|w| w[0] < w[1]));
    let nnz = s.compute_nnz();
    if sorted {
        return csc_from_entries(a.nrows(), a.ncols(), nnz, &|emit| visit_csc(a, emit));
    }
    let longest = (0..s.ncols())
        .map(|j| s.col_range(j).len())
        .max()
        .unwrap_or(0);
    let order = std::cell::RefCell::new(reserved::<usize>(longest)?);
    csc_from_entries(a.nrows(), a.ncols(), nnz, &|emit| {
        let mut order = order.borrow_mut();
        for j in 0..s.ncols() {
            order.clear();
            order.extend(s.col_range(j));
            order.sort_unstable_by_key(|&k| s.row_idx()[k]);
            for &k in order.iter() {
                emit(s.row_idx()[k], j, &a.val()[k]);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::operators::NativeValue;
    use crate::sparse::{SparseData, SparseOwner, Storage};
    use crate::sparse_factors::{
        construction_workspace, numeric_with_par, Analysis, Factors, Plan, SparseFactor,
        SparseSymbolic,
    };
    use crate::workspace::OwnedScratch;
    use faer::sparse::linalg::{cholesky, SupernodalThreshold};
    use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
    use faer::traits::math_utils::{conj, copy, from_f64};
    use faer::traits::ComplexField;
    use faer::{c64, Par, Side};
    use std::sync::Arc;

    // Hermitian positive definite matrix storing both triangles.
    fn full<T: NativeValue>(n: usize) -> SparseColMat<usize, T> {
        let mut e = Vec::new();
        for j in 0..n {
            e.push(Triplet::new(j, j, from_f64::<T>(8.0 + j as f64)));
            for d in [1, 3, 7] {
                if j + d < n {
                    let x = T::from_parts(-1.0 / d as f64, 0.25 / d as f64);
                    e.push(Triplet::new(j + d, j, copy(&x)));
                    e.push(Triplet::new(j, j + d, conj(&x)));
                }
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &e).unwrap()
    }
    fn triangle<T: ComplexField>(
        a: &SparseColMat<usize, T>,
        lower: bool,
    ) -> SparseColMat<usize, T> {
        let e: Vec<_> = a
            .triplet_iter()
            .filter(|e| {
                if lower {
                    e.row >= e.col
                } else {
                    e.row <= e.col
                }
            })
            .map(|e| Triplet::new(e.row, e.col, copy(e.val)))
            .collect();
        SparseColMat::try_new_from_triplets(a.nrows(), a.ncols(), &e).unwrap()
    }
    fn llt_plan<T>(
        a: &SparseColMat<usize, T>,
        lower: bool,
        t: SupernodalThreshold,
    ) -> SparseSymbolic {
        let symbolic = cholesky::factorize_symbolic_cholesky(
            a.symbolic(),
            if lower { Side::Lower } else { Side::Upper },
            cholesky::SymmetricOrdering::Amd,
            cholesky::CholeskySymbolicParams {
                supernodal_flop_ratio_threshold: t,
                ..Default::default()
            },
        )
        .unwrap();
        SparseSymbolic {
            tuning: Default::default(),
            plan: Some(Plan {
                analysis: Analysis::Llt(Arc::new(symbolic)),
                pattern: a.symbolic().to_owned().unwrap(),
                lower,
            }),
        }
    }
    // Reference: the former triplet-based assembly.
    fn via_triplets<T: ComplexField>(a: SparseColMatRef<'_, usize, T>) -> SparseColMat<usize, T> {
        let e: Vec<_> = a
            .triplet_iter()
            .map(|e| Triplet::new(e.row, e.col, copy(e.val)))
            .collect();
        SparseColMat::try_new_from_triplets(a.nrows(), a.ncols(), &e).unwrap()
    }
    fn assert_same_sorted<T: ComplexField>(x: &SparseColMat<usize, T>, y: &SparseColMat<usize, T>) {
        assert_eq!((x.nrows(), x.ncols()), (y.nrows(), y.ncols()));
        assert!(x.symbolic().col_nnz().is_none());
        assert_eq!(x.symbolic().col_ptr(), y.symbolic().col_ptr());
        assert_eq!(x.symbolic().row_idx(), y.symbolic().row_idx());
        for j in 0..x.ncols() {
            assert!(x
                .symbolic()
                .row_idx_of_col_raw(j)
                .windows(2)
                .all(|w| w[0] < w[1]));
        }
        for (u, v) in x.val().iter().zip(y.val()) {
            assert!(u == v);
        }
    }
    type Get<T> = fn(&SparseOwner) -> &SparseColMat<usize, T>;
    type Wrap<T> = fn(crate::sparse_factors::Numeric<T>) -> Factors;
    fn get_f64(o: &SparseOwner) -> &SparseColMat<usize, f64> {
        match o.data.as_ref().unwrap() {
            SparseData::F64(Storage::Csc(a)) => a,
            _ => panic!("expected f64 CSC"),
        }
    }
    fn get_c64(o: &SparseOwner) -> &SparseColMat<usize, c64> {
        match o.data.as_ref().unwrap() {
            SparseData::C64(Storage::Csc(a)) => a,
            _ => panic!("expected c64 CSC"),
        }
    }
    fn check_cholesky<T: NativeValue>(wrap: Wrap<T>, get: Get<T>) {
        let a = full::<T>(30);
        for t in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            for lower in [true, false] {
                for stored in [a.clone(), triangle(&a, lower)] {
                    let storage = Storage::Csc(stored.clone());
                    let plan = llt_plan(&stored, lower, t);
                    let mut req = [0usize; 2];
                    construction_workspace::query(&plan, T::tag(), req.as_mut_ptr(), Par::Seq)
                        .unwrap();
                    let mut scratch = OwnedScratch::new();
                    scratch.scratch_reserve(req[0], req[1]).unwrap();
                    let p = plan.plan().unwrap();
                    // Only the returned numeric value buffer is allocated.
                    let count = crate::allocation_probe::count(|| {
                        numeric_with_par(p, &storage, Some(&mut scratch), Par::Seq)
                            .unwrap()
                            .unwrap();
                    });
                    assert_eq!(count, 1, "t={} lower={lower}", t.0);
                    // Without scratch: the value buffer plus one combined stack.
                    let count = crate::allocation_probe::count(|| {
                        numeric_with_par(p, &storage, None, Par::Seq)
                            .unwrap()
                            .unwrap();
                    });
                    assert_eq!(count, 2);
                    let numeric = numeric_with_par(p, &storage, None, Par::Seq)
                        .unwrap()
                        .unwrap();
                    let factor = SparseFactor {
                        factors: Some(wrap(numeric)),
                        qr_rank_hint: None,
                    };
                    let mut l = SparseOwner { data: None };
                    let count = crate::allocation_probe::count(|| {
                        l.sparse_cholesky_component(&factor, false).unwrap();
                    });
                    assert!(count <= 3, "L assembly allocated {count}");
                    let mut u = SparseOwner { data: None };
                    let count = crate::allocation_probe::count(|| {
                        u.sparse_cholesky_component(&factor, true).unwrap();
                    });
                    assert!(count <= 4, "U assembly allocated {count}");
                    let (l, u) = (get(&l), get(&u));
                    let former = crate::allocation_probe::count(|| {
                        via_triplets(l.as_ref());
                    });
                    assert!(count < former, "direct {count} vs triplets {former}");
                    assert_same_sorted(
                        u,
                        &super::adjoint_from_entries(30, 30, l.compute_nnz(), &|emit| {
                            super::visit_csc(l.as_ref(), emit)
                        })
                        .unwrap(),
                    );
                    assert_same_sorted(l, &via_triplets(l.as_ref()));
                    assert_same_sorted(u, &via_triplets(u.as_ref()));
                    // L L^H reproduces the permuted matrix.
                    let perm = match &p.analysis {
                        Analysis::Llt(s) => s.perm().unwrap().arrays().0.to_vec(),
                        _ => unreachable!(),
                    };
                    let dense = a.to_dense();
                    let product = l.to_dense() * u.to_dense();
                    let tol = from_f64::<T::Real>(1e-3);
                    let expected =
                        faer::Mat::<T>::from_fn(30, 30, |i, j| copy(&dense[(perm[i], perm[j])]));
                    assert!((&product - &expected).norm_l2() < tol);
                }
            }
        }
    }
    #[test]
    fn cholesky_construction_temporaries_f64() {
        check_cholesky::<f64>(Factors::F64, get_f64);
    }
    #[test]
    fn cholesky_construction_temporaries_c64() {
        check_cholesky::<c64>(Factors::C64, get_c64);
    }
    #[test]
    fn simplicial_lu_components_are_sorted_directly() {
        let a = full::<c64>(25);
        let cols = faer::perm::Perm::new_checked(
            (0..25).rev().collect::<Vec<_>>().into_boxed_slice(),
            (0..25).rev().collect::<Vec<_>>().into_boxed_slice(),
            25,
        );
        let f =
            crate::sparse_lu_components::SimplicialFactors::new(a.as_ref(), cols.as_ref()).unwrap();
        let factor = SparseFactor {
            factors: Some(Factors::C64(crate::sparse_factors::Numeric::SimplicialLu(
                f.clone(),
            ))),
            qr_rank_hint: None,
        };
        for (upper, source) in [
            (false, f.numeric.l_factor_unsorted()),
            (true, f.numeric.u_factor_unsorted()),
        ] {
            let mut out = SparseOwner { data: None };
            let count = crate::allocation_probe::count(|| {
                out.sparse_lu_component(&factor, upper).unwrap();
            });
            assert!(count <= 4, "LU component allocated {count}");
            let former = crate::allocation_probe::count(|| {
                via_triplets(source);
            });
            assert!(count < former, "direct {count} vs triplets {former}");
            assert_same_sorted(get_c64(&out), &via_triplets(source));
        }
    }
    // Records the remaining upstream cost: faer's public `LuRef::factors`
    // assembles both factors through triplets for auto/supernodal LU.
    #[test]
    fn native_lu_components_report_faer_triplet_assembly() {
        use faer::sparse::linalg::lu;
        let a = full::<f64>(30);
        for t in [
            SupernodalThreshold::FORCE_SIMPLICIAL,
            SupernodalThreshold::FORCE_SUPERNODAL,
        ] {
            let symbolic = Arc::new(
                lu::factorize_symbolic_lu(
                    a.symbolic(),
                    lu::LuSymbolicParams {
                        supernodal_flop_ratio_threshold: t,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let f = crate::sparse_lu_owned::Factors::new(symbolic, a.as_ref(), Par::Seq).unwrap();
            let factor = SparseFactor {
                factors: Some(Factors::F64(crate::sparse_factors::Numeric::Lu(f))),
                qr_rank_hint: None,
            };
            for upper in [false, true] {
                let mut out = SparseOwner { data: None };
                let count = crate::allocation_probe::count(|| {
                    out.sparse_lu_component(&factor, upper).unwrap();
                });
                let largest = crate::allocation_probe::largest(|| {
                    out.sparse_lu_component(&factor, upper).unwrap();
                });
                let nnz = get_f64(&out).compute_nnz();
                eprintln!(
                    "native LU t={} upper={upper}: {count} allocations, largest {largest} bytes, nnz {nnz}",
                    t.0
                );
                assert!(count > 0);
            }
        }
    }
}
