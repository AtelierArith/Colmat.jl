//! Sparse numeric construction with matching query/execution team shapes.
use super::{
    numeric_with_par, qr_construction_req, rank_hint, Analysis, Factors, SparseFactor,
    SparseSymbolic,
};
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner};
use crate::teams::OwnedTeam;
use crate::workspace::OwnedScratch;
use faer::dyn_stack::StackReq;
use faer::traits::ComplexField;
use faer::{c32, c64, Par};
use rustcall_julia_macros::julia;

fn requirement<T: ComplexField>(analysis: &Analysis, par: &Par<'_>) -> Result<StackReq, String> {
    Ok(match analysis {
        Analysis::Qr(sym) => qr_construction_req::<T>(sym, par),
        Analysis::SimplicialQr(plan) => crate::sparse_qr_components::Factors::<T>::scratch(plan)
            .or(rank_hint::scratch(plan.symbolic.ncols())),
        Analysis::WideQr(plan) => crate::sparse_qr_wide::Factors::<T>::scratch(plan, par),
        Analysis::Llt(sym) => {
            sym.factorize_numeric_llt_scratch::<T>(par.shape(), Default::default())
        }
        Analysis::Lu(sym) => sym.factorize_numeric_lu_scratch::<T>(par.shape(), Default::default()),
        Analysis::SimplicialLu(cols) => {
            crate::sparse_lu_components::SimplicialFactors::<T>::scratch(cols.len())
        }
    })
}
pub(super) fn query(
    plan: &SparseSymbolic,
    tag: i32,
    out: *mut usize,
    par: Par<'_>,
) -> Result<(), String> {
    let analysis = &plan.plan()?.analysis;
    let req = match tag {
        0 => requirement::<f32>(analysis, &par)?,
        1 => requirement::<f64>(analysis, &par)?,
        2 => requirement::<c32>(analysis, &par)?,
        3 => requirement::<c64>(analysis, &par)?,
        _ => return Err("invalid scalar tag".into()),
    };
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("query requires aligned two-entry usize buffer".into());
    }
    // SAFETY: caller preserves two writable usize entries; address/span checked.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
pub(super) fn run(
    owner: &mut SparseFactor,
    analysis: &SparseSymbolic,
    matrix: &SparseOwner,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<usize, String> {
    let plan = analysis.plan()?;
    let analysis = &plan.analysis;
    let req = match matrix.data()? {
        SparseData::F32(_) => requirement::<f32>(analysis, &par)?,
        SparseData::F64(_) => requirement::<f64>(analysis, &par)?,
        SparseData::C32(_) => requirement::<c32>(analysis, &par)?,
        SparseData::C64(_) => requirement::<c64>(analysis, &par)?,
    };
    scratch.stack(req)?;
    let result = match matrix.data()? {
        SparseData::F32(a) => {
            numeric_with_par(plan, a, Some(&mut *scratch), par)?.map(Factors::F32)
        }
        SparseData::F64(a) => {
            numeric_with_par(plan, a, Some(&mut *scratch), par)?.map(Factors::F64)
        }
        SparseData::C32(a) => {
            numeric_with_par(plan, a, Some(&mut *scratch), par)?.map(Factors::C32)
        }
        SparseData::C64(a) => {
            numeric_with_par(plan, a, Some(&mut *scratch), par)?.map(Factors::C64)
        }
    };
    match result {
        Ok(f) => {
            let hint = rank_hint::prepare(&f, scratch.stack(rank_hint::requirements(&f))?);
            owner.factors = Some(f);
            owner.qr_rank_hint = hint;
            Ok(0)
        }
        Err(info) => Ok(info),
    }
}
#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn sparse_qr_factor_requirements_team(
        &self,
        tag: i32,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| team.execute(|par| query(self, tag, out, par))?)
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_qr_factor_workspace_team(
        &mut self,
        analysis: &SparseSymbolic,
        matrix: &SparseOwner,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
    ) -> Result<usize, String> {
        guarded(|| team.execute(|par| run(self, analysis, matrix, scratch, par))?)
    }
}

#[julia]
impl SparseSymbolic {
    #[julia]
    pub fn sparse_factor_requirements(&self, tag: i32, out: *mut usize) -> Result<(), String> {
        guarded(|| query(self, tag, out, Par::Seq))
    }
    #[julia]
    pub fn sparse_factor_requirements_team(
        &self,
        tag: i32,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| team.execute(|par| query(self, tag, out, par))?)
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_factor_workspace(
        &mut self,
        analysis: &SparseSymbolic,
        matrix: &SparseOwner,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| run(self, analysis, matrix, scratch, Par::Seq))
    }
    #[julia]
    pub fn sparse_factor_workspace_team(
        &mut self,
        analysis: &SparseSymbolic,
        matrix: &SparseOwner,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
    ) -> Result<usize, String> {
        guarded(|| team.execute(|par| run(self, analysis, matrix, scratch, par))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::NativeValue;
    use crate::sparse_factors::Plan;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::{linalg::qr, SparseColMat, Triplet};
    use faer::traits::math_utils::copy;
    use std::sync::Arc;

    fn check<T: NativeValue>() {
        for width in [1, 2] {
            let mut team = OwnedTeam::new();
            team.team_initialize(width).unwrap();
            for (m, n) in [(0, 0), (3, 0), (7, 4), (96, 64)] {
                let dense = faer::Mat::from_fn(m, n, |i, j| {
                    T::from_parts(
                        if i == j {
                            4.0
                        } else {
                            0.01 * ((i + 2 * j) as f64).sin()
                        },
                        0.02 * ((2 * i + j) as f64).cos(),
                    )
                });
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| {
                        let a = &dense;
                        (0..m).map(move |i| Triplet {
                            row: i,
                            col: j,
                            val: copy(&a[(i, j)]),
                        })
                    })
                    .collect();
                let a = SparseColMat::try_new_from_triplets(m, n, &entries).unwrap();
                let symbolic =
                    Arc::new(qr::factorize_symbolic_qr(a.symbolic(), Default::default()).unwrap());
                let plan = SparseSymbolic {
                    plan: Some(Plan {
                        analysis: Analysis::Qr(symbolic.clone()),
                        pattern: a.symbolic().to_owned().unwrap(),
                        lower: true,
                    }),
                };
                let mut out = [0usize; 2];
                plan.sparse_qr_factor_requirements_team(T::tag(), out.as_mut_ptr(), &team)
                    .unwrap();
                let f = team
                    .execute(|par| {
                        let req = qr_construction_req::<T>(&symbolic, &par);
                        assert_eq!(out, [req.size_bytes(), req.align_bytes()]);
                        let mut mem = MemBuffer::new(req);
                        crate::sparse_qr_owned::Factors::new_in_parallel(
                            symbolic.clone(),
                            a.as_ref(),
                            MemStack::new(&mut mem),
                            par,
                        )
                        .unwrap()
                    })
                    .unwrap();
                let actual = f.reconstruct().unwrap();
                for j in 0..n {
                    for i in 0..m {
                        let (x, y) = actual[(i, j)].parts();
                        let (u, v) = dense[(i, j)].parts();
                        assert!((x - u).hypot(y - v) < 2e-4, "shape {m}x{n}, width {width}");
                    }
                }
                assert!(plan
                    .sparse_qr_factor_requirements_team(T::tag(), std::ptr::null_mut(), &team)
                    .is_err());
            }
            team.team_close().unwrap();
        }
    }
    fn check_simplicial<T: NativeValue>() {
        for width in [1, 2] {
            let mut team = OwnedTeam::new();
            team.team_initialize(width).unwrap();
            for (m, n) in [(0, 0), (3, 0), (7, 4), (96, 64)] {
                let dense = faer::Mat::from_fn(m, n, |i, j| {
                    T::from_parts(
                        if i == j {
                            5.0
                        } else {
                            0.01 * ((i + 2 * j) as f64).sin()
                        },
                        0.02 * ((2 * i + j) as f64).cos(),
                    )
                });
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| {
                        let a = &dense;
                        (0..m).map(move |i| Triplet {
                            row: i,
                            col: j,
                            val: copy(&a[(i, j)]),
                        })
                    })
                    .collect();
                let a = SparseColMat::try_new_from_triplets(m, n, &entries).unwrap();
                let ids: Vec<_> = (0..n).rev().collect();
                let permutation = faer::perm::Perm::new_checked(
                    ids.clone().into_boxed_slice(),
                    ids.into_boxed_slice(),
                    n,
                );
                let symbolic =
                    crate::sparse_qr_components::Plan::new(a.symbolic(), permutation.as_ref())
                        .unwrap();
                let plan = SparseSymbolic {
                    plan: Some(Plan {
                        analysis: Analysis::SimplicialQr(symbolic.clone()),
                        pattern: a.symbolic().to_owned().unwrap(),
                        lower: true,
                    }),
                };
                let mut out = [0usize; 2];
                plan.sparse_qr_factor_requirements_team(T::tag(), out.as_mut_ptr(), &team)
                    .unwrap();
                let req = crate::sparse_qr_components::Factors::<T>::scratch(&symbolic)
                    .or(rank_hint::scratch(n));
                assert_eq!(out, [req.size_bytes(), req.align_bytes()]);
                let mut mem = MemBuffer::new(req);
                let f = crate::sparse_qr_components::Factors::new_in(
                    &symbolic,
                    a.as_ref(),
                    MemStack::new(&mut mem),
                )
                .unwrap();
                let actual = f.reconstruct();
                for j in 0..n {
                    for i in 0..m {
                        let (x, y) = actual[(i, j)].parts();
                        let (u, v) = dense[(i, j)].parts();
                        assert!((x - u).hypot(y - v) < 2e-4);
                    }
                }
            }
            team.team_close().unwrap();
        }
    }
    #[test]
    fn simplicial_construction_f32() {
        check_simplicial::<f32>();
    }
    #[test]
    fn simplicial_construction_f64() {
        check_simplicial::<f64>();
    }
    #[test]
    fn simplicial_construction_c32() {
        check_simplicial::<c32>();
    }
    #[test]
    fn simplicial_construction_c64() {
        check_simplicial::<c64>();
    }
    fn check_cholesky<T: NativeValue>() {
        use faer::sparse::linalg::{cholesky, SupernodalThreshold};
        use faer::Side;
        for width in [1, 2] {
            let mut team = OwnedTeam::new();
            team.team_initialize(width).unwrap();
            for n in [0, 1, 4, 64] {
                let dense = faer::Mat::from_fn(n, n, |i, j| {
                    T::from_parts(
                        if i == j {
                            5.0
                        } else {
                            0.001 * ((i + j) as f64).sin()
                        },
                        0.001 * (i as f64 - j as f64).sin(),
                    )
                });
                for lower in [false, true] {
                    let side = if lower { Side::Lower } else { Side::Upper };
                    let entries: Vec<_> = (0..n)
                        .flat_map(|j| {
                            let dense = &dense;
                            (0..n).map(move |i| Triplet {
                                row: i,
                                col: j,
                                val: if if lower { i >= j } else { i <= j } {
                                    copy(&dense[(i, j)])
                                } else {
                                    T::from_parts(f64::NAN, f64::NAN)
                                },
                            })
                        })
                        .collect();
                    let a = SparseColMat::try_new_from_triplets(n, n, &entries).unwrap();
                    for threshold in [
                        SupernodalThreshold::AUTO,
                        SupernodalThreshold::FORCE_SIMPLICIAL,
                        SupernodalThreshold::FORCE_SUPERNODAL,
                    ] {
                        let symbolic = Arc::new(
                            cholesky::factorize_symbolic_cholesky(
                                a.symbolic(),
                                side,
                                cholesky::SymmetricOrdering::Identity,
                                cholesky::CholeskySymbolicParams {
                                    supernodal_flop_ratio_threshold: threshold,
                                    ..Default::default()
                                },
                            )
                            .unwrap(),
                        );
                        let plan = SparseSymbolic {
                            plan: Some(Plan {
                                analysis: Analysis::Llt(symbolic.clone()),
                                pattern: a.symbolic().to_owned().unwrap(),
                                lower,
                            }),
                        };
                        let mut out = [0usize; 2];
                        plan.sparse_factor_requirements_team(T::tag(), out.as_mut_ptr(), &team)
                            .unwrap();
                        let mut scratch = OwnedScratch::new();
                        scratch.scratch_reserve(out[0], out[1]).unwrap();
                        let matrix = crate::sparse::Storage::Csc(a.clone());
                        let numeric = team
                            .execute(|par| {
                                let req = symbolic.factorize_numeric_llt_scratch::<T>(
                                    par.shape(),
                                    Default::default(),
                                );
                                assert_eq!(out, [req.size_bytes(), req.align_bytes()]);
                                numeric_with_par(
                                    plan.plan().unwrap(),
                                    &matrix,
                                    Some(&mut scratch),
                                    par,
                                )
                                .unwrap()
                                .unwrap()
                            })
                            .unwrap();
                        let expected =
                            faer::Mat::from_fn(n, 2, |i, j| T::from_parts((i + j + 1) as f64, 0.1));
                        let rhs = &dense * &expected;
                        let actual =
                            crate::sparse_factors::solve(&numeric, rhs.as_ref(), false, false)
                                .unwrap();
                        for j in 0..2 {
                            for i in 0..n {
                                let (x, y) = actual[(i, j)].parts();
                                let (u, v) = expected[(i, j)].parts();
                                assert!((x - u).hypot(y - v) < 3e-4);
                            }
                        }
                        let triangular: Vec<_> = entries
                            .iter()
                            .filter(|e| {
                                if lower {
                                    e.row >= e.col
                                } else {
                                    e.row <= e.col
                                }
                            })
                            .map(|e| Triplet {
                                row: e.row,
                                col: e.col,
                                val: copy(&e.val),
                            })
                            .collect();
                        let triangular =
                            SparseColMat::try_new_from_triplets(n, n, &triangular).unwrap();
                        let mut values =
                            vec![faer::traits::math_utils::zero::<T>(); symbolic.len_val()];
                        team.execute(|mut par| {
                            let req = symbolic.factorize_numeric_llt_scratch::<T>(
                                par.shape(),
                                Default::default(),
                            );
                            symbolic
                                .factorize_numeric_llt(
                                    &mut values,
                                    triangular.as_ref(),
                                    side,
                                    Default::default(),
                                    par.reborrow(),
                                    scratch.stack(req).unwrap(),
                                    Default::default(),
                                )
                                .unwrap();
                            assert_eq!(
                                crate::allocation_probe::count(|| {
                                    symbolic
                                        .factorize_numeric_llt(
                                            &mut values,
                                            triangular.as_ref(),
                                            side,
                                            Default::default(),
                                            par,
                                            scratch.stack(req).unwrap(),
                                            Default::default(),
                                        )
                                        .unwrap();
                                }),
                                0
                            );
                        })
                        .unwrap();
                    }
                }
            }
            team.team_close().unwrap();
        }
    }
    #[test]
    fn failed_cholesky_construction_preserves_native_revision() {
        use crate::sparse::Storage;
        use faer::sparse::linalg::cholesky;
        let entries: Vec<_> = (0..4)
            .flat_map(|j| {
                (0..4).map(move |i| Triplet {
                    row: i,
                    col: j,
                    val: if i == j { 4.0 } else { 0.1 },
                })
            })
            .collect();
        let a = SparseColMat::try_new_from_triplets(4, 4, &entries).unwrap();
        let symbolic = Arc::new(
            cholesky::factorize_symbolic_cholesky(
                a.symbolic(),
                faer::Side::Lower,
                cholesky::SymmetricOrdering::Identity,
                cholesky::CholeskySymbolicParams {
                    supernodal_flop_ratio_threshold:
                        faer::sparse::linalg::SupernodalThreshold::FORCE_SUPERNODAL,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let plan = SparseSymbolic {
            plan: Some(Plan {
                analysis: Analysis::Llt(symbolic),
                pattern: a.symbolic().to_owned().unwrap(),
                lower: true,
            }),
        };
        let good = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(a))),
        };
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut req = [0usize; 2];
        plan.sparse_factor_requirements_team(1, req.as_mut_ptr(), &team)
            .unwrap();
        assert!(plan
            .sparse_factor_requirements_team(1, std::ptr::null_mut(), &team)
            .is_err());
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req[0], req[1]).unwrap();
        let mut owner = SparseFactor::new();
        assert_eq!(
            owner
                .sparse_factor_workspace_team(&plan, &good, &mut work, &team)
                .unwrap(),
            0
        );
        let before = match owner.factors.as_ref().unwrap() {
            Factors::F64(crate::sparse_factors::Numeric::Llt(x)) => x.values.clone(),
            _ => unreachable!(),
        };
        let mut entries = entries;
        entries[0].val = -1.0;
        let bad = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(
                SparseColMat::try_new_from_triplets(4, 4, &entries).unwrap(),
            ))),
        };
        assert_eq!(
            owner
                .sparse_factor_workspace_team(&plan, &bad, &mut work, &team)
                .unwrap(),
            1
        );
        let mut short = OwnedScratch::new();
        short.scratch_reserve(req[0] - 1, req[1]).unwrap();
        assert!(owner
            .sparse_factor_workspace_team(&plan, &good, &mut short, &team)
            .is_err());
        team.team_close().unwrap();
        assert!(owner
            .sparse_factor_workspace_team(&plan, &good, &mut work, &team)
            .is_err());
        let after = match owner.factors.as_ref().unwrap() {
            Factors::F64(crate::sparse_factors::Numeric::Llt(x)) => &x.values,
            _ => unreachable!(),
        };
        assert_eq!(&before, after);
        assert_eq!(owner.qr_rank_hint, None);
    }
    #[test]
    fn cholesky_construction_f32() {
        check_cholesky::<f32>();
    }
    #[test]
    fn cholesky_construction_f64() {
        check_cholesky::<f64>();
    }
    #[test]
    fn cholesky_construction_c32() {
        check_cholesky::<c32>();
    }
    #[test]
    fn cholesky_construction_c64() {
        check_cholesky::<c64>();
    }
    fn check_lu<T: NativeValue>() {
        use faer::sparse::linalg::{lu, SupernodalThreshold};
        for width in [1, 2] {
            let mut team = OwnedTeam::new();
            team.team_initialize(width).unwrap();
            for n in [0, 1, 4, 64] {
                let dense = faer::Mat::from_fn(n, n, |i, j| {
                    T::from_parts(
                        if n - 1 - i == j {
                            5.0
                        } else {
                            0.001 * ((i + 2 * j) as f64).sin()
                        },
                        0.001 * ((2 * i + j) as f64).cos(),
                    )
                });
                let entries: Vec<_> = (0..n)
                    .flat_map(|j| {
                        let a = &dense;
                        (0..n).map(move |i| Triplet {
                            row: i,
                            col: j,
                            val: copy(&a[(i, j)]),
                        })
                    })
                    .collect();
                let a = SparseColMat::try_new_from_triplets(n, n, &entries).unwrap();
                let ids: Vec<_> = (0..n).rev().collect();
                let permutation = faer::perm::Perm::new_checked(
                    ids.clone().into_boxed_slice(),
                    ids.into_boxed_slice(),
                    n,
                );
                let expected =
                    faer::Mat::from_fn(n, 2, |i, j| T::from_parts((i + j + 1) as f64, 0.1));
                let rhs = &dense * &expected;
                for threshold in [
                    SupernodalThreshold::AUTO,
                    SupernodalThreshold::FORCE_SIMPLICIAL,
                    SupernodalThreshold::FORCE_SUPERNODAL,
                ] {
                    let symbolic = Arc::new(
                        lu::factorize_symbolic_lu_with_column_ordering(
                            a.symbolic(),
                            lu::LuSymbolicParams {
                                supernodal_flop_ratio_threshold: threshold,
                                ..Default::default()
                            },
                            Some(permutation.as_ref()),
                        )
                        .unwrap(),
                    );
                    let plan = SparseSymbolic {
                        plan: Some(Plan {
                            analysis: Analysis::Lu(symbolic.clone()),
                            pattern: a.symbolic().to_owned().unwrap(),
                            lower: true,
                        }),
                    };
                    let mut out = [0usize; 2];
                    plan.sparse_factor_requirements_team(T::tag(), out.as_mut_ptr(), &team)
                        .unwrap();
                    let mut scratch = OwnedScratch::new();
                    scratch.scratch_reserve(out[0], out[1]).unwrap();
                    let f = team
                        .execute(|par| {
                            let req = symbolic
                                .factorize_numeric_lu_scratch::<T>(par.shape(), Default::default());
                            assert_eq!(out, [req.size_bytes(), req.align_bytes()]);
                            crate::sparse_lu_owned::Factors::new_in(
                                symbolic.clone(),
                                a.as_ref(),
                                par,
                                scratch.stack(req).unwrap(),
                            )
                            .unwrap()
                        })
                        .unwrap();
                    let (l, u) = f.as_ref().factors().unwrap();
                    let product = &l.to_dense() * &u.to_dense();
                    let rows = f.as_ref().row_perm().arrays().0;
                    let cols = f.as_ref().col_perm().arrays().0;
                    for j in 0..n {
                        for i in 0..n {
                            let (x, y) = product[(i, j)].parts();
                            let (u, v) = dense[(rows[i], cols[j])].parts();
                            assert!((x - u).hypot(y - v) < 3e-4);
                        }
                    }
                    let actual = f.solve(rhs.as_ref(), false, false).unwrap();
                    for j in 0..2 {
                        for i in 0..n {
                            let (x, y) = actual[(i, j)].parts();
                            let (u, v) = expected[(i, j)].parts();
                            assert!((x - u).hypot(y - v) < 3e-4);
                        }
                    }
                }
                let plan = SparseSymbolic {
                    plan: Some(Plan {
                        analysis: Analysis::SimplicialLu(permutation),
                        pattern: a.symbolic().to_owned().unwrap(),
                        lower: true,
                    }),
                };
                let mut out = [0usize; 2];
                plan.sparse_factor_requirements_team(T::tag(), out.as_mut_ptr(), &team)
                    .unwrap();
                let req = crate::sparse_lu_components::SimplicialFactors::<T>::scratch(n);
                assert_eq!(out, [req.size_bytes(), req.align_bytes()]);
                let mut scratch = OwnedScratch::new();
                scratch.scratch_reserve(out[0], out[1]).unwrap();
                let matrix = crate::sparse::Storage::Csc(a);
                let f = team
                    .execute(|par| {
                        numeric_with_par(plan.plan().unwrap(), &matrix, Some(&mut scratch), par)
                            .unwrap()
                            .unwrap()
                    })
                    .unwrap();
                let actual = crate::sparse_factors::solve(&f, rhs.as_ref(), false, false).unwrap();
                for j in 0..2 {
                    for i in 0..n {
                        let (x, y) = actual[(i, j)].parts();
                        let (u, v) = expected[(i, j)].parts();
                        assert!((x - u).hypot(y - v) < 3e-4);
                    }
                }
            }
            team.team_close().unwrap();
        }
    }
    #[test]
    fn lu_construction_f32() {
        check_lu::<f32>();
    }
    #[test]
    fn lu_construction_f64() {
        check_lu::<f64>();
    }
    #[test]
    fn lu_construction_c32() {
        check_lu::<c32>();
    }
    #[test]
    fn lu_construction_c64() {
        check_lu::<c64>();
    }
    #[test]
    fn failed_lu_construction_preserves_native_revision() {
        use crate::sparse::Storage;
        use faer::sparse::linalg::lu;
        let entries: Vec<_> = (0..4)
            .flat_map(|j| {
                (0..4).map(move |i| Triplet {
                    row: i,
                    col: j,
                    val: if i == j { 4.0 } else { 0.1 },
                })
            })
            .collect();
        let a = SparseColMat::try_new_from_triplets(4, 4, &entries).unwrap();
        let symbolic = Arc::new(
            lu::factorize_symbolic_lu(
                a.symbolic(),
                lu::LuSymbolicParams {
                    supernodal_flop_ratio_threshold:
                        faer::sparse::linalg::SupernodalThreshold::FORCE_SUPERNODAL,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let plan = SparseSymbolic {
            plan: Some(Plan {
                analysis: Analysis::Lu(symbolic),
                pattern: a.symbolic().to_owned().unwrap(),
                lower: true,
            }),
        };
        let good = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(a))),
        };
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut req = [0usize; 2];
        plan.sparse_factor_requirements_team(1, req.as_mut_ptr(), &team)
            .unwrap();
        assert!(plan
            .sparse_factor_requirements_team(1, std::ptr::null_mut(), &team)
            .is_err());
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req[0], req[1]).unwrap();
        let mut owner = SparseFactor::new();
        assert_eq!(
            owner
                .sparse_factor_workspace_team(&plan, &good, &mut work, &team)
                .unwrap(),
            0
        );
        let before = match owner.factors.as_ref().unwrap() {
            Factors::F64(crate::sparse_factors::Numeric::Lu(x)) => x.as_ref().factors().unwrap(),
            _ => unreachable!(),
        };
        let zero: Vec<_> = entries
            .iter()
            .map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: 0.0,
            })
            .collect();
        let bad = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(
                SparseColMat::try_new_from_triplets(4, 4, &zero).unwrap(),
            ))),
        };
        assert_eq!(
            owner
                .sparse_factor_workspace_team(&plan, &bad, &mut work, &team)
                .unwrap(),
            1
        );
        let mut short = OwnedScratch::new();
        short.scratch_reserve(req[0] - 1, req[1]).unwrap();
        assert!(owner
            .sparse_factor_workspace_team(&plan, &good, &mut short, &team)
            .is_err());
        let diagonal: Vec<_> = entries
            .iter()
            .filter(|e| e.row == e.col)
            .map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: e.val,
            })
            .collect();
        let mismatch = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(
                SparseColMat::try_new_from_triplets(4, 4, &diagonal).unwrap(),
            ))),
        };
        assert!(owner
            .sparse_factor_workspace_team(&plan, &mismatch, &mut work, &team)
            .is_err());
        team.team_close().unwrap();
        assert!(owner
            .sparse_factor_workspace_team(&plan, &good, &mut work, &team)
            .is_err());
        let after = match owner.factors.as_ref().unwrap() {
            Factors::F64(crate::sparse_factors::Numeric::Lu(x)) => x.as_ref().factors().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!((&before.0.to_dense() - &after.0.to_dense()).norm_max(), 0.0);
        assert_eq!((&before.1.to_dense() - &after.1.to_dense()).norm_max(), 0.0);
        assert_eq!(owner.qr_rank_hint, None);
    }
    #[test]
    fn query_and_numeric_team_f32() {
        check::<f32>();
    }
    #[test]
    fn query_and_numeric_team_f64() {
        check::<f64>();
    }
    #[test]
    fn query_and_numeric_team_c32() {
        check::<c32>();
    }
    #[test]
    fn query_and_numeric_team_c64() {
        check::<c64>();
    }
}
