//! Numeric QR construction with the same scoped team in query and execution.
use super::{
    numeric_with_par, qr_construction_req, rank_hint, Analysis, Factors, SparseFactor,
    SparseSymbolic,
};
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner};
use crate::teams::OwnedTeam;
use crate::workspace::OwnedScratch;
use faer::{c32, c64, Par};
use rustcall_julia_macros::julia;

pub(super) fn query(
    plan: &SparseSymbolic,
    tag: i32,
    out: *mut usize,
    par: Par<'_>,
) -> Result<(), String> {
    let Analysis::Qr(sym) = &plan.plan()?.analysis else {
        return Err("factor workspace requires auto/supernodal QR".into());
    };
    let req = match tag {
        0 => qr_construction_req::<f32>(sym, &par),
        1 => qr_construction_req::<f64>(sym, &par),
        2 => qr_construction_req::<c32>(sym, &par),
        3 => qr_construction_req::<c64>(sym, &par),
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
    let Analysis::Qr(sym) = &plan.analysis else {
        return Err("factor workspace requires auto/supernodal QR".into());
    };
    let req = match matrix.data()? {
        SparseData::F32(_) => qr_construction_req::<f32>(sym, &par),
        SparseData::F64(_) => qr_construction_req::<f64>(sym, &par),
        SparseData::C32(_) => qr_construction_req::<c32>(sym, &par),
        SparseData::C64(_) => qr_construction_req::<c64>(sym, &par),
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
