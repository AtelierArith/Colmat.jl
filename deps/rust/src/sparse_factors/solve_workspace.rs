//! Borrowed sparse LU/LLT solves with caller-owned scratch and execution teams.
use super::{Factors, Numeric, SparseFactor};
use crate::permutations::OwnedPermutation;
use crate::teams::OwnedTeam;
use crate::typed::{MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::sparse::linalg::{cholesky::LltRef, lu::simplicial};
use faer::traits::ComplexField;
use faer::{Conj, Mat, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

fn descriptor<T: ComplexField>(f: &Numeric<T>) -> Result<(usize, usize, i32), String> {
    Ok(match f {
        Numeric::Lu(x) => {
            let s = x.as_ref().symbolic();
            (s.nrows(), s.ncols(), 0)
        }
        Numeric::SimplicialLu(x) => (x.numeric.nrows(), x.numeric.nrows(), 0),
        Numeric::Llt(x) => (x.symbolic.nrows(), x.symbolic.nrows(), 2),
        _ => return Err("direct solve requires sparse LU or Cholesky".into()),
    })
}
fn validate(f: &SparseFactor, m: usize, n: usize, tag: i32, kind: i32) -> Result<(), String> {
    let (dims, actual) = match f.factors()? {
        Factors::F32(x) => (descriptor(x)?, 0),
        Factors::F64(x) => (descriptor(x)?, 1),
        Factors::C32(x) => (descriptor(x)?, 2),
        Factors::C64(x) => (descriptor(x)?, 3),
    };
    if dims != (m, n, kind) || tag != actual || m != n {
        return Err("sparse direct factor native metadata mismatch".into());
    }
    Ok(())
}
fn requirement<T: ComplexField>(
    f: &Numeric<T>,
    k: usize,
    op: i32,
    p: ParShape,
) -> Result<StackReq, String> {
    if !(0..4).contains(&op) {
        return Err("invalid sparse solve transform".into());
    }
    let (m, _, _) = descriptor(f)?;
    if m == 0 || k == 0 {
        return Ok(StackReq::EMPTY);
    }
    let t = op == 1 || op == 2;
    Ok(match f {
        Numeric::Lu(x) => {
            if t {
                x.as_ref()
                    .symbolic()
                    .solve_transpose_in_place_scratch::<T>(k, p)
            } else {
                x.as_ref().symbolic().solve_in_place_scratch::<T>(k, p)
            }
        }
        Numeric::SimplicialLu(_) => {
            if t {
                simplicial::solve_transpose_in_place_scratch::<usize, T>(m, k, p)
            } else {
                simplicial::solve_in_place_scratch::<usize, T>(m, k, p)
            }
        }
        Numeric::Llt(x) => x.symbolic.solve_in_place_scratch::<T>(k, p),
        _ => unreachable!(),
    })
}
fn query(f: &SparseFactor, k: usize, op: i32, p: ParShape, out: *mut usize) -> Result<(), String> {
    let req = match f.factors()? {
        Factors::F32(x) => requirement(x, k, op, p)?,
        Factors::F64(x) => requirement(x, k, op, p)?,
        Factors::C32(x) => requirement(x, k, op, p)?,
        Factors::C64(x) => requirement(x, k, op, p)?,
    };
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("query requires aligned two-entry usize buffer".into());
    }
    // SAFETY: caller preserves two writable usize entries; alignment/span checked.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
fn run<T: ComplexField>(
    f: &Numeric<T>,
    b: MatRef<'_, T>,
    s: &mut OwnedScratch,
    k: usize,
    op: i32,
    p: Par,
) -> Result<Mat<T>, String> {
    let (n, _, _) = descriptor(f)?;
    if b.shape() != (n, k) {
        return Err("sparse solve RHS native shape mismatch".into());
    }
    if !b.is_all_finite() {
        return Err("solve RHS must be finite".into());
    }
    let stack = s.stack(requirement(f, k, op, p.shape())?)?;
    let mut out = b.to_owned();
    if n == 0 || k == 0 {
        return Ok(out);
    }
    let t = op == 1 || op == 2;
    let c = op == 2 || op == 3;
    let conj = if c { Conj::Yes } else { Conj::No };
    match f {
        Numeric::Lu(x) => {
            if t {
                x.as_ref()
                    .solve_transpose_in_place_with_conj(conj, out.as_mut(), p, stack)
            } else {
                x.as_ref()
                    .solve_in_place_with_conj(conj, out.as_mut(), p, stack)
            }
        }
        Numeric::SimplicialLu(x) => {
            if t {
                x.numeric.solve_transpose_in_place_with_conj(
                    x.rows.as_ref(),
                    x.cols.as_ref(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            } else {
                x.numeric.solve_in_place_with_conj(
                    x.rows.as_ref(),
                    x.cols.as_ref(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            }
        }
        Numeric::Llt(x) => LltRef::new(&x.symbolic, &x.values).solve_in_place_with_conj(
            if t ^ c { Conj::Yes } else { Conj::No },
            out.as_mut(),
            p,
            stack,
        ),
        _ => unreachable!(),
    }
    Ok(out)
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_direct_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| validate(self, m, n, tag, kind))
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_direct_requirements(
        &self,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(self, m, n, tag, kind)?;
            query(self, k, op, p.shape(), out)
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_direct_workspace(
        &mut self,
        f: &SparseFactor,
        b: &OwnedMatrix,
        s: &mut OwnedScratch,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(f, m, n, tag, kind)?;
            let Factors::F64(x) = f.factors()? else {
                return Err("sparse solve scalar mismatch".into());
            };
            let result = run(x, b.matrix.as_ref(), s, k, op, p)?;
            self.matrix = result;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_direct_workspace(
        &mut self,
        f: &SparseFactor,
        b: &TypedMatrix,
        s: &mut OwnedScratch,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(f, m, n, tag, kind)?;
            let result = match (f.factors()?, b.data()?) {
                (Factors::F32(x), MatrixData::F32(b)) => {
                    MatrixData::F32(run(x, b.as_ref(), s, k, op, p)?)
                }
                (Factors::F64(x), MatrixData::F64(b)) => {
                    MatrixData::F64(run(x, b.as_ref(), s, k, op, p)?)
                }
                (Factors::C32(x), MatrixData::C32(b)) => {
                    MatrixData::C32(run(x, b.as_ref(), s, k, op, p)?)
                }
                (Factors::C64(x), MatrixData::C64(b)) => {
                    MatrixData::C64(run(x, b.as_ref(), s, k, op, p)?)
                }
                _ => return Err("sparse solve scalar mismatch".into()),
            };
            self.data = Some(result);
            Ok(())
        })
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_direct_requirements_team(
        &self,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(self, m, n, tag, kind)?;
                query(self, k, op, p.shape(), out)
            })?
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_direct_workspace_team(
        &mut self,
        f: &SparseFactor,
        b: &OwnedMatrix,
        s: &mut OwnedScratch,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(f, m, n, tag, kind)?;
                let Factors::F64(x) = f.factors()? else {
                    return Err("sparse solve scalar mismatch".into());
                };
                let result = run(x, b.matrix.as_ref(), s, k, op, p)?;
                self.matrix = result;
                Ok(())
            })?
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_direct_workspace_team(
        &mut self,
        f: &SparseFactor,
        b: &TypedMatrix,
        s: &mut OwnedScratch,
        k: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(f, m, n, tag, kind)?;
                let result = match (f.factors()?, b.data()?) {
                    (Factors::F32(x), MatrixData::F32(b)) => {
                        MatrixData::F32(run(x, b.as_ref(), s, k, op, p)?)
                    }
                    (Factors::F64(x), MatrixData::F64(b)) => {
                        MatrixData::F64(run(x, b.as_ref(), s, k, op, p)?)
                    }
                    (Factors::C32(x), MatrixData::C32(b)) => {
                        MatrixData::C32(run(x, b.as_ref(), s, k, op, p)?)
                    }
                    (Factors::C64(x), MatrixData::C64(b)) => {
                        MatrixData::C64(run(x, b.as_ref(), s, k, op, p)?)
                    }
                    _ => return Err("sparse solve scalar mismatch".into()),
                };
                self.data = Some(result);
                Ok(())
            })?
        })
    }
}

// Numeric LLT retains the symbolic permutation independently of its analysis owner.
fn llt_permutation<T: ComplexField>(f: &Numeric<T>) -> Result<faer::perm::Perm<usize>, String> {
    let Numeric::Llt(x) = f else {
        return Err("requires sparse Cholesky".into());
    };
    let n = x.symbolic.nrows();
    Ok(match x.symbolic.perm() {
        Some(p) => {
            let (a, b) = p.arrays();
            faer::perm::Perm::new_checked(
                a.to_vec().into_boxed_slice(),
                b.to_vec().into_boxed_slice(),
                n,
            )
        }
        None => faer::perm::Perm::new_checked(
            (0..n).collect::<Vec<_>>().into_boxed_slice(),
            (0..n).collect::<Vec<_>>().into_boxed_slice(),
            n,
        ),
    })
}
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn sparse_numeric_cholesky_permutation(&mut self, f: &SparseFactor) -> Result<(), String> {
        guarded(|| {
            let p = match f.factors()? {
                Factors::F32(x) => llt_permutation(x)?,
                Factors::F64(x) => llt_permutation(x)?,
                Factors::C32(x) => llt_permutation(x)?,
                Factors::C64(x) => llt_permutation(x)?,
            };
            self.data = Some(p);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::traits::math_utils::*;
    fn check<T: ComplexField>() {
        let a = SparseColMat::<usize, T>::try_new_from_triplets(
            3,
            3,
            &[
                Triplet {
                    row: 0,
                    col: 0,
                    val: from_f64::<T>(4.0),
                },
                Triplet {
                    row: 1,
                    col: 1,
                    val: from_f64::<T>(5.0),
                },
                Triplet {
                    row: 2,
                    col: 2,
                    val: from_f64::<T>(6.0),
                },
                Triplet {
                    row: 1,
                    col: 0,
                    val: from_f64::<T>(0.4),
                },
                Triplet {
                    row: 0,
                    col: 2,
                    val: from_f64::<T>(0.2),
                },
            ],
        )
        .unwrap();
        let permutation = faer::perm::Perm::new_checked(
            vec![2, 0, 1].into_boxed_slice(),
            vec![1, 2, 0].into_boxed_slice(),
            3,
        );
        let factors = [
            Numeric::Lu(a.sp_lu(Par::Seq).unwrap()),
            Numeric::SimplicialLu(
                crate::sparse_lu_components::SimplicialFactors::new(
                    a.as_ref(),
                    permutation.as_ref(),
                )
                .unwrap(),
            ),
        ];
        let b = Mat::from_fn(3, 4, |i, j| from_f64::<T>((1 + i + j) as f64));
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for f in &factors {
            for op in 0..4 {
                for parallel in [false, true] {
                    let exercise = |mut p: Par| {
                        let req = requirement(f, 4, op, p.shape()).unwrap();
                        let mut work = OwnedScratch::new();
                        work.scratch_reserve(req.size_bytes(), req.align_bytes())
                            .unwrap();
                        let x = run(f, b.as_ref(), &mut work, 4, op, p.reborrow()).unwrap();
                        let dense = a.to_dense();
                        let effective = match op {
                            0 => dense.clone(),
                            1 => dense.transpose().to_owned(),
                            2 => dense.adjoint().to_owned(),
                            _ => dense.conjugate().to_owned(),
                        };
                        let error = (&effective * &x - &b).norm_max();
                        assert!(error < from_f64::<T::Real>(1e-4));
                        if req.size_bytes() > 0 {
                            let mut short = OwnedScratch::new();
                            short
                                .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                                .unwrap();
                            assert!(run(f, b.as_ref(), &mut short, 4, op, p.reborrow()).is_err());
                        }
                        assert!(run(f, b.as_ref(), &mut work, 3, op, p.reborrow()).is_err());
                        work.scratch_close().unwrap();
                        assert!(run(f, b.as_ref(), &mut work, 4, op, p.reborrow()).is_err());
                    };
                    if parallel {
                        team.execute(exercise).unwrap();
                    } else {
                        exercise(Par::Seq);
                    }
                }
            }
        }
        assert!(requirement(&factors[0], 1, 4, ParShape::Seq).is_err());
    }
    #[test]
    fn f32_direct() {
        check::<f32>();
    }
    #[test]
    fn f64_direct() {
        check::<f64>();
    }
    #[test]
    fn c32_direct() {
        check::<faer::c32>();
    }
    #[test]
    fn c64_direct() {
        check::<faer::c64>();
    }
}
