//! Caller-owned kernel workspace and scoped team for Hessenberg/Schur construction.
use super::{hessenberg_in, reduced_in, SchurScalar};
use crate::{
    guarded,
    teams::OwnedTeam,
    typed::{MatrixData, TypedMatrix},
    workspace::OwnedScratch,
    OwnedMatrix,
};
use faer::dyn_stack::StackReq;
use faer::linalg::{evd, householder as hh, qr::no_pivoting::factor as qr};
use faer::traits::ComplexField;
use faer::{Mat, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
pub(super) fn hessenberg_requirement<T: ComplexField>(n: usize, par: ParShape) -> StackReq {
    hessenberg_requirement_tuned::<T>(n, par, super::hessenberg_tuning::Policy::default())
}
pub(crate) fn hessenberg_requirement_tuned<T: ComplexField>(
    n: usize,
    par: ParShape,
    policy: super::hessenberg_tuning::Policy,
) -> StackReq {
    let par = policy.shape(par);
    if n <= 1 {
        return StackReq::EMPTY;
    }
    let b = qr::recommended_block_size::<T>(n - 1, n - 1);
    faer::linalg::temp_mat_scratch::<T>(b, n - 1).and(
        evd::hessenberg::hessenberg_in_place_scratch::<T>(n, b, par, policy.params::<T>()).or(
            hh::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(
                n - 1,
                b,
                n - 1,
            ),
        ),
    )
}
fn requirement<T: SchurScalar>(n: usize, stage: i32, par: ParShape) -> Result<StackReq, String> {
    match stage {
        0 => Ok(hessenberg_requirement::<T>(n, par).or(T::step_scratch(n, par))),
        1 => Ok(hessenberg_requirement::<T>(n, par)),
        2 => Ok(T::step_scratch(n, par)),
        _ => Err("Schur construction stage must be 0, 1 or 2".into()),
    }
}
pub(super) fn buffer<T>(p: *const u8, n: usize) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    let len = n
        .checked_mul(core::mem::size_of::<T>())
        .filter(|v| *v <= isize::MAX as usize)
        .ok_or("Schur output span is too large")?;
    if p.is_null()
        || (p as usize) % core::mem::align_of::<T>() != 0
        || (p as usize).checked_add(len).is_none()
    {
        return Err("Schur output pointer is null, unaligned or has an invalid span".into());
    }
    Ok(())
}
fn query(n: usize, tag: i32, stage: i32, par: ParShape, out: *mut usize) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = match tag {
        0 => requirement::<f32>(n, stage, par)?,
        1 => requirement::<f64>(n, stage, par)?,
        2 => requirement::<faer::c32>(n, stage, par)?,
        3 => requirement::<faer::c64>(n, stage, par)?,
        _ => return Err("Schur scalar tag is invalid".into()),
    };
    // SAFETY: preserved Julia UInt[2] storage; span and alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn schur_construction_requirements(
    n: usize,
    tag: i32,
    stage: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| query(n, tag, stage, ParShape::Seq, out))
}
#[julia]
pub fn schur_construction_team_requirements(
    team: &OwnedTeam,
    n: usize,
    tag: i32,
    stage: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| team.execute(|par| query(n, tag, stage, par.shape(), out))?)
}
fn validate(actual: (usize, usize, i32), m: usize, n: usize, tag: i32) -> Result<(), String> {
    if actual.2 != tag {
        return Err("Schur requested scalar type does not match its native owner".into());
    }
    if (actual.0, actual.1) != (m, n) {
        return Err("Schur current native dimensions do not match requested metadata".into());
    }
    Ok(())
}
fn run<T: SchurScalar>(
    a: MatRef<'_, T>,
    q: MatRef<'_, T>,
    work: &mut OwnedScratch,
    values: *mut u8,
    stage: i32,
    n: usize,
    mut par: Par,
) -> Result<(Mat<T>, Mat<T>), String> {
    if a.shape() != (n, n) || (stage == 2 && q.shape() != (n, n)) {
        return Err("Schur current native dimensions do not match requested metadata".into());
    }
    if stage != 1 {
        buffer::<T::Value>(values, n)?;
    }
    let stack = work.stack(requirement::<T>(n, stage, par.shape())?)?;
    let (h, z) = if stage == 2 {
        reduced_in(a, q, par, stack)?
    } else {
        let (mut h, mut z) = hessenberg_in(a, par.reborrow(), stack)?;
        if stage == 0 {
            T::step_in(&mut h, &mut z, par, stack)?;
        }
        (h, z)
    };
    if stage != 1 {
        // SAFETY: Julia preserves n complex output entries; tag/shape/span checked before execution.
        // The final extraction happens only after successful native computation.
        unsafe {
            T::values(h.as_ref(), values);
        }
    }
    Ok((h, z))
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn schur_input_validate(&self, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| validate((self.matrix.nrows(), self.matrix.ncols(), 1), m, n, tag))
    }
    #[julia]
    pub fn schur_construction_workspace(
        &mut self,
        a: &OwnedMatrix,
        q: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("Schur requested scalar type does not match its native owner".into());
            }
            let mut execute = |par: Par<'_>| {
                run(
                    a.matrix.as_ref(),
                    q.matrix.as_ref(),
                    work,
                    values,
                    stage,
                    n,
                    par,
                )
            };
            let (h, z) = execute(Par::Seq)?;
            self.matrix = h;
            out_q.matrix = z;
            Ok(())
        })
    }
    #[julia]
    pub fn schur_construction_workspace_team(
        &mut self,
        a: &OwnedMatrix,
        q: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("Schur requested scalar type does not match its native owner".into());
            }
            let execute = |par: Par<'_>| {
                run(
                    a.matrix.as_ref(),
                    q.matrix.as_ref(),
                    work,
                    values,
                    stage,
                    n,
                    par,
                )
            };
            let (h, z) = team.execute(execute)??;
            self.matrix = h;
            out_q.matrix = z;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_input_validate(&self, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            let actual = match self.data()? {
                MatrixData::F32(a) => (a.nrows(), a.ncols(), 0),
                MatrixData::F64(a) => (a.nrows(), a.ncols(), 1),
                MatrixData::C32(a) => (a.nrows(), a.ncols(), 2),
                MatrixData::C64(a) => (a.nrows(), a.ncols(), 3),
            };
            validate(actual, m, n, tag)
        })
    }
    #[julia]
    pub fn schur_construction_workspace(
        &mut self,
        a: &TypedMatrix,
        q: &TypedMatrix,
        out_q: &mut TypedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($a:expr,$q:expr,$variant:ident) => {{
                    let mut execute =
                        |par: Par<'_>| run($a.as_ref(), $q.as_ref(), work, values, stage, n, par);
                    let (h, z) = execute(Par::Seq)?;
                    self.data = Some(MatrixData::$variant(h));
                    out_q.data = Some(MatrixData::$variant(z));
                }};
            }
            match (a.data()?, q.data()?) {
                (MatrixData::F32(a), MatrixData::F32(q)) if tag == 0 => one!(a, q, F32),
                (MatrixData::F64(a), MatrixData::F64(q)) if tag == 1 => one!(a, q, F64),
                (MatrixData::C32(a), MatrixData::C32(q)) if tag == 2 => one!(a, q, C32),
                (MatrixData::C64(a), MatrixData::C64(q)) if tag == 3 => one!(a, q, C64),
                _ => return Err("Schur scalar types must match the requested scalar type".into()),
            }
            Ok(())
        })
    }
    #[julia]
    pub fn schur_construction_workspace_team(
        &mut self,
        a: &TypedMatrix,
        q: &TypedMatrix,
        out_q: &mut TypedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($a:expr,$q:expr,$variant:ident) => {{
                    let execute =
                        |par: Par<'_>| run($a.as_ref(), $q.as_ref(), work, values, stage, n, par);
                    let (h, z) = team.execute(execute)??;
                    self.data = Some(MatrixData::$variant(h));
                    out_q.data = Some(MatrixData::$variant(z));
                }};
            }
            match (a.data()?, q.data()?) {
                (MatrixData::F32(a), MatrixData::F32(q)) if tag == 0 => one!(a, q, F32),
                (MatrixData::F64(a), MatrixData::F64(q)) if tag == 1 => one!(a, q, F64),
                (MatrixData::C32(a), MatrixData::C32(q)) if tag == 2 => one!(a, q, C32),
                (MatrixData::C64(a), MatrixData::C64(q)) if tag == 3 => one!(a, q, C64),
                _ => return Err("Schur scalar types must match the requested scalar type".into()),
            }
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::traits::math_utils::*;
    use faer::{c32, c64};
    #[test]
    fn construction_preflight_preserves_owners_and_values() {
        let a = OwnedMatrix {
            matrix: Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 3.0 } else { 0.1 }),
        };
        let mut out = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 42.0),
        };
        let mut q = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 43.0),
        };
        let mut values = [c64::new(44.0, 0.0); 2];
        let req = requirement::<f64>(2, 0, ParShape::Seq).unwrap();
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        assert!(out
            .schur_construction_workspace(
                &a,
                &a,
                &mut q,
                &mut work,
                values.as_mut_ptr().cast(),
                0,
                2,
                1
            )
            .is_err());
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        for (n, tag, stage, ptr) in [
            (3, 1, 0, values.as_mut_ptr().cast()),
            (2, 0, 0, values.as_mut_ptr().cast()),
            (2, 1, 0, core::ptr::null_mut()),
            (2, 1, 0, values.as_mut_ptr().cast::<u8>().wrapping_add(1)),
            (2, 1, 3, values.as_mut_ptr().cast()),
        ] {
            assert!(out
                .schur_construction_workspace(&a, &a, &mut q, &mut work, ptr, stage, n, tag)
                .is_err());
        }
        let team = OwnedTeam::new();
        assert!(out
            .schur_construction_workspace_team(
                &a,
                &a,
                &mut q,
                &mut work,
                values.as_mut_ptr().cast(),
                0,
                2,
                1,
                &team
            )
            .is_err());
        assert_eq!(out.matrix, Mat::<f64>::full(2, 2, 42.0));
        assert_eq!(q.matrix, Mat::<f64>::full(2, 2, 43.0));
        assert_eq!(values, [c64::new(44.0, 0.0); 2]);
        assert!(schur_construction_requirements(2, 1, 0, core::ptr::null_mut()).is_err());
    }
    fn step_alloc<T: SchurScalar>(mut par: Par) {
        let n = 6;
        let a = Mat::<T>::from_fn(n, n, |i, j| {
            from_f64::<T>(if i == j { 2.0 + i as f64 } else { 0.01 })
        });
        let mut mem = faer::dyn_stack::MemBuffer::new(requirement::<T>(n, 0, par.shape()).unwrap());
        for rep in 0..3 {
            let (mut h, mut z) = hessenberg_in(
                a.as_ref(),
                par.reborrow(),
                faer::dyn_stack::MemStack::new(&mut mem),
            )
            .unwrap();
            let mut step = || {
                T::step_in(
                    &mut h,
                    &mut z,
                    par.reborrow(),
                    faer::dyn_stack::MemStack::new(&mut mem),
                )
                .unwrap()
            };
            if rep == 0 {
                step();
            } else {
                assert_eq!(crate::allocation_probe::count(step), 0);
            }
            let err = (&z * &h * z.adjoint() - &a).norm_l2();
            assert!(err <= eps::<T::Real>() * from_f64::<T::Real>(256.0) * a.norm_l2());
        }
    }
    #[test]
    fn eigenvalue_temporary_columns_use_stack() {
        step_alloc::<f32>(Par::Seq);
        step_alloc::<f64>(Par::Seq);
        step_alloc::<c32>(Par::Seq);
        step_alloc::<c64>(Par::Seq);
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.execute(|mut par| {
            step_alloc::<f32>(par.reborrow());
            step_alloc::<f64>(par.reborrow());
            step_alloc::<c32>(par.reborrow());
            step_alloc::<c64>(par);
        })
        .unwrap();
    }
}
