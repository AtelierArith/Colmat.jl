//! External native scratch and scoped teams for compact reflector application.
use super::{apply_in, requirement, OwnedHouseholder, Reflection, ReflectionData};
use crate::{
    guarded,
    typed::{MatrixData, TypedMatrix},
    OwnedMatrix,
};
use crate::{teams::OwnedTeam, workspace::OwnedScratch};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatMut, Par};
use rustcall_julia_macros::julia;
fn validate(source: &OwnedHouseholder, n: usize, tag: i32) -> Result<(), String> {
    macro_rules! check {
        ($s:expr,$t:expr) => {{
            let s = $s;
            if s.basis.nrows() != n || tag != $t {
                return Err(
                    "Householder native metadata does not match requested scalar/shape".into(),
                );
            }
        }};
    }
    match source.data()? {
        ReflectionData::F32(s) => check!(s, 0),
        ReflectionData::F64(s) => check!(s, 1),
        ReflectionData::C32(s) => check!(s, 2),
        ReflectionData::C64(s) => check!(s, 3),
    }
    Ok(())
}
fn query(
    source: &OwnedHouseholder,
    n: usize,
    tag: i32,
    extent: usize,
    right: bool,
    transpose: bool,
    out: *mut usize,
) -> Result<(), String> {
    validate(source, n, tag)?;
    let req = match source.data()? {
        ReflectionData::F32(s) => requirement(s, extent, right, transpose),
        ReflectionData::F64(s) => requirement(s, extent, right, transpose),
        ReflectionData::C32(s) => requirement(s, extent, right, transpose),
        ReflectionData::C64(s) => requirement(s, extent, right, transpose),
    };
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("query needs aligned two-entry output".into());
    }
    // SAFETY: Julia preserves the writable two-entry UInt buffer, after the address checks above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
impl OwnedHouseholder {
    #[julia]
    pub fn householder_requirements(
        &self,
        n: usize,
        tag: i32,
        extent: usize,
        right: bool,
        transpose: bool,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| query(self, n, tag, extent, right, transpose, out))
    }
}
fn run<T: ComplexField>(
    s: &Reflection<T>,
    a: MatMut<'_, T>,
    work: &mut OwnedScratch,
    right: bool,
    transpose: bool,
    _conjugate: bool,
    par: Par,
) -> Result<(), String> {
    if (if right { a.ncols() } else { a.nrows() }) != s.basis.nrows() {
        return Err("Householder application dimensions must match".into());
    }
    let req = requirement(
        s,
        if right { a.nrows() } else { a.ncols() },
        right,
        transpose,
    );
    let stack = work.stack(req)?;
    apply_in(s, a, right, par, stack)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn householder_workspace(
        &mut self,
        source: &OwnedHouseholder,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate(source, n, tag)?;
            match source.data()? {
                ReflectionData::F64(s) => run(
                    s,
                    self.matrix.as_mut(),
                    work,
                    right,
                    transpose,
                    conjugate,
                    par,
                ),
                _ => Err("Householder scalar types must match".into()),
            }
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn householder_workspace_team(
        &mut self,
        source: &OwnedHouseholder,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate(source, n, tag)?;
                match source.data()? {
                    ReflectionData::F64(s) => run(
                        s,
                        self.matrix.as_mut(),
                        work,
                        right,
                        transpose,
                        conjugate,
                        par,
                    ),
                    _ => Err("Householder scalar types must match".into()),
                }
            })?
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn householder_workspace(
        &mut self,
        source: &OwnedHouseholder,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate(source, n, tag)?;
            match (
                source.data()?,
                self.data.as_mut().ok_or("matrix is not initialized")?,
            ) {
                (ReflectionData::F32(s), MatrixData::F32(a)) => {
                    run(s, a.as_mut(), work, right, transpose, conjugate, par)
                }
                (ReflectionData::F64(s), MatrixData::F64(a)) => {
                    run(s, a.as_mut(), work, right, transpose, conjugate, par)
                }
                (ReflectionData::C32(s), MatrixData::C32(a)) => {
                    run(s, a.as_mut(), work, right, transpose, conjugate, par)
                }
                (ReflectionData::C64(s), MatrixData::C64(a)) => {
                    run(s, a.as_mut(), work, right, transpose, conjugate, par)
                }
                _ => Err("Householder scalar types must match".into()),
            }
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn householder_workspace_team(
        &mut self,
        source: &OwnedHouseholder,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate(source, n, tag)?;
                match (
                    source.data()?,
                    self.data.as_mut().ok_or("matrix is not initialized")?,
                ) {
                    (ReflectionData::F32(s), MatrixData::F32(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (ReflectionData::F64(s), MatrixData::F64(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (ReflectionData::C32(s), MatrixData::C32(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (ReflectionData::C64(s), MatrixData::C64(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    _ => Err("Householder scalar types must match".into()),
                }
            })?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::make;
    use super::*;
    use faer::{c32, c64, Mat};
    fn exercise<T: ComplexField + super::super::RealDivide>() {
        let input = (0..7)
            .map(|i| from_f64::<T>((i + 1) as f64))
            .collect::<Vec<_>>();
        let s = make(&input);
        let reference = Mat::from_fn(7, 7, |i, j| {
            (if i == j { one::<T>() } else { zero::<T>() })
                - &s.basis[(i, 0)] * conj(&s.basis[(j, 0)]) / from_real::<T>(&s.tau)
        });
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for right in [false, true] {
            for transpose in [false, true] {
                for conjugated in [false, true] {
                    let b = Mat::from_fn(
                        if right { 3 } else { 7 },
                        if right { 7 } else { 3 },
                        |i, j| from_f64::<T>((1 + i + 2 * j) as f64),
                    );
                    let effective = Mat::from_fn(7, 7, |i, j| {
                        let x = if transpose {
                            &reference[(j, i)]
                        } else {
                            &reference[(i, j)]
                        };
                        if conjugated {
                            conj(x)
                        } else {
                            copy(x)
                        }
                    });
                    // Elementary reflection orientation is encoded in its owner, not run flags.
                    let effective = reference.clone();
                    let expected = if right {
                        &b * &effective
                    } else {
                        &effective * &b
                    };
                    let req = requirement(&s, 3, right, transpose);
                    let mut scratch = OwnedScratch::new();
                    scratch
                        .scratch_reserve(req.size_bytes(), req.align_bytes())
                        .unwrap();
                    for threaded in [false, true] {
                        let mut out = b.clone();
                        if threaded {
                            team.execute(|par| {
                                run(
                                    &s,
                                    out.as_mut(),
                                    &mut scratch,
                                    right,
                                    transpose,
                                    conjugated,
                                    par,
                                )
                            })
                            .unwrap()
                            .unwrap();
                        } else {
                            run(
                                &s,
                                out.as_mut(),
                                &mut scratch,
                                right,
                                transpose,
                                conjugated,
                                Par::Seq,
                            )
                            .unwrap();
                        }
                        for i in 0..out.nrows() {
                            for j in 0..out.ncols() {
                                assert!(
                                    abs(&(&out[(i, j)] - &expected[(i, j)]))
                                        < from_f64::<T::Real>(10000.0) * eps::<T::Real>()
                                );
                            }
                        }
                    }
                    if req.size_bytes() > 0 {
                        let mut short = OwnedScratch::new();
                        short
                            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                            .unwrap();
                        let mut out = b.clone();
                        assert!(run(
                            &s,
                            out.as_mut(),
                            &mut short,
                            right,
                            transpose,
                            conjugated,
                            Par::Seq
                        )
                        .is_err());
                        assert_eq!(out, b);
                    }
                }
            }
        }
        team.team_close().unwrap();
    }
    #[test]
    fn external_scratch_f32() {
        exercise::<f32>();
    }
    #[test]
    fn external_scratch_f64() {
        exercise::<f64>();
    }
    #[test]
    fn external_scratch_c32() {
        exercise::<c32>();
    }
    #[test]
    fn external_scratch_c64() {
        exercise::<c64>();
    }
}
