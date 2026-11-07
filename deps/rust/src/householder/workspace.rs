//! External native scratch and scoped teams for compact reflector application.
use super::{apply_in, requirement, OwnedHouseholder, Reflection, ReflectionData};
use crate::views::{
    action_into, source as source_view, spec_shape, Arm, Borrowed, Spec,
};
use crate::workspace::scratch_or_temporary;
use crate::{
    guarded,
    typed::{MatrixData, TypedMatrix},
    OwnedMatrix,
};
use crate::{
    teams::{with_team, OwnedTeam},
    workspace::OwnedScratch,
};
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
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
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
        })
    }
}
#[julia]
impl OwnedMatrix {}
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
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
        })
    }
}
#[julia]
impl TypedMatrix {}

/// `dest = alpha*H*src + beta*dest` (`right`: `alpha*src*H + beta*dest`)
/// written by faer directly in any view destination; a null `src` applies
/// the reflection in place. See [`crate::views::action_into`] for the
/// coefficient and overlap rules. `work` is the caller's scratch (a
/// temporary one is reserved without it); its capacity is checked before
/// the destination is written.
///
/// # Safety
/// Julia roots the destination, source, descriptors and coefficients.
#[allow(clippy::too_many_arguments)]
#[julia]
pub fn householder_mul_into(
    dest: *mut u8,
    dd: *const isize,
    src: *const u8,
    sd: *const isize,
    source: &OwnedHouseholder,
    right: bool,
    alpha: *const u8,
    beta: *const u8,
    alpha_zero: bool,
    beta_zero: bool,
    work: Option<&mut OwnedScratch>,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn go<T: Arm>(
        s: &Reflection<T>,
        dest: *mut u8,
        ds: Spec,
        src: *const u8,
        ss: Option<Spec>,
        coefficients: (*const u8, *const u8, bool, bool),
        right: bool,
        work: Option<&mut OwnedScratch>,
        par: Par,
    ) -> Result<(), String> {
        let (rows, cols) = spec_shape(ds);
        if (if right { cols } else { rows }) != s.basis.nrows() {
            return Err("Householder application dimensions must match".into());
        }
        let req = requirement(s, if right { rows } else { cols }, right, false);
        let (alpha, beta, alpha_zero, beta_zero) = coefficients;
        scratch_or_temporary(work, req, |work| {
            action_into::<T>(
                dest,
                ds,
                src,
                ss,
                alpha,
                beta,
                alpha_zero,
                beta_zero,
                |c| apply_in(s, c, right, par, work.stack(req)?),
            )
        })
    }
    guarded(|| unsafe {
        let ds = Spec::read(dd)?;
        let ss = if src.is_null() {
            None
        } else {
            Some(Spec::read(sd)?)
        };
        let k = (alpha, beta, alpha_zero, beta_zero);
        with_team(team, |par| match (source.data()?, source_view(dest, ds)?) {
            (ReflectionData::F32(s), Borrowed::F32(_)) => {
                go(s, dest, ds, src, ss, k, right, work, par)
            }
            (ReflectionData::F64(s), Borrowed::F64(_)) => {
                go(s, dest, ds, src, ss, k, right, work, par)
            }
            (ReflectionData::C32(s), Borrowed::C32(_)) => {
                go(s, dest, ds, src, ss, k, right, work, par)
            }
            (ReflectionData::C64(s), Borrowed::C64(_)) => {
                go(s, dest, ds, src, ss, k, right, work, par)
            }
            _ => Err("Householder scalar types must match".into()),
        })
    })
}
#[cfg(test)]
mod tests {
    use super::super::make;
    use super::*;
    use crate::allocation_probe;
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
    // Column-major Julia buffer descriptor (owner kind 4), f64.
    fn raw(m: usize, n: usize, ld: isize) -> Vec<isize> {
        vec![4, 0, 1, m as isize, 0, 1, n as isize, 0, 0, m as isize, n as isize, 1, ld, 1]
    }
    #[test]
    fn mul_into_writes_julia_memory_with_caller_scratch_without_allocating() {
        let input = (0..5).map(|i| (i + 1) as f64).collect::<Vec<_>>();
        let owner = OwnedHouseholder {
            data: Some(ReflectionData::F64(make(&input))),
        };
        let s = match owner.data().unwrap() {
            ReflectionData::F64(s) => s.clone(),
            _ => unreachable!(),
        };
        let h = Mat::from_fn(5, 5, |i, j| {
            (if i == j { 1.0 } else { 0.0 }) - s.basis[(i, 0)] * s.basis[(j, 0)] / s.tau
        });
        let b = Mat::from_fn(5, 3, |i, j| (1 + i + 2 * j) as f64);
        let bd = raw(5, 3, b.col_stride());
        let mut julia = vec![f64::NAN; 7 * 3];
        let dd = raw(5, 3, 7);
        let req = requirement(&s, 3, false, false);
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let (alpha, beta) = (2.0f64, 0.5f64);
        let src = b.as_ptr().cast::<u8>();
        let call = |julia: &mut Vec<f64>, beta_zero: bool, scratch: &mut OwnedScratch| {
            householder_mul_into(
                julia.as_mut_ptr().cast(),
                dd.as_ptr(),
                src,
                bd.as_ptr(),
                &owner,
                false,
                (&alpha as *const f64).cast(),
                (&beta as *const f64).cast(),
                false,
                beta_zero,
                Some(scratch),
                None,
            )
            .unwrap()
        };
        // beta == 0: B is copied into C and reflected there; C (NaN) is not read.
        let count = allocation_probe::count(|| call(&mut julia, true, &mut scratch));
        assert_eq!(count, 0);
        let expected = &h * &b * alpha;
        for j in 0..3 {
            for i in 0..5 {
                assert!((julia[i + 7 * j] - expected[(i, j)]).abs() < 1e-12);
            }
            assert!(julia[5 + 7 * j].is_nan() && julia[6 + 7 * j].is_nan());
        }
        // beta != 0: exactly one native product buffer.
        let before = Mat::from_fn(5, 3, |i, j| julia[i + 7 * j]);
        let count = allocation_probe::count(|| call(&mut julia, false, &mut scratch));
        assert_eq!(count, 1);
        let expected = &h * &b * alpha + &before * beta;
        for j in 0..3 {
            for i in 0..5 {
                assert!((julia[i + 7 * j] - expected[(i, j)]).abs() < 1e-10);
            }
        }
        // In place (null source) also runs on the caller scratch only.
        let mut c = b.clone();
        let cd = raw(5, 3, c.col_stride());
        let count = allocation_probe::count(|| {
            householder_mul_into(
                c.as_ptr_mut().cast(),
                cd.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                &owner,
                false,
                std::ptr::null(),
                std::ptr::null(),
                false,
                true,
                Some(&mut scratch),
                None,
            )
            .unwrap()
        });
        assert_eq!(count, 0);
        assert!((&c - &h * &b).norm_max() < 1e-12);
        // A short scratch fails before the destination is written.
        if req.size_bytes() > 0 {
            let mut short = OwnedScratch::new();
            short
                .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                .unwrap();
            let saved = julia.clone();
            assert!(householder_mul_into(
                julia.as_mut_ptr().cast(),
                dd.as_ptr(),
                src,
                bd.as_ptr(),
                &owner,
                false,
                (&alpha as *const f64).cast(),
                (&beta as *const f64).cast(),
                false,
                true,
                Some(&mut short),
                None,
            )
            .is_err());
            assert!(julia
                .iter()
                .zip(&saved)
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
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
