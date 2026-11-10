//! External native scratch and scoped teams for compact reflector application.
use super::{apply_in, requirement, Data, OwnedHouseholderSequence, Sequence};
use crate::teams::with_team;
use crate::views::{action_into, source as source_view, spec_shape, Arm, Borrowed, Spec};
use crate::workspace::scratch_or_temporary;
use crate::{
    guarded,
    typed::{MatrixData, TypedMatrix},
    OwnedMatrix,
};
use crate::{teams::OwnedTeam, workspace::OwnedScratch};
use faer::traits::ComplexField;
use faer::{MatMut, Par};
use rustcall_julia_macros::julia;
pub(super) fn validate(
    source: &OwnedHouseholderSequence,
    n: usize,
    k: usize,
    blocksize: usize,
    tag: i32,
) -> Result<(), String> {
    macro_rules! check {
        ($s:expr,$t:expr) => {{
            let s = $s;
            if s.basis.ncols() != k
                || s.factor.nrows() != blocksize
                || s.basis.nrows() != n
                || tag != $t
            {
                return Err(
                    "Householder native metadata does not match requested scalar/shape".into(),
                );
            }
        }};
    }
    match source.data()? {
        Data::F32(s) => check!(s, 0),
        Data::F64(s) => check!(s, 1),
        Data::C32(s) => check!(s, 2),
        Data::C64(s) => check!(s, 3),
    }
    Ok(())
}
fn query(
    source: &OwnedHouseholderSequence,
    n: usize,
    k: usize,
    blocksize: usize,
    tag: i32,
    extent: usize,
    right: bool,
    transpose: bool,
    out: *mut usize,
) -> Result<(), String> {
    validate(source, n, k, blocksize, tag)?;
    let req = match source.data()? {
        Data::F32(s) => requirement(s, extent, right, transpose),
        Data::F64(s) => requirement(s, extent, right, transpose),
        Data::C32(s) => requirement(s, extent, right, transpose),
        Data::C64(s) => requirement(s, extent, right, transpose),
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
impl OwnedHouseholderSequence {
    #[julia]
    pub fn sequence_requirements(
        &self,
        n: usize,
        k: usize,
        blocksize: usize,
        tag: i32,
        extent: usize,
        right: bool,
        transpose: bool,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| query(self, n, k, blocksize, tag, extent, right, transpose, out))
    }
}
fn run<T: ComplexField>(
    s: &Sequence<T>,
    a: MatMut<'_, T>,
    work: &mut OwnedScratch,
    right: bool,
    transpose: bool,
    conjugate: bool,
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
    apply_in(s, a, right, transpose, conjugate, par, stack)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sequence_workspace(
        &mut self,
        source: &OwnedHouseholderSequence,
        work: &mut OwnedScratch,
        n: usize,
        k: usize,
        blocksize: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                validate(source, n, k, blocksize, tag)?;
                match source.data()? {
                    Data::F64(s) => run(
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
    pub fn sequence_workspace(
        &mut self,
        source: &OwnedHouseholderSequence,
        work: &mut OwnedScratch,
        n: usize,
        k: usize,
        blocksize: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugate: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                validate(source, n, k, blocksize, tag)?;
                match (
                    source.data()?,
                    self.data.as_mut().ok_or("matrix is not initialized")?,
                ) {
                    (Data::F32(s), MatrixData::F32(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (Data::F64(s), MatrixData::F64(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (Data::C32(s), MatrixData::C32(a)) => {
                        run(s, a.as_mut(), work, right, transpose, conjugate, par)
                    }
                    (Data::C64(s), MatrixData::C64(a)) => {
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

/// `dest = alpha*Q*src + beta*dest` (`right`: `alpha*src*Q + beta*dest`)
/// for the sequence `Q` in the orientation `transpose`/`conjugate`, written
/// by faer directly in any view destination; a null `src` applies `Q` in
/// place. See [`crate::views::action_into`] for the coefficient and overlap
/// rules. Scratch capacity is checked before the destination is written.
///
/// # Safety
/// Julia roots the destination, source, descriptors and coefficients.
#[allow(clippy::too_many_arguments)]
#[julia]
pub fn sequence_mul_into(
    dest: *mut u8,
    dd: *const isize,
    src: *const u8,
    sd: *const isize,
    source: &OwnedHouseholderSequence,
    right: bool,
    transpose: bool,
    conjugate: bool,
    alpha: *const u8,
    beta: *const u8,
    alpha_zero: bool,
    beta_zero: bool,
    work: Option<&mut OwnedScratch>,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    #[allow(clippy::too_many_arguments)]
    unsafe fn go<T: Arm>(
        s: &Sequence<T>,
        dest: *mut u8,
        ds: Spec,
        src: *const u8,
        ss: Option<Spec>,
        coefficients: (*const u8, *const u8, bool, bool),
        orientation: (bool, bool, bool),
        work: Option<&mut OwnedScratch>,
        par: Par,
    ) -> Result<(), String> {
        let (right, transpose, conjugate) = orientation;
        let (rows, cols) = spec_shape(ds);
        if (if right { cols } else { rows }) != s.basis.nrows() {
            return Err("Householder application dimensions must match".into());
        }
        let req = requirement(s, if right { rows } else { cols }, right, transpose);
        let (alpha, beta, alpha_zero, beta_zero) = coefficients;
        scratch_or_temporary(work, req, |work| {
            action_into::<T>(dest, ds, src, ss, alpha, beta, alpha_zero, beta_zero, |c| {
                apply_in(s, c, right, transpose, conjugate, par, work.stack(req)?)
            })
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
        let o = (right, transpose, conjugate);
        with_team(team, |par| match (source.data()?, source_view(dest, ds)?) {
            (Data::F32(s), Borrowed::F32(_)) => go(s, dest, ds, src, ss, k, o, work, par),
            (Data::F64(s), Borrowed::F64(_)) => go(s, dest, ds, src, ss, k, o, work, par),
            (Data::C32(s), Borrowed::C32(_)) => go(s, dest, ds, src, ss, k, o, work, par),
            (Data::C64(s), Borrowed::C64(_)) => go(s, dest, ds, src, ss, k, o, work, par),
            _ => Err("Householder scalar types must match".into()),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::load;
    use super::*;
    use faer::traits::math_utils::*;
    use faer::{c32, c64, Mat};
    fn exercise<T: ComplexField>() {
        let basis = Mat::from_fn(7, 3, |i, j| {
            if i < j {
                zero::<T>()
            } else if i == j {
                one::<T>()
            } else {
                from_f64::<T>(0.1 * (i + j + 1) as f64)
            }
        });
        let tau = (0..3)
            .map(|j| {
                let mut t = zero::<T::Real>();
                for i in j..7 {
                    t = t + abs2(&basis[(i, j)]);
                }
                from_real::<T>(&(t / from_f64::<T::Real>(2.0)))
            })
            .collect::<Vec<_>>();
        let packed = (0..3)
            .flat_map(|j| (0..7).map(move |i| (i, j)))
            .map(|(i, j)| copy(&basis[(i, j)]))
            .collect::<Vec<_>>();
        let s = unsafe {
            load(
                packed.as_ptr().cast(),
                7,
                3,
                tau.as_ptr().cast(),
                2,
                std::ptr::null(),
                Par::Seq,
            )
            .unwrap()
        };
        let mut reference = Mat::<T>::identity(7, 7);
        for j in 0..3 {
            let h = Mat::from_fn(7, 7, |i, l| {
                (if i == l { one::<T>() } else { zero::<T>() })
                    - &basis[(i, j)] * conj(&basis[(l, j)]) / &tau[j]
            });
            reference = &reference * &h;
        }
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
    fn mul_into_runs_on_caller_scratch_with_only_the_kernel_allocations() {
        use crate::allocation_probe::count;
        use crate::views::raw_descriptor;
        let (m, k) = (7usize, 3usize);
        let basis = Mat::from_fn(m, k, |i, j| {
            if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                0.1 * (i + j + 1) as f64
            }
        });
        let tau: Vec<f64> = (0..k)
            .map(|j| (j..m).map(|i| basis[(i, j)] * basis[(i, j)]).sum::<f64>() / 2.0)
            .collect();
        let packed: Vec<f64> = (0..k)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .map(|(i, j)| basis[(i, j)])
            .collect();
        let s = unsafe {
            load::<f64>(
                packed.as_ptr().cast(),
                m,
                k,
                tau.as_ptr().cast(),
                2,
                std::ptr::null(),
                Par::Seq,
            )
        }
        .unwrap();
        let owner = OwnedHouseholderSequence {
            data: Some(Data::F64(s.clone())),
        };
        let req = requirement(&s, 3, false, false);
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let b = Mat::from_fn(m, 3, |i, j| (1 + i + 2 * j) as f64);
        let bd = raw_descriptor(m, 3, b.col_stride(), 1);
        let dd = raw_descriptor(m, 3, 9, 1);
        let mut julia = vec![0.0f64; 9 * 3];
        // Bare kernel on the same padded layout.
        let kernel = {
            let mut probe = vec![1.0f64; 9 * 3];
            let view = MatMut::from_column_major_slice_with_stride_mut(&mut probe, m, 3, 9);
            count(|| {
                apply_in(
                    &s,
                    view,
                    false,
                    false,
                    false,
                    Par::Seq,
                    scratch.stack(req).unwrap(),
                )
                .unwrap()
            })
        };
        // Previous staged entry: owned result plus its own MemBuffer.
        let staged = {
            let mut owned = OwnedMatrix { matrix: b.clone() };
            count(|| owned.sequence_apply(&owner, false, false, false).unwrap())
        };
        let (alpha, beta) = (2.0f64, 0.5f64);
        let mut call = |beta_zero: bool| {
            count(|| {
                sequence_mul_into(
                    julia.as_mut_ptr().cast(),
                    dd.as_ptr(),
                    b.as_ptr().cast(),
                    bd.as_ptr(),
                    &owner,
                    false,
                    false,
                    false,
                    (&alpha as *const f64).cast(),
                    (&beta as *const f64).cast(),
                    false,
                    beta_zero,
                    Some(&mut scratch),
                    None,
                )
                .unwrap()
            })
        };
        assert_eq!(call(true), kernel);
        assert_eq!(call(false), kernel + 1);
        assert!(staged >= kernel + 1);
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
