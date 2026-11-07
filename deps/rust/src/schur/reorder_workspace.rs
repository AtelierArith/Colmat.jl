//! Saved Schur reordering with caller stack and checked owner references.
use super::construction::buffer;
use super::{reordered_in, SchurScalar};
use crate::teams::with_team;
use crate::{
    guarded,
    teams::OwnedTeam,
    typed::{MatrixData, TypedMatrix},
    workspace::OwnedScratch,
    OwnedMatrix,
};
use faer::dyn_stack::StackReq;
use faer::traits::ComplexField;
use faer::{Mat, MatRef, Par};
use rustcall_julia_macros::julia;
pub(super) fn requirement<T: ComplexField>(n: usize) -> StackReq {
    if n == 0 {
        StackReq::EMPTY
    } else {
        StackReq::new::<u8>(n).and(faer::linalg::evd::reorder_schur_scratch::<T>(n))
    }
}
fn query(n: usize, tag: i32, out: *mut usize) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = match tag {
        0 => requirement::<f32>(n),
        1 => requirement::<f64>(n),
        2 => requirement::<faer::c32>(n),
        3 => requirement::<faer::c64>(n),
        _ => return Err("Schur reorder scalar tag is invalid".into()),
    };
    // SAFETY: preserved Julia UInt[2] has writable span/alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn schur_reorder_requirements(
    n: usize,
    tag: i32,
    out: *mut usize,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    // Selected faer reorder kernels accept, but currently ignore, the token.
    // Their declared requirement depends only on n and scalar, not team width.
    guarded(|| with_team(team, |_| query(n, tag, out)))
}

fn run<T: SchurScalar>(
    t: MatRef<'_, T>,
    q: MatRef<'_, T>,
    work: &mut OwnedScratch,
    select: *const u8,
    n: usize,
    values: *mut u8,
    par: Par,
) -> Result<(Mat<T>, Mat<T>), String> {
    if t.shape() != (n, n) || q.shape() != (n, n) {
        return Err("Schur current native dimensions do not match requested metadata".into());
    }
    buffer::<u8>(select, n)?;
    buffer::<T::Value>(values, n)?;
    if n > 0 {
        let a = select as usize;
        let b = values as usize;
        // Span products/address additions were checked by buffer above.
        if a < b + n * core::mem::size_of::<T::Value>() && b < a + n {
            return Err("Schur selection and output spans must not overlap".into());
        }
    }
    let stack = work.stack(requirement::<T>(n))?;
    // SAFETY: Julia preserves n UInt8 flags; native dimensions and address span
    // are checked before slice formation. No borrowed pointer escapes this call.
    let flags = if n == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(select, n) }
    };
    let (h, z) = reordered_in(t, q, flags, par, stack)?;
    // SAFETY: Julia preserves n writable complex eigenvalues, typed span/alignment
    // was checked above. Values are written only after complete successful reorder.
    unsafe {
        T::values(h.as_ref(), values);
    }
    Ok((h, z))
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn schur_reorder_workspace(
        &mut self,
        t: &OwnedMatrix,
        q: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        select: *const u8,
        n: usize,
        values: *mut u8,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("Schur requested scalar type does not match its native owner".into());
            }
            let execute = |par: Par<'_>| {
                run(
                    t.matrix.as_ref(),
                    q.matrix.as_ref(),
                    work,
                    select,
                    n,
                    values,
                    par,
                )
            };
            let (h, z) = with_team(team, execute)?;
            self.matrix = h;
            out_q.matrix = z;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_reorder_workspace(
        &mut self,
        t: &TypedMatrix,
        q: &TypedMatrix,
        out_q: &mut TypedMatrix,
        work: &mut OwnedScratch,
        select: *const u8,
        n: usize,
        values: *mut u8,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($t:expr,$q:expr,$variant:ident) => {{
                    let execute =
                        |par: Par<'_>| run($t.as_ref(), $q.as_ref(), work, select, n, values, par);
                    let (h, z) = with_team(team, execute)?;
                    self.data = Some(MatrixData::$variant(h));
                    out_q.data = Some(MatrixData::$variant(z));
                }};
            }
            match (t.data()?, q.data()?) {
                (MatrixData::F32(t), MatrixData::F32(q)) if tag == 0 => one!(t, q, F32),
                (MatrixData::F64(t), MatrixData::F64(q)) if tag == 1 => one!(t, q, F64),
                (MatrixData::C32(t), MatrixData::C32(q)) if tag == 2 => one!(t, q, C32),
                (MatrixData::C64(t), MatrixData::C64(q)) if tag == 3 => one!(t, q, C64),
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
    fn reorder_preflight_preserves_owners_and_values() {
        let t = OwnedMatrix {
            matrix: Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 3.0 + i as f64 } else { 0.0 }),
        };
        let q = OwnedMatrix {
            matrix: Mat::<f64>::identity(2, 2),
        };
        let mut out = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 42.0),
        };
        let mut out_q = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 43.0),
        };
        let mut values = [c64::new(44.0, 0.0); 2];
        let flags = [0u8, 1];
        let req = requirement::<f64>(2);
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        assert!(out
            .schur_reorder_workspace(
                &t,
                &q,
                &mut out_q,
                &mut work,
                flags.as_ptr(),
                2,
                values.as_mut_ptr().cast(),
                1,
                None
            )
            .is_err());
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        for (n, tag, select, ptr) in [
            (3, 1, flags.as_ptr(), values.as_mut_ptr().cast()),
            (2, 0, flags.as_ptr(), values.as_mut_ptr().cast()),
            (2, 1, core::ptr::null(), values.as_mut_ptr().cast()),
            (2, 1, flags.as_ptr(), core::ptr::null_mut()),
            (
                2,
                1,
                flags.as_ptr(),
                values.as_mut_ptr().cast::<u8>().wrapping_add(1),
            ),
        ] {
            assert!(out
                .schur_reorder_workspace(&t, &q, &mut out_q, &mut work, select, n, ptr, tag, None)
                .is_err());
        }
        let invalid = [0u8, 2];
        assert!(out
            .schur_reorder_workspace(
                &t,
                &q,
                &mut out_q,
                &mut work,
                invalid.as_ptr(),
                2,
                values.as_mut_ptr().cast(),
                1,
                None
            )
            .is_err());
        let overlap = out
            .schur_reorder_workspace(
                &t,
                &q,
                &mut out_q,
                &mut work,
                values.as_ptr().cast(),
                2,
                values.as_mut_ptr().cast(),
                1,
                None,
            )
            .unwrap_err();
        assert!(overlap.contains("overlap"));
        let team = OwnedTeam::new();
        assert!(out
            .schur_reorder_workspace(
                &t,
                &q,
                &mut out_q,
                &mut work,
                flags.as_ptr(),
                2,
                values.as_mut_ptr().cast(),
                1,
                Some(&team)
            )
            .is_err());
        assert_eq!(out.matrix, Mat::<f64>::full(2, 2, 42.0));
        assert_eq!(out_q.matrix, Mat::<f64>::full(2, 2, 43.0));
        assert_eq!(values, [c64::new(44.0, 0.0); 2]);
        assert!(schur_reorder_requirements(2, 1, core::ptr::null_mut(), None).is_err());
    }
    fn allocations<T: SchurScalar>(mut par: Par) {
        let n = 6;
        let h = Mat::<T>::from_fn(n, n, |i, j| {
            from_f64::<T>(if i == j {
                1.0 + i as f64
            } else if i < j {
                0.01
            } else {
                0.0
            })
        });
        let q = Mat::<T>::identity(n, n);
        let flags = [0u8, 1, 0, 1, 0, 1];
        let mut memory = faer::dyn_stack::MemBuffer::new(requirement::<T>(n));
        for rep in 0..3 {
            let mut result = None;
            let mut step = || {
                result = Some(
                    reordered_in(
                        h.as_ref(),
                        q.as_ref(),
                        &flags,
                        par.reborrow(),
                        faer::dyn_stack::MemStack::new(&mut memory),
                    )
                    .unwrap(),
                );
            };
            if rep == 0 {
                step();
            } else {
                // Exactly the two independent owning factor matrices allocate;
                // native flags and block-swap workspace use the caller stack.
                assert_eq!(crate::allocation_probe::count(step), 2);
            }
            let (t, z) = result.unwrap();
            let err = (&z * &t * z.adjoint() - &h).norm_l2();
            assert!(err <= eps::<T::Real>() * from_f64::<T::Real>(256.0) * h.norm_l2());
            for (i, d) in [2.0, 4.0, 6.0].into_iter().enumerate() {
                assert!(
                    abs(&sub(&t[(i, i)], &from_f64::<T>(d)))
                        < eps::<T::Real>() * from_f64::<T::Real>(256.0)
                );
            }
        }
    }
    #[test]
    fn caller_stack_reorder_only_allocates_owning_outputs() {
        allocations::<f32>(Par::Seq);
        allocations::<f64>(Par::Seq);
        allocations::<c32>(Par::Seq);
        allocations::<c64>(Par::Seq);
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.execute(|mut par| {
            allocations::<f32>(par.reborrow());
            allocations::<f64>(par.reborrow());
            allocations::<c32>(par.reborrow());
            allocations::<c64>(par);
        })
        .unwrap();
    }
}
