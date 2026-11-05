//! Reusable tridiagonal reduction and explicit Q construction workspace.
use super::workspace::buffer;
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::{self, evd, householder as hh};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
fn block_size<T: ComplexField>(n: usize, bs: usize) -> Result<usize, String> {
    if bs > n.saturating_sub(1).max(1) {
        return Err("invalid tridiagonal blocksize".into());
    }
    Ok(if n < 2 {
        0
    } else if bs == 0 {
        linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1)
            .min(n - 1)
            .max(1)
    } else {
        bs
    })
}
fn requirement<T: ComplexField>(n: usize, bs: usize, par: ParShape) -> Result<StackReq, String> {
    let bs = block_size::<T>(n, bs)?;
    let work = linalg::temp_mat_scratch::<T>(n, n);
    if n < 2 {
        return Ok(work);
    }
    Ok(work.and(linalg::temp_mat_scratch::<T>(bs, n - 1)).and(
        evd::tridiag::tridiag_in_place_scratch::<T>(n, par, Default::default()).or(
            hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n - 1, bs, n),
        ),
    ))
}
fn dispatch(n: usize, tag: i32, bs: usize, par: ParShape) -> Result<StackReq, String> {
    match tag {
        0 => requirement::<f32>(n, bs, par),
        1 => requirement::<f64>(n, bs, par),
        2 => requirement::<faer::c32>(n, bs, par),
        3 => requirement::<faer::c64>(n, bs, par),
        _ => Err("unsupported tridiagonal scalar".into()),
    }
}
fn query(n: usize, tag: i32, bs: usize, par: ParShape, out: *mut usize) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = dispatch(n, tag, bs, par)?;
    // SAFETY: caller preserves UInt[2]; span/alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn tridiagonalize_requirements(
    n: usize,
    tag: i32,
    bs: usize,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| query(n, tag, bs, ParShape::Seq, out))
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn tridiagonalize_team_requirements(
        &self,
        n: usize,
        tag: i32,
        bs: usize,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| self.execute(|par| query(n, tag, bs, par.shape(), out))?)
    }
}
unsafe fn compute<T: ComplexField>(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    qout: *mut u8,
    dout: *mut u8,
    eout: *mut u8,
    bs: usize,
    mut par: Par,
) -> Result<(), String> {
    let nn = n.checked_mul(n).ok_or("tridiagonal dimensions overflow")?;
    let ranges = [
        buffer::<T>(input, nn)?,
        buffer::<T>(qout, nn)?,
        buffer::<T::Real>(dout, n)?,
        buffer::<T::Real>(eout, n.saturating_sub(1))?,
    ];
    for i in 0..4 {
        for j in i + 1..4 {
            let (a, b) = ranges[i];
            let (c, d) = ranges[j];
            if a < b && c < d && a < d && c < b {
                return Err("tridiagonal reduction buffers overlap".into());
            }
        }
    }
    let stack = scratch.stack(requirement::<T>(n, bs, par.shape())?)?;
    if n == 0 {
        return Ok(());
    }
    let bs = block_size::<T>(n, bs)?;
    // SAFETY: caller preserves a contiguous n*n T input; spans checked above.
    let a = MatRef::from_raw_parts(input.cast::<T>(), n, n, 1, n as isize);
    crate::typed::finite(a)?;
    // SAFETY: every logical entry is copied before any read; native scalars.
    let (mut work, stack) = linalg::temp_mat_uninit::<T, _, _>(n, n, stack);
    work.as_mat_mut().copy_from(a);
    // SAFETY: caller preserves disjoint n*n output storage, checked above.
    // Julia may supply undef storage: initialize via raw writes before forming
    // a MatMut whose logical entries must already be valid T values.
    for k in 0..nn {
        qout.cast::<T>().add(k).write(zero::<T>());
    }
    let mut q = MatMut::from_raw_parts_mut(qout.cast::<T>(), n, n, 1, n as isize);
    for i in 0..n {
        q[(i, i)] = one::<T>();
    }
    if n > 1 {
        let scaling = crate::schur::scale_in(work.as_mat_mut());
        let (mut factor, stack) = linalg::temp_mat_zeroed::<T, _, _>(bs, n - 1, stack);
        evd::tridiag::tridiag_in_place(
            work.as_mat_mut(),
            factor.as_mat_mut(),
            par.reborrow(),
            stack,
            Default::default(),
        );
        crate::householder_sequence::apply_compact_in(
            work.as_mat_ref().submatrix(1, 0, n - 1, n - 1),
            factor.as_mat_ref(),
            q.rb_mut().subrows_mut(1, n - 1),
            par,
            stack,
        )?;
        if !T::IS_REAL {
            let mut phase = one::<T>();
            for i in 1..n {
                let e = &work.as_mat_ref()[(i, i - 1)];
                let norm = abs(e);
                if norm != zero::<T::Real>() {
                    phase = &phase * (e / from_real::<T>(&norm));
                    let pnorm = abs(&phase);
                    phase = &phase / from_real::<T>(&pnorm);
                }
                for row in 0..n {
                    q[(row, i)] = &q[(row, i)] * &phase;
                }
            }
        }
        if let Some(scale) = scaling {
            // Scaling is an exactly representable power of two. Real scalar
            // multiplication avoids complex division's squared denominator,
            // which can underflow while the restored components are finite.
            let inv_scale = recip(&scale);
            for i in 0..n {
                let v = copy(&work.as_mat_ref()[(i, i)]);
                work.as_mat_mut()[(i, i)] = mul_real(&v, &inv_scale);
                if i + 1 < n {
                    let v = copy(&work.as_mat_ref()[(i + 1, i)]);
                    work.as_mat_mut()[(i + 1, i)] = mul_real(&v, &inv_scale);
                }
            }
        }
    }
    for i in 0..n {
        dout.cast::<T::Real>()
            .add(i)
            .write(real(&work.as_mat_ref()[(i, i)]));
        if i + 1 < n {
            eout.cast::<T::Real>().add(i).write(if T::IS_REAL {
                real(&work.as_mat_ref()[(i + 1, i)])
            } else {
                abs(&work.as_mat_ref()[(i + 1, i)])
            });
        }
    }
    Ok(())
}
fn run(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    q: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
    par: Par,
) -> Result<(), String> {
    // SAFETY: each branch checks every buffer before typed access.
    unsafe {
        match tag {
            0 => compute::<f32>(scratch, input, n, q, d, e, bs, par),
            1 => compute::<f64>(scratch, input, n, q, d, e, bs, par),
            2 => compute::<faer::c32>(scratch, input, n, q, d, e, bs, par),
            3 => compute::<faer::c64>(scratch, input, n, q, d, e, bs, par),
            _ => Err("unsupported tridiagonal scalar".into()),
        }
    }
}
#[julia]
pub fn tridiagonalize_workspace(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    q: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
) -> Result<(), String> {
    guarded(|| run(scratch, input, n, q, d, e, tag, bs, Par::Seq))
}
#[julia]
pub fn tridiagonalize_workspace_team(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    q: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
    team: &OwnedTeam,
) -> Result<(), String> {
    guarded(|| team.execute(|par| run(scratch, input, n, q, d, e, tag, bs, par))?)
}
pub(super) fn allocate(
    input: *const u8,
    n: usize,
    q: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
) -> Result<(), String> {
    let req = dispatch(n, tag, bs, ParShape::Seq)?;
    let mut scratch = OwnedScratch::new();
    scratch.scratch_reserve(req.size_bytes(), req.align_bytes())?;
    run(&mut scratch, input, n, q, d, e, tag, bs, Par::Seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: ComplexField>() {
        let n = 4;
        let input: Vec<T> = (0..n * n)
            .map(|k| {
                from_f64::<T>(if k % n == k / n {
                    (k % n + 1) as f64
                } else {
                    0.0
                })
            })
            .collect();
        let mut q: Vec<T> = (0..n * n).map(|_| from_f64::<T>(-9.0)).collect();
        let mut d: Vec<T::Real> = (0..n).map(|_| from_f64::<T::Real>(-9.0)).collect();
        let mut e: Vec<T::Real> = (0..n - 1).map(|_| from_f64::<T::Real>(-9.0)).collect();
        let req = requirement::<T>(n, 2, ParShape::Seq).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                n,
                q.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(q.iter().all(|x| *x == from_f64::<T>(-9.0)));
        assert!(d
            .iter()
            .chain(e.iter())
            .all(|x| *x == from_f64::<T::Real>(-9.0)));
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                n,
                q.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
            .unwrap();
        }
        let allocations = crate::allocation_probe::count(|| unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                n,
                q.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
            .unwrap();
        });
        assert_eq!(allocations, 0);
        for i in 0..n {
            assert!(
                abs(&sub(&d[i], &from_f64::<T::Real>((i + 1) as f64))) < from_f64::<T::Real>(1e-5)
            );
        }
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                std::ptr::null(),
                n,
                q.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast::<u8>().add(1),
                n,
                q.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                n,
                input.as_ptr().cast_mut().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                2,
                Par::Seq,
            )
        }
        .is_err());
    }
    #[test]
    fn construction_f32() {
        check::<f32>();
    }
    #[test]
    fn construction_f64() {
        check::<f64>();
    }
    #[test]
    fn construction_c32() {
        check::<faer::c32>();
    }
    #[test]
    fn construction_c64() {
        check::<faer::c64>();
    }
}
