//! Dedicated tridiagonal EVD: caller scratch and ephemeral team authority.
use crate::teams::with_team;
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::{self, evd};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Auto, MatMut, Par, ParShape};
use rustcall_julia_macros::julia;
fn params<T: ComplexField>(
    threshold: usize,
) -> Result<faer::Spec<evd::SelfAdjointEvdParams, T>, String> {
    if threshold < 2 {
        return Err("tridiagonal recursion threshold must be at least two".into());
    }
    Ok(evd::SelfAdjointEvdParams {
        recursion_threshold: threshold,
        ..<evd::SelfAdjointEvdParams as Auto<T>>::auto()
    }
    .into())
}
fn requirement<T: ComplexField>(
    n: usize,
    compute: bool,
    threshold: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    let params = params::<T>(threshold)?;
    if n == 0 {
        return Ok(StackReq::EMPTY);
    }
    Ok(StackReq::all_of(&[
        linalg::temp_mat_scratch::<T>(n, 2),
        linalg::temp_mat_scratch::<T>(n, 1),
        evd::tridiagonal_self_adjoint_evd_scratch::<T>(
            n,
            if compute {
                evd::ComputeEigenvectors::Yes
            } else {
                evd::ComputeEigenvectors::No
            },
            par,
            params,
        ),
    ]))
}
fn dispatch(
    n: usize,
    tag: i32,
    compute: bool,
    threshold: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    match tag {
        0 => requirement::<f32>(n, compute, threshold, par),
        1 => requirement::<f64>(n, compute, threshold, par),
        2 => requirement::<faer::c32>(n, compute, threshold, par),
        3 => requirement::<faer::c64>(n, compute, threshold, par),
        _ => Err("unsupported tridiagonal scalar".into()),
    }
}
pub(super) fn buffer<T>(p: *const u8, n: usize) -> Result<(usize, usize), String> {
    let first = p as usize;
    if n == 0 {
        return Ok((first, first));
    }
    let bytes = n
        .checked_mul(core::mem::size_of::<T>())
        .filter(|&b| b <= isize::MAX as usize)
        .ok_or("tridiagonal buffer span overflow")?;
    let end = first
        .checked_add(bytes)
        .ok_or("tridiagonal buffer address overflow")?;
    if p.is_null() || first % core::mem::align_of::<T>() != 0 {
        return Err("tridiagonal buffer is null or unaligned".into());
    }
    Ok((first, end))
}
fn query(
    n: usize,
    tag: i32,
    compute: bool,
    threshold: usize,
    par: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = dispatch(n, tag, compute, threshold, par)?;
    // SAFETY: caller preserves UInt[2]; span and alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn tridiagonal_evd_requirements(
    n: usize,
    tag: i32,
    compute: bool,
    threshold: usize,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| {
        with_team(team, |par| {
            query(n, tag, compute, threshold, par.shape(), out)
        })
    })
}
unsafe fn compute<T: ComplexField>(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    vectors: *mut u8,
    want_vectors: bool,
    threshold: usize,
    par: Par<'_>,
) -> Result<(), String> {
    let nn = if want_vectors {
        n.checked_mul(n)
            .ok_or("tridiagonal vector dimensions overflow")?
    } else {
        0
    };
    let ranges = [
        buffer::<T>(d, n)?,
        buffer::<T>(e, n.saturating_sub(1))?,
        buffer::<T::Real>(values, n)?,
        buffer::<T>(vectors, nn)?,
    ];
    for i in 0..4 {
        for j in i + 1..4 {
            if i == 0 && j == 1 {
                continue;
            } // two immutable coefficient inputs may alias
            let (a, b) = ranges[i];
            let (c, d) = ranges[j];
            if a < b && c < d && a < d && c < b {
                return Err("tridiagonal input/output buffers overlap".into());
            }
        }
    }
    let stack = scratch.stack(requirement::<T>(n, want_vectors, threshold, par.shape())?)?;
    if n == 0 {
        return Ok(());
    }
    let (mut band, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 2, stack);
    for i in 0..n {
        band.as_mat_mut()[(i, 0)] = copy(&*d.cast::<T>().add(i));
        if i + 1 < n {
            band.as_mat_mut()[(i, 1)] = copy(&*e.cast::<T>().add(i));
        }
    }
    crate::typed::finite(band.as_mat_ref())?;
    for i in 0..n {
        if imag(&band.as_mat_ref()[(i, 0)]) != zero::<T::Real>() {
            return Err("Hermitian tridiagonal diagonal must be real".into());
        }
    }
    let scaling = crate::schur::scale_in(band.as_mat_mut());
    let (mut vals, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
    // SAFETY: caller preserves an independently allocated, aligned n*n T output.
    // All coefficient inputs were copied and checked before output is borrowed.
    let vecs = if want_vectors {
        Some(MatMut::from_raw_parts_mut(
            vectors.cast::<T>(),
            n,
            n,
            1,
            n as isize,
        ))
    } else {
        None
    };
    evd::tridiagonal_self_adjoint_evd(
        band.as_mat_ref().col(0).as_diagonal(),
        band.as_mat_ref().col(1).get(..n - 1).as_diagonal(),
        vals.as_mat_mut().col_mut(0).as_diagonal_mut(),
        vecs,
        par,
        stack,
        params::<T>(threshold)?,
    )
    .map_err(|e| format!("{e:?}"))?;
    for i in 0..n {
        let v = real(&vals.as_mat_ref()[(i, 0)]);
        values
            .cast::<T::Real>()
            .add(i)
            .write(if let Some(ref f) = scaling { v / f } else { v });
    }
    Ok(())
}
fn run(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    vectors: *mut u8,
    tag: i32,
    want_vectors: bool,
    threshold: usize,
    par: Par<'_>,
) -> Result<(), String> {
    // SAFETY: each branch uses the caller's scalar tag and checks all spans before access.
    unsafe {
        match tag {
            0 => compute::<f32>(
                scratch,
                d,
                e,
                n,
                values,
                vectors,
                want_vectors,
                threshold,
                par,
            ),
            1 => compute::<f64>(
                scratch,
                d,
                e,
                n,
                values,
                vectors,
                want_vectors,
                threshold,
                par,
            ),
            2 => compute::<faer::c32>(
                scratch,
                d,
                e,
                n,
                values,
                vectors,
                want_vectors,
                threshold,
                par,
            ),
            3 => compute::<faer::c64>(
                scratch,
                d,
                e,
                n,
                values,
                vectors,
                want_vectors,
                threshold,
                par,
            ),
            _ => Err("unsupported tridiagonal scalar".into()),
        }
    }
}
#[julia]
pub fn tridiagonal_evd_workspace(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    vectors: *mut u8,
    tag: i32,
    want_vectors: bool,
    threshold: usize,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            run(
                scratch,
                d,
                e,
                n,
                values,
                vectors,
                tag,
                want_vectors,
                threshold,
                par,
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: ComplexField>() {
        let n = 6;
        let d: Vec<T> = (0..n).map(|i| from_f64::<T>((i + 1) as f64)).collect();
        let e: Vec<T> = (0..n - 1).map(|_| zero::<T>()).collect();
        let mut vals: Vec<T::Real> = (0..n).map(|_| from_f64::<T::Real>(-99.0)).collect();
        let mut vecs: Vec<T> = (0..n * n).map(|_| from_f64::<T>(-99.0)).collect();
        let req = requirement::<T>(n, true, 2, ParShape::Seq).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        let mut run = |scratch: &mut OwnedScratch| unsafe {
            compute::<T>(
                scratch,
                d.as_ptr().cast(),
                e.as_ptr().cast(),
                n,
                vals.as_mut_ptr().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
        };
        assert!(run(&mut scratch).is_err());
        assert!(vals.iter().all(|v| *v == from_f64::<T::Real>(-99.0)));
        assert!(vecs.iter().all(|v| *v == from_f64::<T>(-99.0)));
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // Warm setup outside the calling-thread allocation observation.
        unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast(),
                e.as_ptr().cast(),
                n,
                vals.as_mut_ptr().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
            .unwrap();
        }
        let allocations = crate::allocation_probe::count(|| unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast(),
                e.as_ptr().cast(),
                n,
                vals.as_mut_ptr().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
            .unwrap();
        });
        assert_eq!(allocations, 0);
        for i in 0..n {
            assert!(
                abs(&sub(&vals[i], &from_f64::<T::Real>((i + 1) as f64)))
                    < from_f64::<T::Real>(1e-5)
            );
        }
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                std::ptr::null(),
                e.as_ptr().cast(),
                n,
                vals.as_mut_ptr().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast::<u8>().add(1),
                e.as_ptr().cast(),
                n,
                vals.as_mut_ptr().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast(),
                e.as_ptr().cast(),
                n,
                d.as_ptr().cast_mut().cast(),
                vecs.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
        }
        .is_err());
    }
    #[test]
    fn scratch_f32() {
        check::<f32>();
    }
    #[test]
    fn scratch_f64() {
        check::<f64>();
    }
    #[test]
    fn scratch_c32() {
        check::<faer::c32>();
    }
    #[test]
    fn scratch_c64() {
        check::<faer::c64>();
    }
}
