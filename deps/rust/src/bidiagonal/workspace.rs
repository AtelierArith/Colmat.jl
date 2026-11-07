//! Square bidiagonal SVD with caller scratch and scoped team authority.
use crate::teams::with_team;
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::svd;
use faer::traits::{math_utils::*, ComplexField};
use faer::{Auto, MatMut, Par, ParShape};
use rustcall_julia_macros::julia;

fn params<T: ComplexField>(threshold: usize) -> Result<faer::Spec<svd::SvdParams, T>, String> {
    if threshold < 2 {
        return Err("bidiagonal recursion threshold must be at least two".into());
    }
    Ok(svd::SvdParams {
        recursion_threshold: threshold,
        ..<svd::SvdParams as Auto<T>>::auto()
    }
    .into())
}
fn requirement<T: ComplexField>(
    n: usize,
    threshold: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    let params = params::<T>(threshold)?;
    if n == 0 {
        return Ok(StackReq::EMPTY);
    }
    Ok(StackReq::all_of(&[
        StackReq::new::<T>(n),
        StackReq::new::<T>(n - 1),
        svd::bidiagonal_svd_scratch::<T>(n, par, params),
    ]))
}
fn dispatch(n: usize, tag: i32, threshold: usize, par: ParShape) -> Result<StackReq, String> {
    match tag {
        0 => requirement::<f32>(n, threshold, par),
        1 => requirement::<f64>(n, threshold, par),
        2 => requirement::<faer::c32>(n, threshold, par),
        3 => requirement::<faer::c64>(n, threshold, par),
        _ => Err("unsupported bidiagonal scalar".into()),
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
        .ok_or("bidiagonal buffer span overflow")?;
    let end = first
        .checked_add(bytes)
        .ok_or("bidiagonal buffer address overflow")?;
    if p.is_null() || first % core::mem::align_of::<T>() != 0 {
        return Err("bidiagonal buffer is null or unaligned".into());
    }
    Ok((first, end))
}
fn query(
    n: usize,
    tag: i32,
    threshold: usize,
    par: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = dispatch(n, tag, threshold, par)?;
    // SAFETY: caller supplies a preserved UInt[2], with checked span/alignment.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn bidiagonal_svd_requirements(
    n: usize,
    tag: i32,
    threshold: usize,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| with_team(team, |par| query(n, tag, threshold, par.shape(), out)))
}
unsafe fn compute<T: ComplexField>(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    u: *mut u8,
    v: *mut u8,
    upper: bool,
    threshold: usize,
    par: Par<'_>,
) -> Result<(), String> {
    let nn = n
        .checked_mul(n)
        .ok_or("bidiagonal vector dimensions overflow")?;
    let ranges = [
        buffer::<T>(d, n)?,
        buffer::<T>(e, n.saturating_sub(1))?,
        buffer::<T::Real>(values, n)?,
        buffer::<T>(u, nn)?,
        buffer::<T>(v, nn)?,
    ];
    for i in 0..5 {
        for j in i + 1..5 {
            if i == 0 && j == 1 {
                continue;
            } // immutable inputs may alias
            let (a, b) = ranges[i];
            let (c, d) = ranges[j];
            if a < b && c < d && a < d && c < b {
                return Err("bidiagonal input/output buffers overlap".into());
            }
        }
    }
    let stack = scratch.stack(requirement::<T>(n, threshold, par.shape())?)?;
    if n == 0 {
        return Ok(());
    }
    let (mut diag, stack) = stack.make_with::<T>(n, |i| copy(&*d.cast::<T>().add(i)));
    let (off, stack) = stack.make_with::<T>(n - 1, |i| copy(&*e.cast::<T>().add(i)));
    if !diag.iter().chain(off.iter()).all(|x| is_finite(x)) {
        return Err("bidiagonal coefficients must be finite".into());
    }
    // Initialize raw output storage before creating MatMut references. This also
    // accepts preserved Julia undef buffers; no uninitialized T reference exists.
    for i in 0..nn {
        u.cast::<T>().add(i).write(zero());
        v.cast::<T>().add(i).write(zero());
    }
    let umat = MatMut::from_raw_parts_mut(u.cast::<T>(), n, n, 1, n as isize);
    let vmat = MatMut::from_raw_parts_mut(v.cast::<T>(), n, n, 1, n as isize);
    svd::bidiagonal_svd(
        &mut diag,
        &off,
        upper,
        umat,
        vmat,
        par,
        stack,
        params::<T>(threshold)?,
    )
    .map_err(|e| format!("{e:?}"))?;
    for i in 0..n {
        values.cast::<T::Real>().add(i).write(real(&diag[i]));
    }
    Ok(())
}
fn run(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    u: *mut u8,
    v: *mut u8,
    tag: i32,
    upper: bool,
    threshold: usize,
    par: Par<'_>,
) -> Result<(), String> {
    // SAFETY: checked tagged spans, lifetimes supplied by the preserved caller.
    unsafe {
        match tag {
            0 => compute::<f32>(scratch, d, e, n, values, u, v, upper, threshold, par),
            1 => compute::<f64>(scratch, d, e, n, values, u, v, upper, threshold, par),
            2 => compute::<faer::c32>(scratch, d, e, n, values, u, v, upper, threshold, par),
            3 => compute::<faer::c64>(scratch, d, e, n, values, u, v, upper, threshold, par),
            _ => Err("unsupported bidiagonal scalar".into()),
        }
    }
}
#[julia]
pub fn bidiagonal_svd_workspace(
    scratch: &mut OwnedScratch,
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    u: *mut u8,
    v: *mut u8,
    tag: i32,
    upper: bool,
    threshold: usize,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            run(scratch, d, e, n, values, u, v, tag, upper, threshold, par)
        })
    })
}

pub(super) fn allocate(
    d: *const u8,
    e: *const u8,
    n: usize,
    values: *mut u8,
    u: *mut u8,
    v: *mut u8,
    tag: i32,
    upper: bool,
    threshold: usize,
) -> Result<(), String> {
    let req = dispatch(n, tag, threshold, ParShape::Seq)?;
    let mut scratch = OwnedScratch::new();
    scratch.scratch_reserve(req.size_bytes(), req.align_bytes())?;
    run(
        &mut scratch,
        d,
        e,
        n,
        values,
        u,
        v,
        tag,
        upper,
        threshold,
        Par::Seq,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: ComplexField>() {
        let n = 6;
        let d: Vec<T> = (0..n).map(|i| from_f64::<T>((i + 1) as f64)).collect();
        let e: Vec<T> = (0..n - 1).map(|_| zero::<T>()).collect();
        let mut values: Vec<T::Real> = (0..n).map(|_| from_f64::<T::Real>(-99.0)).collect();
        let mut u: Vec<T> = (0..n * n).map(|_| from_f64::<T>(-99.0)).collect();
        let mut v = u.clone();
        let req = requirement::<T>(n, 2, ParShape::Seq).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        macro_rules! call {
            ($s:expr) => {
                unsafe {
                    compute::<T>(
                        $s,
                        d.as_ptr().cast(),
                        e.as_ptr().cast(),
                        n,
                        values.as_mut_ptr().cast(),
                        u.as_mut_ptr().cast(),
                        v.as_mut_ptr().cast(),
                        true,
                        2,
                        Par::Seq,
                    )
                }
            };
        }
        assert!(call!(&mut scratch).is_err());
        assert!(values.iter().all(|x| *x == from_f64::<T::Real>(-99.0)));
        assert!(u.iter().chain(v.iter()).all(|x| *x == from_f64::<T>(-99.0)));
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        call!(&mut scratch).unwrap();
        let allocations = crate::allocation_probe::count(|| {
            call!(&mut scratch).unwrap();
        });
        assert_eq!(allocations, 0);
        for i in 0..n {
            assert!(
                abs(&sub(&values[i], &from_f64::<T::Real>((n - i) as f64)))
                    < from_f64::<T::Real>(1e-5)
            );
        }
        let before = (values.clone(), u.clone(), v.clone());
        for bad in [std::ptr::null(), unsafe { d.as_ptr().cast::<u8>().add(1) }] {
            assert!(unsafe {
                compute::<T>(
                    &mut scratch,
                    bad,
                    e.as_ptr().cast(),
                    n,
                    values.as_mut_ptr().cast(),
                    u.as_mut_ptr().cast(),
                    v.as_mut_ptr().cast(),
                    true,
                    2,
                    Par::Seq,
                )
            }
            .is_err());
        }
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast(),
                e.as_ptr().cast(),
                n,
                values.as_mut_ptr().cast(),
                u.as_mut_ptr().cast(),
                u.as_mut_ptr().cast(),
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
                u.as_mut_ptr().cast(),
                v.as_mut_ptr().cast(),
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
                usize::MAX,
                values.as_mut_ptr().cast(),
                u.as_mut_ptr().cast(),
                v.as_mut_ptr().cast(),
                true,
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(values, before.0);
        assert_eq!(u, before.1);
        assert_eq!(v, before.2);
        // Aliased immutable coefficients are copied before any mutable borrow.
        unsafe {
            compute::<T>(
                &mut scratch,
                d.as_ptr().cast(),
                d.as_ptr().cast(),
                n,
                values.as_mut_ptr().cast(),
                u.as_mut_ptr().cast(),
                v.as_mut_ptr().cast(),
                false,
                2,
                Par::Seq,
            )
        }
        .unwrap();
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
