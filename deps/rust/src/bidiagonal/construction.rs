//! Bidiagonal reduction and explicit bases with caller workspace.
use super::workspace::buffer;
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::{self, householder as hh, svd};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

fn block_size<T: ComplexField>(rows: usize, k: usize, bs: usize) -> Result<usize, String> {
    if bs > k.max(1) {
        return Err("invalid bidiagonal blocksize".into());
    }
    Ok(if k == 0 {
        0
    } else if bs == 0 {
        linalg::qr::no_pivoting::factor::recommended_block_size::<T>(rows, k)
            .min(k)
            .max(1)
    } else {
        bs
    })
}
fn requirement<T: ComplexField>(
    m: usize,
    n: usize,
    full: bool,
    bs: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    if m > isize::MAX as usize || n > isize::MAX as usize {
        return Err("bidiagonal dimensions overflow".into());
    }
    let (rows, k) = (m.max(n), m.min(n));
    let bs = block_size::<T>(rows, k, bs)?;
    if k == 0 {
        return Ok(StackReq::EMPTY);
    }
    let mut kernel = svd::bidiag::bidiag_in_place_scratch::<T>(rows, k, par, Default::default())
        .or(
            hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                rows,
                bs,
                if full { rows } else { k },
            ),
        );
    if k > 1 {
        kernel = kernel.or(
            hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(k - 1, bs, k),
        );
    }
    Ok(StackReq::all_of(&[
        linalg::temp_mat_scratch::<T>(rows, k),
        linalg::temp_mat_scratch::<T>(bs, k),
        linalg::temp_mat_scratch::<T>(bs, k - 1),
        kernel,
    ]))
}
fn dispatch(
    m: usize,
    n: usize,
    tag: i32,
    full: bool,
    bs: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    match tag {
        0 => requirement::<f32>(m, n, full, bs, par),
        1 => requirement::<f64>(m, n, full, bs, par),
        2 => requirement::<faer::c32>(m, n, full, bs, par),
        3 => requirement::<faer::c64>(m, n, full, bs, par),
        _ => Err("unsupported bidiagonal scalar".into()),
    }
}
fn query(
    m: usize,
    n: usize,
    tag: i32,
    full: bool,
    bs: usize,
    par: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = dispatch(m, n, tag, full, bs, par)?;
    // SAFETY: caller preserves UInt[2]; checked span/alignment.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn bidiagonalize_requirements(
    m: usize,
    n: usize,
    tag: i32,
    full: bool,
    bs: usize,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| query(m, n, tag, full, bs, ParShape::Seq, out))
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn bidiagonalize_team_requirements(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        full: bool,
        bs: usize,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| self.execute(|par| query(m, n, tag, full, bs, par.shape(), out))?)
    }
}
unsafe fn compute<T: ComplexField>(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    uout: *mut u8,
    vout: *mut u8,
    dout: *mut u8,
    eout: *mut u8,
    full: bool,
    bs: usize,
    mut par: Par,
) -> Result<(), String> {
    let k = m.min(n);
    let uc = if full { m } else { k };
    let vc = if full { n } else { k };
    let mn = m.checked_mul(n).ok_or("bidiagonal dimensions overflow")?;
    let nu = m
        .checked_mul(uc)
        .ok_or("bidiagonal U dimensions overflow")?;
    let nv = n
        .checked_mul(vc)
        .ok_or("bidiagonal V dimensions overflow")?;
    let ranges = [
        buffer::<T>(input, mn)?,
        buffer::<T>(uout, nu)?,
        buffer::<T>(vout, nv)?,
        buffer::<T>(dout, k)?,
        buffer::<T>(eout, k.saturating_sub(1))?,
    ];
    for i in 0..5 {
        for j in i + 1..5 {
            let (a, b) = ranges[i];
            let (c, d) = ranges[j];
            if a < b && c < d && a < d && c < b {
                return Err("bidiagonal reduction buffers overlap".into());
            }
        }
    }
    let stack = scratch.stack(requirement::<T>(m, n, full, bs, par.shape())?)?;
    let a = if k == 0 {
        None
    } else {
        let a = MatRef::from_raw_parts(input.cast::<T>(), m, n, 1, m as isize);
        crate::typed::finite(a)?;
        Some(a)
    };
    // Raw initialization precedes MatMut; empty buffers use aligned dangling
    // pointers rather than relying on a caller's null zero-length pointer.
    for i in 0..nu {
        uout.cast::<T>().add(i).write(zero());
    }
    for i in 0..nv {
        vout.cast::<T>().add(i).write(zero());
    }
    let up = if nu == 0 {
        core::ptr::NonNull::<T>::dangling().as_ptr()
    } else {
        uout.cast::<T>()
    };
    let vp = if nv == 0 {
        core::ptr::NonNull::<T>::dangling().as_ptr()
    } else {
        vout.cast::<T>()
    };
    let mut u = MatMut::from_raw_parts_mut(up, m, uc, 1, m as isize);
    let mut v = MatMut::from_raw_parts_mut(vp, n, vc, 1, n as isize);
    for i in 0..m.min(uc) {
        u[(i, i)] = one();
    }
    for i in 0..n.min(vc) {
        v[(i, i)] = one();
    }
    if k == 0 {
        return Ok(());
    }
    let wide = m < n;
    let rows = m.max(n);
    let bs = block_size::<T>(rows, k, bs)?;
    let a = a.unwrap();
    // SAFETY: every logical work entry is initialized by copy_from before read.
    let (mut work, stack) = linalg::temp_mat_uninit::<T, _, _>(rows, k, stack);
    if wide {
        work.as_mat_mut().copy_from(a.adjoint());
    } else {
        work.as_mat_mut().copy_from(a);
    }
    let scaling = crate::schur::scale_in(work.as_mat_mut());
    let (mut hl, stack) = linalg::temp_mat_zeroed::<T, _, _>(bs, k, stack);
    let (mut hr, stack) = linalg::temp_mat_zeroed::<T, _, _>(bs, k - 1, stack);
    svd::bidiag::bidiag_in_place(
        work.as_mat_mut(),
        hl.as_mat_mut(),
        hr.as_mat_mut(),
        par.reborrow(),
        stack,
        Default::default(),
    );
    crate::typed::finite(work.as_mat_ref())?;
    let (left, mut right) = if wide {
        (v.rb_mut(), u.rb_mut())
    } else {
        (u.rb_mut(), v.rb_mut())
    };
    crate::householder_sequence::apply_compact_in(
        work.as_mat_ref(),
        hl.as_mat_ref(),
        left,
        par.reborrow(),
        stack,
    )?;
    if k > 1 {
        crate::householder_sequence::apply_compact_conj_in(
            work.as_mat_ref().submatrix(0, 1, k - 1, k - 1).transpose(),
            hr.as_mat_ref(),
            right.rb_mut().subrows_mut(1, k - 1),
            true,
            par,
            stack,
        )?;
    }
    crate::typed::finite(u.rb())?;
    crate::typed::finite(v.rb())?;
    for i in 0..k {
        let mut d = copy(&work.as_mat_ref()[(i, i)]);
        let mut e = if i + 1 < k {
            copy(&work.as_mat_ref()[(i, i + 1)])
        } else {
            zero()
        };
        if let Some(ref f) = scaling {
            let inv = recip(f);
            d = mul_real(&d, &inv);
            e = mul_real(&e, &inv);
        }
        dout.cast::<T>()
            .add(i)
            .write(if wide { conj(&d) } else { d });
        if i + 1 < k {
            eout.cast::<T>()
                .add(i)
                .write(if wide { conj(&e) } else { e });
        }
    }
    Ok(())
}
fn run(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    u: *mut u8,
    v: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    full: bool,
    bs: usize,
    par: Par,
) -> Result<(), String> {
    // SAFETY: tagged branches check every buffer before typed access.
    unsafe {
        match tag {
            0 => compute::<f32>(scratch, input, m, n, u, v, d, e, full, bs, par),
            1 => compute::<f64>(scratch, input, m, n, u, v, d, e, full, bs, par),
            2 => compute::<faer::c32>(scratch, input, m, n, u, v, d, e, full, bs, par),
            3 => compute::<faer::c64>(scratch, input, m, n, u, v, d, e, full, bs, par),
            _ => Err("unsupported bidiagonal scalar".into()),
        }
    }
}
#[julia]
pub fn bidiagonalize_workspace(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    u: *mut u8,
    v: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    full: bool,
    bs: usize,
) -> Result<(), String> {
    guarded(|| run(scratch, input, m, n, u, v, d, e, tag, full, bs, Par::Seq))
}
#[julia]
pub fn bidiagonalize_workspace_team(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    u: *mut u8,
    v: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    full: bool,
    bs: usize,
    team: &OwnedTeam,
) -> Result<(), String> {
    guarded(|| team.execute(|par| run(scratch, input, m, n, u, v, d, e, tag, full, bs, par))?)
}
pub(super) fn allocate(
    input: *const u8,
    m: usize,
    n: usize,
    u: *mut u8,
    v: *mut u8,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    full: bool,
    bs: usize,
) -> Result<(), String> {
    let req = dispatch(m, n, tag, full, bs, ParShape::Seq)?;
    let mut scratch = OwnedScratch::new();
    scratch.scratch_reserve(req.size_bytes(), req.align_bytes())?;
    run(
        &mut scratch,
        input,
        m,
        n,
        u,
        v,
        d,
        e,
        tag,
        full,
        bs,
        Par::Seq,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: ComplexField>() {
        let (m, n) = (5, 3);
        let input: Vec<T> = (0..m * n)
            .map(|j| {
                from_f64::<T>(if j % m == j / m {
                    (j % m + 1) as f64
                } else {
                    0.0
                })
            })
            .collect();
        let mut u: Vec<T> = (0..m * n).map(|_| from_f64::<T>(-9.0)).collect();
        let mut v: Vec<T> = (0..n * n).map(|_| from_f64::<T>(-9.0)).collect();
        let mut d: Vec<T> = (0..n).map(|_| from_f64::<T>(-9.0)).collect();
        let mut e: Vec<T> = (0..n - 1).map(|_| from_f64::<T>(-9.0)).collect();
        let req = requirement::<T>(m, n, false, 2, ParShape::Seq).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        macro_rules! call {
            ($s:expr) => {
                unsafe {
                    compute::<T>(
                        $s,
                        input.as_ptr().cast(),
                        m,
                        n,
                        u.as_mut_ptr().cast(),
                        v.as_mut_ptr().cast(),
                        d.as_mut_ptr().cast(),
                        e.as_mut_ptr().cast(),
                        false,
                        2,
                        Par::Seq,
                    )
                }
            };
        }
        assert!(call!(&mut scratch).is_err());
        assert!(u
            .iter()
            .chain(v.iter())
            .chain(d.iter())
            .chain(e.iter())
            .all(|x| *x == from_f64::<T>(-9.0)));
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
                abs(&sub(&abs(&d[i]), &from_f64::<T::Real>((i + 1) as f64)))
                    < from_f64::<T::Real>(1e-5)
            );
        }
        let before = (u.clone(), v.clone(), d.clone(), e.clone());
        let mut bad = input.clone();
        bad[0] = from_f64::<T>(f64::NAN);
        for p in [
            std::ptr::null(),
            unsafe { input.as_ptr().cast::<u8>().add(1) },
            bad.as_ptr().cast(),
        ] {
            assert!(unsafe {
                compute::<T>(
                    &mut scratch,
                    p,
                    m,
                    n,
                    u.as_mut_ptr().cast(),
                    v.as_mut_ptr().cast(),
                    d.as_mut_ptr().cast(),
                    e.as_mut_ptr().cast(),
                    false,
                    2,
                    Par::Seq,
                )
            }
            .is_err());
        }
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                m,
                n,
                input.as_ptr().cast_mut().cast(),
                v.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                false,
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert!(unsafe {
            compute::<T>(
                &mut scratch,
                input.as_ptr().cast(),
                m,
                n,
                u.as_mut_ptr().cast(),
                u.as_mut_ptr().cast(),
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                false,
                2,
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(u, before.0);
        assert_eq!(v, before.1);
        assert_eq!(d, before.2);
        assert_eq!(e, before.3);
        // Full empty reductions still initialize the surviving complete basis.
        let mut empty_u: Vec<T> = (0..9).map(|_| from_f64::<T>(-9.0)).collect();
        unsafe {
            compute::<T>(
                &mut scratch,
                std::ptr::null(),
                3,
                0,
                empty_u.as_mut_ptr().cast(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                true,
                0,
                Par::Seq,
            )
        }
        .unwrap();
        for j in 0..3 {
            for i in 0..3 {
                assert_eq!(
                    empty_u[i + j * 3],
                    if i == j { one::<T>() } else { zero::<T>() }
                );
            }
        }
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
