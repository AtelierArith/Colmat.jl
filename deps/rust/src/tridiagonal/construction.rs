//! Reusable tridiagonal reduction and explicit Q construction workspace.
use super::workspace::buffer;
use crate::teams::{with_team, OwnedTeam};
use crate::{guarded, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::{self, evd, householder as hh};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::utils::cost::ParThreshold;
use faer::{Auto, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
pub(super) fn params<T: ComplexField>(
    threshold: usize,
) -> faer::Spec<evd::tridiag::TridiagParams, T> {
    evd::tridiag::TridiagParams {
        par_threshold: ParThreshold::from(threshold),
        ..<evd::tridiag::TridiagParams as Auto<T>>::auto()
    }
    .into()
}
// A real norm must not be routed through complex division: its squared
// denominator can underflow. Lifting subnormal norms also keeps the real
// reciprocal finite, without changing the represented phase.
pub(super) fn normalize_phase<T: ComplexField>(value: &T, norm: &T::Real) -> T {
    let inverse = recip(norm);
    if is_finite(&inverse) {
        mul_real(value, &inverse)
    } else {
        let lift = recip(&min_positive::<T::Real>());
        mul_real(&mul_real(value, &lift), &recip(&mul(norm, &lift)))
    }
}
// Scaling by the largest component also preserves subnormal magnitudes;
// the generic faer complex magnitude can erase them (tensor4all/faer#274).
pub(super) fn phase_magnitude<T: ComplexField>(value: &T) -> T::Real {
    let re = abs(&real(value));
    let im = abs(&imag(value));
    if !is_finite(&re) || !is_finite(&im) {
        return abs(value);
    }
    let (large, small) = if re >= im { (re, im) } else { (im, re) };
    if large == zero::<T::Real>() {
        return large;
    }
    let ratio = div(&small, &large);
    mul(&large, &sqrt(&add(&one::<T::Real>(), &mul(&ratio, &ratio))))
}
pub(super) fn real_subdiagonal<T: ComplexField>(value: &T) -> T::Real {
    if T::IS_REAL {
        real(value)
    } else {
        phase_magnitude(value)
    }
}
pub(super) fn validate_real_bands<T: ComplexField>(
    packed: MatRef<'_, T>,
    inverse_scale: Option<&T::Real>,
) -> Result<(), String> {
    let restored = |value: &T| match inverse_scale {
        Some(scale) => mul_real(value, scale),
        None => copy(value),
    };
    for i in 0..packed.nrows() {
        if !is_finite(&real(&restored(&packed[(i, i)])))
            || (i + 1 < packed.nrows()
                && !is_finite(&real_subdiagonal(&restored(&packed[(i + 1, i)]))))
        {
            return Err(
                "tridiagonal real bands overflowed the working type; scale the input first".into(),
            );
        }
    }
    Ok(())
}
pub(super) fn block_size<T: ComplexField>(n: usize, bs: usize) -> Result<usize, String> {
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
fn requirement<T: ComplexField>(
    n: usize,
    bs: usize,
    par: ParShape,
    threshold: usize,
) -> Result<StackReq, String> {
    let par = if threshold == usize::MAX {
        ParShape::Seq
    } else {
        par
    };
    let bs = block_size::<T>(n, bs)?;
    let work = linalg::temp_mat_scratch::<T>(n, n);
    if n < 2 {
        return Ok(work);
    }
    Ok(work.and(linalg::temp_mat_scratch::<T>(bs, n - 1)).and(
        evd::tridiag::tridiag_in_place_scratch::<T>(n, par, params::<T>(threshold)).or(
            hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n - 1, bs, n),
        ),
    ))
}
fn dispatch(
    n: usize,
    tag: i32,
    bs: usize,
    par: ParShape,
    threshold: usize,
) -> Result<StackReq, String> {
    match tag {
        0 => requirement::<f32>(n, bs, par, threshold),
        1 => requirement::<f64>(n, bs, par, threshold),
        2 => requirement::<faer::c32>(n, bs, par, threshold),
        3 => requirement::<faer::c64>(n, bs, par, threshold),
        _ => Err("unsupported tridiagonal scalar".into()),
    }
}
fn query(
    n: usize,
    tag: i32,
    bs: usize,
    par: ParShape,
    threshold: usize,
    out: *mut usize,
) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = dispatch(n, tag, bs, par, threshold)?;
    // SAFETY: caller preserves UInt[2]; span/alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
/// Workspace of `tridiagonalize_workspace`; `threshold` 0 keeps faer's derived
/// parallel threshold, and `team` borrows a caller team's shape.
#[julia]
pub fn tridiagonalize_requirements(
    n: usize,
    tag: i32,
    bs: usize,
    threshold: usize,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| with_team(team, |par| query(n, tag, bs, par.shape(), threshold, out)))
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
    threshold: usize,
) -> Result<(), String> {
    if threshold == usize::MAX {
        par = Par::Seq;
    }
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
    let stack = scratch.stack(requirement::<T>(n, bs, par.shape(), threshold)?)?;
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
    // INVARIANT: exactly one branch initializes Q, after checking every real band.
    // SAFETY: qout has a disjoint checked n*n span. Raw writes initialize every
    // element before forming a MatMut, including when Julia supplied undef storage.
    let initialize_q = || unsafe {
        for k in 0..nn {
            qout.cast::<T>().add(k).write(zero::<T>());
        }
        let mut q = MatMut::from_raw_parts_mut(qout.cast::<T>(), n, n, 1, n as isize);
        for i in 0..n {
            q[(i, i)] = one::<T>();
        }
        q
    };
    if n > 1 {
        let scaling = crate::schur::scale_in(work.as_mat_mut());
        let (mut factor, stack) = linalg::temp_mat_zeroed::<T, _, _>(bs, n - 1, stack);
        evd::tridiag::tridiag_in_place(
            work.as_mat_mut(),
            factor.as_mat_mut(),
            par.reborrow(),
            stack,
            params::<T>(threshold),
        );
        let inverse_scale = scaling.as_ref().map(recip);
        validate_real_bands(work.as_mat_ref(), inverse_scale.as_ref())?;
        let mut q = initialize_q();
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
                let norm = phase_magnitude(e);
                if norm != zero::<T::Real>() {
                    phase = &phase * normalize_phase::<T>(e, &norm);
                    let pnorm = phase_magnitude(&phase);
                    phase = normalize_phase::<T>(&phase, &pnorm);
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
    } else {
        validate_real_bands(work.as_mat_ref(), None)?;
        let _ = initialize_q();
    }
    for i in 0..n {
        dout.cast::<T::Real>()
            .add(i)
            .write(real(&work.as_mat_ref()[(i, i)]));
        if i + 1 < n {
            eout.cast::<T::Real>()
                .add(i)
                .write(real_subdiagonal(&work.as_mat_ref()[(i + 1, i)]));
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
    threshold: usize,
) -> Result<(), String> {
    // SAFETY: each branch checks every buffer before typed access.
    unsafe {
        match tag {
            0 => compute::<f32>(scratch, input, n, q, d, e, bs, par, threshold),
            1 => compute::<f64>(scratch, input, n, q, d, e, bs, par, threshold),
            2 => compute::<faer::c32>(scratch, input, n, q, d, e, bs, par, threshold),
            3 => compute::<faer::c64>(scratch, input, n, q, d, e, bs, par, threshold),
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
    threshold: usize,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            run(scratch, input, n, q, d, e, tag, bs, par, threshold)
        })
    })
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
    let req = dispatch(n, tag, bs, ParShape::Seq, 0)?;
    let mut scratch = OwnedScratch::new();
    scratch.scratch_reserve(req.size_bytes(), req.align_bytes())?;
    run(&mut scratch, input, n, q, d, e, tag, bs, Par::Seq, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reject_unrepresentable_bands<T: ComplexField>(value: T) {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for n in [2, 3] {
            let early = mul_real(
                &one::<T>(),
                &mul(&real(&value), &from_f64::<T::Real>(0.125)),
            );
            let input: Vec<T> = (0..n * n)
                .map(|index| {
                    let (i, j) = (index % n, index / n);
                    if i == j + 1 {
                        copy(if j + 2 == n { &value } else { &early })
                    } else if j == i + 1 {
                        conj(if i + 2 == n { &value } else { &early })
                    } else {
                        zero::<T>()
                    }
                })
                .collect();
            for use_team in [false, true] {
                let run = |par: Par<'_>| {
                    let req = requirement::<T>(n, 1, par.shape(), 0).unwrap();
                    let mut work = OwnedScratch::new();
                    work.scratch_reserve(req.size_bytes(), req.align_bytes())
                        .unwrap();
                    let mut q: Vec<T> = (0..n * n).map(|_| from_f64(11.0)).collect();
                    let mut diagonal: Vec<T::Real> = (0..n).map(|_| from_f64(17.0)).collect();
                    let mut off: Vec<T::Real> = (0..n - 1).map(|_| from_f64(23.0)).collect();
                    let before_q = q.clone();
                    let before_d = diagonal.clone();
                    let before_e = off.clone();
                    // SAFETY: all initialized numeric buffers have disjoint checked spans.
                    let error = unsafe {
                        compute::<T>(
                            &mut work,
                            input.as_ptr().cast(),
                            n,
                            q.as_mut_ptr().cast(),
                            diagonal.as_mut_ptr().cast(),
                            off.as_mut_ptr().cast(),
                            1,
                            par,
                            0,
                        )
                    }
                    .err()
                    .expect("unrepresentable bands must fail");
                    assert!(error.contains("real bands"), "{error}");
                    assert_eq!(q, before_q);
                    assert_eq!(diagonal, before_d);
                    assert_eq!(off, before_e);
                };
                if use_team {
                    team.execute(run).unwrap();
                } else {
                    run(Par::Seq);
                }
            }
        }
    }
    #[test]
    fn explicit_real_band_overflow_c32_preserves_outputs() {
        reject_unrepresentable_bands(faer::c32::new(3.0e38, 3.0e38));
    }
    #[test]
    fn explicit_real_band_overflow_c64_preserves_outputs() {
        reject_unrepresentable_bands(faer::c64::new(1.3e308, 1.3e308));
    }
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
        let req = requirement::<T>(n, 2, ParShape::Seq, 0).unwrap();
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
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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
                0,
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
    #[test]
    fn phase_scalar_c32_full_range() {
        for x in [1e-30_f32, f32::MIN_POSITIVE, f32::from_bits(1), 1e30_f32] {
            for sign in [-1.0_f32, 1.0] {
                let value = faer::c32::new(0.0, sign * x);
                let norm = phase_magnitude(&value);
                assert_eq!(norm, x);
                let phase = normalize_phase(&value, &norm);
                assert!(phase.re.abs() <= 4.0 * f32::EPSILON);
                assert!((phase.im - sign).abs() <= 4.0 * f32::EPSILON);
            }
        }
        assert_eq!(phase_magnitude(&faer::c32::new(0.0, 0.0)), 0.0);
    }
    #[test]
    fn phase_scalar_c64_full_range() {
        for x in [1e-300_f64, f64::MIN_POSITIVE, f64::from_bits(1), 1e300_f64] {
            for sign in [-1.0_f64, 1.0] {
                let value = faer::c64::new(0.0, sign * x);
                let norm = phase_magnitude(&value);
                assert_eq!(norm, x);
                let phase = normalize_phase(&value, &norm);
                assert!(phase.re.abs() <= 4.0 * f64::EPSILON);
                assert!((phase.im - sign).abs() <= 4.0 * f64::EPSILON);
            }
        }
        assert_eq!(phase_magnitude(&faer::c64::new(0.0, 0.0)), 0.0);
    }
}
