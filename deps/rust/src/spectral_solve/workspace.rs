//! Spectral projection and coefficient promotion inside caller-owned scratch.
use super::{threshold, Rhs};
use crate::pseudoinverse::ToleranceReal;
use crate::spectral_factors::{Decomposition, OwnedSpectral, SpectralData};
use crate::teams::with_team;
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::{matmul::matmul_with_conj, temp_mat_scratch, temp_mat_zeroed};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;
fn req<T: ComplexField>(
    m: usize,
    n: usize,
    k: usize,
    extent: usize,
    right: bool,
    promoted: bool,
) -> StackReq {
    if k == 0 || extent == 0 {
        return StackReq::EMPTY;
    }
    temp_mat_scratch::<T>(k, 1)
        .and(if right {
            temp_mat_scratch::<T>(extent, k)
        } else {
            temp_mat_scratch::<T>(k, extent)
        })
        .and(if promoted {
            temp_mat_scratch::<T>(m, k).and(temp_mat_scratch::<T>(n, k))
        } else {
            StackReq::EMPTY
        })
}
pub(super) fn requirement_for(
    f: &OwnedSpectral,
    m: usize,
    n: usize,
    source: i32,
    target: i32,
    eigen: bool,
    op: i32,
    right: bool,
    extent: usize,
) -> Result<StackReq, String> {
    f.check(source, eigen)?;
    if !(0..4).contains(&op) {
        return Err("invalid spectral solve operation".into());
    }
    if !(source == target || matches!((source, target), (0, 1) | (0, 2) | (0, 3) | (1, 3) | (2, 3)))
    {
        return Err("unsupported spectral solve scalar promotion".into());
    }
    let (rows, cols, k) = match f.data()? {
        SpectralData::F32(f) => (f.u().nrows(), f.v().nrows(), f.s().dim()),
        SpectralData::F64(f) => (f.u().nrows(), f.v().nrows(), f.s().dim()),
        SpectralData::C32(f) => (f.u().nrows(), f.v().nrows(), f.s().dim()),
        SpectralData::C64(f) => (f.u().nrows(), f.v().nrows(), f.s().dim()),
    };
    if (rows, cols) != (m, n) {
        return Err("spectral factor native shape mismatch".into());
    }
    Ok(match target {
        0 => req::<f32>(m, n, k, extent, right, source != target),
        1 => req::<f64>(m, n, k, extent, right, source != target),
        2 => req::<faer::c32>(m, n, k, extent, right, source != target),
        3 => req::<faer::c64>(m, n, k, extent, right, source != target),
        _ => return Err("invalid spectral solve target scalar".into()),
    })
}
pub(crate) fn project<T: ComplexField>(
    mut out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    u: MatRef<'_, T>,
    v: MatRef<'_, T>,
    weights: MatRef<'_, T>,
    mut projected: MatMut<'_, T>,
    op: i32,
    right: bool,
    mut par: Par,
) {
    let k = weights.nrows();
    if !(0..k).any(|i| weights[(i, 0)] != zero::<T>()) {
        out.fill(zero::<T>());
        return;
    }
    let (input, output) = if op == 1 || op == 2 { (v, u) } else { (u, v) };
    let input_conj = if op == 0 || op == 2 {
        Conj::Yes
    } else {
        Conj::No
    };
    let output_conj = if op == 1 || op == 3 {
        Conj::Yes
    } else {
        Conj::No
    };
    if right {
        matmul_with_conj(
            projected.rb_mut(),
            Accum::Replace,
            b,
            Conj::No,
            output,
            output_conj,
            one::<T>(),
            par.reborrow(),
        );
        for j in 0..k {
            for i in 0..b.nrows() {
                projected[(i, j)] = if weights[(j, 0)] == zero::<T>() {
                    zero::<T>()
                } else {
                    mul_real(&projected[(i, j)], &real(&weights[(j, 0)]))
                };
            }
        }
        matmul_with_conj(
            out,
            Accum::Replace,
            projected.rb(),
            Conj::No,
            input.transpose(),
            input_conj,
            one::<T>(),
            par,
        );
    } else {
        matmul_with_conj(
            projected.rb_mut(),
            Accum::Replace,
            input.transpose(),
            input_conj,
            b,
            Conj::No,
            one::<T>(),
            par.reborrow(),
        );
        for j in 0..b.ncols() {
            for i in 0..k {
                projected[(i, j)] = if weights[(i, 0)] == zero::<T>() {
                    zero::<T>()
                } else {
                    mul_real(&projected[(i, j)], &real(&weights[(i, 0)]))
                };
            }
        }
        matmul_with_conj(
            out,
            Accum::Replace,
            output,
            output_conj,
            projected.rb(),
            Conj::No,
            one::<T>(),
            par,
        );
    }
}
fn dimensions(
    m: usize,
    n: usize,
    brows: usize,
    bcols: usize,
    op: i32,
    right: bool,
) -> (usize, usize) {
    let (m, n) = if op == 1 || op == 2 { (n, m) } else { (m, n) };
    if right {
        (brows, m)
    } else {
        (n, bcols)
    }
}
pub(super) fn same<T: ComplexField>(
    f: &Decomposition<T>,
    b: MatRef<'_, T>,
    op: i32,
    right: bool,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    par: Par,
    scratch: &mut OwnedScratch,
) -> Result<Mat<T>, String>
where
    T::Real: ToleranceReal,
{
    finite(b)?;
    let (m, n, k) = (f.u().nrows(), f.v().nrows(), f.s().dim());
    let extent = if right { b.nrows() } else { b.ncols() };
    let cutoff = threshold(f, atol, rtol, relative_f32)?;
    let stack = scratch.stack(req::<T>(m, n, k, extent, right, false))?;
    let dims = dimensions(m, n, b.nrows(), b.ncols(), op, right);
    let mut out = Mat::zeros(dims.0, dims.1);
    if k == 0 || extent == 0 {
        return Ok(out);
    }
    let (mut weights, stack) = temp_mat_zeroed::<T, _, _>(k, 1, stack);
    let (mut projected, _) = temp_mat_zeroed::<T, _, _>(
        if right { extent } else { k },
        if right { k } else { extent },
        stack,
    );
    for i in 0..k {
        let s = real(&f.s()[i]);
        weights.as_mat_mut()[(i, 0)] = if abs(&s) > cutoff {
            from_real::<T>(&recip(&s))
        } else {
            zero::<T>()
        };
    }
    project(
        out.as_mut(),
        b,
        f.u().subcols(0, k),
        f.v().subcols(0, k),
        weights.as_mat_ref(),
        projected.as_mat_mut(),
        op,
        right,
        par,
    );
    Ok(out)
}
pub(super) fn converted<F: ComplexField, P: ComplexField>(
    f: &Decomposition<F>,
    b: MatRef<'_, P>,
    cast: fn(&F) -> P,
    op: i32,
    right: bool,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    par: Par,
    scratch: &mut OwnedScratch,
) -> Result<Mat<P>, String>
where
    F::Real: ToleranceReal,
{
    finite(b)?;
    let (m, n, k) = (f.u().nrows(), f.v().nrows(), f.s().dim());
    let extent = if right { b.nrows() } else { b.ncols() };
    let cutoff = threshold(f, atol, rtol, relative_f32)?;
    let stack = scratch.stack(req::<P>(m, n, k, extent, right, true))?;
    let dims = dimensions(m, n, b.nrows(), b.ncols(), op, right);
    let mut out = Mat::zeros(dims.0, dims.1);
    if k == 0 || extent == 0 {
        return Ok(out);
    }
    let (mut weights, stack) = temp_mat_zeroed::<P, _, _>(k, 1, stack);
    let (mut projected, stack) = temp_mat_zeroed::<P, _, _>(
        if right { extent } else { k },
        if right { k } else { extent },
        stack,
    );
    let (mut u, stack) = temp_mat_zeroed::<P, _, _>(m, k, stack);
    let (mut v, _) = temp_mat_zeroed::<P, _, _>(n, k, stack);
    for j in 0..k {
        weights.as_mat_mut()[(j, 0)] = if abs(&real(&f.s()[j])) > cutoff {
            from_real::<P>(&recip(&real(&cast(&f.s()[j]))))
        } else {
            zero::<P>()
        };
        for i in 0..m {
            u.as_mat_mut()[(i, j)] = cast(&f.u()[(i, j)]);
        }
        for i in 0..n {
            v.as_mat_mut()[(i, j)] = cast(&f.v()[(i, j)]);
        }
    }
    project(
        out.as_mut(),
        b,
        u.as_mat_ref(),
        v.as_mat_ref(),
        weights.as_mat_ref(),
        projected.as_mat_mut(),
        op,
        right,
        par,
    );
    Ok(out)
}
fn query(
    f: &OwnedSpectral,
    m: usize,
    n: usize,
    source: i32,
    target: i32,
    eigen: bool,
    op: i32,
    right: bool,
    extent: usize,
    out: *mut usize,
) -> Result<(), String> {
    let req = requirement_for(f, m, n, source, target, eigen, op, right, extent)?;
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("spectral scratch query requires aligned two-entry buffer".into());
    }
    // SAFETY: Julia preserves a writable two-entry UInt buffer; address and span checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
impl OwnedSpectral {
    #[julia]
    pub fn spectral_solve_requirements(
        &self,
        m: usize,
        n: usize,
        source: i32,
        target: i32,
        eigen: bool,
        op: i32,
        right: bool,
        extent: usize,
        out: *mut usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |_| {
                query(self, m, n, source, target, eigen, op, right, extent, out)
            })
        })
    }
}
#[julia]
impl OwnedSpectral {}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn spectral_solve_workspace(
        &mut self,
        f: &OwnedSpectral,
        b: &OwnedMatrix,
        m: usize,
        n: usize,
        rows: usize,
        cols: usize,
        source: i32,
        target: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = Rhs::F64(b.matrix.as_ref());
            let result = with_team(team, |par| {
                f.solve(
                    rhs,
                    m,
                    n,
                    rows,
                    cols,
                    source,
                    target,
                    eigen,
                    op,
                    right,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                    Some(scratch),
                )
            })?;
            if let MatrixData::F64(a) = result {
                self.matrix = a;
                Ok(())
            } else {
                Err("spectral output scalar mismatch".into())
            }
        })
    }
}
#[julia]
impl OwnedMatrix {}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn spectral_solve_workspace(
        &mut self,
        f: &OwnedSpectral,
        b: &TypedMatrix,
        m: usize,
        n: usize,
        rows: usize,
        cols: usize,
        source: i32,
        target: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = match b.data()? {
                MatrixData::F32(b) => Rhs::F32(b.as_ref()),
                MatrixData::F64(b) => Rhs::F64(b.as_ref()),
                MatrixData::C32(b) => Rhs::C32(b.as_ref()),
                MatrixData::C64(b) => Rhs::C64(b.as_ref()),
            };
            let result = with_team(team, |par| {
                f.solve(
                    rhs,
                    m,
                    n,
                    rows,
                    cols,
                    source,
                    target,
                    eigen,
                    op,
                    right,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                    Some(scratch),
                )
            })?;
            self.data = Some(result);
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::{c32, c64, Side};
    fn exercise<T: ComplexField>()
    where
        T::Real: ToleranceReal,
    {
        let a = Mat::from_fn(3, 3, |i, j| {
            if i != j {
                zero::<T>()
            } else {
                from_f64::<T>([4.0, -0.01, 0.0][i])
            }
        });
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for eigen in [false, true] {
            let f = if eigen {
                Decomposition::Eigen(a.self_adjoint_eigen(Side::Lower, Par::Seq).unwrap())
            } else {
                Decomposition::Svd(a.thin_svd(Par::Seq).unwrap())
            };
            for right in [false, true] {
                for op in 0..4 {
                    let b = Mat::from_fn(
                        if right { 2 } else { 3 },
                        if right { 3 } else { 2 },
                        |i, j| from_f64::<T>((1 + i + j) as f64),
                    );
                    let expected = Mat::from_fn(b.nrows(), b.ncols(), |i, j| {
                        let k = if right { j } else { i };
                        if k == 2 {
                            zero::<T>()
                        } else {
                            &b[(i, j)] / from_f64::<T>([4.0, -0.01][k])
                        }
                    });
                    let request = req::<T>(3, 3, 3, 2, right, false);
                    let mut scratch = OwnedScratch::new();
                    scratch
                        .scratch_reserve(request.size_bytes(), request.align_bytes())
                        .unwrap();
                    for threaded in [false, true] {
                        let out = if threaded {
                            team.execute(|p| {
                                same(&f, b.as_ref(), op, right, 0.0, 0.0, false, p, &mut scratch)
                            })
                            .unwrap()
                            .unwrap()
                        } else {
                            same(
                                &f,
                                b.as_ref(),
                                op,
                                right,
                                0.0,
                                0.0,
                                false,
                                Par::Seq,
                                &mut scratch,
                            )
                            .unwrap()
                        };
                        assert!(
                            (&out - &expected).norm_max()
                                <= from_f64::<T::Real>(1000.0)
                                    * eps::<T::Real>()
                                    * expected.norm_max()
                        );
                    }
                    let mut short = OwnedScratch::new();
                    short
                        .scratch_reserve(request.size_bytes() - 1, request.align_bytes())
                        .unwrap();
                    assert!(same(
                        &f,
                        b.as_ref(),
                        op,
                        right,
                        0.0,
                        0.0,
                        false,
                        Par::Seq,
                        &mut short
                    )
                    .is_err());
                }
            }
        }
        // Only the preallocated projection kernel is measured, not result ownership or worker heaps.
        let u = Mat::<T>::identity(3, 3);
        let weights = Mat::from_fn(3, 1, |i, _| from_f64::<T>([0.25, -100.0, 0.0][i]));
        for right in [false, true] {
            let b = Mat::from_fn(
                if right { 2 } else { 3 },
                if right { 3 } else { 2 },
                |i, j| from_f64::<T>((i + j + 1) as f64),
            );
            let mut out = Mat::zeros(b.nrows(), b.ncols());
            let mut projection = Mat::zeros(if right { 2 } else { 3 }, if right { 3 } else { 2 });
            let mut run = || {
                project(
                    out.as_mut(),
                    b.as_ref(),
                    u.as_ref(),
                    u.as_ref(),
                    weights.as_ref(),
                    projection.as_mut(),
                    0,
                    right,
                    Par::Seq,
                )
            };
            run();
            assert_eq!(
                crate::allocation_probe::count(|| {
                    run();
                    run();
                }),
                0
            );
            drop(run);
            let mut run = || {
                team.execute(|p| {
                    project(
                        out.as_mut(),
                        b.as_ref(),
                        u.as_ref(),
                        u.as_ref(),
                        weights.as_ref(),
                        projection.as_mut(),
                        0,
                        right,
                        p,
                    )
                })
                .unwrap()
            };
            run();
            assert_eq!(
                crate::allocation_probe::count(|| {
                    run();
                    run();
                }),
                0
            );
        }
    }
    #[test]
    fn saved_projection_f32() {
        exercise::<f32>();
    }
    #[test]
    fn saved_projection_f64() {
        exercise::<f64>();
    }
    #[test]
    fn saved_projection_c32() {
        exercise::<c32>();
    }
    #[test]
    fn saved_projection_c64() {
        exercise::<c64>();
    }
    #[test]
    fn promoted_projection_preserves_source_cutoff_and_zeros_discarded_values() {
        let a = Mat::<f32>::from_fn(3, 2, |i, j| {
            if i == j {
                if i == 0 {
                    1.0
                } else {
                    1e-8
                }
            } else {
                0.0
            }
        });
        let f = Decomposition::Svd(a.thin_svd(Par::Seq).unwrap());
        for right in [false, true] {
            for op in 0..4 {
                let trans = op == 1 || op == 2;
                let (m, n) = if trans { (2, 3) } else { (3, 2) };
                let b = Mat::<c64>::full(
                    if right { 2 } else { m },
                    if right { n } else { 2 },
                    c64::new(1.0, 0.2),
                );
                let r = req::<c64>(3, 2, 2, 2, right, true);
                let mut scratch = OwnedScratch::new();
                scratch
                    .scratch_reserve(r.size_bytes(), r.align_bytes())
                    .unwrap();
                let out = converted(
                    &f,
                    b.as_ref(),
                    |x| c64::new(*x as f64, 0.0),
                    op,
                    right,
                    0.0,
                    2.0 * f32::EPSILON as f64,
                    true,
                    Par::Seq,
                    &mut scratch,
                )
                .unwrap();
                for i in 0..out.nrows() {
                    for j in 0..out.ncols() {
                        assert_eq!(
                            out[(i, j)],
                            if (if right { j } else { i }) == 0 {
                                c64::new(1.0, 0.2)
                            } else {
                                c64::new(0.0, 0.0)
                            }
                        );
                    }
                }
                let huge = Mat::<c64>::full(b.nrows(), b.ncols(), c64::new(f64::MAX, 0.0));
                let out = converted(
                    &f,
                    huge.as_ref(),
                    |x| c64::new(*x as f64, 0.0),
                    op,
                    right,
                    f64::INFINITY,
                    0.0,
                    false,
                    Par::Seq,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(out.norm_max(), 0.0);
            }
        }
    }
    #[test]
    fn spectral_workspace_checks_current_metadata_and_stages_failures() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(3, 2, |i, j| if i == j { 4.0 + j as f64 } else { 0.0 }),
        };
        let mut f = OwnedSpectral::new();
        a.spectral_factor(&mut f, false, true, false, None).unwrap();
        let b = OwnedMatrix {
            matrix: Mat::full(3, 2, 1.0),
        };
        let mut out = OwnedMatrix {
            matrix: Mat::identity(2, 2),
        };
        let mut info = [0; 2];
        f.spectral_solve_requirements(3, 2, 1, 1, false, 0, false, 2, info.as_mut_ptr(), None)
            .unwrap();
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0], info[1]).unwrap();
        out.spectral_solve_workspace(
            &f,
            &b,
            3,
            2,
            2,
            2,
            1,
            1,
            false,
            0,
            false,
            0.0,
            0.0,
            false,
            &mut scratch,
            None,
        )
        .unwrap();
        let saved = out.matrix.clone();
        for (m, n, source, target, eigen, op) in [
            (4, 2, 1, 1, false, 0),
            (3, 2, 0, 1, false, 0),
            (3, 2, 1, 0, false, 0),
            (3, 2, 1, 1, true, 0),
            (3, 2, 1, 1, false, 9),
        ] {
            assert!(f
                .spectral_solve_requirements(
                    m,
                    n,
                    source,
                    target,
                    eigen,
                    op,
                    false,
                    2,
                    info.as_mut_ptr(),
                    None
                )
                .is_err());
            assert!(out
                .spectral_solve_workspace(
                    &f,
                    &b,
                    m,
                    n,
                    2,
                    2,
                    source,
                    target,
                    eigen,
                    op,
                    false,
                    0.0,
                    0.0,
                    false,
                    &mut scratch,
                    None
                )
                .is_err());
            assert_eq!(out.matrix, saved);
        }
        assert!(f
            .spectral_solve_requirements(3, 2, 1, 1, false, 0, false, 2, std::ptr::null_mut(), None)
            .is_err());
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.team_close().unwrap();
        assert!(f
            .spectral_solve_requirements(
                3,
                2,
                1,
                1,
                false,
                0,
                false,
                2,
                info.as_mut_ptr(),
                Some(&team)
            )
            .is_err());
        scratch.scratch_close().unwrap();
        assert!(out
            .spectral_solve_workspace(
                &f,
                &b,
                3,
                2,
                2,
                2,
                1,
                1,
                false,
                0,
                false,
                0.0,
                0.0,
                false,
                &mut scratch,
                None
            )
            .is_err());
        assert_eq!(out.matrix, saved);
        f.spectral_close().unwrap();
        assert!(f
            .spectral_solve_requirements(3, 2, 1, 1, false, 0, false, 2, info.as_mut_ptr(), None)
            .is_err());
    }
}
