//! Iterative solver and independent diagnostic scratch with scoped teams.
use super::{
    solve_bicg_resources, solve_lsmr_resources, solve_resources, Data, Operator,
    OwnedLinearOperator,
};
use crate::teams::OwnedTeam;
use crate::typed::{MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
use faer::mat::{AsMatMut, AsMatRef};
use faer::matrix_free::{BiLinOp, LinOp};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
type Observer = extern "C" fn(*const u8, usize, usize, isize, isize) -> bool;
fn normal_req<T: ComplexField>(
    a: &Operator<T>,
    m: &Operator<T>,
    k: usize,
    p: ParShape,
) -> StackReq {
    temp_mat_scratch::<T>(a.ncols(), k)
        .and(temp_mat_scratch::<T>(m.ncols(), k))
        .and(
            a.transpose_apply_scratch(k, p)
                .or(m.transpose_apply_scratch(k, p)),
        )
}
pub(super) fn requirement<T: ComplexField>(
    a: &Operator<T>,
    m: &Operator<T>,
    n: &Operator<T>,
    kind: i32,
    k: usize,
    p: ParShape,
) -> Result<StackReq, String> {
    if !(0..3).contains(&kind) {
        return Err("invalid iterative solver kind".into());
    }
    if kind != 2 && a.nrows() != a.ncols() {
        return Err("iterative square-operator shape mismatch".into());
    }
    let size = a.ncols();
    if m.shape() != (size, size) || n.shape() != (size, size) {
        return Err("iterative preconditioner shape mismatch".into());
    }
    let native = match kind {
        0 => faer::matrix_free::conjugate_gradient::conjugate_gradient_scratch(m, a, k, p),
        1 => faer::matrix_free::bicgstab::bicgstab_scratch(m, n, a, k, p),
        2 => {
            if a.nrows() == 0 || a.ncols() == 0 || k == 0 {
                StackReq::EMPTY
            } else {
                faer::matrix_free::lsmr::lsmr_scratch(m, a, k, p)
            }
        }
        _ => unreachable!(),
    };
    let normal = if kind == 2 {
        normal_req(a, m, k, p)
    } else {
        StackReq::EMPTY
    };
    Ok(native
        .or(normal)
        .or(temp_mat_scratch::<T>(a.nrows(), k).and(a.apply_scratch(k, p).or(normal))))
}
pub(super) fn normal_norm<T: ComplexField>(
    a: &Operator<T>,
    m: &Operator<T>,
    b: MatRef<'_, T>,
    mut par: Par<'_>,
    stack: &mut MemStack,
) -> T::Real {
    let (mut atb, stack) = temp_mat_zeroed::<T, _, _>(a.ncols(), b.ncols(), stack);
    let (mut result, stack) = temp_mat_zeroed::<T, _, _>(m.ncols(), b.ncols(), stack);
    a.adjoint_apply(atb.as_mat_mut(), b, par.reborrow(), stack);
    m.adjoint_apply(result.as_mat_mut(), atb.as_mat_ref(), par, stack);
    result.as_mat_ref().norm_l2()
}
pub(super) fn residual_norms<T: ComplexField>(
    a: &Operator<T>,
    m: &Operator<T>,
    x: MatRef<'_, T>,
    b: MatRef<'_, T>,
    kind: i32,
    mut par: Par<'_>,
    stack: &mut MemStack,
) -> (T::Real, T::Real) {
    let (mut residual, stack) = temp_mat_zeroed::<T, _, _>(a.nrows(), b.ncols(), stack);
    let mut residual = residual.as_mat_mut();
    a.apply(residual.as_mut(), x, par.reborrow(), stack);
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            residual[(i, j)] = &b[(i, j)] - &residual[(i, j)];
        }
    }
    let absolute = residual.as_ref().norm_l2();
    let normal = if kind == 2 {
        normal_norm(a, m, residual.as_ref(), par, stack)
    } else {
        zero::<T::Real>()
    };
    (absolute, normal)
}
fn query(
    a: &OwnedLinearOperator,
    m: &OwnedLinearOperator,
    n: &OwnedLinearOperator,
    kind: i32,
    k: usize,
    rows: usize,
    cols: usize,
    tag: i32,
    p: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    super::apply_workspace::validate(a, rows, cols, tag)?;
    super::apply_workspace::validate(m, cols, cols, tag)?;
    super::apply_workspace::validate(n, cols, cols, tag)?;
    let req = match (a.data()?, m.data()?, n.data()?) {
        (Data::F32(a), Data::F32(m), Data::F32(n)) => requirement(a, m, n, kind, k, p)?,
        (Data::F64(a), Data::F64(m), Data::F64(n)) => requirement(a, m, n, kind, k, p)?,
        (Data::C32(a), Data::C32(m), Data::C32(n)) => requirement(a, m, n, kind, k, p)?,
        (Data::C64(a), Data::C64(m), Data::C64(n)) => requirement(a, m, n, kind, k, p)?,
        _ => return Err("iterative scalar types must match".into()),
    };
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("iterative query requires aligned two-entry usize output".into());
    }
    // SAFETY: caller retains two writable usize slots, with address/span checked.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn iterative_requirements(
        &self,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        k: usize,
        rows: usize,
        cols: usize,
        tag: i32,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| query(self, m, n, kind, k, rows, cols, tag, ParShape::Seq, out))
    }
    #[julia]
    pub fn iterative_requirements_team(
        &self,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        k: usize,
        rows: usize,
        cols: usize,
        tag: i32,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| query(self, m, n, kind, k, rows, cols, tag, par.shape(), out))?
        })
    }
}
fn run<T: ComplexField>(
    a: &Operator<T>,
    m: &Operator<T>,
    n: &Operator<T>,
    kind: i32,
    b: MatRef<'_, T>,
    x: MatRef<'_, T>,
    atol: f64,
    rtol: f64,
    maxiter: usize,
    stats: *mut u8,
    iterations: *mut usize,
    observing: bool,
    observer: Observer,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<Mat<T>, String> {
    let count = if kind == 2 { 7 } else { 5 };
    if stats.is_null()
        || (stats as usize) % std::mem::align_of::<T>() != 0
        || (stats as usize)
            .checked_add(count * std::mem::size_of::<T>())
            .is_none()
        || iterations.is_null()
        || (iterations as usize) % std::mem::align_of::<usize>() != 0
        || (iterations as usize)
            .checked_add(std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("invalid iterative statistics output buffer".into());
    }
    let req = requirement(a, m, n, kind, b.ncols(), par.shape())?;
    scratch.stack(req)?;
    match kind {
        0 => solve_resources(
            a,
            Some(m),
            b,
            x,
            atol,
            rtol,
            maxiter,
            stats,
            iterations,
            observing,
            observer,
            Some(scratch),
            par,
        ),
        1 => solve_bicg_resources(
            a,
            Some(m),
            Some(n),
            b,
            x,
            atol,
            rtol,
            maxiter,
            stats,
            iterations,
            observing,
            observer,
            Some(scratch),
            par,
        ),
        2 => solve_lsmr_resources(
            a,
            Some(m),
            b,
            x,
            atol,
            rtol,
            maxiter,
            stats,
            iterations,
            observing,
            observer,
            Some(scratch),
            par,
        ),
        _ => Err("invalid iterative kind".into()),
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn iterative_workspace(
        &mut self,
        a: &OwnedLinearOperator,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        rhs: &OwnedMatrix,
        initial: &OwnedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            let (Data::F64(a), Data::F64(m), Data::F64(n)) = (a.data()?, m.data()?, n.data()?)
            else {
                return Err("iterative scalar types must match".into());
            };
            self.matrix = run(
                a,
                m,
                n,
                kind,
                rhs.matrix.as_ref(),
                initial.matrix.as_ref(),
                atol,
                rtol,
                maxiter,
                stats,
                iterations,
                observing,
                observer,
                scratch,
                par,
            )?;
            Ok(())
        })
    }
    #[julia]
    pub fn iterative_workspace_team(
        &mut self,
        a: &OwnedLinearOperator,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        rhs: &OwnedMatrix,
        initial: &OwnedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                let (Data::F64(a), Data::F64(m), Data::F64(n)) = (a.data()?, m.data()?, n.data()?)
                else {
                    return Err("iterative scalar types must match".into());
                };
                self.matrix = run(
                    a,
                    m,
                    n,
                    kind,
                    rhs.matrix.as_ref(),
                    initial.matrix.as_ref(),
                    atol,
                    rtol,
                    maxiter,
                    stats,
                    iterations,
                    observing,
                    observer,
                    scratch,
                    par,
                )?;
                Ok(())
            })?
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn iterative_workspace(
        &mut self,
        a: &OwnedLinearOperator,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        rhs: &TypedMatrix,
        initial: &TypedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            self.data = Some(
                match (
                    a.data()?,
                    m.data()?,
                    n.data()?,
                    rhs.data()?,
                    initial.data()?,
                ) {
                    (
                        Data::F32(a),
                        Data::F32(m),
                        Data::F32(n),
                        MatrixData::F32(b),
                        MatrixData::F32(x),
                    ) => MatrixData::F32(run(
                        a,
                        m,
                        n,
                        kind,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                        observing,
                        observer,
                        scratch,
                        par,
                    )?),
                    (
                        Data::C32(a),
                        Data::C32(m),
                        Data::C32(n),
                        MatrixData::C32(b),
                        MatrixData::C32(x),
                    ) => MatrixData::C32(run(
                        a,
                        m,
                        n,
                        kind,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                        observing,
                        observer,
                        scratch,
                        par,
                    )?),
                    (
                        Data::C64(a),
                        Data::C64(m),
                        Data::C64(n),
                        MatrixData::C64(b),
                        MatrixData::C64(x),
                    ) => MatrixData::C64(run(
                        a,
                        m,
                        n,
                        kind,
                        b.as_ref(),
                        x.as_ref(),
                        atol,
                        rtol,
                        maxiter,
                        stats,
                        iterations,
                        observing,
                        observer,
                        scratch,
                        par,
                    )?),
                    _ => return Err("iterative scalar types must match".into()),
                },
            );
            Ok(())
        })
    }
    #[julia]
    pub fn iterative_workspace_team(
        &mut self,
        a: &OwnedLinearOperator,
        m: &OwnedLinearOperator,
        n: &OwnedLinearOperator,
        kind: i32,
        rhs: &TypedMatrix,
        initial: &TypedMatrix,
        atol: f64,
        rtol: f64,
        maxiter: usize,
        stats: *mut u8,
        iterations: *mut usize,
        observing: bool,
        observer: extern "C" fn(*const u8, usize, usize, isize, isize) -> bool,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                self.data = Some(
                    match (
                        a.data()?,
                        m.data()?,
                        n.data()?,
                        rhs.data()?,
                        initial.data()?,
                    ) {
                        (
                            Data::F32(a),
                            Data::F32(m),
                            Data::F32(n),
                            MatrixData::F32(b),
                            MatrixData::F32(x),
                        ) => MatrixData::F32(run(
                            a,
                            m,
                            n,
                            kind,
                            b.as_ref(),
                            x.as_ref(),
                            atol,
                            rtol,
                            maxiter,
                            stats,
                            iterations,
                            observing,
                            observer,
                            scratch,
                            par,
                        )?),
                        (
                            Data::C32(a),
                            Data::C32(m),
                            Data::C32(n),
                            MatrixData::C32(b),
                            MatrixData::C32(x),
                        ) => MatrixData::C32(run(
                            a,
                            m,
                            n,
                            kind,
                            b.as_ref(),
                            x.as_ref(),
                            atol,
                            rtol,
                            maxiter,
                            stats,
                            iterations,
                            observing,
                            observer,
                            scratch,
                            par,
                        )?),
                        (
                            Data::C64(a),
                            Data::C64(m),
                            Data::C64(n),
                            MatrixData::C64(b),
                            MatrixData::C64(x),
                        ) => MatrixData::C64(run(
                            a,
                            m,
                            n,
                            kind,
                            b.as_ref(),
                            x.as_ref(),
                            atol,
                            rtol,
                            maxiter,
                            stats,
                            iterations,
                            observing,
                            observer,
                            scratch,
                            par,
                        )?),
                        _ => return Err("iterative scalar types must match".into()),
                    },
                );
                Ok(())
            })?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{Base, NativeValue};
    use faer::matrix_free::IdentityPrecond;
    use std::cell::Cell;
    thread_local! {static CALLER:Cell<bool>=const{Cell::new(false)};static CALLS:Cell<usize>=const{Cell::new(0)};}
    extern "C" fn observer(_: *const u8, _: usize, _: usize, _: isize, _: isize) -> bool {
        if !CALLER.with(Cell::get) {
            return false;
        }
        CALLS.with(|n| n.set(n.get() + 1));
        true
    }
    trait Value: NativeValue + Sized {
        fn data(x: Operator<Self>) -> Data;
    }
    impl Value for f32 {
        fn data(x: Operator<Self>) -> Data {
            Data::F32(x)
        }
    }
    impl Value for f64 {
        fn data(x: Operator<Self>) -> Data {
            Data::F64(x)
        }
    }
    impl Value for faer::c32 {
        fn data(x: Operator<Self>) -> Data {
            Data::C32(x)
        }
    }
    impl Value for faer::c64 {
        fn data(x: Operator<Self>) -> Data {
            Data::C64(x)
        }
    }
    fn check<T: Value>() {
        for width in [1, 2] {
            let mut team = OwnedTeam::new();
            team.team_initialize(width).unwrap();
            for kind in 0..3 {
                for (rows, cols) in [(0, 0), (4, 4), (8, 4), (4, 8), (0, 4), (4, 0), (96, 96)] {
                    if kind != 2 && rows != cols {
                        continue;
                    }
                    let dense = Mat::from_fn(rows, cols, |i, j| {
                        T::from_parts(
                            if i == j {
                                5.0
                            } else {
                                0.001 * ((i + j) as f64).sin()
                            },
                            0.001 * (i as f64 - j as f64).sin(),
                        )
                    });
                    let a = Operator::new(Base::Dense(dense.clone()));
                    let m = Operator::<T>::new(Base::Identity(IdentityPrecond { dim: cols }));
                    let n = m.clone();
                    let ao = OwnedLinearOperator {
                        data: Some(T::data(a.clone())),
                    };
                    let mo = OwnedLinearOperator {
                        data: Some(T::data(m.clone())),
                    };
                    let no = OwnedLinearOperator {
                        data: Some(T::data(n.clone())),
                    };
                    let mut info = [0usize; 2];
                    ao.iterative_requirements_team(
                        &mo,
                        &no,
                        kind,
                        2,
                        rows,
                        cols,
                        T::tag(),
                        info.as_mut_ptr(),
                        &team,
                    )
                    .unwrap();
                    let mut work = OwnedScratch::new();
                    work.scratch_reserve(info[0], info[1]).unwrap();
                    let b = Mat::from_fn(rows, 2, |i, j| {
                        T::from_parts(((i + j + 1) as f64).sin(), 0.1)
                    });
                    let x = Mat::<T>::zeros(cols, 2);
                    let mut stats = vec![zero::<T>(); 7];
                    let mut iterations = 0usize;
                    let tolerance = if T::tag() == 0 || T::tag() == 2 {
                        1e-3
                    } else {
                        1e-8
                    };
                    CALLER.with(|c| c.set(true));
                    CALLS.with(|c| c.set(0));
                    let out = team
                        .execute(|par| {
                            let req = requirement(&a, &m, &n, kind, 2, par.shape()).unwrap();
                            assert_eq!(info, [req.size_bytes(), req.align_bytes()]);
                            run(
                                &a,
                                &m,
                                &n,
                                kind,
                                b.as_ref(),
                                x.as_ref(),
                                0.0,
                                tolerance,
                                256,
                                stats.as_mut_ptr().cast(),
                                &mut iterations,
                                true,
                                observer,
                                &mut work,
                                par,
                            )
                            .unwrap()
                        })
                        .unwrap();
                    CALLER.with(|c| c.set(false));
                    assert_eq!(
                        stats[0].parts().0,
                        0.0,
                        "kind {kind}, shape {rows}x{cols}, width {width}"
                    );
                    if rows > 0 && cols > 0 {
                        assert!(CALLS.with(Cell::get) > 0);
                    }
                    let reference = if rows == 0 || cols == 0 {
                        Mat::zeros(cols, 2)
                    } else {
                        dense.thin_svd(Par::Seq).unwrap().pseudoinverse(Par::Seq) * &b
                    };
                    for j in 0..2 {
                        for i in 0..cols {
                            let (x, y) = out[(i, j)].parts();
                            let (u, v) = reference[(i, j)].parts();
                            assert!((x - u).hypot(y - v) < 3e-3);
                        }
                    }
                    team.execute(|mut par| {
                        let req = requirement(&a, &m, &n, kind, 2, par.shape()).unwrap();
                        residual_norms(
                            &a,
                            &m,
                            out.as_ref(),
                            b.as_ref(),
                            kind,
                            par.reborrow(),
                            work.stack(req).unwrap(),
                        );
                        assert_eq!(
                            crate::allocation_probe::count(|| {
                                residual_norms(
                                    &a,
                                    &m,
                                    out.as_ref(),
                                    b.as_ref(),
                                    kind,
                                    par,
                                    work.stack(req).unwrap(),
                                );
                            }),
                            0
                        );
                    })
                    .unwrap();
                }
            }
            team.team_close().unwrap();
        }
    }
    #[test]
    fn iterative_workspace_f32() {
        check::<f32>();
    }
    #[test]
    fn iterative_workspace_f64() {
        check::<f64>();
    }
    #[test]
    fn iterative_workspace_c32() {
        check::<faer::c32>();
    }
    #[test]
    fn iterative_workspace_c64() {
        check::<faer::c64>();
    }
    extern "C" fn keep(_: *const u8, _: usize, _: usize, _: isize, _: isize) -> bool {
        true
    }
    #[test]
    fn invalid_resources_preserve_output_and_statistics() {
        let a = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::Dense(Mat::from_fn(
                4,
                4,
                |i, j| if i == j { 5.0 } else { 0.1 },
            ))))),
        };
        let m = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::Identity(IdentityPrecond {
                dim: 4,
            })))),
        };
        let n = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::Identity(IdentityPrecond {
                dim: 4,
            })))),
        };
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut req = [0usize; 2];
        a.iterative_requirements_team(&m, &n, 0, 1, 4, 4, 1, req.as_mut_ptr(), &team)
            .unwrap();
        assert!(a
            .iterative_requirements_team(&m, &n, 0, 1, 4, 4, 1, std::ptr::null_mut(), &team)
            .is_err());
        assert!(a
            .iterative_requirements_team(&m, &n, 0, 1, 3, 4, 1, req.as_mut_ptr(), &team)
            .is_err());
        assert!(a
            .iterative_requirements_team(&m, &n, 0, 1, 4, 4, 0, req.as_mut_ptr(), &team)
            .is_err());
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req[0] - 1, req[1]).unwrap();
        let mut out = OwnedMatrix::new();
        out.matrix = Mat::from_fn(4, 1, |_, _| 19.0);
        let mut rhs = OwnedMatrix::new();
        rhs.matrix = Mat::from_fn(4, 1, |_, _| 1.0);
        let mut initial = OwnedMatrix::new();
        initial.matrix = Mat::zeros(4, 1);
        let mut stats = [23.0f64; 5];
        let mut count = 99usize;
        assert!(out
            .iterative_workspace_team(
                &a,
                &m,
                &n,
                0,
                &rhs,
                &initial,
                0.0,
                1e-8,
                64,
                stats.as_mut_ptr().cast(),
                &mut count,
                false,
                keep,
                &mut work,
                &team
            )
            .is_err());
        assert_eq!(stats, [23.0; 5]);
        assert_eq!(count, 99);
        assert_eq!(out.matrix[(0, 0)], 19.0);
        work.scratch_reserve(req[0], req[1]).unwrap();
        team.team_close().unwrap();
        assert!(out
            .iterative_workspace_team(
                &a,
                &m,
                &n,
                0,
                &rhs,
                &initial,
                0.0,
                1e-8,
                64,
                stats.as_mut_ptr().cast(),
                &mut count,
                false,
                keep,
                &mut work,
                &team
            )
            .is_err());
        assert_eq!(stats, [23.0; 5]);
        assert_eq!(out.matrix[(0, 0)], 19.0);
    }
}
