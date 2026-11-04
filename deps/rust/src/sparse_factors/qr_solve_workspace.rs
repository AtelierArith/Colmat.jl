//! Rank-aware sparse QR solves using a source-precision R SVD in caller scratch.
use super::{Factors, Numeric, SparseFactor};
use crate::operators::NativeValue;
use crate::pseudoinverse::ToleranceReal;
use crate::teams::OwnedTeam;
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::{svd, temp_mat_scratch, temp_mat_zeroed};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Mat, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
fn dims<S: ComplexField>(f: &Numeric<S>) -> Result<(usize, usize), String> {
    Ok(match f {
        Numeric::SimplicialQr(x) => (x.h.nrows(), x.r.ncols()),
        Numeric::Qr(x) => x.dimensions(),
        Numeric::WideQr(x) => x.dimensions(),
        _ => return Err("requires sparse QR".into()),
    })
}
fn validate(
    f: &SparseFactor,
    m: usize,
    n: usize,
    source: i32,
    target: i32,
    op: i32,
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    let (actual, tag) = match f.factors()? {
        Factors::F32(x) => (dims(x)?, 0),
        Factors::F64(x) => (dims(x)?, 1),
        Factors::C32(x) => (dims(x)?, 2),
        Factors::C64(x) => (dims(x)?, 3),
    };
    if actual != (m, n) || tag != source {
        return Err("sparse QR factor native metadata mismatch".into());
    }
    if !(0..4).contains(&op) {
        return Err("invalid QR solve transform".into());
    }
    if !(source == target || matches!((source, target), (0, 1) | (0, 2) | (0, 3) | (1, 3) | (2, 3)))
    {
        return Err("invalid QR solve promotion".into());
    }
    if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
        return Err("invalid QR solve tolerances".into());
    }
    Ok(())
}
fn req<S: ComplexField, T: ComplexField>(m: usize, n: usize, c: usize, p: ParShape) -> StackReq {
    let k = m.min(n);
    if k == 0 || c == 0 {
        return StackReq::EMPTY;
    }
    StackReq::all_of(&[
        temp_mat_scratch::<S>(k, n),
        temp_mat_scratch::<S>(k, k),
        temp_mat_scratch::<S>(n, k),
        temp_mat_scratch::<S>(k, 1),
        temp_mat_scratch::<S>(m, k),
        temp_mat_scratch::<T>(m, k),
        temp_mat_scratch::<T>(n, k),
        temp_mat_scratch::<T>(k, 1),
        temp_mat_scratch::<T>(k, c),
        svd::svd_scratch::<S>(
            k,
            n,
            svd::ComputeSvdVectors::Thin,
            svd::ComputeSvdVectors::Thin,
            p,
            Default::default(),
        ),
    ])
}
fn parts<S: ComplexField>(
    f: &Numeric<S>,
) -> Result<
    (
        Mat<S>,
        faer::sparse::SparseColMat<usize, S>,
        faer::perm::Perm<usize>,
    ),
    String,
> {
    Ok(match f {
        Numeric::SimplicialQr(x) => (x.q(), x.r.clone(), x.plan.cols.clone()),
        Numeric::Qr(x) => (x.q()?, x.r()?, x.permutation()),
        Numeric::WideQr(x) => (x.q()?, x.r(), x.permutation()),
        _ => return Err("requires sparse QR".into()),
    })
}
fn run<S: NativeValue, T: NativeValue>(
    f: &Numeric<S>,
    b: MatRef<'_, T>,
    work: &mut OwnedScratch,
    columns: usize,
    op: i32,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    mut par: Par,
) -> Result<Mat<T>, String>
where
    S::Real: ToleranceReal,
{
    let (m, n) = dims(f)?;
    let k = m.min(n);
    let tr = op == 1 || op == 2;
    if b.shape() != (if tr { n } else { m }, columns) {
        return Err("QR RHS native shape mismatch".into());
    }
    finite(b)?;
    let stack = work.stack(req::<S, T>(m, n, columns, par.shape()))?;
    let mut out = Mat::zeros(if tr { m } else { n }, columns);
    if k == 0 || columns == 0 {
        return Ok(out);
    }
    // Independent Q/R/permutation materialization is component preparation, outside
    // solve scratch. No input reconstruction or explicit pseudoinverse is formed.
    let (q, r, permutation) = parts(f)?;
    let (mut rdense, stack) = temp_mat_zeroed::<S, _, _>(k, n, stack);
    for e in r.as_ref().triplet_iter() {
        rdense.as_mat_mut()[(e.row, e.col)] = copy(e.val);
    }
    let (mut ur, stack) = temp_mat_zeroed::<S, _, _>(k, k, stack);
    let (mut vr, stack) = temp_mat_zeroed::<S, _, _>(n, k, stack);
    let (mut values, stack) = temp_mat_zeroed::<S, _, _>(k, 1, stack);
    let (mut qu, stack) = temp_mat_zeroed::<S, _, _>(m, k, stack);
    let (mut u, stack) = temp_mat_zeroed::<T, _, _>(m, k, stack);
    let (mut v, stack) = temp_mat_zeroed::<T, _, _>(n, k, stack);
    let (mut weights, stack) = temp_mat_zeroed::<T, _, _>(k, 1, stack);
    let (mut projection, stack) = temp_mat_zeroed::<T, _, _>(k, columns, stack);
    svd::svd(
        rdense.as_mat_ref(),
        values.as_mat_mut().col_mut(0).as_diagonal_mut(),
        Some(ur.as_mat_mut()),
        Some(vr.as_mat_mut()),
        par.reborrow(),
        stack,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let cutoff = values
        .as_mat_ref()
        .col(0)
        .norm_max()
        .cutoff(atol, rtol, relative_f32);
    for j in 0..k {
        let s = copy(&values.as_mat_ref()[(j, 0)]);
        let (re, im) = s.parts();
        let converted = T::from_parts(re, im);
        weights.as_mat_mut()[(j, 0)] = if abs(&real(&s)) > cutoff {
            from_real::<T>(&recip(&real(&converted)))
        } else {
            zero::<T>()
        };
    }
    if !(0..k).any(|i| weights.as_mat_ref()[(i, 0)] != zero::<T>()) {
        return Ok(out);
    }
    faer::linalg::matmul::matmul(
        qu.as_mat_mut(),
        Accum::Replace,
        q.as_ref(),
        ur.as_mat_ref(),
        one::<S>(),
        par.reborrow(),
    );
    let p = permutation.as_ref().arrays().0;
    for j in 0..k {
        for i in 0..m {
            let (re, im) = qu.as_mat_ref()[(i, j)].parts();
            u.as_mat_mut()[(i, j)] = T::from_parts(re, im);
        }
        for i in 0..n {
            let (re, im) = vr.as_mat_ref()[(i, j)].parts();
            v.as_mat_mut()[(p[i], j)] = T::from_parts(re, im);
        }
    }
    crate::spectral_solve::workspace::project(
        out.as_mut(),
        b,
        u.as_mat_ref(),
        v.as_mat_ref(),
        weights.as_mat_ref(),
        projection.as_mat_mut(),
        op,
        false,
        par,
    );
    Ok(out)
}
fn query(
    f: &SparseFactor,
    m: usize,
    n: usize,
    c: usize,
    target: i32,
    p: ParShape,
    out: *mut usize,
    fast: bool,
) -> Result<(), String> {
    macro_rules! for_source {
        ($s:ty) => {
            match target {
                0 => req::<$s, f32>(m, n, c, p),
                1 => req::<$s, f64>(m, n, c, p),
                2 => req::<$s, faer::c32>(m, n, c, p),
                3 => req::<$s, faer::c64>(m, n, c, p),
                _ => return Err("invalid QR solve target".into()),
            }
        };
    }
    let req = if fast {
        match f.factors()? {
            Factors::F32(x) => fast_req(x, c, p)?,
            Factors::F64(x) => fast_req(x, c, p)?,
            Factors::C32(x) => fast_req(x, c, p)?,
            Factors::C64(x) => fast_req(x, c, p)?,
        }
    } else {
        match f.factors()? {
            Factors::F32(_) => for_source!(f32),
            Factors::F64(_) => for_source!(f64),
            Factors::C32(_) => for_source!(faer::c32),
            Factors::C64(_) => for_source!(faer::c64),
        }
    };
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("query requires aligned two-entry usize buffer".into());
    }
    // SAFETY: caller preserves two writable usize entries, address/span checked.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}

fn fast_req<T: ComplexField>(f: &Numeric<T>, c: usize, p: ParShape) -> Result<StackReq, String> {
    Ok(match f {
        Numeric::Qr(x) => x.solve_requirements_par(c, p),
        Numeric::SimplicialQr(x) => x.plan.symbolic.solve_in_place_scratch::<T>(c, p),
        _ => return Err("fast QR solve requires tall/square QR factors".into()),
    })
}
fn fast_validate(fast: bool, source: i32, target: i32, op: i32) -> Result<(), String> {
    if fast && (source != target || op != 0) {
        return Err("fast QR solve requires matching scalar and normal orientation".into());
    }
    Ok(())
}
fn fast_run<T: ComplexField>(
    f: &Numeric<T>,
    b: MatRef<'_, T>,
    work: &mut OwnedScratch,
    c: usize,
    par: Par,
) -> Result<Mat<T>, String> {
    let (m, n) = dims(f)?;
    if b.shape() != (m, c) {
        return Err("QR RHS native shape mismatch".into());
    }
    finite(b)?;
    let stack = work.stack(fast_req(f, c, par.shape())?)?;
    if m.min(n) == 0 || c == 0 {
        return Ok(Mat::zeros(n, c));
    }
    match f {
        Numeric::Qr(x) => x.solve_in_par(b, par, stack),
        Numeric::SimplicialQr(x) => {
            let mut out = b.to_owned();
            x.as_ref()
                .solve_in_place_with_conj(faer::Conj::No, out.as_mut(), par, stack);
            let inverse = x.plan.cols.as_ref().arrays().1;
            Ok(Mat::from_fn(n, c, |i, j| copy(&out[(inverse[i], j)])))
        }
        _ => Err("fast QR solve requires tall/square QR".into()),
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_qr_rank_requirements(
        &self,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        fast: bool,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(self, m, n, source, target, op, atol, rtol)?;
            fast_validate(fast, source, target, op)?;
            query(self, m, n, c, target, p.shape(), out, fast)
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_qr_rank_workspace(
        &mut self,
        f: &SparseFactor,
        b: &OwnedMatrix,
        work: &mut OwnedScratch,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        fast: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(f, m, n, source, target, op, atol, rtol)?;
            fast_validate(fast, source, target, op)?;
            if target != 1 {
                return Err("QR output scalar mismatch".into());
            }
            let result = match f.factors()? {
                Factors::F32(x) => run(
                    x,
                    b.matrix.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?,
                Factors::F64(x) => {
                    if fast {
                        fast_run(x, b.matrix.as_ref(), work, c, p)?
                    } else {
                        run(
                            x,
                            b.matrix.as_ref(),
                            work,
                            c,
                            op,
                            atol,
                            rtol,
                            relative_f32,
                            p,
                        )?
                    }
                }
                _ => return Err("QR scalar mismatch".into()),
            };
            self.matrix = result;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_qr_rank_workspace(
        &mut self,
        f: &SparseFactor,
        b: &TypedMatrix,
        work: &mut OwnedScratch,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        fast: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let p = Par::Seq;
            validate(f, m, n, source, target, op, atol, rtol)?;
            fast_validate(fast, source, target, op)?;
            let actual = match b.data()? {
                MatrixData::F32(_) => 0,
                MatrixData::F64(_) => 1,
                MatrixData::C32(_) => 2,
                MatrixData::C64(_) => 3,
            };
            if actual != target {
                return Err("QR RHS scalar metadata mismatch".into());
            }
            let result = match (f.factors()?, b.data()?) {
                (Factors::F32(x), MatrixData::F32(b)) => MatrixData::F32(if fast {
                    fast_run(x, b.as_ref(), work, c, p)?
                } else {
                    run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                }),
                (Factors::F32(x), MatrixData::F64(b)) => MatrixData::F64(run(
                    x,
                    b.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?),
                (Factors::F32(x), MatrixData::C32(b)) => MatrixData::C32(run(
                    x,
                    b.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?),
                (Factors::F32(x), MatrixData::C64(b)) => MatrixData::C64(run(
                    x,
                    b.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?),
                (Factors::F64(x), MatrixData::F64(b)) => MatrixData::F64(if fast {
                    fast_run(x, b.as_ref(), work, c, p)?
                } else {
                    run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                }),
                (Factors::F64(x), MatrixData::C64(b)) => MatrixData::C64(run(
                    x,
                    b.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?),
                (Factors::C32(x), MatrixData::C32(b)) => MatrixData::C32(if fast {
                    fast_run(x, b.as_ref(), work, c, p)?
                } else {
                    run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                }),
                (Factors::C32(x), MatrixData::C64(b)) => MatrixData::C64(run(
                    x,
                    b.as_ref(),
                    work,
                    c,
                    op,
                    atol,
                    rtol,
                    relative_f32,
                    p,
                )?),
                (Factors::C64(x), MatrixData::C64(b)) => MatrixData::C64(if fast {
                    fast_run(x, b.as_ref(), work, c, p)?
                } else {
                    run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                }),
                _ => return Err("QR scalar mismatch".into()),
            };
            self.data = Some(result);
            Ok(())
        })
    }
}
#[julia]
impl SparseFactor {
    #[julia]
    pub fn sparse_qr_rank_requirements_team(
        &self,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        fast: bool,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(self, m, n, source, target, op, atol, rtol)?;
                fast_validate(fast, source, target, op)?;
                query(self, m, n, c, target, p.shape(), out, fast)
            })?
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_qr_rank_workspace_team(
        &mut self,
        f: &SparseFactor,
        b: &OwnedMatrix,
        work: &mut OwnedScratch,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        fast: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(f, m, n, source, target, op, atol, rtol)?;
                fast_validate(fast, source, target, op)?;
                if target != 1 {
                    return Err("QR output scalar mismatch".into());
                }
                let result = match f.factors()? {
                    Factors::F32(x) => run(
                        x,
                        b.matrix.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?,
                    Factors::F64(x) => {
                        if fast {
                            fast_run(x, b.matrix.as_ref(), work, c, p)?
                        } else {
                            run(
                                x,
                                b.matrix.as_ref(),
                                work,
                                c,
                                op,
                                atol,
                                rtol,
                                relative_f32,
                                p,
                            )?
                        }
                    }
                    _ => return Err("QR scalar mismatch".into()),
                };
                self.matrix = result;
                Ok(())
            })?
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_qr_rank_workspace_team(
        &mut self,
        f: &SparseFactor,
        b: &TypedMatrix,
        work: &mut OwnedScratch,
        m: usize,
        n: usize,
        c: usize,
        source: i32,
        target: i32,
        op: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        fast: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|p| {
                validate(f, m, n, source, target, op, atol, rtol)?;
                fast_validate(fast, source, target, op)?;
                let actual = match b.data()? {
                    MatrixData::F32(_) => 0,
                    MatrixData::F64(_) => 1,
                    MatrixData::C32(_) => 2,
                    MatrixData::C64(_) => 3,
                };
                if actual != target {
                    return Err("QR RHS scalar metadata mismatch".into());
                }
                let result = match (f.factors()?, b.data()?) {
                    (Factors::F32(x), MatrixData::F32(b)) => MatrixData::F32(if fast {
                        fast_run(x, b.as_ref(), work, c, p)?
                    } else {
                        run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                    }),
                    (Factors::F32(x), MatrixData::F64(b)) => MatrixData::F64(run(
                        x,
                        b.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?),
                    (Factors::F32(x), MatrixData::C32(b)) => MatrixData::C32(run(
                        x,
                        b.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?),
                    (Factors::F32(x), MatrixData::C64(b)) => MatrixData::C64(run(
                        x,
                        b.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?),
                    (Factors::F64(x), MatrixData::F64(b)) => MatrixData::F64(if fast {
                        fast_run(x, b.as_ref(), work, c, p)?
                    } else {
                        run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                    }),
                    (Factors::F64(x), MatrixData::C64(b)) => MatrixData::C64(run(
                        x,
                        b.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?),
                    (Factors::C32(x), MatrixData::C32(b)) => MatrixData::C32(if fast {
                        fast_run(x, b.as_ref(), work, c, p)?
                    } else {
                        run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                    }),
                    (Factors::C32(x), MatrixData::C64(b)) => MatrixData::C64(run(
                        x,
                        b.as_ref(),
                        work,
                        c,
                        op,
                        atol,
                        rtol,
                        relative_f32,
                        p,
                    )?),
                    (Factors::C64(x), MatrixData::C64(b)) => MatrixData::C64(if fast {
                        fast_run(x, b.as_ref(), work, c, p)?
                    } else {
                        run(x, b.as_ref(), work, c, op, atol, rtol, relative_f32, p)?
                    }),
                    _ => return Err("QR scalar mismatch".into()),
                };
                self.data = Some(result);
                Ok(())
            })?
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    fn check<S: NativeValue>()
    where
        S::Real: ToleranceReal,
    {
        let a = SparseColMat::<usize, S>::try_new_from_triplets(
            4,
            3,
            &[
                Triplet {
                    row: 0,
                    col: 0,
                    val: S::from_parts(2.0, 0.0),
                },
                Triplet {
                    row: 2,
                    col: 2,
                    val: S::from_parts(4.0, 0.0),
                },
            ],
        )
        .unwrap();
        let sym = std::sync::Arc::new(
            faer::sparse::linalg::qr::factorize_symbolic_qr(a.symbolic(), Default::default())
                .unwrap(),
        );
        let f = Numeric::Qr(crate::sparse_qr_owned::Factors::new(sym, a.as_ref()).unwrap());
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for op in 0..4 {
            for parallel in [false, true] {
                let exercise = |mut p: Par| {
                    let rows = if op == 1 || op == 2 { 3 } else { 4 };
                    let output = if rows == 3 { 4 } else { 3 };
                    let b = Mat::from_fn(rows, 2, |_, _| faer::c64::new(1.0, 0.3));
                    let req = req::<S, faer::c64>(4, 3, 2, p.shape());
                    let mut work = OwnedScratch::new();
                    work.scratch_reserve(req.size_bytes(), req.align_bytes())
                        .unwrap();
                    let x = run(
                        &f,
                        b.as_ref(),
                        &mut work,
                        2,
                        op,
                        0.0,
                        1e-5,
                        false,
                        p.reborrow(),
                    )
                    .unwrap();
                    assert_eq!(x.shape(), (output, 2));
                    for j in 0..2 {
                        for i in 0..output {
                            let expected = if i == 0 {
                                faer::c64::new(0.5, 0.15)
                            } else if i == 2 {
                                faer::c64::new(0.25, 0.075)
                            } else {
                                faer::c64::new(0.0, 0.0)
                            };
                            assert!((x[(i, j)] - expected).norm() < 1e-4);
                        }
                    }
                    let mut short = OwnedScratch::new();
                    short
                        .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                        .unwrap();
                    assert!(run(
                        &f,
                        b.as_ref(),
                        &mut short,
                        2,
                        op,
                        0.0,
                        1e-5,
                        false,
                        p.reborrow()
                    )
                    .is_err());
                    work.scratch_close().unwrap();
                    assert!(run(
                        &f,
                        b.as_ref(),
                        &mut work,
                        2,
                        op,
                        0.0,
                        1e-5,
                        false,
                        p.reborrow()
                    )
                    .is_err());
                };
                if parallel {
                    team.execute(exercise).unwrap();
                } else {
                    exercise(Par::Seq);
                }
            }
        }
    }
    #[test]
    fn f32_rank() {
        check::<f32>();
    }
    #[test]
    fn f64_rank() {
        check::<f64>();
    }
    #[test]
    fn c32_rank() {
        check::<faer::c32>();
    }
    #[test]
    fn c64_rank() {
        check::<faer::c64>();
    }
}
