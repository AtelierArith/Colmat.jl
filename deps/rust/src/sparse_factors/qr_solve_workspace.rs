//! Rank-aware sparse QR solves using a source-precision R SVD in caller scratch.
use super::{Factors, Numeric, SparseFactor};
use crate::operators::NativeValue;
use crate::pseudoinverse::ToleranceReal;
use crate::teams::with_team;
use crate::teams::OwnedTeam;
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::{svd, temp_mat_scratch, temp_mat_zeroed};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatMut, MatRef, Par, ParShape};
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
fn req<S: ComplexField, T: ComplexField>(
    m: usize,
    n: usize,
    c: usize,
    p: ParShape,
    projection: StackReq,
) -> StackReq {
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
        )
        .or(projection),
    ])
}
fn q_scratch<S: ComplexField>(f: &Numeric<S>, k: usize) -> Result<StackReq, String> {
    Ok(match f {
        Numeric::SimplicialQr(x) => {
            crate::sparse_qr_owned::borrowed::simplicial_scratch(x.h.nrows(), x.r.ncols())
        }
        Numeric::Qr(x) => x.apply_q_scratch(k),
        Numeric::WideQr(x) => x.head_ref().apply_q_scratch(k),
        _ => return Err("requires sparse QR".into()),
    })
}
fn req_for<S: ComplexField, T: ComplexField>(
    f: &Numeric<S>,
    c: usize,
    p: ParShape,
) -> Result<StackReq, String> {
    let (m, n) = dims(f)?;
    let k = m.min(n);
    if k == 0 || c == 0 {
        return Ok(StackReq::EMPTY);
    }
    Ok(req::<S, T>(m, n, c, p, q_scratch(f, k)?))
}
fn write_r<S: ComplexField>(f: &Numeric<S>, mut r: MatMut<'_, S>) -> Result<(), String> {
    r.fill(zero::<S>());
    match f {
        Numeric::SimplicialQr(x) => {
            for e in x.r.as_ref().triplet_iter() {
                r[(e.row, e.col)] = add(&r[(e.row, e.col)], e.val);
            }
        }
        Numeric::Qr(x) => x.write_r(r),
        Numeric::WideQr(x) => {
            for e in x.r_ref().triplet_iter() {
                r[(e.row, e.col)] = add(&r[(e.row, e.col)], e.val);
            }
        }
        _ => return Err("requires sparse QR".into()),
    }
    Ok(())
}
fn apply_q<S: ComplexField>(
    f: &Numeric<S>,
    q: MatMut<'_, S>,
    u: MatRef<'_, S>,
    p: Par,
    stack: &mut faer::dyn_stack::MemStack,
) -> Result<(), String> {
    match f {
        Numeric::SimplicialQr(x) => {
            crate::sparse_qr_owned::borrowed::simplicial_apply(x.h.as_ref(), &x.tau, q, u, stack)
        }
        Numeric::Qr(x) => x.apply_q_in(q, u, p, stack),
        Numeric::WideQr(x) => x.head_ref().apply_q_in(q, u, p, stack),
        _ => return Err("requires sparse QR".into()),
    }
    Ok(())
}
fn permutation<S: ComplexField>(f: &Numeric<S>) -> Result<faer::perm::PermRef<'_, usize>, String> {
    Ok(match f {
        Numeric::SimplicialQr(x) => x.plan.cols.as_ref(),
        Numeric::Qr(x) => x.permutation_ref(),
        Numeric::WideQr(x) => x.permutation_ref(),
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
    let stack = work.stack(req_for::<S, T>(f, columns, par.shape())?)?;
    let mut out = Mat::zeros(if tr { m } else { n }, columns);
    if k == 0 || columns == 0 {
        return Ok(out);
    }
    let (mut rdense, stack) = temp_mat_zeroed::<S, _, _>(k, n, stack);
    write_r(f, rdense.as_mat_mut())?;
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
    apply_q(f, qu.as_mat_mut(), ur.as_mat_ref(), par.reborrow(), stack)?;
    let p = permutation(f)?.arrays().0;
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
    _m: usize,
    _n: usize,
    c: usize,
    target: i32,
    p: ParShape,
    out: *mut usize,
    fast: bool,
) -> Result<(), String> {
    macro_rules! for_source {
        ($s:ty,$factor:expr) => {
            match target {
                0 => req_for::<$s, f32>($factor, c, p)?,
                1 => req_for::<$s, f64>($factor, c, p)?,
                2 => req_for::<$s, faer::c32>($factor, c, p)?,
                3 => req_for::<$s, faer::c64>($factor, c, p)?,
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
            Factors::F32(x) => for_source!(f32, x),
            Factors::F64(x) => for_source!(f64, x),
            Factors::C32(x) => for_source!(faer::c32, x),
            Factors::C64(x) => for_source!(faer::c64, x),
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
        out: &mut [usize],
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            with_team(team, |p| {
                validate(self, m, n, source, target, op, atol, rtol)?;
                fast_validate(fast, source, target, op)?;
                query(self, m, n, c, target, p.shape(), out, fast)
            })
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |p| {
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
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |p| {
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
        })
    }
}
#[julia]
impl SparseFactor {}
#[julia]
impl OwnedMatrix {}
#[julia]
impl TypedMatrix {}
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
                    let req = req_for::<S, faer::c64>(&f, 2, p.shape()).unwrap();
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

#[cfg(test)]
mod borrowed_tests {
    use super::*;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::sparse::linalg::{
        qr::{self, QrSymbolicParams},
        SupernodalThreshold,
    };
    use faer::sparse::{SparseColMat, Triplet};
    fn source<S: NativeValue>(
        m: usize,
        n: usize,
        mode: usize,
        deficient: bool,
    ) -> (Numeric<S>, SparseColMat<usize, S>) {
        let entries: Vec<_> = (0..n)
            .flat_map(|j| {
                (0..m).filter_map(move |i| {
                    if deficient && (j == 0 || j == n / 2) {
                        return None;
                    }
                    let re = if i == j {
                        5.0
                    } else {
                        0.03 * ((i + 2 * j + 1) as f64).sin()
                    };
                    let v = S::from_parts(re, 0.02 * ((2 * i + j + 1) as f64).cos());
                    Some(Triplet {
                        row: i,
                        col: j,
                        val: v,
                    })
                })
            })
            .collect();
        let a = SparseColMat::<usize, S>::try_new_from_triplets(m, n, &entries).unwrap();
        let indices: Vec<_> = (0..n).rev().collect();
        let perm = faer::perm::Perm::new_checked(
            indices.clone().into_boxed_slice(),
            indices.into_boxed_slice(),
            n,
        );
        let params = QrSymbolicParams {
            supernodal_flop_ratio_threshold: if mode == 2 {
                SupernodalThreshold::FORCE_SUPERNODAL
            } else if mode == 1 {
                SupernodalThreshold::FORCE_SIMPLICIAL
            } else {
                SupernodalThreshold::AUTO
            },
            ..Default::default()
        };
        let f = if m < n {
            let plan = crate::sparse_qr_wide::Plan::new(a.symbolic(), perm, params).unwrap();
            Numeric::WideQr(crate::sparse_qr_wide::Factors::new(&plan, a.as_ref()).unwrap())
        } else if mode == 1 {
            let plan = crate::sparse_qr_components::Plan::new(a.symbolic(), perm.as_ref()).unwrap();
            Numeric::SimplicialQr(
                crate::sparse_qr_components::Factors::new(&plan, a.as_ref()).unwrap(),
            )
        } else {
            let sym = std::sync::Arc::new(
                qr::factorize_symbolic_qr_with_column_ordering(
                    a.symbolic(),
                    params,
                    Some(perm.as_ref()),
                )
                .unwrap(),
            );
            Numeric::Qr(crate::sparse_qr_owned::Factors::new(sym, a.as_ref()).unwrap())
        };
        (f, a)
    }
    fn legacy<S: ComplexField>(f: &Numeric<S>) -> (Mat<S>, Mat<S>, faer::perm::Perm<usize>) {
        match f {
            Numeric::SimplicialQr(x) => (x.q(), x.r.to_dense(), x.plan.cols.clone()),
            Numeric::Qr(x) => (x.q().unwrap(), x.r().unwrap().to_dense(), x.permutation()),
            Numeric::WideQr(x) => (x.q().unwrap(), x.r().to_dense(), x.permutation()),
            _ => unreachable!(),
        }
    }
    fn check<S: NativeValue>()
    where
        S::Real: ToleranceReal,
    {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for mode in 0..3 {
            for (m, n) in [
                (0, 0),
                (0, 3),
                (3, 0),
                (7, 4),
                (4, 7),
                (4, 4),
                (96, 64),
                (64, 96),
            ] {
                for deficient in [false, true] {
                    let (f, a) = source::<S>(m, n, mode, deficient);
                    let k = m.min(n);
                    let tolerance =
                        eps::<S::Real>() * from_f64::<S::Real>(20.0 * m.max(n).max(1) as f64);
                    let (q, r, perm) = legacy(&f);
                    let mut staged = Mat::<S>::zeros(k, n);
                    write_r(&f, staged.as_mut()).unwrap();
                    assert_eq!((&staged - &r).norm_max(), zero::<S::Real>());
                    assert_eq!(
                        permutation(&f).unwrap().arrays().0,
                        perm.as_ref().arrays().0
                    );
                    // Independent input reconstruction checks factor coordinates as well as Q.
                    let reconstructed = &q * &staged;
                    let p = perm.as_ref().arrays().0;
                    let dense = a.to_dense();
                    for j in 0..n {
                        for i in 0..m {
                            assert!(
                                abs(&(copy(&reconstructed[(i, j)]) - copy(&dense[(i, p[j])])))
                                    < tolerance
                            );
                        }
                    }
                    let b =
                        Mat::from_fn(k, 3, |i, j| S::from_parts((i + j + 1) as f64 * 0.02, 0.01));
                    let expected = &q * &b;
                    for parallel in [false, true] {
                        let mut exercise = |mut par: Par| {
                            let req = q_scratch(&f, 3).unwrap();
                            let mut mem = MemBuffer::new(req);
                            let mut out = Mat::<S>::zeros(m, 3);
                            apply_q(
                                &f,
                                out.as_mut(),
                                b.as_ref(),
                                par.reborrow(),
                                MemStack::new(&mut mem),
                            )
                            .unwrap();
                            assert!((&out - &expected).norm_max() < tolerance);
                            let allocations = crate::allocation_probe::count(|| {
                                write_r(&f, staged.as_mut()).unwrap();
                                apply_q(
                                    &f,
                                    out.as_mut(),
                                    b.as_ref(),
                                    par.reborrow(),
                                    MemStack::new(&mut mem),
                                )
                                .unwrap();
                                let _ = permutation(&f).unwrap();
                            });
                            assert_eq!(
                                allocations, 0,
                                "borrowed components allocate on calling thread"
                            );
                        };
                        if parallel {
                            team.execute(exercise).unwrap();
                        } else {
                            exercise(Par::Seq);
                        }
                    }
                }
            }
        }
    }
    fn rank_preparation<S: NativeValue>()
    where
        S::Real: ToleranceReal,
    {
        let epsilon = if S::tag() == 0 || S::tag() == 2 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        for mode in 0..3 {
            for (m, n) in [(0, 0), (3, 0), (1, 1), (7, 4), (4, 7), (96, 64), (64, 96)] {
                for deficient in [false, true] {
                    let (f, _) = source::<S>(m, n, mode, deficient);
                    let (_, r, _) = legacy(&f);
                    let expected = if m < n {
                        false
                    } else {
                        let entries: Vec<_> = (0..n)
                            .flat_map(|j| {
                                let r = &r;
                                (0..r.nrows()).map(move |i| Triplet {
                                    row: i,
                                    col: j,
                                    val: copy(&r[(i, j)]),
                                })
                            })
                            .collect();
                        let sparse =
                            SparseColMat::<usize, S>::try_new_from_triplets(r.nrows(), n, &entries)
                                .unwrap();
                        crate::sparse_factors::qr_needs_spectral_solve(sparse.as_ref(), epsilon)
                    };
                    let mut memory = MemBuffer::new(crate::sparse_factors::rank_hint::scratch(n));
                    assert_eq!(
                        crate::sparse_factors::rank_hint::needs(
                            &f,
                            epsilon,
                            MemStack::new(&mut memory)
                        ),
                        Some(expected)
                    );
                    assert_eq!(
                        crate::allocation_probe::count(|| {
                            assert_eq!(
                                crate::sparse_factors::rank_hint::needs(
                                    &f,
                                    epsilon,
                                    MemStack::new(&mut memory)
                                ),
                                Some(expected)
                            );
                        }),
                        0
                    );
                }
            }
        }
    }
    #[test]
    fn construction_rank_f32() {
        rank_preparation::<f32>();
    }
    #[test]
    fn construction_rank_f64() {
        rank_preparation::<f64>();
    }
    #[test]
    fn construction_rank_c32() {
        rank_preparation::<faer::c32>();
    }
    #[test]
    fn construction_rank_c64() {
        rank_preparation::<faer::c64>();
    }
    #[test]
    fn construction_workspace_errors_preserve_numeric_revision() {
        use crate::sparse::{SparseData, SparseOwner, Storage};
        use crate::sparse_factors::{Analysis, Plan, SparseSymbolic};
        let (_, a) = source::<f64>(7, 4, 2, false);
        let symbolic = std::sync::Arc::new(
            qr::factorize_symbolic_qr(a.symbolic(), Default::default()).unwrap(),
        );
        let plan = SparseSymbolic {
            tuning: Default::default(),
            plan: Some(Plan {
                analysis: Analysis::Qr(symbolic),
                pattern: a.symbolic().to_owned().unwrap(),
                lower: true,
            }),
        };
        let matrix = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(a.clone()))),
        };
        let mut query = [0usize; 2];
        plan.sparse_qr_factor_requirements(1, &mut query[..], None)
            .unwrap();
        assert!(plan
            .sparse_qr_factor_requirements(1, &mut [], None)
            .is_err());
        let mut work = OwnedScratch::new();
        work.scratch_reserve(query[0], query[1]).unwrap();
        let mut owner = SparseFactor::new();
        assert_eq!(
            owner
                .sparse_qr_factor_workspace(&plan, &matrix, &mut work, None)
                .unwrap(),
            0
        );
        let mut before = Mat::zeros(4, 4);
        let Factors::F64(f) = owner.factors().unwrap() else {
            unreachable!()
        };
        write_r(f, before.as_mut()).unwrap();
        let hint = owner.sparse_qr_needs_spectral_solve().unwrap();
        let mut short = OwnedScratch::new();
        short.scratch_reserve(query[0] - 1, query[1]).unwrap();
        assert!(owner
            .sparse_qr_factor_workspace(&plan, &matrix, &mut short, None)
            .is_err());
        let entries: Vec<_> = a
            .as_ref()
            .triplet_iter()
            .filter(|e| !(e.row == 0 && e.col == 0))
            .map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: copy(e.val),
            })
            .collect();
        let bad = SparseOwner {
            data: Some(SparseData::F64(Storage::Csc(
                SparseColMat::try_new_from_triplets(7, 4, &entries).unwrap(),
            ))),
        };
        assert!(owner
            .sparse_qr_factor_workspace(&plan, &bad, &mut work, None)
            .is_err());
        assert_eq!(owner.sparse_qr_needs_spectral_solve().unwrap(), hint);
        let mut after = Mat::zeros(4, 4);
        let Factors::F64(f) = owner.factors().unwrap() else {
            unreachable!()
        };
        write_r(f, after.as_mut()).unwrap();
        assert_eq!((&before - &after).norm_max(), 0.0);
    }
    #[test]
    fn rank_certificate_prepared_per_numeric_revision() {
        let (full, _) = source::<f64>(7, 4, 2, false);
        let (deficient, _) = source::<f64>(7, 4, 2, true);
        let mut owner = SparseFactor::new();
        owner.replace_factors(Factors::F64(full)).unwrap();
        assert!(!owner.sparse_qr_needs_spectral_solve().unwrap());
        let mut independent = SparseFactor::new();
        independent.sparse_factor_clone(&owner).unwrap();
        assert_eq!(
            crate::allocation_probe::count(|| {
                assert!(!owner.sparse_qr_needs_spectral_solve().unwrap());
                assert!(!independent.sparse_qr_needs_spectral_solve().unwrap());
            }),
            0
        );
        owner.replace_factors(Factors::F64(deficient)).unwrap();
        assert!(owner.sparse_qr_needs_spectral_solve().unwrap());
        assert!(!independent.sparse_qr_needs_spectral_solve().unwrap());
        assert_eq!(
            crate::allocation_probe::count(|| {
                assert!(owner.sparse_qr_needs_spectral_solve().unwrap());
            }),
            0
        );
    }
    #[test]
    fn borrowed_f32() {
        check::<f32>();
    }
    #[test]
    fn borrowed_f64() {
        check::<f64>();
    }
    #[test]
    fn borrowed_c32() {
        check::<faer::c32>();
    }
    #[test]
    fn borrowed_c64() {
        check::<faer::c64>();
    }
}
