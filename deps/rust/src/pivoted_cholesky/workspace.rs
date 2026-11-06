//! Pivoted LLT with caller-owned permutation/kernel temporaries.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{
    guarded,
    teams::{with_team, OwnedTeam},
    workspace::OwnedScratch,
    OwnedMatrix,
};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::cholesky::llt_pivoting::factor;
use faer::mat::AsMatMut;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

fn params<T: ComplexField>(
    block_size: usize,
) -> Result<faer::Spec<factor::PivLltParams, T>, String> {
    if block_size == 0 {
        return Err("pivoted Cholesky block_size must be positive".into());
    }
    Ok(factor::PivLltParams {
        block_size,
        ..Default::default()
    }
    .into())
}
pub(crate) fn requirement<T: ComplexField>(
    n: usize,
    tol: f64,
    block_size: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    let params = params::<T>(block_size)?;
    if !tol.is_finite() {
        return Err("pivoted Cholesky tolerance must be finite".into());
    }
    n.checked_mul(n)
        .and_then(|nn| nn.checked_mul(core::mem::size_of::<T>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("pivoted Cholesky dimensions overflow")?;
    let numerical = if tol < 0.0 {
        StackReq::new::<usize>(n).and(factor::cholesky_in_place_scratch::<usize, T>(
            n, par, params,
        ))
    } else {
        faer::linalg::temp_mat_scratch::<T>(n, n).and(StackReq::new::<T::Real>(n))
    };
    Ok(StackReq::new::<usize>(n).and(numerical))
}

// The explicit-tolerance path groups columns into panels. Pending panel
// products are subtracted from each pivot column; the trailing Schur update
// uses the caller's team once per panel. Thus block_size controls both paths.
fn absolute_factor<T: ComplexField>(
    mut a: MatMut<'_, T>,
    forward: &mut [usize],
    tol: f64,
    block_size: usize,
    mut par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<T>, usize), String> {
    let n = a.nrows();
    let (mut diagonal, _) = stack.make_with::<T::Real>(n, |_| zero());
    let threshold = from_f64::<T::Real>(tol);
    let mut l = Mat::<T>::zeros(n, n);
    let mut rank = 0;
    let mut start = 0;
    while start < n {
        let end = start + block_size.min(n - start);
        for i in start..n {
            diagonal[i] = real(&a[(i, i)]);
        }
        for j in start..end {
            let mut pivot = j;
            for i in j + 1..n {
                if diagonal[i] > diagonal[pivot] {
                    pivot = i;
                }
            }
            let d = copy(&diagonal[pivot]);
            if !is_finite(&d) {
                return Err("pivoted Cholesky diagonal became nonfinite".into());
            }
            if d <= threshold || d <= zero::<T::Real>() {
                return Ok((l, rank));
            }
            if pivot != j {
                for col in 0..n {
                    let value = copy(&a[(j, col)]);
                    a[(j, col)] = copy(&a[(pivot, col)]);
                    a[(pivot, col)] = value;
                }
                for row in 0..n {
                    let value = copy(&a[(row, j)]);
                    a[(row, j)] = copy(&a[(row, pivot)]);
                    a[(row, pivot)] = value;
                }
                for col in 0..j {
                    let value = copy(&l[(j, col)]);
                    l[(j, col)] = copy(&l[(pivot, col)]);
                    l[(pivot, col)] = value;
                }
                diagonal.swap(j, pivot);
                forward.swap(j, pivot);
            }
            let root = sqrt(&d);
            l[(j, j)] = from_real::<T>(&root);
            for i in j + 1..n {
                let mut column = copy(&a[(i, j)]);
                for k in start..j {
                    column = sub(&column, &mul(&l[(i, k)], &conj(&l[(j, k)])));
                }
                l[(i, j)] = mul_real(&column, &recip(&root));
                diagonal[i] = sub(&diagonal[i], &abs2(&l[(i, j)]));
            }
            rank += 1;
        }
        let trailing = n - end;
        faer::linalg::matmul::matmul(
            a.rb_mut().submatrix_mut(end, end, trailing, trailing),
            Accum::Add,
            l.as_ref().submatrix(end, start, trailing, end - start),
            l.as_ref()
                .submatrix(end, start, trailing, end - start)
                .adjoint(),
            -one::<T>(),
            par.reborrow(),
        );
        start = end;
    }
    Ok((l, rank))
}

pub(crate) fn factorize<T: ComplexField>(
    a: MatRef<'_, T>,
    lower: bool,
    work: &mut OwnedScratch,
    permutation: &mut [usize],
    tol: f64,
    block_size: usize,
    par: Par,
) -> Result<(Mat<T>, usize), String> {
    let n = a.nrows();
    if a.ncols() != n || permutation.len() != n {
        return Err("pivoted Cholesky dimensions/permutation length mismatch".into());
    }
    let params = params::<T>(block_size)?;
    let stack = work.stack(requirement::<T>(n, tol, block_size, par.shape())?)?;
    let (mut forward, stack) = stack.make_with::<usize>(n, |i| i);
    let (l, rank) = if tol >= 0.0 {
        let (mut copied, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
        let mut copied = copied.as_mat_mut();
        for j in 0..n {
            for i in 0..n {
                copied[(i, j)] = if i == j {
                    from_real::<T>(&real(&a[(i, i)]))
                } else if (lower && i >= j) || (!lower && i <= j) {
                    copy(&a[(i, j)])
                } else {
                    conj(&a[(j, i)])
                };
            }
        }
        crate::typed::finite(copied.rb())?;
        absolute_factor(copied, &mut forward, tol, block_size, par, stack)?
    } else {
        let mut a = crate::typed::hermitian_copy(a, lower)?;
        let (mut inverse, stack) = stack.make_with::<usize>(n, |_| 0);
        let (info, _) =
            factor::cholesky_in_place(a.as_mut(), &mut forward, &mut inverse, par, stack, params)
                .map_err(|e| format!("pivoted Cholesky: {e:?}"))?;
        for j in 0..n {
            for i in 0..n {
                if i < j || j >= info.rank {
                    a[(i, j)] = zero();
                }
            }
        }
        (a, info.rank)
    };
    crate::typed::finite(l.as_ref())?;
    permutation.copy_from_slice(&forward);
    Ok((l, rank))
}

#[julia]
pub fn pivoted_llt_requirements(
    n: usize,
    tag: i32,
    tol: f64,
    block_size: usize,
    team: Option<&OwnedTeam>,
    out: &mut [usize],
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            if out.len() != 2 {
                return Err("pivoted Cholesky requirements output must have length two".into());
            }
            let req = match tag {
                0 => requirement::<f32>(n, tol, block_size, par.shape())?,
                1 => requirement::<f64>(n, tol, block_size, par.shape())?,
                2 => requirement::<faer::c32>(n, tol, block_size, par.shape())?,
                3 => requirement::<faer::c64>(n, tol, block_size, par.shape())?,
                _ => return Err("invalid pivoted Cholesky scalar".into()),
            };
            out[0] = req.size_bytes();
            out[1] = req.align_bytes();
            Ok(())
        })
    })
}
fn apply_requirement<T: ComplexField>(
    n: usize,
    k: usize,
    op: i32,
    par: ParShape,
) -> Result<StackReq, String> {
    use faer::linalg::cholesky::llt_pivoting::{inverse, reconstruct, solve};
    if n > isize::MAX as usize || k > isize::MAX as usize {
        return Err("pivoted Cholesky dimensions overflow".into());
    }
    n.checked_mul(n.max(k))
        .and_then(|v| v.checked_mul(core::mem::size_of::<T>()))
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("pivoted Cholesky dimensions overflow")?;
    let req = match op {
        0 | 3 => solve::solve_in_place_scratch::<usize, T>(n, k, par),
        1 if k == 0 => inverse::inverse_scratch::<usize, T>(n, par),
        2 if k == 0 => reconstruct::reconstruct_scratch::<usize, T>(n, par),
        _ => return Err("invalid pivoted Cholesky operation/RHS width".into()),
    };
    Ok(StackReq::new::<usize>(n).and(req))
}
#[julia]
pub fn pivoted_llt_apply_requirements(
    n: usize,
    k: usize,
    tag: i32,
    op: i32,
    team: Option<&OwnedTeam>,
    out: &mut [usize],
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            if out.len() != 2 {
                return Err("pivoted Cholesky requirements output must have length two".into());
            }
            let req = match tag {
                0 => apply_requirement::<f32>(n, k, op, par.shape())?,
                1 => apply_requirement::<f64>(n, k, op, par.shape())?,
                2 => apply_requirement::<faer::c32>(n, k, op, par.shape())?,
                3 => apply_requirement::<faer::c64>(n, k, op, par.shape())?,
                _ => return Err("invalid pivoted Cholesky scalar".into()),
            };
            out[0] = req.size_bytes();
            out[1] = req.align_bytes();
            Ok(())
        })
    })
}
fn apply<T: ComplexField>(
    l: MatRef<'_, T>,
    lower: bool,
    indices: &[usize],
    rhs: Option<MatRef<'_, T>>,
    n: usize,
    k: usize,
    op: i32,
    work: &mut OwnedScratch,
    par: Par,
) -> Result<Mat<T>, String> {
    use faer::linalg::cholesky::llt_pivoting::{inverse, reconstruct, solve};
    if l.shape() != (n, n) || indices.len() != n {
        return Err("pivoted Cholesky current native metadata mismatch".into());
    }
    let stack = work.stack(apply_requirement::<T>(n, k, op, par.shape())?)?;
    let (mut backward, stack) = stack.make_with::<usize>(n, |_| usize::MAX);
    for (i, &j) in indices.iter().enumerate() {
        if j >= n || backward[j] != usize::MAX {
            return Err("invalid pivoted Cholesky permutation".into());
        }
        backward[j] = i;
    }
    let perm = faer::perm::PermRef::new_checked(indices, &backward, n);
    let upper_copy;
    let l = if lower {
        l
    } else {
        upper_copy = l.adjoint().to_owned();
        upper_copy.as_ref()
    };
    for j in 0..n {
        for i in j..n {
            if !is_finite(&l[(i, j)])
                || (op != 2
                    && i == j
                    && (imag(&l[(i, j)]) != zero::<T::Real>()
                        || real(&l[(i, j)]) <= zero::<T::Real>()))
            {
                return Err("invalid pivoted Cholesky factor entries".into());
            }
        }
    }
    let mut result = match (op, rhs) {
        (0, Some(rhs)) if rhs.shape() == (n, k) => {
            crate::typed::finite(rhs)?;
            rhs.to_owned()
        }
        (3, Some(rhs)) if rhs.shape() == (k, n) => {
            crate::typed::finite(rhs)?;
            rhs.to_owned()
        }
        (1 | 2, None) => Mat::<T>::zeros(n, n),
        _ => return Err("pivoted Cholesky RHS current native metadata mismatch".into()),
    };
    match op {
        0 => solve::solve_in_place(l, perm, result.as_mut(), par, stack),
        3 => solve::solve_in_place_with_conj(
            l,
            perm,
            faer::Conj::Yes,
            result.as_mut().transpose_mut(),
            par,
            stack,
        ),
        1 => inverse::inverse(result.as_mut(), l, perm, par, stack),
        _ => reconstruct::reconstruct(result.as_mut(), l, perm, par, stack),
    }
    if op == 1 || op == 2 {
        for j in 0..n {
            for i in 0..j {
                result[(i, j)] = conj(&result[(j, i)]);
            }
        }
    }
    crate::typed::finite(result.as_ref())?;
    Ok(result)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn pivoted_llt_workspace(
        &self,
        out: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        permutation: &mut [usize],
        lower: bool,
        n: usize,
        tag: i32,
        tol: f64,
        block_size: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            if tag != 1 || self.matrix.shape() != (n, n) {
                return Err("pivoted Cholesky current native metadata mismatch".into());
            }
            let (l, rank) = with_team(team, |par| {
                factorize(
                    self.matrix.as_ref(),
                    lower,
                    work,
                    permutation,
                    tol,
                    block_size,
                    par,
                )
            })?;
            out.matrix = l;
            Ok(rank)
        })
    }
    #[julia]
    pub fn pivoted_llt_apply_workspace(
        &self,
        out: &mut OwnedMatrix,
        rhs: Option<&OwnedMatrix>,
        work: &mut OwnedScratch,
        permutation: &[usize],
        lower: bool,
        n: usize,
        k: usize,
        tag: i32,
        op: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("pivoted Cholesky scalar metadata mismatch".into());
            }
            let result = with_team(team, |par| {
                apply(
                    self.matrix.as_ref(),
                    lower,
                    permutation,
                    rhs.map(|r| r.matrix.as_ref()),
                    n,
                    k,
                    op,
                    work,
                    par,
                )
            })?;
            out.matrix = result;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn pivoted_llt_workspace(
        &self,
        out: &mut TypedMatrix,
        work: &mut OwnedScratch,
        permutation: &mut [usize],
        lower: bool,
        n: usize,
        tag: i32,
        tol: f64,
        block_size: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let (data, rank) = match self.data()? {
                MatrixData::F32(a) if tag == 0 && a.shape() == (n, n) => {
                    let (l, rank) = with_team(team, |par| {
                        factorize(a.as_ref(), lower, work, permutation, tol, block_size, par)
                    })?;
                    (MatrixData::F32(l), rank)
                }
                MatrixData::F64(a) if tag == 1 && a.shape() == (n, n) => {
                    let (l, rank) = with_team(team, |par| {
                        factorize(a.as_ref(), lower, work, permutation, tol, block_size, par)
                    })?;
                    (MatrixData::F64(l), rank)
                }
                MatrixData::C32(a) if tag == 2 && a.shape() == (n, n) => {
                    let (l, rank) = with_team(team, |par| {
                        factorize(a.as_ref(), lower, work, permutation, tol, block_size, par)
                    })?;
                    (MatrixData::C32(l), rank)
                }
                MatrixData::C64(a) if tag == 3 && a.shape() == (n, n) => {
                    let (l, rank) = with_team(team, |par| {
                        factorize(a.as_ref(), lower, work, permutation, tol, block_size, par)
                    })?;
                    (MatrixData::C64(l), rank)
                }
                _ => return Err("pivoted Cholesky current native metadata mismatch".into()),
            };
            out.data = Some(data);
            Ok(rank)
        })
    }
    #[julia]
    pub fn pivoted_llt_apply_workspace(
        &self,
        out: &mut TypedMatrix,
        rhs: Option<&TypedMatrix>,
        work: &mut OwnedScratch,
        permutation: &[usize],
        lower: bool,
        n: usize,
        k: usize,
        tag: i32,
        op: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = rhs.map(|r| r.data()).transpose()?;
            let result = match self.data()? {
                MatrixData::F32(a) if tag == 0 => {
                    let rhs = match rhs {
                        Some(MatrixData::F32(r)) => Some(r.as_ref()),
                        None => None,
                        _ => return Err("pivoted Cholesky RHS scalar mismatch".into()),
                    };
                    MatrixData::F32(with_team(team, |par| {
                        apply(a.as_ref(), lower, permutation, rhs, n, k, op, work, par)
                    })?)
                }
                MatrixData::F64(a) if tag == 1 => {
                    let rhs = match rhs {
                        Some(MatrixData::F64(r)) => Some(r.as_ref()),
                        None => None,
                        _ => return Err("pivoted Cholesky RHS scalar mismatch".into()),
                    };
                    MatrixData::F64(with_team(team, |par| {
                        apply(a.as_ref(), lower, permutation, rhs, n, k, op, work, par)
                    })?)
                }
                MatrixData::C32(a) if tag == 2 => {
                    let rhs = match rhs {
                        Some(MatrixData::C32(r)) => Some(r.as_ref()),
                        None => None,
                        _ => return Err("pivoted Cholesky RHS scalar mismatch".into()),
                    };
                    MatrixData::C32(with_team(team, |par| {
                        apply(a.as_ref(), lower, permutation, rhs, n, k, op, work, par)
                    })?)
                }
                MatrixData::C64(a) if tag == 3 => {
                    let rhs = match rhs {
                        Some(MatrixData::C64(r)) => Some(r.as_ref()),
                        None => None,
                        _ => return Err("pivoted Cholesky RHS scalar mismatch".into()),
                    };
                    MatrixData::C64(with_team(team, |par| {
                        apply(a.as_ref(), lower, permutation, rhs, n, k, op, work, par)
                    })?)
                }
                _ => return Err("pivoted Cholesky scalar metadata mismatch".into()),
            };
            out.data = Some(result);
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::teams::OwnedTeam;
    use faer::{c32, c64};
    fn check<T: ComplexField>(imaginary: T, tolerance: f64) {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for n in [0, 1, 2, 17, 65] {
            let x = Mat::<T>::from_fn(n, 3, |i, j| {
                add(
                    &from_f64::<T>(((i * 7 + j * 11 + 3) % 19) as f64 / 19.0),
                    &mul(
                        &imaginary,
                        &from_f64::<T>(((i * 3 + j * 5) % 13) as f64 / 13.0),
                    ),
                )
            });
            let a = &x * x.adjoint() + Mat::<T>::identity(n, n);
            for tol in [-1.0, 0.0, 1e-6] {
                for block_size in [1, 2, 7, 128, usize::MAX] {
                    for use_team in [false, true] {
                        let execute = |par: Par<'_>| {
                            let req = requirement::<T>(n, tol, block_size, par.shape()).unwrap();
                            let mut work = OwnedScratch::new();
                            work.scratch_reserve(req.size_bytes(), req.align_bytes())
                                .unwrap();
                            let mut perm = vec![usize::MAX; n];
                            let (l, rank) = factorize(
                                a.as_ref(),
                                true,
                                &mut work,
                                &mut perm,
                                tol,
                                block_size,
                                par,
                            )
                            .unwrap();
                            assert_eq!(rank, n);
                            let permuted =
                                Mat::<T>::from_fn(n, n, |i, j| copy(&a[(perm[i], perm[j])]));
                            assert!(
                                (&l * l.adjoint() - permuted).norm_max()
                                    <= from_f64::<T::Real>(tolerance)
                            );
                            let mut sorted = perm.clone();
                            sorted.sort_unstable();
                            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
                            for j in 0..n {
                                for i in 0..j {
                                    assert!(l[(i, j)] == zero::<T>());
                                }
                            }
                        };
                        if use_team {
                            team.execute(execute).unwrap();
                        } else {
                            execute(Par::Seq);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn pivoted_workspace_four_scalars_and_blocking() {
        check::<f32>(0.0, 2e-4);
        check::<f64>(0.0, 2e-12);
        check::<c32>(c32::new(0.0, 1.0), 2e-4);
        check::<c64>(c64::new(0.0, 1.0), 2e-12);
    }
    #[test]
    fn pivoted_workspace_rank_and_transactional_errors() {
        for tol in [-1.0, 0.0, 1e-6] {
            for block_size in [1, 2, 128, usize::MAX] {
                let a = Mat::<f64>::from_fn(
                    3,
                    3,
                    |i, j| if i == j { [1.0, 1e-8, 4.0][i] } else { 0.0 },
                );
                let req = requirement::<f64>(3, tol, block_size, ParShape::Seq).unwrap();
                let mut work = OwnedScratch::new();
                work.scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                    .unwrap();
                let mut perm = vec![42; 3];
                assert!(factorize(
                    a.as_ref(),
                    true,
                    &mut work,
                    &mut perm,
                    tol,
                    block_size,
                    Par::Seq
                )
                .is_err());
                assert_eq!(perm, [42, 42, 42]);
                work.scratch_reserve(req.size_bytes(), req.align_bytes())
                    .unwrap();
                let (l, rank) = factorize(
                    a.as_ref(),
                    true,
                    &mut work,
                    &mut perm,
                    tol,
                    block_size,
                    Par::Seq,
                )
                .unwrap();
                assert_eq!(rank, if tol == 1e-6 { 2 } else { 3 });
                assert_eq!(perm[0], 2);
                assert!(l.is_all_finite());
                let mut bad = a.clone();
                bad[(2, 0)] = f64::NAN;
                perm.fill(42);
                assert!(factorize(
                    bad.as_ref(),
                    true,
                    &mut work,
                    &mut perm,
                    tol,
                    block_size,
                    Par::Seq
                )
                .is_err());
                assert_eq!(perm, [42, 42, 42]);
            }
        }
        assert!(requirement::<f64>(3, 0.0, 0, ParShape::Seq).is_err());
        assert!(requirement::<f64>(3, f64::NAN, 1, ParShape::Seq).is_err());
        assert!(requirement::<f64>(usize::MAX, 0.0, 1, ParShape::Seq).is_err());
    }
}

#[cfg(test)]
mod apply_tests {
    use super::*;
    fn check<T: ComplexField>(imaginary: T, epsilon: f64) {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let x = Mat::<T>::from_fn(3, 2, |i, j| {
            add(
                &from_f64::<T>((i + j + 1) as f64),
                &mul(&imaginary, &from_f64::<T>((i + 2 * j) as f64 / 7.0)),
            )
        });
        let a = &x * x.adjoint() + Mat::<T>::identity(3, 3);
        for lower in [true, false] {
            let mut constructor = OwnedScratch::new();
            let req = requirement::<T>(3, 0.0, 2, ParShape::Seq).unwrap();
            constructor
                .scratch_reserve(req.size_bytes(), req.align_bytes())
                .unwrap();
            let mut p = vec![usize::MAX; 3];
            let (mut l, rank) =
                factorize(a.as_ref(), true, &mut constructor, &mut p, 0.0, 2, Par::Seq).unwrap();
            assert_eq!(rank, 3);
            l[(0, 2)] = from_f64(f64::NAN);
            if !lower {
                l = l.adjoint().to_owned();
            }
            for op in 0..=3 {
                for use_team in [false, true] {
                    let (rows, cols) = if op == 3 { (2, 3) } else { (3, 2) };
                    let rhs = Mat::<T>::from_fn(rows, cols, |i, j| {
                        add(
                            &from_f64::<T>((i + j + 1) as f64),
                            &mul(&imaginary, &from_f64::<T>((i + 3 * j) as f64 / 5.0)),
                        )
                    });
                    let k = if op == 0 || op == 3 { 2 } else { 0 };
                    let execute = |par: Par<'_>| {
                        let req = apply_requirement::<T>(3, k, op, par.shape()).unwrap();
                        let mut work = OwnedScratch::new();
                        work.scratch_reserve(req.size_bytes(), req.align_bytes())
                            .unwrap();
                        let result = apply(
                            l.as_ref(),
                            lower,
                            &p,
                            if op == 0 || op == 3 {
                                Some(rhs.as_ref())
                            } else {
                                None
                            },
                            3,
                            k,
                            op,
                            &mut work,
                            par,
                        )
                        .unwrap();
                        let error = match op {
                            0 => (&a * &result - &rhs).norm_max(),
                            1 => (&a * &result - Mat::<T>::identity(3, 3)).norm_max(),
                            2 => (&result - &a).norm_max(),
                            _ => (&result * &a - &rhs).norm_max(),
                        };
                        assert!(error <= from_f64::<T::Real>(epsilon));
                    };
                    if use_team {
                        team.execute(execute).unwrap();
                    } else {
                        execute(Par::Seq);
                    }
                }
            }
        }
    }
    #[test]
    fn pivoted_workspace_saved_operations_four_scalars() {
        check::<f32>(0.0, 2e-4);
        check::<f64>(0.0, 2e-12);
        check::<faer::c32>(faer::c32::new(0.0, 1.0), 2e-4);
        check::<faer::c64>(faer::c64::new(0.0, 1.0), 2e-12);
    }
}
