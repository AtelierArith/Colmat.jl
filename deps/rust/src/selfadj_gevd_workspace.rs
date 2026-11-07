//! Definite self-adjoint generalized EVD using LLT reduction and reusable scratch.
use crate::teams::with_team;
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::evd::{self, ComputeEigenvectors as Vectors};
use faer::linalg::{cholesky::llt::factor as llt, triangular_solve as triangular};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::Conj;
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
fn req<T: ComplexField>(n: usize, values: bool, par: ParShape) -> StackReq {
    StackReq::all_of(&[
        faer::linalg::temp_mat_scratch::<T>(n, n),
        faer::linalg::temp_mat_scratch::<T>(n, n),
        faer::linalg::temp_mat_scratch::<T>(n, 1),
        StackReq::any_of(&[
            llt::cholesky_in_place_scratch::<T>(n, par, Default::default()),
            evd::self_adjoint_evd_scratch::<T>(
                n,
                if values { Vectors::No } else { Vectors::Yes },
                par,
                Default::default(),
            ),
        ]),
    ])
}
fn dispatch(n: usize, tag: i32, values: bool, par: ParShape) -> Result<StackReq, String> {
    Ok(match tag {
        0 => req::<f32>(n, values, par),
        1 => req::<f64>(n, values, par),
        2 => req::<faer::c32>(n, values, par),
        3 => req::<faer::c64>(n, values, par),
        _ => return Err("invalid EVD scalar".into()),
    })
}
fn write_req(req: StackReq, out: *mut usize) {
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
#[julia]
pub fn selfadj_gevd_requirements(
    n: usize,
    tag: i32,
    values: bool,
    out: *mut usize,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| {
        let req = with_team(team, |par| dispatch(n, tag, values, par.shape()))?;
        write_req(req, out);
        Ok(())
    })
}
unsafe fn run<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
    u: *mut u8,
    s: *mut u8,
    values: bool,
    lower: bool,
    other_lower: bool,
    n: usize,
    tag: i32,
    actual: i32,
) -> Result<(), String> {
    if a.shape() != (n, n) || b.shape() != (n, n) || tag != actual {
        return Err("EVD owner metadata mismatch".into());
    }
    let mut compute = |mut par: Par<'_>| -> Result<(), String> {
        let stack = scratch.stack(req::<T>(n, values, par.shape()))?;
        let (mut canonical, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
        for j in 0..n {
            for i in j..n {
                canonical.as_mat_mut()[(i, j)] = if i == j {
                    from_real::<T>(&real(&a[(i, i)]))
                } else if lower {
                    copy(&a[(i, j)])
                } else {
                    conj(&a[(j, i)])
                };
            }
        }
        if !canonical.as_mat_ref().is_all_finite() {
            return Err("EVD selected triangle must be finite".into());
        }
        for j in 0..n {
            for i in 0..j {
                canonical.as_mat_mut()[(i, j)] = conj(&canonical.as_mat_ref()[(j, i)]);
            }
        }
        let (mut l, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
        for j in 0..n {
            for i in j..n {
                l.as_mat_mut()[(i, j)] = if i == j {
                    from_real::<T>(&real(&b[(i, i)]))
                } else if other_lower {
                    copy(&b[(i, j)])
                } else {
                    conj(&b[(j, i)])
                };
            }
        }
        if !l.as_mat_ref().is_all_finite() {
            return Err("generalized EVD selected B triangle must be finite".into());
        }
        let (mut eigenvalues, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
        llt::cholesky_in_place(
            l.as_mat_mut(),
            Default::default(),
            par.reborrow(),
            &mut *stack,
            Default::default(),
        )
        .map_err(|e| match e {
            // Stable token parsed by Julia into PosDefException(pivot).
            llt::LltError::NonPositivePivot { index } => {
                format!(
                    "generalized EVD B is not positive definite at pivot {}",
                    index + 1
                )
            }
        })?;
        triangular::solve_lower_triangular_in_place(
            l.as_mat_ref(),
            canonical.as_mat_mut(),
            par.reborrow(),
        );
        triangular::solve_lower_triangular_in_place_with_conj(
            l.as_mat_ref(),
            Conj::Yes,
            canonical.as_mat_mut().transpose_mut(),
            par.reborrow(),
        );
        if !canonical.as_mat_ref().is_all_finite() {
            return Err("generalized EVD reduction produced nonfinite entries".into());
        }
        let u_ptr = u;
        let u = if values {
            None
        } else {
            Some(MatMut::from_raw_parts_mut(
                u.cast::<T>(),
                n,
                n,
                1,
                n as isize,
            ))
        };
        evd::self_adjoint_evd(
            canonical.as_mat_ref(),
            eigenvalues.as_mat_mut().col_mut(0).as_diagonal_mut(),
            u,
            par.reborrow(),
            &mut *stack,
            Default::default(),
        )
        .map_err(|e| format!("EVD: {e:?}"))?;
        if !values {
            let output = MatMut::from_raw_parts_mut(u_ptr.cast::<T>(), n, n, 1, n as isize);
            triangular::solve_upper_triangular_in_place(l.as_mat_ref().adjoint(), output, par);
        }
        for i in 0..n {
            s.cast::<T::Real>()
                .add(i)
                .write(real(&eigenvalues.as_mat_ref()[(i, 0)]));
        }
        Ok(())
    };
    if let Some(team) = team {
        team.execute(compute)?
    } else {
        compute(Par::Seq)
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn selfadj_gevd_workspace(
        &self,
        b: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        other_lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            run(
                self.matrix.as_ref(),
                b.matrix.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                other_lower,
                n,
                tag,
                1,
            )
        })
    }
}
impl TypedMatrix {
    unsafe fn workspace_selfadj_gevd(
        &self,
        b: &TypedMatrix,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        other_lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        match (self.data()?, b.data()?) {
            (MatrixData::F32(a), MatrixData::F32(b)) => run(
                a.as_ref(),
                b.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                other_lower,
                n,
                tag,
                0,
            ),
            (MatrixData::F64(a), MatrixData::F64(b)) => run(
                a.as_ref(),
                b.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                other_lower,
                n,
                tag,
                1,
            ),
            (MatrixData::C32(a), MatrixData::C32(b)) => run(
                a.as_ref(),
                b.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                other_lower,
                n,
                tag,
                2,
            ),
            (MatrixData::C64(a), MatrixData::C64(b)) => run(
                a.as_ref(),
                b.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                other_lower,
                n,
                tag,
                3,
            ),
            _ => Err("generalized EVD scalar mismatch".into()),
        }
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn selfadj_gevd_workspace(
        &self,
        b: &TypedMatrix,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        other_lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.workspace_selfadj_gevd(b, scratch, team, u, s, values, lower, other_lower, n, tag)
        })
    }
}
