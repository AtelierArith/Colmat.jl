//! Self-adjoint EVD with explicit scratch and ephemeral team authority.
use crate::teams::with_team;
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::evd::{self, ComputeEigenvectors as Vectors};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
fn req<T: ComplexField>(n: usize, values: bool, par: ParShape) -> StackReq {
    StackReq::all_of(&[
        faer::linalg::temp_mat_scratch::<T>(n, n),
        faer::linalg::temp_mat_scratch::<T>(n, 1),
        evd::self_adjoint_evd_scratch::<T>(
            n,
            if values { Vectors::No } else { Vectors::Yes },
            par,
            crate::evd_tuning::self_adjoint_params::<T>(),
        ),
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
pub fn evd_requirements(
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
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
    u: *mut u8,
    s: *mut u8,
    values: bool,
    lower: bool,
    n: usize,
    tag: i32,
    actual: i32,
) -> Result<(), String> {
    if a.shape() != (n, n) || tag != actual {
        return Err("EVD owner metadata mismatch".into());
    }
    let mut compute = |par: Par<'_>| -> Result<(), String> {
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
        let (mut eigenvalues, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
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
            par,
            stack,
            crate::evd_tuning::self_adjoint_params::<T>(),
        )
        .map_err(|e| format!("EVD: {e:?}"))?;
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
    pub fn evd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            run(
                self.matrix.as_ref(),
                scratch,
                team,
                u,
                s,
                values,
                lower,
                n,
                tag,
                1,
            )
        })
    }
}
impl TypedMatrix {
    unsafe fn workspace_evd(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        match self.data()? {
            MatrixData::F32(a) => run(a.as_ref(), scratch, team, u, s, values, lower, n, tag, 0),
            MatrixData::F64(a) => run(a.as_ref(), scratch, team, u, s, values, lower, n, tag, 1),
            MatrixData::C32(a) => run(a.as_ref(), scratch, team, u, s, values, lower, n, tag, 2),
            MatrixData::C64(a) => run(a.as_ref(), scratch, team, u, s, values, lower, n, tag, 3),
        }
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn evd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        lower: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe { self.workspace_evd(scratch, team, u, s, values, lower, n, tag) })
    }
}
