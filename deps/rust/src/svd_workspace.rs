//! SVD with caller-owned workspace and ephemeral team authority.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::svd::{self, ComputeSvdVectors as Vectors};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::real, ComplexField};
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
fn mode(full: bool, values: bool) -> Vectors {
    if values {
        Vectors::No
    } else if full {
        Vectors::Full
    } else {
        Vectors::Thin
    }
}
fn req<T: ComplexField>(m: usize, n: usize, full: bool, values: bool, par: ParShape) -> StackReq {
    StackReq::all_of(&[
        faer::linalg::temp_mat_scratch::<T>(m.min(n), 1),
        svd::svd_scratch::<T>(
            m,
            n,
            mode(full, values),
            mode(full, values),
            par,
            crate::svd_tuning::svd_params::<T>(),
        ),
    ])
}
fn dispatch_req(
    m: usize,
    n: usize,
    tag: i32,
    full: bool,
    values: bool,
    par: ParShape,
) -> Result<StackReq, String> {
    Ok(match tag {
        0 => req::<f32>(m, n, full, values, par),
        1 => req::<f64>(m, n, full, values, par),
        2 => req::<faer::c32>(m, n, full, values, par),
        3 => req::<faer::c64>(m, n, full, values, par),
        _ => return Err("invalid SVD scalar".into()),
    })
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn svd_team_requirements(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        full: bool,
        values: bool,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            let req = self.execute(|par| dispatch_req(m, n, tag, full, values, par.shape()))??;
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
}
#[julia]
pub fn svd_requirements(
    m: usize,
    n: usize,
    tag: i32,
    full: bool,
    values: bool,
    width: usize,
    out: &mut [usize],
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| {
        let par = if width <= 1 {
            ParShape::Seq
        } else {
            ParShape::team(width)
        };
        let req = dispatch_req(m, n, tag, full, values, par)?;
        unsafe {
            out.write(req.size_bytes());
            out.add(1).write(req.align_bytes());
        }
        Ok(())
    })
}
unsafe fn run<T: ComplexField>(
    a: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
    u: *mut u8,
    s: *mut u8,
    v: *mut u8,
    full: bool,
    values: bool,
    m: usize,
    n: usize,
    tag: i32,
    actual: i32,
) -> Result<(), String> {
    if a.shape() != (m, n) || tag != actual {
        return Err("SVD owner metadata mismatch".into());
    }
    if !a.is_all_finite() {
        return Err("SVD input must be finite".into());
    }
    let k = m.min(n);
    let mut compute = |par: Par<'_>| -> Result<(), String> {
        let stack = scratch.stack(req::<T>(m, n, full, values, par.shape()))?;
        let (mut singular, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(k, 1, stack);
        let u = if values {
            None
        } else {
            Some(MatMut::from_raw_parts_mut(
                u.cast::<T>(),
                m,
                if full { m } else { k },
                1,
                m as isize,
            ))
        };
        let v = if values {
            None
        } else {
            Some(MatMut::from_raw_parts_mut(
                v.cast::<T>(),
                n,
                if full { n } else { k },
                1,
                n as isize,
            ))
        };
        svd::svd(
            a,
            singular.as_mat_mut().col_mut(0).as_diagonal_mut(),
            u,
            v,
            par,
            stack,
            crate::svd_tuning::svd_params::<T>(),
        )
        .map_err(|e| format!("SVD: {e:?}"))?;
        for i in 0..k {
            s.cast::<T::Real>()
                .add(i)
                .write(real(&singular.as_mat_ref()[(i, 0)]));
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
    pub fn svd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        v: *mut u8,
        full: bool,
        values: bool,
        m: usize,
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
                v,
                full,
                values,
                m,
                n,
                tag,
                1,
            )
        })
    }
}
impl TypedMatrix {
    unsafe fn workspace_svd(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        v: *mut u8,
        full: bool,
        values: bool,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        match self.data()? {
            MatrixData::F32(a) => run(
                a.as_ref(),
                scratch,
                team,
                u,
                s,
                v,
                full,
                values,
                m,
                n,
                tag,
                0,
            ),
            MatrixData::F64(a) => run(
                a.as_ref(),
                scratch,
                team,
                u,
                s,
                v,
                full,
                values,
                m,
                n,
                tag,
                1,
            ),
            MatrixData::C32(a) => run(
                a.as_ref(),
                scratch,
                team,
                u,
                s,
                v,
                full,
                values,
                m,
                n,
                tag,
                2,
            ),
            MatrixData::C64(a) => run(
                a.as_ref(),
                scratch,
                team,
                u,
                s,
                v,
                full,
                values,
                m,
                n,
                tag,
                3,
            ),
        }
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn svd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        v: *mut u8,
        full: bool,
        values: bool,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe { self.workspace_svd(scratch, team, u, s, v, full, values, m, n, tag) })
    }
}
