//! General EVD workspace, including real conjugate-pair conversion.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::evd::{self, ComputeEigenvectors as Vectors};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::ComplexField;
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
trait General: ComplexField {
    fn requirement(n: usize, values: bool, par: ParShape) -> StackReq;
    unsafe fn compute(
        a: MatRef<'_, Self>,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        par: Par<'_>,
    ) -> Result<(), String>;
}
macro_rules! real_scalar {
    ($r:ty,$c:ty) => {
        impl General for $r {
            fn requirement(n: usize, values: bool, par: ParShape) -> StackReq {
                StackReq::all_of(&[
                    faer::linalg::temp_mat_scratch::<Self>(n, 2),
                    faer::linalg::temp_mat_scratch::<Self>(n, if values { 0 } else { n }),
                    evd::evd_scratch::<Self>(
                        n,
                        Vectors::No,
                        if values { Vectors::No } else { Vectors::Yes },
                        par,
                        Default::default(),
                    ),
                ])
            }
            unsafe fn compute(
                a: MatRef<'_, Self>,
                scratch: &mut OwnedScratch,
                u: *mut u8,
                s: *mut u8,
                values: bool,
                par: Par<'_>,
            ) -> Result<(), String> {
                let n = a.nrows();
                let stack = scratch.stack(Self::requirement(n, values, par.shape()))?;
                let (mut vals, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, 2, stack);
                let (mut vecs, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(
                    n,
                    if values { 0 } else { n },
                    stack,
                );
                let (re, im) = vals.as_mat_mut().two_cols_mut(0, 1);
                evd::evd_real(
                    a,
                    re.as_diagonal_mut(),
                    im.as_diagonal_mut(),
                    None,
                    if values {
                        None
                    } else {
                        Some(vecs.as_mat_mut())
                    },
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("general EVD: {e:?}"))?;
                for j in 0..n {
                    s.cast::<$c>().add(j).write(<$c>::new(
                        vals.as_mat_ref()[(j, 0)],
                        vals.as_mat_ref()[(j, 1)],
                    ));
                }
                if !values {
                    let mut j = 0;
                    while j < n {
                        if vals.as_mat_ref()[(j, 1)] == 0.0 {
                            for i in 0..n {
                                u.cast::<$c>()
                                    .add(i + j * n)
                                    .write(<$c>::new(vecs.as_mat_ref()[(i, j)], 0.0));
                            }
                            j += 1;
                        } else {
                            if j + 1 >= n {
                                return Err("invalid real EVD conjugate-pair layout".into());
                            }
                            for i in 0..n {
                                let re = vecs.as_mat_ref()[(i, j)];
                                let im = vecs.as_mat_ref()[(i, j + 1)];
                                u.cast::<$c>().add(i + j * n).write(<$c>::new(re, im));
                                u.cast::<$c>()
                                    .add(i + (j + 1) * n)
                                    .write(<$c>::new(re, -im));
                            }
                            j += 2;
                        }
                    }
                }
                Ok(())
            }
        }
    };
}
macro_rules! complex_scalar {
    ($c:ty) => {
        impl General for $c {
            fn requirement(n: usize, values: bool, par: ParShape) -> StackReq {
                evd::evd_scratch::<Self>(
                    n,
                    Vectors::No,
                    if values { Vectors::No } else { Vectors::Yes },
                    par,
                    Default::default(),
                )
            }
            unsafe fn compute(
                a: MatRef<'_, Self>,
                scratch: &mut OwnedScratch,
                u: *mut u8,
                s: *mut u8,
                values: bool,
                par: Par<'_>,
            ) -> Result<(), String> {
                let n = a.nrows();
                let stack = scratch.stack(Self::requirement(n, values, par.shape()))?;
                evd::evd_cplx(
                    a,
                    faer::diag::DiagMut::from_raw_parts_mut(s.cast::<Self>(), n, 1),
                    None,
                    if values {
                        None
                    } else {
                        Some(MatMut::from_raw_parts_mut(
                            u.cast::<Self>(),
                            n,
                            n,
                            1,
                            n as isize,
                        ))
                    },
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("general EVD: {e:?}"))
            }
        }
    };
}
real_scalar!(f32, faer::c32);
real_scalar!(f64, faer::c64);
complex_scalar!(faer::c32);
complex_scalar!(faer::c64);
fn dispatch(n: usize, tag: i32, values: bool, par: ParShape) -> Result<StackReq, String> {
    Ok(match tag {
        0 => f32::requirement(n, values, par),
        1 => f64::requirement(n, values, par),
        2 => faer::c32::requirement(n, values, par),
        3 => faer::c64::requirement(n, values, par),
        _ => return Err("invalid general EVD scalar".into()),
    })
}
fn write_req(req: StackReq, out: *mut usize) {
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
#[julia]
pub fn general_evd_requirements(
    n: usize,
    tag: i32,
    values: bool,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| {
        write_req(dispatch(n, tag, values, ParShape::Seq)?, out);
        Ok(())
    })
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn general_evd_team_requirements(
        &self,
        n: usize,
        tag: i32,
        values: bool,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            write_req(
                self.execute(|par| dispatch(n, tag, values, par.shape()))??,
                out,
            );
            Ok(())
        })
    }
}
unsafe fn run<T: General>(
    a: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
    u: *mut u8,
    s: *mut u8,
    values: bool,
    n: usize,
    tag: i32,
    actual: i32,
) -> Result<(), String> {
    if a.shape() != (n, n) || tag != actual {
        return Err("general EVD owner metadata mismatch".into());
    }
    if !a.is_all_finite() {
        return Err("general EVD input must be finite".into());
    }
    let mut compute = |par: Par<'_>| T::compute(a, scratch, u, s, values, par);
    if let Some(team) = team {
        team.execute(compute)?
    } else {
        compute(Par::Seq)
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn general_evd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe { run(self.matrix.as_ref(), scratch, None, u, s, values, n, tag, 1) })
    }
    #[julia]
    pub fn general_evd_workspace_team(
        &self,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            run(
                self.matrix.as_ref(),
                scratch,
                Some(team),
                u,
                s,
                values,
                n,
                tag,
                1,
            )
        })
    }
}
impl TypedMatrix {
    unsafe fn workspace_general(
        &self,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        match self.data()? {
            MatrixData::F32(a) => run(a.as_ref(), scratch, team, u, s, values, n, tag, 0),
            MatrixData::F64(a) => run(a.as_ref(), scratch, team, u, s, values, n, tag, 1),
            MatrixData::C32(a) => run(a.as_ref(), scratch, team, u, s, values, n, tag, 2),
            MatrixData::C64(a) => run(a.as_ref(), scratch, team, u, s, values, n, tag, 3),
        }
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn general_evd_workspace(
        &self,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe { self.workspace_general(scratch, None, u, s, values, n, tag) })
    }
    #[julia]
    pub fn general_evd_workspace_team(
        &self,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
        u: *mut u8,
        s: *mut u8,
        values: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe { self.workspace_general(scratch, Some(team), u, s, values, n, tag) })
    }
}
