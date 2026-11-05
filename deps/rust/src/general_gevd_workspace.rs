//! Generalized EVD workspace, including real conjugate-pair conversion.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::gevd::{self, ComputeEigenvectors as Vectors};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::ComplexField;
use faer::{MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
trait General: ComplexField {
    fn requirement(n: usize, values: bool, both: bool, par: ParShape) -> StackReq;
    unsafe fn compute(
        a: MatRef<'_, Self>,
        b: MatRef<'_, Self>,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
        par: Par<'_>,
    ) -> Result<(), String>;
}
macro_rules! real_scalar {
    ($r:ty,$c:ty) => {
        impl General for $r {
            fn requirement(n: usize, values: bool, both: bool, par: ParShape) -> StackReq {
                StackReq::all_of(&[
                    faer::linalg::temp_mat_scratch::<Self>(n, n),
                    faer::linalg::temp_mat_scratch::<Self>(n, n),
                    faer::linalg::temp_mat_scratch::<Self>(n, 2),
                    faer::linalg::temp_mat_scratch::<Self>(n, 1),
                    faer::linalg::temp_mat_scratch::<Self>(n, if values { 0 } else { n }),
                    faer::linalg::temp_mat_scratch::<Self>(n, if both && !values { n } else { 0 }),
                    StackReq::any_of(&[
                        crate::qz::real_qz_scratch::<Self>(n, faer::auto!(Self), par),
                        gevd::gevd_scratch::<Self>(
                            n,
                            Vectors::No,
                            if values { Vectors::No } else { Vectors::Yes },
                            par,
                            Default::default(),
                        ),
                    ]),
                ])
            }
            unsafe fn compute(
                a: MatRef<'_, Self>,
                b: MatRef<'_, Self>,
                scratch: &mut OwnedScratch,
                u: *mut u8,
                v: *mut u8,
                s: *mut u8,
                t: *mut u8,
                values: bool,
                left: bool,
                both: bool,
                par: Par<'_>,
            ) -> Result<(), String> {
                let n = a.nrows();
                let stack = scratch.stack(Self::requirement(n, values, both, par.shape()))?;
                let (mut vals, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, 2, stack);
                let (mut vecs, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(
                    n,
                    if values { 0 } else { n },
                    stack,
                );
                let (mut val_beta, stack) =
                    faer::linalg::temp_mat_zeroed::<Self, _, _>(n, 1, stack);
                let (re, im) = vals.as_mat_mut().two_cols_mut(0, 1);
                let (mut ac, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, n, stack);
                let (mut bc, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, n, stack);
                ac.as_mat_mut().copy_from(a);
                bc.as_mat_mut().copy_from(b);
                let (mut other, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(
                    n,
                    if both && !values { n } else { 0 },
                    stack,
                );
                let (ul, ur) = if values {
                    (None, None)
                } else if both {
                    (Some(vecs.as_mat_mut()), Some(other.as_mat_mut()))
                } else if left {
                    (Some(vecs.as_mat_mut()), None)
                } else {
                    (None, Some(vecs.as_mat_mut()))
                };
                gevd::gevd_real(
                    ac.as_mat_mut(),
                    bc.as_mat_mut(),
                    re.as_diagonal_mut(),
                    im.as_diagonal_mut(),
                    val_beta.as_mat_mut().col_mut(0).as_diagonal_mut(),
                    ul,
                    ur,
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("generalized EVD: {e:?}"))?;
                for j in 0..n {
                    s.cast::<$c>().add(j).write(<$c>::new(
                        vals.as_mat_ref()[(j, 0)],
                        vals.as_mat_ref()[(j, 1)],
                    ));
                    t.cast::<Self>().add(j).write(val_beta.as_mat_ref()[(j, 0)]);
                }
                if !values {
                    for (index, (data, out, conjugate)) in [
                        (vecs.as_mat_ref(), u, left || both),
                        (other.as_mat_ref(), v, false),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if index == 1 && !both {
                            continue;
                        }
                        let mut j = 0;
                        while j < n {
                            if vals.as_mat_ref()[(j, 1)] == 0.0 {
                                for i in 0..n {
                                    out.cast::<$c>()
                                        .add(i + j * n)
                                        .write(<$c>::new(data[(i, j)], 0.0));
                                }
                                j += 1;
                            } else {
                                if j + 1 >= n {
                                    return Err("invalid real EVD conjugate-pair layout".into());
                                }
                                for i in 0..n {
                                    let re = data[(i, j)];
                                    let im = if conjugate {
                                        -data[(i, j + 1)]
                                    } else {
                                        data[(i, j + 1)]
                                    };
                                    out.cast::<$c>().add(i + j * n).write(<$c>::new(re, im));
                                    out.cast::<$c>()
                                        .add(i + (j + 1) * n)
                                        .write(<$c>::new(re, -im));
                                }
                                j += 2;
                            }
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
            fn requirement(n: usize, values: bool, both: bool, par: ParShape) -> StackReq {
                StackReq::all_of(&[
                    faer::linalg::temp_mat_scratch::<Self>(n, n),
                    faer::linalg::temp_mat_scratch::<Self>(n, n),
                    faer::linalg::temp_mat_scratch::<Self>(n, 1),
                    gevd::gevd_scratch::<Self>(
                        n,
                        if both && !values {
                            Vectors::Yes
                        } else {
                            Vectors::No
                        },
                        if values { Vectors::No } else { Vectors::Yes },
                        par,
                        Default::default(),
                    ),
                ])
            }
            unsafe fn compute(
                a: MatRef<'_, Self>,
                b: MatRef<'_, Self>,
                scratch: &mut OwnedScratch,
                u: *mut u8,
                v: *mut u8,
                s: *mut u8,
                t: *mut u8,
                values: bool,
                left: bool,
                both: bool,
                par: Par<'_>,
            ) -> Result<(), String> {
                let n = a.nrows();
                let stack = scratch.stack(Self::requirement(n, values, both, par.shape()))?;
                let (mut ac, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, n, stack);
                let (mut bc, stack) = faer::linalg::temp_mat_zeroed::<Self, _, _>(n, n, stack);
                ac.as_mat_mut().copy_from(a);
                bc.as_mat_mut().copy_from(b);
                let (mut val_beta, stack) =
                    faer::linalg::temp_mat_zeroed::<Self, _, _>(n, 1, stack);
                let output = if values {
                    None
                } else {
                    Some(MatMut::from_raw_parts_mut(
                        u.cast::<Self>(),
                        n,
                        n,
                        1,
                        n as isize,
                    ))
                };
                let second = if both && !values {
                    Some(MatMut::from_raw_parts_mut(
                        v.cast::<Self>(),
                        n,
                        n,
                        1,
                        n as isize,
                    ))
                } else {
                    None
                };
                let (ul, ur) = if both {
                    (output, second)
                } else if left {
                    (output, None)
                } else {
                    (None, output)
                };
                gevd::gevd_cplx(
                    ac.as_mat_mut(),
                    bc.as_mat_mut(),
                    faer::diag::DiagMut::from_raw_parts_mut(s.cast::<Self>(), n, 1),
                    val_beta.as_mat_mut().col_mut(0).as_diagonal_mut(),
                    ul,
                    ur,
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("generalized EVD: {e:?}"))?;
                for j in 0..n {
                    t.cast::<Self>().add(j).write(val_beta.as_mat_ref()[(j, 0)]);
                }
                Ok(())
            }
        }
    };
}
real_scalar!(f32, faer::c32);
real_scalar!(f64, faer::c64);
complex_scalar!(faer::c32);
complex_scalar!(faer::c64);
fn dispatch(
    n: usize,
    tag: i32,
    values: bool,
    both: bool,
    par: ParShape,
) -> Result<StackReq, String> {
    Ok(match tag {
        0 => f32::requirement(n, values, both, par),
        1 => f64::requirement(n, values, both, par),
        2 => faer::c32::requirement(n, values, both, par),
        3 => faer::c64::requirement(n, values, both, par),
        _ => return Err("invalid generalized EVD scalar".into()),
    })
}
fn write_req(req: StackReq, out: *mut usize) {
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
#[julia]
pub fn general_gevd_requirements(
    n: usize,
    tag: i32,
    values: bool,
    both: bool,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| {
        write_req(dispatch(n, tag, values, both, ParShape::Seq)?, out);
        Ok(())
    })
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn general_gevd_team_requirements(
        &self,
        n: usize,
        tag: i32,
        values: bool,
        both: bool,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            write_req(
                self.execute(|par| dispatch(n, tag, values, both, par.shape()))??,
                out,
            );
            Ok(())
        })
    }
}
unsafe fn run<T: General>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
    u: *mut u8,
    v: *mut u8,
    s: *mut u8,
    t: *mut u8,
    values: bool,
    left: bool,
    both: bool,
    n: usize,
    tag: i32,
    actual: i32,
) -> Result<(), String> {
    if a.shape() != (n, n) || b.shape() != (n, n) || tag != actual {
        return Err("generalized EVD owner metadata mismatch".into());
    }
    if !a.is_all_finite() || !b.is_all_finite() {
        return Err("generalized EVD input must be finite".into());
    }
    let mut compute = |par: Par<'_>| T::compute(a, b, scratch, u, v, s, t, values, left, both, par);
    if let Some(team) = team {
        team.execute(compute)?
    } else {
        compute(Par::Seq)
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn general_gevd_workspace(
        &self,
        b: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            run(
                self.matrix.as_ref(),
                b.matrix.as_ref(),
                scratch,
                None,
                u,
                v,
                s,
                t,
                values,
                left,
                both,
                n,
                tag,
                1,
            )
        })
    }
    #[julia]
    pub fn general_gevd_workspace_team(
        &self,
        b: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            run(
                self.matrix.as_ref(),
                b.matrix.as_ref(),
                scratch,
                Some(team),
                u,
                v,
                s,
                t,
                values,
                left,
                both,
                n,
                tag,
                1,
            )
        })
    }
}
impl TypedMatrix {
    unsafe fn workspace_generalized(
        &self,
        b: &TypedMatrix,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
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
                v,
                s,
                t,
                values,
                left,
                both,
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
                v,
                s,
                t,
                values,
                left,
                both,
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
                v,
                s,
                t,
                values,
                left,
                both,
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
                v,
                s,
                t,
                values,
                left,
                both,
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
    pub fn general_gevd_workspace(
        &self,
        b: &TypedMatrix,
        scratch: &mut OwnedScratch,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.workspace_generalized(b, scratch, None, u, v, s, t, values, left, both, n, tag)
        })
    }
    #[julia]
    pub fn general_gevd_workspace_team(
        &self,
        b: &TypedMatrix,
        scratch: &mut OwnedScratch,
        team: &OwnedTeam,
        u: *mut u8,
        v: *mut u8,
        s: *mut u8,
        t: *mut u8,
        values: bool,
        left: bool,
        both: bool,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.workspace_generalized(
                b,
                scratch,
                Some(team),
                u,
                v,
                s,
                t,
                values,
                left,
                both,
                n,
                tag,
            )
        })
    }
}
