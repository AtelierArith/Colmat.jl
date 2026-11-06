//! One policy for Hessenberg workspace queries, reduction, and explicit Q.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::linalg::evd::hessenberg::HessenbergParams;
use faer::traits::ComplexField;
use faer::utils::cost::ParThreshold;
use faer::{Auto, Mat, MatRef, Par, ParShape, Spec};
use rustcall_julia_macros::julia;

#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub parallel: usize,
    pub blocking: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            parallel: 0,
            blocking: 256 * 256,
        }
    }
}
impl Policy {
    pub fn params<T: ComplexField>(self) -> Spec<HessenbergParams, T> {
        HessenbergParams {
            par_threshold: ParThreshold::from(self.parallel),
            blocking_threshold: self.blocking,
            ..<HessenbergParams as Auto<T>>::auto()
        }
        .into()
    }
    pub fn shape(self, par: ParShape) -> ParShape {
        if self.parallel == usize::MAX {
            ParShape::Seq
        } else {
            par
        }
    }
    pub fn execution(self, par: Par<'_>) -> Par<'_> {
        if self.parallel == usize::MAX {
            Par::Seq
        } else {
            par
        }
    }
}

fn checked_requirement<T: ComplexField>(
    n: usize,
    par: ParShape,
    policy: Policy,
) -> Result<faer::dyn_stack::StackReq, String> {
    n.checked_mul(n)
        .and_then(|nn| nn.checked_mul(core::mem::size_of::<T>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("Hessenberg dimensions overflow")?;
    Ok(super::construction::hessenberg_requirement_tuned::<T>(
        n, par, policy,
    ))
}
fn query(
    n: usize,
    tag: i32,
    policy: Policy,
    par: ParShape,
    out: &mut [usize],
) -> Result<(), String> {
    if out.len() != 2 {
        return Err("Hessenberg requirements output must have length two".into());
    }
    let req = match tag {
        0 => checked_requirement::<f32>(n, par, policy)?,
        1 => checked_requirement::<f64>(n, par, policy)?,
        2 => checked_requirement::<faer::c32>(n, par, policy)?,
        3 => checked_requirement::<faer::c64>(n, par, policy)?,
        _ => return Err("unsupported Hessenberg scalar".into()),
    };
    out[0] = req.size_bytes();
    out[1] = req.align_bytes();
    Ok(())
}
pub(crate) use crate::teams::with_team;
#[julia]
pub fn hessenberg_tuned_requirements(
    n: usize,
    tag: i32,
    parallel: usize,
    blocking: usize,
    team: Option<&OwnedTeam>,
    out: &mut [usize],
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| {
            query(n, tag, Policy { parallel, blocking }, par.shape(), out)
        })
    })
}
fn compute<T: ComplexField>(
    a: MatRef<'_, T>,
    work: &mut OwnedScratch,
    n: usize,
    policy: Policy,
    par: Par<'_>,
) -> Result<(Mat<T>, Mat<T>), String> {
    if a.shape() != (n, n) {
        return Err("Hessenberg current native dimensions do not match requested metadata".into());
    }
    let req = checked_requirement::<T>(n, par.shape(), policy)?;
    super::hessenberg_in_tuned(a, par, work.stack(req)?, policy)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn hessenberg_tuned_workspace(
        &mut self,
        a: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        parallel: usize,
        blocking: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let policy = Policy { parallel, blocking };
            if tag != 1 {
                return Err("Hessenberg scalar metadata mismatch".into());
            }
            let (h, q) = with_team(team, |par| compute(a.matrix.as_ref(), work, n, policy, par))?;
            self.matrix = h;
            out_q.matrix = q;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn hessenberg_tuned_workspace(
        &mut self,
        a: &TypedMatrix,
        out_q: &mut TypedMatrix,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        parallel: usize,
        blocking: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let policy = Policy { parallel, blocking };
            let (h, q) = match a.data()? {
                MatrixData::F32(a) if tag == 0 => {
                    let (h, q) = with_team(team, |par| compute(a.as_ref(), work, n, policy, par))?;
                    (MatrixData::F32(h), MatrixData::F32(q))
                }
                MatrixData::F64(a) if tag == 1 => {
                    let (h, q) = with_team(team, |par| compute(a.as_ref(), work, n, policy, par))?;
                    (MatrixData::F64(h), MatrixData::F64(q))
                }
                MatrixData::C32(a) if tag == 2 => {
                    let (h, q) = with_team(team, |par| compute(a.as_ref(), work, n, policy, par))?;
                    (MatrixData::C32(h), MatrixData::C32(q))
                }
                MatrixData::C64(a) if tag == 3 => {
                    let (h, q) = with_team(team, |par| compute(a.as_ref(), work, n, policy, par))?;
                    (MatrixData::C64(h), MatrixData::C64(q))
                }
                _ => return Err("Hessenberg scalar metadata mismatch".into()),
            };
            self.data = Some(h);
            out_q.data = Some(q);
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::traits::math_utils::*;
    use faer::{c32, c64, Mat};

    fn check<T: ComplexField>(tolerance: f64, imaginary: T) {
        let mut team = crate::teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for n in [0, 1, 2, 17, 65] {
            let a = Mat::<T>::from_fn(n, n, |i, j| {
                add(
                    &from_f64::<T>(((i * 7 + j * 11 + 3) % 19) as f64 / 19.0),
                    &mul(
                        &imaginary,
                        &from_f64::<T>(((i * 3 + j * 5) % 13) as f64 / 13.0),
                    ),
                )
            });
            for parallel in [0, 1, usize::MAX - 1, usize::MAX] {
                for blocking in [0, 256 * 256, usize::MAX] {
                    let policy = Policy { parallel, blocking };
                    for use_team in [false, true] {
                        let run = |par: Par<'_>| {
                            let req = super::super::construction::hessenberg_requirement_tuned::<T>(
                                n,
                                par.shape(),
                                policy,
                            );
                            let mut memory = MemBuffer::new(req);
                            let (h, q) = super::super::hessenberg_in_tuned(
                                a.as_ref(),
                                par,
                                MemStack::new(&mut memory),
                                policy,
                            )
                            .unwrap();
                            let reconstructed = &q * &h * q.adjoint();
                            assert!(
                                (&reconstructed - &a).norm_max() <= from_f64::<T::Real>(tolerance)
                            );
                            let identity = q.adjoint() * &q;
                            assert!(
                                (&identity - Mat::<T>::identity(n, n)).norm_max()
                                    <= from_f64::<T::Real>(tolerance)
                            );
                            for j in 0..n {
                                for i in j + 2..n {
                                    assert!(h[(i, j)] == zero::<T>());
                                }
                            }
                        };
                        if use_team {
                            team.execute(run).unwrap();
                        } else {
                            run(Par::Seq);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn explicit_tuned_hessenberg_four_scalars() {
        check::<f32>(2e-4, 0.0);
        check::<f64>(2e-12, 0.0);
        check::<c32>(2e-4, c32::new(0.0, 1.0));
        check::<c64>(2e-12, c64::new(0.0, 1.0));
    }
}
