//! Caller-owned reusable teams; execution tokens never escape a native call.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::one, ComplexField};
use faer::{Accum, Mat, MatRef, Par};
use rustcall_julia_macros::julia;
#[julia]
pub struct OwnedTeam {
    team: Option<threadteam::Team>,
    width: usize,
}
impl OwnedTeam {
    pub(crate) fn execute<R>(&self, f: impl FnOnce(Par<'_>) -> R) -> Result<R, String> {
        if self.width == 0 {
            return Err("team is closed or uninitialized".into());
        }
        if self.width == 1 {
            return Ok(f(Par::Seq));
        }
        let team = self.team.as_ref().ok_or("missing team reservation")?;
        let mut exec = team.exec(self.width);
        exec.team(|t| f(Par::Team(t)))
            .map_err(|e| format!("team execution: {e:?}"))
    }
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn new() -> Self {
        Self {
            team: None,
            width: 0,
        }
    }
    #[julia]
    pub fn team_initialize(&mut self, width: usize) -> Result<(), String> {
        guarded(|| {
            if width == 0 {
                return Err("team width must be positive".into());
            }
            let team = if width == 1 {
                None
            } else {
                let pool = threadteam::Pool::builder()
                    .threads(width)
                    .acquire_timeout(std::time::Duration::from_secs(30))
                    .build()
                    .map_err(|e| format!("team pool: {e:?}"))?;
                Some(
                    pool.into_team(width)
                        .map_err(|e| format!("team reservation: {:?}", e.reason()))?,
                )
            };
            self.team = team;
            self.width = width;
            Ok(())
        })
    }
    #[julia]
    pub fn team_close(&mut self) -> Result<(), String> {
        guarded(|| {
            self.width = 0;
            self.team = None;
            Ok(())
        })
    }
}
fn product<T: ComplexField>(
    team: &OwnedTeam,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<Mat<T>, String> {
    if a.ncols() != b.nrows() {
        return Err("team product inner dimensions must match".into());
    }
    let mut out = Mat::zeros(a.nrows(), b.ncols());
    team.execute(|par| {
        faer::linalg::matmul::matmul(out.as_mut(), Accum::Replace, a, b, one::<T>(), par)
    })?;
    Ok(out)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn team_product(
        &mut self,
        team: &OwnedTeam,
        a: &OwnedMatrix,
        b: &OwnedMatrix,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = product(team, a.matrix.as_ref(), b.matrix.as_ref())?;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn team_product(
        &mut self,
        team: &OwnedTeam,
        a: &TypedMatrix,
        b: &TypedMatrix,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match (a.data()?, b.data()?) {
                (MatrixData::F32(a), MatrixData::F32(b)) => {
                    MatrixData::F32(product(team, a.as_ref(), b.as_ref())?)
                }
                (MatrixData::C32(a), MatrixData::C32(b)) => {
                    MatrixData::C32(product(team, a.as_ref(), b.as_ref())?)
                }
                (MatrixData::C64(a), MatrixData::C64(b)) => {
                    MatrixData::C64(product(team, a.as_ref(), b.as_ref())?)
                }
                _ => return Err("team matrix scalar types must match".into()),
            });
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reusable_team_dispatches_reserved_workers() {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for _ in 0..2 {
            let ids = std::sync::Mutex::new(Vec::new());
            team.execute(|par| match par {
                Par::Team(t) => t.run(|_, _| ids.lock().unwrap().push(std::thread::current().id())),
                _ => panic!("width two must lend a team token"),
            })
            .unwrap();
            let ids = ids.into_inner().unwrap();
            assert_eq!(ids.len(), 2);
            assert_ne!(ids[0], ids[1]);
        }
        team.team_close().unwrap();
        assert!(team.execute(|_| ()).is_err());
    }
}
