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
pub(crate) fn with_team<R>(
    team: Option<&OwnedTeam>,
    f: impl FnOnce(Par<'_>) -> Result<R, String>,
) -> Result<R, String> {
    if let Some(team) = team {
        team.execute(f)?
    } else {
        f(Par::Seq)
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
        self.team_initialize_with(width, 30.0, 0, 0, 1, &[0, 0])
    }
    /// Builds the reservation with pool options. `set` flags the overridden
    /// entries: `[spin_polls, qos]`; `qos` is 0 default, 1 utility, 2
    /// user-initiated, 3 user-interactive (applied on Apple targets only).
    #[julia]
    pub fn team_initialize_with(
        &mut self,
        width: usize,
        acquire_timeout: f64,
        spin_polls: u32,
        qos: i32,
        pin: u8,
        set: &[u8],
    ) -> Result<(), String> {
        guarded(|| {
            if width == 0 {
                return Err("team width must be positive".into());
            }
            if set.len() != 2 {
                return Err("team options need two flags".into());
            }
            if !(acquire_timeout.is_finite() && acquire_timeout > 0.0) {
                return Err("acquire_timeout must be positive and finite".into());
            }
            let qos = match qos {
                0 => threadteam::Qos::Default,
                1 => threadteam::Qos::Utility,
                2 => threadteam::Qos::UserInitiated,
                3 => threadteam::Qos::UserInteractive,
                _ => return Err("unknown team qos".into()),
            };
            let mut builder = threadteam::Pool::builder()
                .threads(width)
                .acquire_timeout(std::time::Duration::from_secs_f64(acquire_timeout))
                .pin_participants(pin != 0);
            if set[0] != 0 {
                builder = builder.spin_polls(spin_polls);
            }
            if set[1] != 0 {
                builder = builder.qos(qos);
            }
            let team = if width == 1 {
                None
            } else {
                let pool = builder.build().map_err(|e| format!("team pool: {e:?}"))?;
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
    /// Writes `[reserved width, spin polls, pins participants, resident
    /// members, pin skips, join sleeps, barrier sleeps, member parks]`. A
    /// width-one team runs sequentially and has no pool: all but the width are 0.
    #[julia]
    pub fn team_state(&self, values: &mut [u64]) -> Result<(), String> {
        guarded(|| {
            if values.len() != 8 {
                return Err("team state has eight values".into());
            }
            if self.width == 0 {
                return Err("team is closed or uninitialized".into());
            }
            values.fill(0);
            values[0] = self.width as u64;
            if let Some(team) = &self.team {
                let pool = team.pool();
                let w = pool.wait_counters();
                values[0] = team.size().get() as u64;
                values[1] = pool.spin_polls() as u64;
                values[2] = pool.pin_participants() as u64;
                values[3] = pool.resident_members() as u64;
                values[4] = pool.pin_skips();
                values[5] = w.join_sleeps;
                values[6] = w.barrier_sleeps;
                values[7] = w.member_parks;
            }
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
