//! Native construction and saved-factor upgrades borrow the caller's team.
use super::{load, Data, OwnedHouseholderSequence, Sequence};
use crate::{guarded, teams::OwnedTeam};
use faer::linalg::householder as hh;
use faer::traits::ComplexField;
use faer::{Mat, Par};
use rustcall_julia_macros::julia;
pub(super) fn span<T>(p: *const u8, len: usize) -> Result<(), String> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or("Householder buffer byte size overflow")?;
    if len > 0
        && (p.is_null()
            || (p as usize) % std::mem::align_of::<T>() != 0
            || bytes > isize::MAX as usize
            || (p as usize).checked_add(bytes).is_none())
    {
        return Err("invalid Householder buffer address/alignment/span".into());
    }
    Ok(())
}
fn reblocked<T: ComplexField>(
    s: &Sequence<T>,
    b: usize,
    prev: usize,
    mut par: Par<'_>,
) -> Result<Sequence<T>, String> {
    let old = s.factor.nrows();
    let k = s.basis.ncols();
    if b == 0 || prev == 0 || b % prev != 0 || old % prev != 0 {
        return Err(
            "Householder prev_block_size must positively divide old and new blocksize".into(),
        );
    }
    b.checked_mul(k).ok_or("Householder factor size overflow")?;
    let basis = s.basis.clone();
    let mut factor = Mat::zeros(b, k);
    // Each copied previous block is fully contained in both old and new blocks.
    for j in 0..k {
        let offset = j % prev;
        for i in 0..=offset {
            factor[(j % b - offset + i, j)] = s.factor[(j % old - offset + i, j)].clone();
        }
    }
    let mut start = 0;
    while start < k {
        let width = b.min(k - start);
        hh::upgrade_householder_factor(
            factor.as_mut().submatrix_mut(0, start, width, width),
            basis
                .as_ref()
                .submatrix(start, start, basis.nrows() - start, width),
            width,
            prev,
            par.reborrow(),
        );
        start += width;
    }
    Ok(Sequence { basis, factor })
}
fn reblock(
    source: &OwnedHouseholderSequence,
    b: usize,
    prev: usize,
    n: usize,
    k: usize,
    old: usize,
    tag: i32,
    par: Par<'_>,
) -> Result<Data, String> {
    super::workspace::validate(source, n, k, old, tag)?;
    Ok(match source.data()? {
        Data::F32(s) => Data::F32(reblocked(s, b, prev, par)?),
        Data::F64(s) => Data::F64(reblocked(s, b, prev, par)?),
        Data::C32(s) => Data::C32(reblocked(s, b, prev, par)?),
        Data::C64(s) => Data::C64(reblocked(s, b, prev, par)?),
    })
}
#[julia]
impl OwnedHouseholderSequence {
    #[julia]
    pub fn sequence_validate(&self, n: usize, k: usize, b: usize, tag: i32) -> Result<(), String> {
        guarded(|| super::workspace::validate(self, n, k, b, tag))
    }
    #[julia]
    pub fn sequence_load_team(
        &mut self,
        basis: *const u8,
        m: usize,
        k: usize,
        tau: *const u8,
        b: usize,
        tag: i32,
        provided: *const u8,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                self.data = Some(unsafe {
                    match tag {
                        0 => Data::F32(load(basis, m, k, tau, b, provided, par)?),
                        1 => Data::F64(load(basis, m, k, tau, b, provided, par)?),
                        2 => Data::C32(load(basis, m, k, tau, b, provided, par)?),
                        3 => Data::C64(load(basis, m, k, tau, b, provided, par)?),
                        _ => return Err("unsupported Householder scalar".into()),
                    }
                });
                Ok(())
            })?
        })
    }
    #[julia]
    pub fn sequence_reblock_from(
        &mut self,
        source: &OwnedHouseholderSequence,
        b: usize,
        prev: usize,
        n: usize,
        k: usize,
        old: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(reblock(source, b, prev, n, k, old, tag, Par::Seq)?);
            Ok(())
        })
    }
    #[julia]
    pub fn sequence_reblock_from_team(
        &mut self,
        source: &OwnedHouseholderSequence,
        b: usize,
        prev: usize,
        n: usize,
        k: usize,
        old: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                self.data = Some(reblock(source, b, prev, n, k, old, tag, par)?);
                Ok(())
            })?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::traits::math_utils::*;
    macro_rules! scalar {
        ($name:ident,$t:ty,$variant:ident,$tag:expr,$tol:expr) => {
            #[test]
            fn $name() {
                for width in [1, 2] {
                    let mut team = OwnedTeam::new();
                    team.team_initialize(width).unwrap();
                    for (m, k) in [(0, 0), (4, 0), (1, 1), (13, 11), (96, 65), (8192, 32)] {
                        let mut basis = vec![zero::<$t>(); m * k];
                        let mut tau = vec![zero::<$t>(); k];
                        for j in 0..k {
                            let mut norm = 1.0;
                            for i in j + 1..m {
                                let value = 0.01 * ((i + 2 * j) as f64).sin();
                                basis[i + j * m] = from_f64(value);
                                norm += value * value;
                            }
                            tau[j] = from_f64(norm / 2.0);
                        }
                        let mut source = OwnedHouseholderSequence::new();
                        source
                            .sequence_load_team(
                                basis.as_ptr().cast(),
                                m,
                                k,
                                tau.as_ptr().cast(),
                                8,
                                $tag,
                                std::ptr::null(),
                                &team,
                            )
                            .unwrap();
                        for b in [1, 3, 8, 16, 24] {
                            for prev in [1, 8] {
                                if b % prev != 0 {
                                    continue;
                                }
                                let mut out = OwnedHouseholderSequence::new();
                                out.sequence_reblock_from_team(
                                    &source, b, prev, m, k, 8, $tag, &team,
                                )
                                .unwrap();
                                let Data::$variant(s) = out.data().unwrap() else {
                                    panic!("wrong scalar")
                                };
                                for j in 0..k {
                                    assert!(
                                        abs(&(s.factor[(j % b, j)].clone() - tau[j].clone()))
                                            < from_f64($tol)
                                    );
                                    for offset in 1..=j % b {
                                        let i = j - offset;
                                        let mut expected = 0.01 * ((j + 2 * i) as f64).sin();
                                        for row in j + 1..m {
                                            expected += 0.01
                                                * ((row + 2 * i) as f64).sin()
                                                * 0.01
                                                * ((row + 2 * j) as f64).sin();
                                        }
                                        assert!(
                                            abs(&(s.factor[(i % b, j)].clone()
                                                - from_f64::<$t>(expected)))
                                                < from_f64($tol)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    scalar!(construct_reblock_f32, f32, F32, 0, 1e-4);
    scalar!(construct_reblock_f64, f64, F64, 1, 1e-10);
    scalar!(construct_reblock_c32, faer::c32, C32, 2, 1e-4);
    scalar!(construct_reblock_c64, faer::c64, C64, 3, 1e-10);
    #[test]
    fn rejected_construction_and_upgrades_preserve_output() {
        let basis = [1.0, 0.2, 0.3, 0.0, 1.0, 0.4];
        let tau = [0.565, 0.58];
        let mut source = OwnedHouseholderSequence::new();
        source
            .sequence_load(
                basis.as_ptr().cast(),
                3,
                2,
                tau.as_ptr().cast(),
                8,
                1,
                std::ptr::null(),
            )
            .unwrap();
        let mut out = OwnedHouseholderSequence::new();
        out.sequence_clone(&source).unwrap();
        let before = match out.data().unwrap() {
            Data::F64(s) => s.factor.clone(),
            _ => unreachable!(),
        };
        assert!(out
            .sequence_reblock_from(&source, 16, 3, 3, 2, 8, 1)
            .is_err());
        assert!(out
            .sequence_reblock_from(&source, 16, 8, 4, 2, 8, 1)
            .is_err());
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        assert!(out
            .sequence_load_team(
                std::ptr::null(),
                3,
                2,
                tau.as_ptr().cast(),
                8,
                1,
                std::ptr::null(),
                &team
            )
            .is_err());
        team.team_close().unwrap();
        assert!(out
            .sequence_reblock_from_team(&source, 16, 8, 3, 2, 8, 1, &team)
            .is_err());
        let Data::F64(s) = out.data().unwrap() else {
            unreachable!()
        };
        assert_eq!(s.factor, before);
    }
}
