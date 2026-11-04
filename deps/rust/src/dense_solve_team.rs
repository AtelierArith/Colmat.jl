//! Checked retained-factor solves borrowing a caller-owned team.
use crate::teams::OwnedTeam;
use crate::typed::{self, Factor, FactorData, MatrixData, TypedFactor, TypedMatrix};
use crate::{guarded, DenseFactor, OwnedFactor, OwnedMatrix};
use faer::linalg::solvers::ShapeCore;
use faer::traits::ComplexField;
use rustcall_julia_macros::julia;
fn typed_shape<T: ComplexField>(f: &Factor<T>) -> (usize, usize, i32) {
    macro_rules! shape {
        ($x:ident,$k:expr) => {
            ($x.nrows(), $x.ncols(), $k)
        };
    }
    match f {
        Factor::Lu(x) => shape!(x, 0),
        Factor::FullLu(x) => shape!(x, 1),
        Factor::Qr(x) => shape!(x, 2),
        Factor::PivotQr(x) => shape!(x, 3),
        Factor::Llt(x) => shape!(x, 4),
        Factor::Ldlt(x) => shape!(x, 5),
        Factor::Lblt(x) => shape!(x, 6),
        Factor::Svd(x) => shape!(x, 7),
    }
}
fn validate_owned(f: &OwnedFactor, m: usize, n: usize, tag: i32, kind: i32) -> Result<(), String> {
    let f = f.get()?;
    let k = match f {
        DenseFactor::Lu(_) => 0,
        DenseFactor::FullLu(_) => 1,
        DenseFactor::Qr(_) => 2,
        DenseFactor::PivotQr(_) => 3,
        DenseFactor::Llt(_) => 4,
        DenseFactor::Ldlt(_) => 5,
        DenseFactor::Lblt(_) => 6,
        DenseFactor::Svd(_) => 7,
    };
    if (
        dispatch_factor!(f,x=>x.nrows()),
        dispatch_factor!(f,x=>x.ncols()),
        1,
        k,
    ) != (m, n, tag, kind)
    {
        return Err("factor native dtype/dimensions/kind do not match requested metadata".into());
    }
    Ok(())
}
fn validate_typed(f: &TypedFactor, m: usize, n: usize, tag: i32, kind: i32) -> Result<(), String> {
    let (shape, t) = match f.data()? {
        FactorData::F32(x) => (typed_shape(x), 0),
        FactorData::F64(x) => (typed_shape(x), 1),
        FactorData::C32(x) => (typed_shape(x), 2),
        FactorData::C64(x) => (typed_shape(x), 3),
    };
    if (shape.0, shape.1, t, shape.2) != (m, n, tag, kind) {
        return Err("factor native dtype/dimensions/kind do not match requested metadata".into());
    }
    Ok(())
}
#[julia]
impl OwnedFactor {
    #[julia]
    pub fn factor_solve_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| validate_owned(self, m, n, tag, kind))
    }
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn factor_solve_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| validate_typed(self, m, n, tag, kind))
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn factor_solve_team(
        &mut self,
        f: &OwnedFactor,
        rhs: &OwnedMatrix,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            validate_owned(f, m, n, tag, kind)?;
            if !(0..=3).contains(&op) || rhs.matrix.ncols() != columns {
                return Err("invalid factor solve transform or RHS columns".into());
            }
            self.matrix = team.execute(|par| {
                crate::dense_factor_solve_with_par(f, rhs, op == 1 || op == 2, par)
            })??;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn factor_solve_team(
        &mut self,
        f: &TypedFactor,
        rhs: &TypedMatrix,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            validate_typed(f, m, n, tag, kind)?;
            f.solve_ready()?;
            macro_rules! solve {
                ($f:expr,$b:expr,$variant:ident) => {{
                    if $b.ncols() != columns {
                        return Err("factor RHS columns do not match requested metadata".into());
                    }
                    MatrixData::$variant(
                        team.execute(|par| typed::factor_solve($f, $b.as_ref(), op, par))??,
                    )
                }};
            }
            self.data = Some(match (f.data()?, rhs.data()?) {
                (FactorData::F32(f), MatrixData::F32(b)) => solve!(f, b, F32),
                (FactorData::C32(f), MatrixData::C32(b)) => solve!(f, b, C32),
                (FactorData::C64(f), MatrixData::C64(b)) => solve!(f, b, C64),
                _ => return Err("factor and RHS native scalar types must match".into()),
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{c32, Mat};
    #[test]
    fn owned_team_solve_validates_current_factor_rhs_and_closed_team() {
        let a = OwnedMatrix {
            matrix: Mat::identity(4, 4),
        };
        let mut f = OwnedFactor::new();
        f.initialize(&a, 0, true, 4, 4, 1).unwrap();
        let mut out = OwnedMatrix {
            matrix: Mat::zeros(4, 4),
        };
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for _ in 0..2 {
            out.factor_solve_team(&f, &a, 0, 4, 4, 4, 1, 0, &team)
                .unwrap();
            assert_eq!(out.matrix, a.matrix);
        }
        for (m, n, k, tag, kind, op) in [
            (5, 4, 4, 1, 0, 0),
            (4, 4, 5, 1, 0, 0),
            (4, 4, 4, 0, 0, 0),
            (4, 4, 4, 1, 2, 0),
            (4, 4, 4, 1, 0, 8),
        ] {
            assert!(out
                .factor_solve_team(&f, &a, op, m, n, k, tag, kind, &team)
                .is_err());
            assert_eq!(out.matrix, a.matrix);
        }
        team.team_close().unwrap();
        assert!(out
            .factor_solve_team(&f, &a, 0, 4, 4, 4, 1, 0, &team)
            .is_err());
        assert_eq!(out.matrix, a.matrix);
    }
    #[test]
    fn typed_team_solve_checks_dtype_kind_and_stages_failed_output() {
        let a = TypedMatrix {
            data: Some(MatrixData::C32(Mat::identity(4, 4))),
        };
        let mut f = TypedFactor::new();
        f.initialize(&a, 2, true, 4, 4, 2).unwrap();
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut out = TypedMatrix::new();
        for op in 0..4 {
            out.factor_solve_team(&f, &a, op, 4, 4, 4, 2, 2, &team)
                .unwrap();
            match out.data().unwrap() {
                MatrixData::C32(x) => assert_eq!(x, &Mat::<c32>::identity(4, 4)),
                _ => panic!("wrong output scalar"),
            }
        }
        assert!(f.factor_solve_validate(4, 4, 0, 2).is_err());
        assert!(f.factor_solve_validate(4, 4, 2, 0).is_err());
        let bad = TypedMatrix {
            data: Some(MatrixData::F32(Mat::identity(4, 4))),
        };
        assert!(out
            .factor_solve_team(&f, &bad, 0, 4, 4, 4, 2, 2, &team)
            .is_err());
        match out.data().unwrap() {
            MatrixData::C32(x) => assert_eq!(x, &Mat::<c32>::identity(4, 4)),
            _ => panic!("output changed after failure"),
        }
        team.team_close().unwrap();
    }
}

mod workspace;

mod array_metadata;
