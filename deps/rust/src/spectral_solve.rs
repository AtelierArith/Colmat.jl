//! Apply retained spectral factors without materializing a pseudoinverse.
use crate::pseudoinverse::ToleranceReal;
use crate::spectral_factors::{Decomposition, OwnedSpectral, SpectralData};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatRef, Par};
use rustcall_julia_macros::julia;
mod workspace;
use workspace::{converted, same};
pub(crate) fn threshold<T: ComplexField>(
    f: &Decomposition<T>,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
) -> Result<T::Real, String>
where
    T::Real: ToleranceReal,
{
    if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
        return Err("tolerances must be nonnegative and not NaN".into());
    }
    Ok(if f.s().dim() == 0 {
        zero::<T::Real>()
    } else {
        f.s()
            .column_vector()
            .norm_max()
            .cutoff(atol, rtol, relative_f32)
    })
}
enum Rhs<'a> {
    F32(MatRef<'a, f32>),
    F64(MatRef<'a, f64>),
    C32(MatRef<'a, faer::c32>),
    C64(MatRef<'a, faer::c64>),
}
impl Rhs<'_> {
    fn info(&self) -> (i32, usize, usize) {
        match self {
            Self::F32(b) => (0, b.nrows(), b.ncols()),
            Self::F64(b) => (1, b.nrows(), b.ncols()),
            Self::C32(b) => (2, b.nrows(), b.ncols()),
            Self::C64(b) => (3, b.nrows(), b.ncols()),
        }
    }
}
impl OwnedSpectral {
    fn solve(
        &self,
        b: Rhs<'_>,
        factor_rows: usize,
        factor_cols: usize,
        rows: usize,
        cols: usize,
        factor_tag: i32,
        tag: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
        scratch: Option<&mut OwnedScratch>,
    ) -> Result<MatrixData, String> {
        self.check(factor_tag, eigen)?;
        if !(0..=3).contains(&op) {
            return Err("invalid spectral solve operation".into());
        }
        let actual = match self.data()? {
            SpectralData::F32(f) => (f.u().nrows(), f.v().nrows()),
            SpectralData::F64(f) => (f.u().nrows(), f.v().nrows()),
            SpectralData::C32(f) => (f.u().nrows(), f.v().nrows()),
            SpectralData::C64(f) => (f.u().nrows(), f.v().nrows()),
        };
        if actual != (factor_rows, factor_cols) {
            return Err("spectral factor shape mismatch".into());
        }
        let shape = if op == 1 || op == 2 {
            (factor_cols, factor_rows)
        } else {
            (factor_rows, factor_cols)
        };
        let (actual_tag, br, bc) = b.info();
        if actual_tag != tag {
            return Err("spectral RHS dtype mismatch".into());
        }
        if (right && (bc != shape.1 || (rows, cols) != (br, shape.0)))
            || (!right && (br != shape.0 || (rows, cols) != (shape.1, bc)))
        {
            return Err("spectral solve dimensions mismatch".into());
        }
        let req = workspace::requirement_for(
            self,
            factor_rows,
            factor_cols,
            factor_tag,
            tag,
            eigen,
            op,
            right,
            if right { br } else { bc },
        )?;
        let mut temporary = OwnedScratch::new();
        let scratch = match scratch {
            Some(s) => s,
            None => {
                temporary.scratch_reserve(req.size_bytes(), req.align_bytes())?;
                &mut temporary
            }
        };
        Ok(match (self.data()?, b) {
            (SpectralData::F32(f), Rhs::F32(b)) => MatrixData::F32(same(
                f,
                b,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::F64(f), Rhs::F64(b)) => MatrixData::F64(same(
                f,
                b,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::C32(f), Rhs::C32(b)) => MatrixData::C32(same(
                f,
                b,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::C64(f), Rhs::C64(b)) => MatrixData::C64(same(
                f,
                b,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::F32(f), Rhs::F64(b)) => MatrixData::F64(converted(
                f,
                b,
                |x| *x as f64,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::F32(f), Rhs::C32(b)) => MatrixData::C32(converted(
                f,
                b,
                |x| faer::c32::new(*x, 0.0),
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::F32(f), Rhs::C64(b)) => MatrixData::C64(converted(
                f,
                b,
                |x| faer::c64::new(*x as f64, 0.0),
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::F64(f), Rhs::C64(b)) => MatrixData::C64(converted(
                f,
                b,
                |x| faer::c64::new(*x, 0.0),
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            (SpectralData::C32(f), Rhs::C64(b)) => MatrixData::C64(converted(
                f,
                b,
                |x| faer::c64::new(x.re as f64, x.im as f64),
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                scratch,
            )?),
            _ => return Err("unsupported spectral solve scalar promotion".into()),
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn spectral_solve(
        &mut self,
        f: &OwnedSpectral,
        b: &OwnedMatrix,
        factor_rows: usize,
        factor_cols: usize,
        rows: usize,
        cols: usize,
        factor_tag: i32,
        tag: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = Rhs::F64(b.matrix.as_ref());
            let par = Par::Seq;
            let result = f.solve(
                rhs,
                factor_rows,
                factor_cols,
                rows,
                cols,
                factor_tag,
                tag,
                eigen,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                None,
            )?;
            if let MatrixData::F64(a) = result {
                self.matrix = a;
            } else {
                return Err("Float64 spectral solve output dtype mismatch".into());
            }
            Ok(())
        })
    }
    #[julia]
    pub fn spectral_solve_team(
        &mut self,
        f: &OwnedSpectral,
        b: &OwnedMatrix,
        factor_rows: usize,
        factor_cols: usize,
        rows: usize,
        cols: usize,
        factor_tag: i32,
        tag: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = Rhs::F64(b.matrix.as_ref());
            let result = team.execute(|par| {
                f.solve(
                    rhs,
                    factor_rows,
                    factor_cols,
                    rows,
                    cols,
                    factor_tag,
                    tag,
                    eigen,
                    op,
                    right,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                    None,
                )
            })??;
            if let MatrixData::F64(a) = result {
                self.matrix = a;
            } else {
                return Err("Float64 spectral solve output dtype mismatch".into());
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn spectral_solve(
        &mut self,
        f: &OwnedSpectral,
        b: &TypedMatrix,
        factor_rows: usize,
        factor_cols: usize,
        rows: usize,
        cols: usize,
        factor_tag: i32,
        tag: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = match b.data()? {
                MatrixData::F32(b) => Rhs::F32(b.as_ref()),
                MatrixData::F64(b) => Rhs::F64(b.as_ref()),
                MatrixData::C32(b) => Rhs::C32(b.as_ref()),
                MatrixData::C64(b) => Rhs::C64(b.as_ref()),
            };
            let par = Par::Seq;
            let result = f.solve(
                rhs,
                factor_rows,
                factor_cols,
                rows,
                cols,
                factor_tag,
                tag,
                eigen,
                op,
                right,
                atol,
                rtol,
                relative_f32,
                par,
                None,
            )?;
            self.data = Some(result);
            Ok(())
        })
    }
    #[julia]
    pub fn spectral_solve_team(
        &mut self,
        f: &OwnedSpectral,
        b: &TypedMatrix,
        factor_rows: usize,
        factor_cols: usize,
        rows: usize,
        cols: usize,
        factor_tag: i32,
        tag: i32,
        eigen: bool,
        op: i32,
        right: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            let rhs = match b.data()? {
                MatrixData::F32(b) => Rhs::F32(b.as_ref()),
                MatrixData::F64(b) => Rhs::F64(b.as_ref()),
                MatrixData::C32(b) => Rhs::C32(b.as_ref()),
                MatrixData::C64(b) => Rhs::C64(b.as_ref()),
            };
            let result = team.execute(|par| {
                f.solve(
                    rhs,
                    factor_rows,
                    factor_cols,
                    rows,
                    cols,
                    factor_tag,
                    tag,
                    eigen,
                    op,
                    right,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                    None,
                )
            })??;
            self.data = Some(result);
            Ok(())
        })
    }
}
