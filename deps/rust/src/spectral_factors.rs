//! Reusable owned faer SVD and self-adjoint EVD objects.
use crate::pseudoinverse::ToleranceReal;
use crate::teams::with_team;
use crate::typed::{finite, hermitian_copy, MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, teams::OwnedTeam, OwnedMatrix};
use faer::diag::DiagRef;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::solvers::{DenseSolveCore, SelfAdjointEigen, Svd};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par, ParShape, Side};
use rustcall_julia_macros::julia;
#[derive(Clone)]
pub(crate) enum Decomposition<T> {
    Svd(Svd<T>),
    Eigen(SelfAdjointEigen<T>),
}
impl<T: ComplexField> Decomposition<T>
where
    T::Real: ToleranceReal,
{
    fn new(
        a: MatRef<'_, T>,
        eigen: bool,
        lower: bool,
        full: bool,
        par: Par<'_>,
    ) -> Result<Self, String> {
        if eigen {
            let a = hermitian_copy(a, lower)?;
            Ok(Self::Eigen(
                SelfAdjointEigen::new(a.as_ref(), Side::Lower, par)
                    .map_err(|e| format!("spectral EVD: {e:?}"))?,
            ))
        } else {
            finite(a)?;
            Ok(Self::Svd(
                if full {
                    Svd::new(a, par)
                } else {
                    Svd::new_thin(a, par)
                }
                .map_err(|e| format!("spectral SVD: {e:?}"))?,
            ))
        }
    }
    pub(crate) fn u(&self) -> MatRef<'_, T> {
        match self {
            Self::Svd(f) => f.U(),
            Self::Eigen(f) => f.U(),
        }
    }
    pub(crate) fn v(&self) -> MatRef<'_, T> {
        match self {
            Self::Svd(f) => f.V(),
            Self::Eigen(f) => f.U(),
        }
    }
    pub(crate) fn s(&self) -> DiagRef<'_, T> {
        match self {
            Self::Svd(f) => f.S(),
            Self::Eigen(f) => f.S(),
        }
    }
    pub(crate) fn eigen(&self) -> bool {
        matches!(self, Self::Eigen(_))
    }
    fn pinv_requirement(&self, par: ParShape) -> StackReq {
        let (m, n) = (self.u().nrows(), self.v().nrows());
        if m == 0 || n == 0 {
            return StackReq::EMPTY;
        }
        match self {
            Self::Svd(_) => faer::linalg::svd::pseudoinverse_from_svd_scratch::<T>(m, n, par),
            Self::Eigen(_) => {
                faer::linalg::evd::pseudoinverse_from_self_adjoint_evd_scratch::<T>(n, par)
            }
        }
    }
    fn pinv_in(
        &self,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Mat<T>, String> {
        if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
            return Err("tolerances must be nonnegative and not NaN".into());
        }
        let (m, n) = (self.u().nrows(), self.v().nrows());
        let mut out = Mat::zeros(n, m);
        if m == 0 || n == 0 {
            return Ok(out);
        }
        let cutoff = self
            .s()
            .column_vector()
            .norm_max()
            .cutoff(atol, rtol, relative_f32);
        match self {
            Self::Svd(f) => faer::linalg::svd::pseudoinverse_from_svd_with_tolerance(
                out.as_mut(),
                f.S(),
                f.U(),
                f.V(),
                cutoff,
                zero::<T::Real>(),
                par,
                stack,
            ),
            Self::Eigen(f) => {
                faer::linalg::evd::pseudoinverse_from_self_adjoint_evd_with_tolerance(
                    out.as_mut(),
                    f.S(),
                    f.U(),
                    cutoff,
                    zero::<T::Real>(),
                    par,
                    stack,
                )
            }
        }
        Ok(out)
    }
    fn part(
        &self,
        part: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
    ) -> Result<Mat<T>, String> {
        match part {
            0 => Ok(match self {
                Self::Svd(f) => f.reconstruct(par),
                Self::Eigen(f) => f.reconstruct(par),
            }),
            1 => Ok(self.u().to_owned()),
            2 => Ok(self.v().to_owned()),
            4 => Ok(match self {
                Self::Svd(f) => f.inverse(par),
                Self::Eigen(f) => f.inverse(par),
            }),
            3 => {
                let mut memory = MemBuffer::new(self.pinv_requirement(par.shape()));
                self.pinv_in(atol, rtol, relative_f32, par, MemStack::new(&mut memory))
            }
            _ => Err("unknown spectral factor component".into()),
        }
    }
}
#[derive(Clone)]
pub(crate) enum SpectralData {
    F32(Decomposition<f32>),
    F64(Decomposition<f64>),
    C32(Decomposition<faer::c32>),
    C64(Decomposition<faer::c64>),
}
#[julia]
#[derive(Default)]
pub struct OwnedSpectral {
    data: Option<SpectralData>,
}
impl OwnedSpectral {
    pub(crate) fn data(&self) -> Result<&SpectralData, String> {
        self.data
            .as_ref()
            .ok_or("spectral factor is closed or uninitialized".into())
    }
    pub(crate) fn check(&self, tag: i32, eigen: bool) -> Result<(), String> {
        let (actual, mode) = match self.data()? {
            SpectralData::F32(f) => (0, f.eigen()),
            SpectralData::F64(f) => (1, f.eigen()),
            SpectralData::C32(f) => (2, f.eigen()),
            SpectralData::C64(f) => (3, f.eigen()),
        };
        if actual != tag || mode != eigen {
            return Err("spectral factor metadata mismatch".into());
        }
        Ok(())
    }
    fn part(
        &self,
        part: i32,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
    ) -> Result<MatrixData, String> {
        self.check(tag, eigen)?;
        let data = match self.data()? {
            SpectralData::F32(f) => MatrixData::F32(f.part(part, atol, rtol, relative_f32, par)?),
            SpectralData::F64(f) => MatrixData::F64(f.part(part, atol, rtol, relative_f32, par)?),
            SpectralData::C32(f) => MatrixData::C32(f.part(part, atol, rtol, relative_f32, par)?),
            SpectralData::C64(f) => MatrixData::C64(f.part(part, atol, rtol, relative_f32, par)?),
        };
        let shape = match &data {
            MatrixData::F32(a) => a.shape(),
            MatrixData::F64(a) => a.shape(),
            MatrixData::C32(a) => a.shape(),
            MatrixData::C64(a) => a.shape(),
        };
        if shape != (rows, cols) {
            return Err("spectral result shape mismatch".into());
        }
        Ok(data)
    }
}
#[julia]
impl OwnedSpectral {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn spectral_close(&mut self) -> Result<(), String> {
        guarded(|| {
            self.data = None;
            Ok(())
        })
    }
    #[julia]
    pub fn spectral_clone(&mut self, other: &OwnedSpectral) -> Result<(), String> {
        guarded(|| {
            self.data = Some(other.data()?.clone());
            Ok(())
        })
    }
    #[julia]
    pub fn spectral_values(
        &self,
        out: *mut u8,
        len: usize,
        tag: i32,
        eigen: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.check(tag, eigen)?;
            unsafe fn write<T: ComplexField>(
                f: &Decomposition<T>,
                out: *mut u8,
                len: usize,
            ) -> Result<(), String>
            where
                T::Real: ToleranceReal,
            {
                if f.s().dim() != len {
                    return Err("spectral values length mismatch".into());
                }
                for i in 0..len {
                    out.cast::<T::Real>().add(i).write(real(&f.s()[i]));
                }
                Ok(())
            }
            match self.data()? {
                SpectralData::F32(f) => unsafe { write(f, out, len) },
                SpectralData::F64(f) => unsafe { write(f, out, len) },
                SpectralData::C32(f) => unsafe { write(f, out, len) },
                SpectralData::C64(f) => unsafe { write(f, out, len) },
            }
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn spectral_factor(
        &self,
        out: &mut OwnedSpectral,
        eigen: bool,
        lower: bool,
        full: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            out.data = Some(with_team(team, |par| -> Result<SpectralData, String> {
                Ok(SpectralData::F64(Decomposition::new(
                    self.matrix.as_ref(),
                    eigen,
                    lower,
                    full,
                    par,
                )?))
            })?);
            Ok(())
        })
    }

    #[julia]
    pub fn spectral_part(
        &mut self,
        f: &OwnedSpectral,
        part: i32,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let result = with_team(team, |par| {
                f.part(part, rows, cols, tag, eigen, atol, rtol, relative_f32, par)
            })?;
            if let MatrixData::F64(a) = result {
                self.matrix = a;
            } else {
                return Err("Float64 spectral output dtype mismatch".into());
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn spectral_factor(
        &self,
        out: &mut OwnedSpectral,
        eigen: bool,
        lower: bool,
        full: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            out.data = Some(with_team(team, |par| -> Result<SpectralData, String> {
                Ok(match self.data()? {
                    MatrixData::F32(a) => {
                        SpectralData::F32(Decomposition::new(a.as_ref(), eigen, lower, full, par)?)
                    }
                    MatrixData::F64(a) => {
                        SpectralData::F64(Decomposition::new(a.as_ref(), eigen, lower, full, par)?)
                    }
                    MatrixData::C32(a) => {
                        SpectralData::C32(Decomposition::new(a.as_ref(), eigen, lower, full, par)?)
                    }
                    MatrixData::C64(a) => {
                        SpectralData::C64(Decomposition::new(a.as_ref(), eigen, lower, full, par)?)
                    }
                })
            })?);
            Ok(())
        })
    }

    #[julia]
    pub fn spectral_part(
        &mut self,
        f: &OwnedSpectral,
        part: i32,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let result = with_team(team, |par| {
                f.part(part, rows, cols, tag, eigen, atol, rtol, relative_f32, par)
            })?;
            self.data = Some(result);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_pseudoinverse_survives_source_changes() {
        let expected =
            Mat::<f64>::from_fn(3, 3, |i, j| if i == j { [4.0, 1e-4, 0.0][i] } else { 0.0 });
        for eigen in [false, true] {
            let mut source = expected.clone();
            let factor = Decomposition::new(source.as_ref(), eigen, true, false, Par::Seq).unwrap();
            source.fill(0.0);
            let reconstructed = factor.part(0, 0.0, 0.0, false, Par::Seq).unwrap();
            assert!((&reconstructed - &expected).norm_max() < 1e-12);
            let inverse = factor.part(3, 1e-3, 0.0, false, Par::Seq).unwrap();
            let wanted = Mat::from_fn(3, 3, |i, j| if i == 0 && j == 0 { 0.25 } else { 0.0 });
            assert!((&inverse - &wanted).norm_max() < 1e-12);
        }
    }
    #[test]
    fn values_metadata_and_close_errors_do_not_write() {
        let a = Mat::<f32>::identity(3, 3);
        let mut factor = OwnedSpectral {
            data: Some(SpectralData::F32(
                Decomposition::new(a.as_ref(), false, true, false, Par::Seq).unwrap(),
            )),
        };
        // Sufficient storage/alignment for either real scalar prevents the test
        // itself from relying on an out-of-bounds write even if validation regresses.
        let mut output = [0x1234_5678_1234_5678u64; 4];
        let original = output;
        assert!(factor
            .spectral_values(output.as_mut_ptr().cast(), 3, 1, false)
            .is_err());
        assert_eq!(output, original);
        assert!(factor
            .spectral_values(output.as_mut_ptr().cast(), 2, 0, false)
            .is_err());
        assert_eq!(output, original);
        factor.spectral_close().unwrap();
        assert!(factor
            .spectral_values(output.as_mut_ptr().cast(), 3, 0, false)
            .is_err());
        assert_eq!(output, original);
    }
}

impl OwnedSpectral {
    fn pinv_requirement(
        &self,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        par: ParShape,
    ) -> Result<StackReq, String> {
        self.check(tag, eigen)?;
        match self.data()? {
            SpectralData::F32(f) => {
                if (f.u().nrows(), f.v().nrows()) != (rows, cols) {
                    return Err("spectral factor shape mismatch".into());
                }
                Ok(f.pinv_requirement(par))
            }
            SpectralData::F64(f) => {
                if (f.u().nrows(), f.v().nrows()) != (rows, cols) {
                    return Err("spectral factor shape mismatch".into());
                }
                Ok(f.pinv_requirement(par))
            }
            SpectralData::C32(f) => {
                if (f.u().nrows(), f.v().nrows()) != (rows, cols) {
                    return Err("spectral factor shape mismatch".into());
                }
                Ok(f.pinv_requirement(par))
            }
            SpectralData::C64(f) => {
                if (f.u().nrows(), f.v().nrows()) != (rows, cols) {
                    return Err("spectral factor shape mismatch".into());
                }
                Ok(f.pinv_requirement(par))
            }
        }
    }
    fn pinv_workspace(
        &self,
        scratch: &mut OwnedScratch,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
    ) -> Result<MatrixData, String> {
        let req = self.pinv_requirement(cols, rows, tag, eigen, par.shape())?;
        let stack = scratch.stack(req)?;
        Ok(match self.data()? {
            SpectralData::F32(f) => {
                MatrixData::F32(f.pinv_in(atol, rtol, relative_f32, par, stack)?)
            }
            SpectralData::F64(f) => {
                MatrixData::F64(f.pinv_in(atol, rtol, relative_f32, par, stack)?)
            }
            SpectralData::C32(f) => {
                MatrixData::C32(f.pinv_in(atol, rtol, relative_f32, par, stack)?)
            }
            SpectralData::C64(f) => {
                MatrixData::C64(f.pinv_in(atol, rtol, relative_f32, par, stack)?)
            }
        })
    }
}
#[julia]
impl OwnedSpectral {
    #[julia]
    pub fn spectral_pinv_requirements(
        &self,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        out: *mut usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let req = with_team(team, |par| {
                self.pinv_requirement(rows, cols, tag, eigen, par.shape())
            })?;
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn spectral_pinv_workspace(
        &mut self,
        f: &OwnedSpectral,
        scratch: &mut OwnedScratch,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let result = with_team(team, |par| {
                f.pinv_workspace(
                    scratch,
                    rows,
                    cols,
                    tag,
                    eigen,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                )
            })?;
            if let MatrixData::F64(a) = result {
                self.matrix = a;
            } else {
                return Err("Float64 spectral output dtype mismatch".into());
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn spectral_pinv_workspace(
        &mut self,
        f: &OwnedSpectral,
        scratch: &mut OwnedScratch,
        rows: usize,
        cols: usize,
        tag: i32,
        eigen: bool,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let result = with_team(team, |par| {
                f.pinv_workspace(
                    scratch,
                    rows,
                    cols,
                    tag,
                    eigen,
                    atol,
                    rtol,
                    relative_f32,
                    par,
                )
            })?;
            self.data = Some(result);
            Ok(())
        })
    }
}
