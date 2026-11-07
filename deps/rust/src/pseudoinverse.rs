use crate::typed::{finite, hermitian_copy, MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatRef, Par, Side};
use rustcall_julia_macros::julia;

// Compare native singular/eigenvalues against the same promoted cutoff as
// Julia. Rounding a Float64 threshold upward to Float32 would incorrectly
// discard a Float32 singular value just above that threshold.
pub(crate) trait ToleranceReal: Sized {
    fn cutoff(self, atol: f64, rtol: f64, relative_f32: bool) -> Self;
}
impl ToleranceReal for f64 {
    fn cutoff(self, atol: f64, rtol: f64, _relative_f32: bool) -> Self {
        atol.max(rtol * self)
    }
}
impl ToleranceReal for f32 {
    fn cutoff(self, atol: f64, rtol: f64, relative_f32: bool) -> Self {
        let rel = if relative_f32 {
            (rtol as f32 * self) as f64
        } else {
            rtol * self as f64
        };
        let cutoff = atol.max(rel);
        let rounded = cutoff as f32;
        if rounded as f64 > cutoff {
            rounded.next_down()
        } else {
            rounded
        }
    }
}

fn pseudoinverse<T: ComplexField>(
    a: MatRef<'_, T>,
    mode: i32,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    mut par: Par<'_>,
) -> Result<Mat<T>, String>
where
    T::Real: ToleranceReal,
{
    if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
        return Err("tolerances must be nonnegative and not NaN".into());
    }
    let (m, n) = a.shape();
    let mut out = Mat::zeros(n, m);
    if mode == 0 {
        finite(a)?;
        if m == 0 || n == 0 {
            return Ok(out);
        }
        let f = a.thin_svd(par.reborrow()).map_err(|e| format!("{e:?}"))?;
        let mut memory = faer::dyn_stack::MemBuffer::new(
            faer::linalg::svd::pseudoinverse_from_svd_scratch::<T>(m, n, par.shape()),
        );
        faer::linalg::svd::pseudoinverse_from_svd_with_tolerance(
            out.as_mut(),
            f.S(),
            f.U(),
            f.V(),
            f.S()
                .column_vector()
                .norm_max()
                .cutoff(atol, rtol, relative_f32),
            zero::<T::Real>(),
            par.reborrow(),
            faer::dyn_stack::MemStack::new(&mut memory),
        );
    } else if mode == 1 || mode == 2 {
        let a = hermitian_copy(a, mode == 1)?;
        if m == 0 {
            return Ok(out);
        }
        let f = a
            .self_adjoint_eigen(Side::Lower, par.reborrow())
            .map_err(|e| format!("{e:?}"))?;
        let mut memory = faer::dyn_stack::MemBuffer::new(
            faer::linalg::evd::pseudoinverse_from_self_adjoint_evd_scratch::<T>(n, par.shape()),
        );
        faer::linalg::evd::pseudoinverse_from_self_adjoint_evd_with_tolerance(
            out.as_mut(),
            f.S(),
            f.U(),
            f.S()
                .column_vector()
                .norm_max()
                .cutoff(atol, rtol, relative_f32),
            zero::<T::Real>(),
            par.reborrow(),
            faer::dyn_stack::MemStack::new(&mut memory),
        );
    } else {
        return Err("unknown pseudoinverse mode".into());
    }
    Ok(out)
}
unsafe fn operator_norm<T: ComplexField>(
    a: MatRef<'_, T>,
    out: *mut u8,
    rows: bool,
) -> Result<(), String> {
    let a = if rows { a.transpose() } else { a };
    let mut max = zero::<T::Real>();
    for j in 0..a.ncols() {
        let mut sum = zero::<T::Real>();
        for i in 0..a.nrows() {
            sum = add(&sum, &abs(&a[(i, j)]));
        }
        if is_nan(&sum) || sum > max {
            max = sum;
        }
    }
    out.cast::<T::Real>().write(max);
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn pseudoinverse_from(
        &mut self,
        source: &OwnedMatrix,
        mode: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source;
            self.matrix = crate::teams::with_team(team, |par| {
                pseudoinverse(a.matrix.as_ref(), mode, atol, rtol, relative_f32, par)
            })?;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn pseudoinverse_from(
        &mut self,
        source: &TypedMatrix,
        mode: i32,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        team: Option<&crate::teams::OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(crate::teams::with_team(team, |par| {
                Ok(match a {
                    MatrixData::F32(x) => MatrixData::F32(pseudoinverse(
                        x.as_ref(),
                        mode,
                        atol,
                        rtol,
                        relative_f32,
                        par,
                    )?),
                    MatrixData::F64(x) => MatrixData::F64(pseudoinverse(
                        x.as_ref(),
                        mode,
                        atol,
                        rtol,
                        relative_f32,
                        par,
                    )?),
                    MatrixData::C32(x) => MatrixData::C32(pseudoinverse(
                        x.as_ref(),
                        mode,
                        atol,
                        rtol,
                        relative_f32,
                        par,
                    )?),
                    MatrixData::C64(x) => MatrixData::C64(pseudoinverse(
                        x.as_ref(),
                        mode,
                        atol,
                        rtol,
                        relative_f32,
                        par,
                    )?),
                })
            })?);
            Ok(())
        })
    }
    #[julia]
    pub fn operator_norm(&self, out: *mut u8, rows: bool) -> Result<(), String> {
        guarded(|| unsafe {
            match self.data()? {
                MatrixData::F32(x) => operator_norm(x.as_ref(), out, rows),
                MatrixData::F64(x) => operator_norm(x.as_ref(), out, rows),
                MatrixData::C32(x) => operator_norm(x.as_ref(), out, rows),
                MatrixData::C64(x) => operator_norm(x.as_ref(), out, rows),
            }
        })
    }
}
