//! Caller-owned workspaces and scoped teams for native operator application.
use super::{Data, Operator, OwnedLinearOperator};
use crate::teams::with_team;
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::matrix_free::LinOp;
use faer::traits::ComplexField;
use crate::views::{
    destination, destination_span, operand, source as source_view, spec_shape, Arm, Borrowed,
    Spec,
};
use crate::workspace::scratch_or_temporary;
use faer::prelude::ReborrowMut;
use faer::traits::math_utils;
use faer::{Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

pub(super) fn validate(
    a: &OwnedLinearOperator,
    m: usize,
    n: usize,
    tag: i32,
) -> Result<(), String> {
    let (rows, cols, actual_tag) = match a.data()? {
        Data::F32(a) => (a.nrows(), a.ncols(), 0),
        Data::F64(a) => (a.nrows(), a.ncols(), 1),
        Data::C32(a) => (a.nrows(), a.ncols(), 2),
        Data::C64(a) => (a.nrows(), a.ncols(), 3),
    };
    if (rows, cols, actual_tag) != (m, n, tag) {
        return Err("operator native dtype/dimensions do not match requested metadata".into());
    }
    Ok(())
}
fn requirement<T: ComplexField>(
    a: &Operator<T>,
    columns: usize,
    m: usize,
    n: usize,
    par: ParShape,
) -> Result<StackReq, String> {
    if (a.nrows(), a.ncols()) != (m, n) {
        return Err("operator native dimensions do not match requested metadata".into());
    }
    Ok(a.apply_scratch(columns, par))
}
fn query(
    a: &OwnedLinearOperator,
    columns: usize,
    m: usize,
    n: usize,
    tag: i32,
    par: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    let req = match (a.data()?, tag) {
        (Data::F32(a), 0) => requirement(a, columns, m, n, par)?,
        (Data::F64(a), 1) => requirement(a, columns, m, n, par)?,
        (Data::C32(a), 2) => requirement(a, columns, m, n, par)?,
        (Data::C64(a), 3) => requirement(a, columns, m, n, par)?,
        _ => return Err("operator native scalar does not match requested metadata".into()),
    };
    let address = out as usize;
    if out.is_null()
        || address % std::mem::align_of::<usize>() != 0
        || address
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err(
            "operator workspace query output must be an aligned two-entry usize buffer".into(),
        );
    }
    // SAFETY: Julia preserves the writable two-entry UInt buffer. Alignment,
    // nonnull address and representable span were checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
impl OwnedLinearOperator {
    #[julia]
    pub fn operator_requirements(
        &self,
        columns: usize,
        m: usize,
        n: usize,
        tag: i32,
        out: &mut [usize],
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| with_team(team, |p| query(self, columns, m, n, tag, p.shape(), out)))
    }
}
fn run<T: ComplexField>(
    a: &Operator<T>,
    b: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    m: usize,
    n: usize,
    columns: usize,
    par: Par<'_>,
) -> Result<Mat<T>, String> {
    if b.shape() != (n, columns) {
        return Err("operator RHS native dimensions do not match requested metadata".into());
    }
    let req = requirement(a, columns, m, n, par.shape())?;
    let stack = scratch.stack(req)?;
    // The owned result stages errors/aliasing and is outside the kernel query.
    let mut result = Mat::zeros(m, columns);
    a.apply(result.as_mut(), b, par, stack);
    Ok(result)
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn operator_workspace(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                self.operator_workspace_impl(operator, rhs, scratch, m, n, columns, tag, par)
            })
        })
    }
}
impl OwnedMatrix {
    fn operator_workspace_impl(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        par: Par<'_>,
    ) -> Result<(), String> {
        match (operator.data()?, tag) {
            (Data::F64(a), 1) => {
                self.matrix = run(a, rhs.matrix.as_ref(), scratch, m, n, columns, par)?;
                Ok(())
            }
            _ => Err("operator scalar types must match the requested scalar".into()),
        }
    }
}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn operator_workspace(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &TypedMatrix,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                self.operator_workspace_impl(operator, rhs, scratch, m, n, columns, tag, par)
            })
        })
    }
}
impl TypedMatrix {
    fn operator_workspace_impl(
        &mut self,
        operator: &OwnedLinearOperator,
        rhs: &TypedMatrix,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        par: Par<'_>,
    ) -> Result<(), String> {
        self.data = Some(match (operator.data()?, rhs.data()?, tag) {
            (Data::F32(a), MatrixData::F32(b), 0) => {
                MatrixData::F32(run(a, b.as_ref(), scratch, m, n, columns, par)?)
            }
            (Data::F64(a), MatrixData::F64(b), 1) => {
                MatrixData::F64(run(a, b.as_ref(), scratch, m, n, columns, par)?)
            }
            (Data::C32(a), MatrixData::C32(b), 2) => {
                MatrixData::C32(run(a, b.as_ref(), scratch, m, n, columns, par)?)
            }
            (Data::C64(a), MatrixData::C64(b), 3) => {
                MatrixData::C64(run(a, b.as_ref(), scratch, m, n, columns, par)?)
            }
            _ => return Err("operator/RHS scalar types must match the requested scalar".into()),
        });
        Ok(())
    }
}

fn run_scratch<T: ComplexField>(a: &Operator<T>, k: usize, par: ParShape, t: bool) -> StackReq {
    if t {
        faer::matrix_free::BiLinOp::transpose_apply_scratch(a, k, par)
    } else {
        a.apply_scratch(k, par)
    }
}
fn scale_into<T: ComplexField>(mut c: MatMut<'_, T>, s: &T, zero: bool) {
    for j in 0..c.ncols() {
        for i in 0..c.nrows() {
            c[(i, j)] = if zero {
                math_utils::zero::<T>()
            } else {
                math_utils::mul(s, &c[(i, j)])
            };
        }
    }
}
#[allow(clippy::too_many_arguments)]
unsafe fn mul_into<T: Arm>(
    a: &Operator<T>,
    dest: *mut u8,
    ds: Spec,
    src: *const u8,
    ss: Spec,
    coefficients: (*const u8, *const u8, bool, bool),
    transpose: bool,
    work: Option<&mut OwnedScratch>,
    par: Par<'_>,
) -> Result<(), String> {
    let (rows, cols) = spec_shape(ds);
    let (m, n) = if transpose {
        (a.ncols(), a.nrows())
    } else {
        (a.nrows(), a.ncols())
    };
    if spec_shape(ss) != (n, cols) || rows != m {
        return Err("operator product dimensions must match".into());
    }
    let (alpha, beta, alpha_zero, beta_zero) = coefficients;
    let alpha = math_utils::copy(&*alpha.cast::<T>());
    let beta = math_utils::copy(&*beta.cast::<T>());
    let req = run_scratch(a, cols, par.shape(), transpose);
    scratch_or_temporary(work, req, |work| {
        let x = operand::<T>(src, ss, destination_span(dest, ds)?)?;
        let conjugated;
        let rhs = match x.get() {
            (b, false) => b,
            (b, true) => {
                conjugated = b.conjugate().to_owned();
                conjugated.as_ref()
            }
        };
        let mut c =
            T::pick_mut(destination(dest, ds)?).ok_or("destination scalar types must match")?;
        let stack = work.stack(req)?;
        if alpha_zero {
            scale_into(c, &beta, beta_zero);
            return Ok(());
        }
        if beta_zero {
            a.run(c.rb_mut(), rhs, par, stack, transpose, false);
            if alpha != math_utils::one::<T>() {
                scale_into(c, &alpha, false);
            }
            return Ok(());
        }
        let mut product = Mat::<T>::zeros(rows, cols);
        a.run(product.as_mut(), rhs, par, stack, transpose, false);
        for j in 0..cols {
            for i in 0..rows {
                let y = math_utils::add(
                    &math_utils::mul(&alpha, &product[(i, j)]),
                    &math_utils::mul(&beta, &c[(i, j)]),
                );
                c[(i, j)] = y;
            }
        }
        Ok(())
    })
}

/// `dest = alpha*op*src + beta*dest`, or with `transpose` the transposed
/// action `alpha*op^T*src + beta*dest` (Julia writes right products
/// `Y = alpha*X*op + beta*Y` as `Y^T = alpha*op^T*X^T + beta*Y^T` through
/// transposed descriptors). The native operator writes its `MatMut`
/// output straight into the destination when `beta_zero` (LinOp outputs
/// are replaced, never read); otherwise one native product buffer of the
/// destination's shape holds the action before the axpby. A source that
/// overlaps the destination is snapshot first. `alpha_zero` never reads
/// `src` and does not run the operator. Scratch capacity and shapes are
/// checked before the destination is written.
///
/// # Safety
/// Julia roots the destination, source, descriptors and coefficients.
#[allow(clippy::too_many_arguments)]
#[julia]
pub fn operator_mul_into(
    dest: *mut u8,
    dd: *const isize,
    src: *const u8,
    sd: *const isize,
    operator: &OwnedLinearOperator,
    transpose: bool,
    alpha: *const u8,
    beta: *const u8,
    alpha_zero: bool,
    beta_zero: bool,
    work: Option<&mut OwnedScratch>,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| unsafe {
        let (ds, ss) = (Spec::read(dd)?, Spec::read(sd)?);
        let k = (alpha, beta, alpha_zero, beta_zero);
        let t = transpose;
        with_team(team, |par| match (operator.data()?, source_view(dest, ds)?) {
            (Data::F32(a), Borrowed::F32(_)) => mul_into(a, dest, ds, src, ss, k, t, work, par),
            (Data::F64(a), Borrowed::F64(_)) => mul_into(a, dest, ds, src, ss, k, t, work, par),
            (Data::C32(a), Borrowed::C32(_)) => mul_into(a, dest, ds, src, ss, k, t, work, par),
            (Data::C64(a), Borrowed::C64(_)) => mul_into(a, dest, ds, src, ss, k, t, work, par),
            _ => Err("operator and destination scalar types must match".into()),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::{Base, InverseFactors};
    use super::*;
    #[test]
    fn query_rejects_bad_metadata_and_output_addresses() {
        let op = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::Dense(Mat::identity(2, 2))))),
        };
        assert!(op
            .operator_requirements(1, 2, 2, 0, &mut [], None)
            .is_err());
        assert!(op
            .operator_requirements(1, 3, 2, 1, &mut [], None)
            .is_err());
        assert!(op
            .operator_requirements(1, 2, 2, 1, &mut [], None)
            .is_err());
        // Slices carry their alignment; a short output is rejected by length.
        assert!(op
            .operator_requirements(1, 2, 2, 1, &mut [0usize; 1], None)
            .is_err());
        assert!(op
            .operator_requirements(1, 2, 2, 1, &mut [0usize; 3], None)
            .is_err());
    }
    #[test]
    fn workspace_and_team_rejection_preserve_previous_output() {
        let a = Mat::from_fn(2, 2, |i, j| if i == j { 4.0 } else { 1.0 });
        let lu = crate::lu_qr_factors::PartialPivLu::new(a.as_ref(), Par::Seq);
        let op = OwnedLinearOperator {
            data: Some(Data::F64(Operator::new(Base::InverseFactors(Box::new(
                InverseFactors::new(&lu).unwrap(),
            ))))),
        };
        let rhs = OwnedMatrix {
            matrix: Mat::from_fn(2, 2, |_, _| 1.0),
        };
        let mut out = OwnedMatrix {
            matrix: Mat::from_fn(2, 2, |_, _| 42.0),
        };
        let mut info = [0usize; 2];
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        op.operator_requirements(2, 2, 2, 1, &mut info[..], Some(&team))
            .unwrap();
        assert!(info[0] > 0);
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0] - 1, info[1]).unwrap();
        assert!(out
            .operator_workspace(&op, &rhs, &mut scratch, 2, 2, 2, 1, Some(&team))
            .is_err());
        assert_eq!(out.matrix[(0, 0)], 42.0);
        scratch.scratch_reserve(info[0], info[1]).unwrap();
        assert!(out
            .operator_workspace(&op, &rhs, &mut scratch, 2, 2, 3, 1, Some(&team))
            .is_err());
        assert_eq!(out.matrix[(0, 0)], 42.0);
        out.operator_workspace(&op, &rhs, &mut scratch, 2, 2, 2, 1, Some(&team))
            .unwrap();
        assert!((&a * &out.matrix - &rhs.matrix).norm_max() < 1e-12);
        let saved = out.matrix.clone();
        team.team_close().unwrap();
        assert!(out
            .operator_workspace(&op, &rhs, &mut scratch, 2, 2, 2, 1, Some(&team))
            .is_err());
        assert_eq!(out.matrix[(0, 0)], saved[(0, 0)]);
    }
}
