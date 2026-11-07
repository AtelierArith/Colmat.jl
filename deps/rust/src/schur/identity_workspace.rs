//! QR iteration on a saved Hessenberg matrix with an owned identity initial Z.
//! Separate entrypoints keep the legacy stage0/1/2 contract unchanged.
use super::{construction, reduced_with_initial, SchurScalar};
use crate::teams::with_team;
use crate::{
    guarded,
    teams::OwnedTeam,
    typed::{MatrixData, TypedMatrix},
    workspace::OwnedScratch,
    OwnedMatrix,
};
use faer::{Mat, MatRef, Par};
use rustcall_julia_macros::julia;
fn run<T: SchurScalar>(
    a: MatRef<'_, T>,
    work: &mut OwnedScratch,
    values: *mut u8,
    n: usize,
    par: Par,
) -> Result<(Mat<T>, Mat<T>), String> {
    if a.shape() != (n, n) {
        return Err("saved Hessenberg native dimensions mismatch".into());
    }
    construction::buffer::<T::Value>(values, n)?;
    let stack = work.stack(T::step_scratch(n, par.shape()))?;
    let (h, z) = reduced_with_initial(a, None, par, stack)?;
    // SAFETY: caller preserves n eigenvalue entries after scalar/span checks.
    unsafe {
        T::values(h.as_ref(), values);
    }
    Ok((h, z))
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn schur_identity_workspace(
        &mut self,
        a: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let execute = |par: Par| -> Result<(Mat<f64>, Mat<f64>), String> {
                if tag != 1 {
                    return Err("Schur requested scalar does not match native owner".into());
                }
                let (h, z) = run(a.matrix.as_ref(), work, values, n, par)?;
                Ok((h, z))
            };
            let (h, z) = with_team(team, execute)?;
            self.matrix = h;
            out_q.matrix = z;
            Ok(())
        })
    }
}

#[julia]
impl OwnedMatrix {}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_identity_workspace(
        &mut self,
        a: &TypedMatrix,
        out_q: &mut TypedMatrix,
        work: &mut OwnedScratch,
        values: *mut u8,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            let execute = |par: Par| match (a.data()?, tag) {
                (MatrixData::F32(a), 0) => {
                    let (h, z) = run::<f32>(a.as_ref(), work, values, n, par)?;
                    Ok::<_, String>((MatrixData::F32(h), MatrixData::F32(z)))
                }
                (MatrixData::F64(a), 1) => {
                    let (h, z) = run::<f64>(a.as_ref(), work, values, n, par)?;
                    Ok::<_, String>((MatrixData::F64(h), MatrixData::F64(z)))
                }
                (MatrixData::C32(a), 2) => {
                    let (h, z) = run::<faer::c32>(a.as_ref(), work, values, n, par)?;
                    Ok::<_, String>((MatrixData::C32(h), MatrixData::C32(z)))
                }
                (MatrixData::C64(a), 3) => {
                    let (h, z) = run::<faer::c64>(a.as_ref(), work, values, n, par)?;
                    Ok::<_, String>((MatrixData::C64(h), MatrixData::C64(z)))
                }
                _ => Err("Schur requested scalar does not match native owner".into()),
            };
            let (h, z) = with_team(team, execute)?;
            self.data = Some(h);
            out_q.data = Some(z);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{c32, c64, traits::math_utils::*};

    fn check<T: SchurScalar>(n: usize, tolerance: f64)
    where
        T::Value: faer::traits::ComplexField,
    {
        let a = Mat::from_fn(n, n, |i, j| {
            if i > j + 1 {
                zero::<T>()
            } else {
                from_f64::<T>(if i == j { i as f64 + 2.0 } else { 0.15 })
            }
        });
        let mut work = OwnedScratch::new();
        let req = T::step_scratch(n, faer::ParShape::Seq);
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let mut values: Vec<T::Value> = (0..n).map(|_| zero()).collect();
        // run checks the scalar output span before computing and publishes
        // values only after the entire saved-Hessenberg step succeeds.
        let (h, z) = run(
            a.as_ref(),
            &mut work,
            values.as_mut_ptr().cast(),
            n,
            Par::Seq,
        )
        .unwrap();
        assert!((&z * &h * z.adjoint() - &a).norm_l2() <= from_f64::<T::Real>(tolerance));
        assert!(values.iter().all(is_finite));
    }

    #[test]
    fn identity_initial_schur_four_scalars_and_empty_boundaries() {
        for n in [0, 1, 2, 7] {
            check::<f32>(n, 1e-3);
            check::<f64>(n, 1e-11);
            check::<c32>(n, 1e-3);
            check::<c64>(n, 1e-11);
        }
    }

    #[test]
    fn identity_initial_schur_failure_preserves_outputs() {
        let mut source = OwnedMatrix::new();
        source.matrix = Mat::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.1 });
        let mut out = OwnedMatrix::new();
        out.matrix = Mat::full(3, 3, 71.0);
        let mut q = OwnedMatrix::new();
        q.matrix = Mat::full(3, 3, 72.0);
        let mut values = [c64::new(73.0, 0.0); 3];
        let mut work = OwnedScratch::new();
        let req = <f64 as SchurScalar>::step_scratch(3, faer::ParShape::Seq);
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // Invalid lower bandwidth is rejected after all output capacities are
        // checked, without replacing either matrix or publishing eigenvalues.
        assert!(out
            .schur_identity_workspace(
                &source,
                &mut q,
                &mut work,
                values.as_mut_ptr().cast(),
                3,
                1,
                None
            )
            .is_err());
        assert_eq!(out.matrix, Mat::full(3, 3, 71.0));
        assert_eq!(q.matrix, Mat::full(3, 3, 72.0));
        assert_eq!(values, [c64::new(73.0, 0.0); 3]);
        source.matrix[(2, 0)] = 0.0;
        for (n, tag) in [(2, 1), (3, 0)] {
            assert!(out
                .schur_identity_workspace(
                    &source,
                    &mut q,
                    &mut work,
                    values.as_mut_ptr().cast(),
                    n,
                    tag,
                    None
                )
                .is_err());
            assert_eq!(out.matrix, Mat::full(3, 3, 71.0));
            assert_eq!(q.matrix, Mat::full(3, 3, 72.0));
            assert_eq!(values, [c64::new(73.0, 0.0); 3]);
        }
        source.matrix[(0, 0)] = f64::NAN;
        assert!(out
            .schur_identity_workspace(
                &source,
                &mut q,
                &mut work,
                values.as_mut_ptr().cast(),
                3,
                1,
                None
            )
            .is_err());
        assert_eq!(values, [c64::new(73.0, 0.0); 3]);
    }
}

#[julia]
impl TypedMatrix {}
