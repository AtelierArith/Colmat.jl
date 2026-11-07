//! Declared kernel workspace for right eigenvectors of saved Schur factors.
use super::construction::buffer;
use super::{validate, SchurScalar};
use crate::teams::with_team;
use crate::{
    guarded,
    teams::OwnedTeam,
    typed::{MatrixData, TypedMatrix},
    workspace::OwnedScratch,
    OwnedMatrix,
};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{self, evd};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::ComplexField;
use faer::{Mat, MatRef, Par};
use rustcall_julia_macros::julia;
trait SavedEigen: SchurScalar {
    type Complex: ComplexField<Real = Self::Real>;
    fn requirement(n: usize) -> StackReq;
    fn compute(
        t: MatRef<'_, Self>,
        q: MatRef<'_, Self>,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<Mat<Self::Complex>, String>;
}
macro_rules! real {
    ($r:ty,$c:ty) => {
        impl SavedEigen for $r {
            type Complex = $c;
            fn requirement(n: usize) -> StackReq {
                if n == 0 {
                    StackReq::EMPTY
                } else {
                    linalg::temp_mat_scratch::<Self>(n, n)
                        .and(evd::eigenvectors_from_schur_scratch::<Self>(n))
                }
            }
            fn compute(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<Mat<$c>, String> {
                let n = t.nrows();
                if n == 0 {
                    return Ok(Mat::zeros(0, 0));
                }
                let (mut x, stack) = linalg::temp_mat_zeroed::<Self, _, _>(n, n, stack);
                evd::eigenvectors_from_schur_real(
                    t,
                    Some(q),
                    x.as_mat_mut(),
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("{e:?}"))?;
                let x = x.as_mat_ref();
                let mut out = Mat::<$c>::zeros(n, n);
                let mut j = 0;
                while j < n {
                    if Self::block_size(t, j) == 2 {
                        for i in 0..n {
                            out[(i, j)] = <$c>::new(x[(i, j)], x[(i, j + 1)]);
                            out[(i, j + 1)] = <$c>::new(x[(i, j)], -x[(i, j + 1)]);
                        }
                        j += 2;
                    } else {
                        for i in 0..n {
                            out[(i, j)] = <$c>::new(x[(i, j)], 0.0);
                        }
                        j += 1;
                    }
                }
                Ok(out)
            }
        }
    };
}
macro_rules! complex {
    ($c:ty) => {
        impl SavedEigen for $c {
            type Complex = $c;
            fn requirement(n: usize) -> StackReq {
                if n == 0 {
                    StackReq::EMPTY
                } else {
                    evd::eigenvectors_from_schur_scratch::<Self>(n)
                }
            }
            fn compute(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<Mat<Self>, String> {
                let n = t.nrows();
                let mut out = Mat::<Self>::zeros(n, n);
                if n > 0 {
                    evd::eigenvectors_from_schur_cplx(
                        t,
                        Some(q),
                        out.as_mut(),
                        par,
                        stack,
                        Default::default(),
                    )
                    .map_err(|e| format!("{e:?}"))?;
                }
                Ok(out)
            }
        }
    };
}
real!(f32, faer::c32);
real!(f64, faer::c64);
complex!(faer::c32);
complex!(faer::c64);
fn query(n: usize, tag: i32, out: *mut usize) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = match tag {
        0 => f32::requirement(n),
        1 => f64::requirement(n),
        2 => faer::c32::requirement(n),
        3 => faer::c64::requirement(n),
        _ => return Err("Schur eigenvectors scalar tag is invalid".into()),
    };
    // SAFETY: preserved Julia UInt[2] output, checked for span/alignment above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn schur_eigenvectors_requirements(
    n: usize,
    tag: i32,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| with_team(team, |_| query(n, tag, out)))
}

fn run<T: SavedEigen>(
    t: MatRef<'_, T>,
    q: MatRef<'_, T>,
    work: &mut OwnedScratch,
    n: usize,
    par: Par,
) -> Result<Mat<T::Complex>, String> {
    if t.shape() != (n, n) || q.shape() != (n, n) {
        return Err("Schur current native dimensions do not match requested metadata".into());
    }
    let stack = work.stack(T::requirement(n))?;
    validate(t, q)?;
    T::compute(t, q, par, stack)
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_eigenvectors_owned_workspace(
        &mut self,
        t: &OwnedMatrix,
        q: &OwnedMatrix,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("Schur requested scalar type does not match its native owner".into());
            }
            let execute = |par: Par<'_>| run(t.matrix.as_ref(), q.matrix.as_ref(), work, n, par);
            let out = with_team(team, execute)?;
            self.data = Some(MatrixData::C64(out));
            Ok(())
        })
    }

    #[julia]
    pub fn schur_eigenvectors_typed_workspace(
        &mut self,
        t: &TypedMatrix,
        q: &TypedMatrix,
        work: &mut OwnedScratch,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($t:expr,$q:expr,$out:ident) => {{
                    let execute = |par: Par<'_>| run($t.as_ref(), $q.as_ref(), work, n, par);
                    let out = with_team(team, execute)?;
                    self.data = Some(MatrixData::$out(out));
                }};
            }
            match (t.data()?, q.data()?) {
                (MatrixData::F32(t), MatrixData::F32(q)) if tag == 0 => one!(t, q, C32),
                (MatrixData::F64(t), MatrixData::F64(q)) if tag == 1 => one!(t, q, C64),
                (MatrixData::C32(t), MatrixData::C32(q)) if tag == 2 => one!(t, q, C32),
                (MatrixData::C64(t), MatrixData::C64(q)) if tag == 3 => one!(t, q, C64),
                _ => return Err("Schur scalar types must match the requested scalar type".into()),
            }
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::traits::math_utils::*;
    use faer::{c32, c64};
    #[test]
    fn native_preflight_preserves_existing_output() {
        let t = OwnedMatrix {
            matrix: Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 + i as f64 } else { 0.0 }),
        };
        let q = OwnedMatrix {
            matrix: Mat::<f64>::identity(2, 2),
        };
        let mut out = TypedMatrix {
            data: Some(MatrixData::C64(Mat::<c64>::full(2, 2, c64::new(42.0, 0.0)))),
        };
        let req = f64::requirement(2);
        let mut w = OwnedScratch::new();
        w.scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        assert!(out
            .schur_eigenvectors_owned_workspace(&t, &q, &mut w, 2, 1, None)
            .is_err());
        w.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        assert!(out
            .schur_eigenvectors_owned_workspace(&t, &q, &mut w, 3, 1, None)
            .is_err());
        assert!(out
            .schur_eigenvectors_owned_workspace(&t, &q, &mut w, 2, 0, None)
            .is_err());
        let team = OwnedTeam::new();
        assert!(out
            .schur_eigenvectors_owned_workspace(&t, &q, &mut w, 2, 1, Some(&team))
            .is_err());
        w.scratch_close().unwrap();
        assert!(out
            .schur_eigenvectors_owned_workspace(&t, &q, &mut w, 2, 1, None)
            .is_err());
        match out.data().unwrap() {
            MatrixData::C64(m) => assert_eq!(m, &Mat::<c64>::full(2, 2, c64::new(42.0, 0.0))),
            _ => panic!("output scalar changed"),
        }
        assert!(schur_eigenvectors_requirements(2, 1, &mut [], None).is_err());
    }
    fn allocation_probe<T: SavedEigen>(mut par: Par) {
        let n = 6;
        let base = Mat::<T>::from_fn(n, n, |i, j| {
            from_f64::<T>(if i == j {
                1.0 + i as f64
            } else if i < j {
                0.03
            } else {
                0.0
            })
        });
        let q = Mat::<T>::identity(n, n);
        let base_c = Mat::<T::Complex>::from_fn(n, n, |i, j| {
            from_f64::<T::Complex>(if i == j {
                1.0 + i as f64
            } else if i < j {
                0.03
            } else {
                0.0
            })
        });
        let range = if core::mem::size_of::<T::Real>() == 4 {
            [1.0, 1e30, 1e-30]
        } else {
            [1.0, 1e300, 1e-300]
        };
        let mut mem = faer::dyn_stack::MemBuffer::new(T::requirement(n));
        for (case, s) in range.into_iter().enumerate() {
            let s = from_f64::<T::Real>(s);
            let h = Mat::<T>::from_fn(n, n, |i, j| mul_real(&base[(i, j)], &s));
            for rep in 0..3 {
                let mut out = None;
                let mut compute = || {
                    out = Some(
                        T::compute(
                            h.as_ref(),
                            q.as_ref(),
                            par.reborrow(),
                            MemStack::new(&mut mem),
                        )
                        .unwrap(),
                    );
                };
                if rep == 0 {
                    compute();
                } else {
                    // One owning output remains. Selected faer adds one internal scaled
                    // Schur copy for extreme cases (#270); declared scratch cannot cover it.
                    assert_eq!(
                        crate::allocation_probe::count(compute),
                        if case == 0 { 1 } else { 2 }
                    );
                }
                let x = out.unwrap();
                let diagonal = Mat::<T::Complex>::from_fn(n, n, |i, j| {
                    from_f64::<T::Complex>(if i == j { (i + 1) as f64 } else { 0.0 })
                });
                let residual = &base_c * &x - &x * &diagonal;
                assert!(
                    residual.norm_l2()
                        <= eps::<T::Real>()
                            * from_f64::<T::Real>(256.0 * n as f64)
                            * base_c.norm_l2()
                );
                for j in 0..n {
                    assert!(
                        abs(&sub(&x.col(j).norm_l2(), &one::<T::Real>()))
                            < eps::<T::Real>() * from_f64::<T::Real>(256.0)
                    );
                }
            }
        }
    }
    #[test]
    fn declared_workspace_excludes_only_output_and_reported_scaled_copy() {
        allocation_probe::<f32>(Par::Seq);
        allocation_probe::<f64>(Par::Seq);
        allocation_probe::<c32>(Par::Seq);
        allocation_probe::<c64>(Par::Seq);
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.execute(|mut p| {
            allocation_probe::<f32>(p.reborrow());
            allocation_probe::<f64>(p.reborrow());
            allocation_probe::<c32>(p.reborrow());
            allocation_probe::<c64>(p);
        })
        .unwrap();
    }
}
