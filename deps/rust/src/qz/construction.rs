//! Declared caller workspace for generalized Hessenberg and QZ construction.
use super::{compute_in, Pair, QzScalar};
use crate::sparsity::{Pattern, SparsityOwner};
use crate::teams::OwnedTeam;
use crate::typed::{MatrixData, TypedMatrix};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::{gevd, householder as hh, qr::no_pivoting::factor as qr};
use faer::{MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
pub(super) fn requirement<T: QzScalar>(
    n: usize,
    stage: i32,
    par: ParShape,
) -> Result<StackReq, String> {
    if !(0..=2).contains(&stage) {
        return Err("QZ construction stage must be 0, 1 or 2".into());
    }
    if n == 0 {
        return Ok(StackReq::EMPTY);
    }
    let preparation = if stage == 2 {
        StackReq::EMPTY
    } else {
        let b = qr::recommended_block_size::<T>(n, n);
        faer::linalg::temp_mat_scratch::<T>(b, n)
            .and(
                qr::qr_in_place_scratch::<T>(n, n, b, par, Default::default()).or(
                    hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(n, b, n),
                ),
            )
            .or(faer::linalg::temp_mat_scratch::<T>(n, n))
            .or(gevd::gen_hessenberg::generalized_hessenberg_scratch::<T>(
                n,
                faer::auto!(T),
            ))
    };
    Ok(if stage == 1 {
        preparation
    } else {
        preparation.or(T::qz_scratch(n, par))
    })
}
fn query(n: usize, tag: i32, stage: i32, par: ParShape, out: *mut usize) -> Result<(), String> {
    super::reorder_buffer::<usize>(out.cast(), 2)?;
    let req = match tag {
        0 => requirement::<f32>(n, stage, par)?,
        1 => requirement::<f64>(n, stage, par)?,
        2 => requirement::<faer::c32>(n, stage, par)?,
        3 => requirement::<faer::c64>(n, stage, par)?,
        _ => return Err("QZ scalar tag is invalid".into()),
    };
    // SAFETY: the preserved Julia output has two writable UInt entries; span/alignment checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn qz_construction_requirements(
    n: usize,
    tag: i32,
    stage: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| query(n, tag, stage, ParShape::Seq, out))
}
#[julia]
pub fn qz_construction_team_requirements(
    team: &OwnedTeam,
    n: usize,
    tag: i32,
    stage: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| team.execute(|par| query(n, tag, stage, par.shape(), out))?)
}
#[julia]
impl SparsityOwner {
    #[julia]
    pub fn qz_pattern_validate(
        &self,
        m: usize,
        n: usize,
        csr: bool,
        stored: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let actual = match self.data()? {
                Pattern::Csc(a) => (a.nrows(), a.ncols(), false, a.compute_nnz()),
                Pattern::Csr(a) => (a.nrows(), a.ncols(), true, a.compute_nnz()),
            };
            if actual != (m, n, csr, stored) {
                return Err("QZ pattern input native metadata mismatch".into());
            }
            Ok(())
        })
    }
}
fn run<T: QzScalar>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    q: MatRef<'_, T>,
    z: MatRef<'_, T>,
    work: &mut OwnedScratch,
    alpha: *mut u8,
    beta: *mut u8,
    stage: i32,
    n: usize,
    par: Par,
) -> Result<Pair<T>, String> {
    if a.shape() != (n, n)
        || b.shape() != (n, n)
        || (stage == 2 && (q.shape() != (n, n) || z.shape() != (n, n)))
    {
        return Err("QZ native shapes do not match the requested metadata".into());
    }
    let stack = work.stack(requirement::<T>(n, stage, par.shape())?)?;
    // SAFETY: Julia preserves homogeneous arrays. compute_in checks their scalar
    // alignment, lengths/address spans and non-overlap before writing; no pointer escapes.
    unsafe {
        compute_in(
            a,
            b,
            if stage == 2 { Some((q, z)) } else { None },
            alpha,
            beta,
            stage == 1,
            n,
            par,
            stack,
        )
    }
}
fn validate_metadata(
    actual: (usize, usize, i32),
    expected: (usize, usize, i32),
) -> Result<(), String> {
    if actual.2 != expected.2 {
        return Err("QZ requested scalar type does not match its native owner".into());
    }
    if (actual.0, actual.1) != (expected.0, expected.1) {
        return Err("QZ current native dimensions do not match requested output".into());
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn qz_input_validate(&self, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| validate_metadata((self.matrix.nrows(), self.matrix.ncols(), 1), (m, n, tag)))
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn qz_input_validate(&self, m: usize, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            let actual = match self.data()? {
                MatrixData::F32(a) => (a.nrows(), a.ncols(), 0),
                MatrixData::F64(a) => (a.nrows(), a.ncols(), 1),
                MatrixData::C32(a) => (a.nrows(), a.ncols(), 2),
                MatrixData::C64(a) => (a.nrows(), a.ncols(), 3),
            };
            validate_metadata(actual, (m, n, tag))
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn qz_construction_workspace(
        &mut self,
        a: &OwnedMatrix,
        b: &OwnedMatrix,
        iq: &OwnedMatrix,
        iz: &OwnedMatrix,
        out_t: &mut OwnedMatrix,
        out_q: &mut OwnedMatrix,
        out_z: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        alpha: *mut u8,
        beta: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("QZ native scalar does not match requested metadata".into());
            }
            let mut execute = |par: Par<'_>| {
                run(
                    a.matrix.as_ref(),
                    b.matrix.as_ref(),
                    iq.matrix.as_ref(),
                    iz.matrix.as_ref(),
                    work,
                    alpha,
                    beta,
                    stage,
                    n,
                    par,
                )
            };
            let p = execute(Par::Seq)?;
            self.matrix = p.s;
            out_t.matrix = p.t;
            out_q.matrix = p.q;
            out_z.matrix = p.z;
            Ok(())
        })
    }
    #[julia]
    pub fn qz_construction_workspace_team(
        &mut self,
        a: &OwnedMatrix,
        b: &OwnedMatrix,
        iq: &OwnedMatrix,
        iz: &OwnedMatrix,
        out_t: &mut OwnedMatrix,
        out_q: &mut OwnedMatrix,
        out_z: &mut OwnedMatrix,
        work: &mut OwnedScratch,
        alpha: *mut u8,
        beta: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("QZ native scalar does not match requested metadata".into());
            }
            let execute = |par: Par<'_>| {
                run(
                    a.matrix.as_ref(),
                    b.matrix.as_ref(),
                    iq.matrix.as_ref(),
                    iz.matrix.as_ref(),
                    work,
                    alpha,
                    beta,
                    stage,
                    n,
                    par,
                )
            };
            let p = team.execute(execute)??;
            self.matrix = p.s;
            out_t.matrix = p.t;
            out_q.matrix = p.q;
            out_z.matrix = p.z;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn qz_construction_workspace(
        &mut self,
        a: &TypedMatrix,
        b: &TypedMatrix,
        iq: &TypedMatrix,
        iz: &TypedMatrix,
        out_t: &mut TypedMatrix,
        out_q: &mut TypedMatrix,
        out_z: &mut TypedMatrix,
        work: &mut OwnedScratch,
        alpha: *mut u8,
        beta: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($a:expr,$b:expr,$q:expr,$z:expr,$v:ident) => {{
                    let mut execute = |par: Par<'_>| {
                        run(
                            $a.as_ref(),
                            $b.as_ref(),
                            $q.as_ref(),
                            $z.as_ref(),
                            work,
                            alpha,
                            beta,
                            stage,
                            n,
                            par,
                        )
                    };
                    let p = execute(Par::Seq)?;
                    out_t.data = Some(MatrixData::$v(p.t));
                    out_q.data = Some(MatrixData::$v(p.q));
                    out_z.data = Some(MatrixData::$v(p.z));
                    MatrixData::$v(p.s)
                }};
            }
            self.data = Some(match (a.data()?, b.data()?, iq.data()?, iz.data()?) {
                (
                    MatrixData::F32(a),
                    MatrixData::F32(b),
                    MatrixData::F32(q),
                    MatrixData::F32(z),
                ) if tag == 0 => one!(a, b, q, z, F32),
                (
                    MatrixData::F64(a),
                    MatrixData::F64(b),
                    MatrixData::F64(q),
                    MatrixData::F64(z),
                ) if tag == 1 => one!(a, b, q, z, F64),
                (
                    MatrixData::C32(a),
                    MatrixData::C32(b),
                    MatrixData::C32(q),
                    MatrixData::C32(z),
                ) if tag == 2 => one!(a, b, q, z, C32),
                (
                    MatrixData::C64(a),
                    MatrixData::C64(b),
                    MatrixData::C64(q),
                    MatrixData::C64(z),
                ) if tag == 3 => one!(a, b, q, z, C64),
                _ => return Err("QZ native scalar does not match requested metadata".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn qz_construction_workspace_team(
        &mut self,
        a: &TypedMatrix,
        b: &TypedMatrix,
        iq: &TypedMatrix,
        iz: &TypedMatrix,
        out_t: &mut TypedMatrix,
        out_q: &mut TypedMatrix,
        out_z: &mut TypedMatrix,
        work: &mut OwnedScratch,
        alpha: *mut u8,
        beta: *mut u8,
        stage: i32,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! one {
                ($a:expr,$b:expr,$q:expr,$z:expr,$v:ident) => {{
                    let execute = |par: Par<'_>| {
                        run(
                            $a.as_ref(),
                            $b.as_ref(),
                            $q.as_ref(),
                            $z.as_ref(),
                            work,
                            alpha,
                            beta,
                            stage,
                            n,
                            par,
                        )
                    };
                    let p = team.execute(execute)??;
                    out_t.data = Some(MatrixData::$v(p.t));
                    out_q.data = Some(MatrixData::$v(p.q));
                    out_z.data = Some(MatrixData::$v(p.z));
                    MatrixData::$v(p.s)
                }};
            }
            self.data = Some(match (a.data()?, b.data()?, iq.data()?, iz.data()?) {
                (
                    MatrixData::F32(a),
                    MatrixData::F32(b),
                    MatrixData::F32(q),
                    MatrixData::F32(z),
                ) if tag == 0 => one!(a, b, q, z, F32),
                (
                    MatrixData::F64(a),
                    MatrixData::F64(b),
                    MatrixData::F64(q),
                    MatrixData::F64(z),
                ) if tag == 1 => one!(a, b, q, z, F64),
                (
                    MatrixData::C32(a),
                    MatrixData::C32(b),
                    MatrixData::C32(q),
                    MatrixData::C32(z),
                ) if tag == 2 => one!(a, b, q, z, C32),
                (
                    MatrixData::C64(a),
                    MatrixData::C64(b),
                    MatrixData::C64(q),
                    MatrixData::C64(z),
                ) if tag == 3 => one!(a, b, q, z, C64),
                _ => return Err("QZ native scalar does not match requested metadata".into()),
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::traits::math_utils::*;
    use faer::{c32, c64, Mat};
    #[test]
    fn native_preflight_keeps_owner_and_value_outputs() {
        let a = OwnedMatrix {
            matrix: Mat::<f64>::identity(2, 2),
        };
        let b = OwnedMatrix {
            matrix: Mat::<f64>::identity(2, 2),
        };
        let mut s = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 42.0),
        };
        let mut t = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 43.0),
        };
        let mut q = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 44.0),
        };
        let mut z = OwnedMatrix {
            matrix: Mat::<f64>::full(2, 2, 45.0),
        };
        let mut alpha = [c64::new(41.0, 0.0); 2];
        let mut beta = [40.0; 2];
        let req = requirement::<f64>(2, 0, ParShape::Seq).unwrap();
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req.size_bytes() - 1, req.align_bytes())
            .unwrap();
        assert!(s
            .qz_construction_workspace(
                &a,
                &b,
                &a,
                &a,
                &mut t,
                &mut q,
                &mut z,
                &mut work,
                alpha.as_mut_ptr().cast(),
                beta.as_mut_ptr().cast(),
                0,
                2,
                1
            )
            .is_err());
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        assert!(s
            .qz_construction_workspace(
                &a,
                &b,
                &a,
                &a,
                &mut t,
                &mut q,
                &mut z,
                &mut work,
                alpha.as_mut_ptr().cast(),
                beta.as_mut_ptr().cast(),
                0,
                3,
                1
            )
            .is_err());
        assert!(s
            .qz_construction_workspace(
                &a,
                &b,
                &a,
                &a,
                &mut t,
                &mut q,
                &mut z,
                &mut work,
                alpha.as_mut_ptr().cast(),
                beta.as_mut_ptr().cast(),
                0,
                2,
                0
            )
            .is_err());
        let team = OwnedTeam::new();
        assert!(s
            .qz_construction_workspace_team(
                &a,
                &b,
                &a,
                &a,
                &mut t,
                &mut q,
                &mut z,
                &mut work,
                alpha.as_mut_ptr().cast(),
                beta.as_mut_ptr().cast(),
                0,
                2,
                1,
                &team
            )
            .is_err());
        assert_eq!(s.matrix[(0, 0)], 42.0);
        assert_eq!(t.matrix[(0, 0)], 43.0);
        assert_eq!(q.matrix[(0, 0)], 44.0);
        assert_eq!(z.matrix[(0, 0)], 45.0);
        assert_eq!(alpha, [c64::new(41.0, 0.0); 2]);
        assert_eq!(beta, [40.0; 2]);
        assert!(qz_construction_requirements(2, 1, 0, std::ptr::null_mut()).is_err());
    }
    fn qz_step_alloc<T: QzScalar>(mut par: Par)
    where
        T::Alpha: faer::traits::ComplexField,
    {
        let n = 6;
        let a = Mat::<T>::from_fn(n, n, |i, j| {
            from_f64::<T>(if i == j { 2.0 + i as f64 } else { 0.01 })
        });
        let b = Mat::<T>::from_fn(n, n, |i, j| from_f64::<T>(if i == j { 2.0 } else { 0.002 }));
        let req = requirement::<T>(n, 0, par.shape()).unwrap();
        let mut memory = faer::dyn_stack::MemBuffer::new(req);
        let mut alpha = vec![zero::<T::Alpha>(); n];
        let mut beta = vec![zero::<T>(); n];
        for rep in 0..3 {
            let mut p = crate::qz::prepare(
                a.as_ref(),
                b.as_ref(),
                par.reborrow(),
                faer::dyn_stack::MemStack::new(&mut memory),
            )
            .unwrap();
            let mut step = || {
                // SAFETY: vectors hold n writable, nonoverlapping correctly typed entries.
                unsafe {
                    T::qz(
                        &mut p,
                        alpha.as_mut_ptr().cast(),
                        beta.as_mut_ptr().cast(),
                        par.reborrow(),
                        faer::dyn_stack::MemStack::new(&mut memory),
                    )
                    .unwrap();
                }
            };
            if rep == 0 {
                step();
            } else {
                assert_eq!(crate::allocation_probe::count(step), 0);
            }
            let err = (&p.q * &p.s * p.z.adjoint() - &a).norm_l2();
            let bound = eps::<T::Real>() * from_f64::<T::Real>(256.0) * a.norm_l2();
            assert!(err <= bound);
        }
    }
    #[test]
    fn qz_homogeneous_temporary_columns_use_stack() {
        qz_step_alloc::<f32>(Par::Seq);
        qz_step_alloc::<f64>(Par::Seq);
        qz_step_alloc::<c32>(Par::Seq);
        qz_step_alloc::<c64>(Par::Seq);
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.execute(|mut par| {
            qz_step_alloc::<f32>(par.reborrow());
            qz_step_alloc::<f64>(par.reborrow());
            qz_step_alloc::<c32>(par.reborrow());
            qz_step_alloc::<c64>(par);
        })
        .unwrap();
    }
}
