//! Retained LLᴴ and LDLᴴ factors with caller parameters and regularization.
//!
//! faer's public `solvers::Llt::new`/`Ldlt::new` always pass the default
//! `LltParams`/`LdltParams` and no dynamic regularization, and the solver
//! types cannot be assembled from an existing factor. Like `crate::lblt`, these
//! owners call the low-level `cholesky_in_place` with the parameters and the
//! regularization of `crate::selfadjoint_tuning`, and expose the accessors and
//! solver traits of the faer types they replace. With default settings the
//! factors are bit-identical to faer's.
use crate::selfadjoint_tuning as tuning;
use faer::diag::{Diag, DiagRef};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::{ldlt, llt};
use faer::linalg::solvers::{DenseSolveCore, LdltError, LltError, ShapeCore, SolveCore};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, Mat, MatMut, MatRef, Par, Side};

#[derive(Clone, Debug)]
pub struct Llt<T> {
    l: Mat<T>,
}
#[derive(Clone, Debug)]
pub struct Ldlt<T> {
    l: Mat<T>,
    d: Diag<T>,
}

/// The lower factor input faer's self-adjoint solvers start from.
fn lower_copy<T: ComplexField>(a: MatRef<'_, T>, side: Side) -> Mat<T> {
    assert!(a.nrows() == a.ncols());
    let n = a.nrows();
    let mut l = Mat::zeros(n, n);
    match side {
        Side::Lower => l.copy_from_triangular_lower(a),
        Side::Upper => l.copy_from_triangular_lower(a.adjoint()),
    }
    l
}
fn zero_strict_upper<T: ComplexField>(l: &mut Mat<T>) {
    for j in 0..l.ncols() {
        for i in 0..j {
            l[(i, j)] = zero();
        }
    }
}
fn make_self_adjoint<T: ComplexField>(mut a: MatMut<'_, T>) {
    for j in 0..a.ncols() {
        a[(j, j)] = from_real(&real(&a[(j, j)]));
        for i in 0..j {
            a[(i, j)] = conj(&a[(j, i)]);
        }
    }
}

impl<T: ComplexField> Llt<T> {
    /// LLᴴ of the `side` triangle; records faer's regularization count.
    pub fn new(a: MatRef<'_, T>, side: Side, mut par: Par<'_>) -> Result<Self, LltError> {
        tuning::reset_regularization_count();
        let mut l = lower_copy(a, side);
        let n = l.nrows();
        let params = tuning::llt_params::<T>();
        let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<T>(
            n,
            par.shape(),
            params.into(),
        ));
        let stack = MemStack::new(&mut mem);
        let count = match tuning::regularized_llt_in_place(l.as_mut(), par.reborrow(), stack) {
            Some(Ok(count)) => count,
            Some(Err(index)) => return Err(LltError::NonPositivePivot { index }),
            None => {
                llt::factor::cholesky_in_place(
                    l.as_mut(),
                    tuning::llt_regularization::<T>(),
                    par,
                    stack,
                    params.into(),
                )?
                .dynamic_regularization_count
            }
        };
        tuning::record_regularization_count(count);
        zero_strict_upper(&mut l);
        Ok(Self { l })
    }
    #[allow(non_snake_case)]
    pub fn L(&self) -> MatRef<'_, T> {
        self.l.as_ref()
    }
    fn solve_with(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<T>(
            self.l.nrows(),
            rhs.ncols(),
            par.shape(),
        ));
        llt::solve::solve_in_place_with_conj(self.L(), conj, rhs, par, MemStack::new(&mut mem));
    }
}

impl<T: ComplexField> Ldlt<T> {
    /// LDLᴴ of the `side` triangle; records faer's regularization count. The
    /// outer error reports invalid regularization signs.
    pub fn new(
        a: MatRef<'_, T>,
        side: Side,
        par: Par<'_>,
    ) -> Result<Result<Self, LdltError>, String> {
        tuning::reset_regularization_count();
        let mut l = lower_copy(a, side);
        let n = l.nrows();
        let params = tuning::ldlt_params::<T>();
        let mut mem = MemBuffer::new(ldlt::factor::cholesky_in_place_scratch::<T>(
            n,
            par.shape(),
            params.into(),
        ));
        let result = tuning::with_ldlt_regularization::<T, _>(n, |reg| {
            ldlt::factor::cholesky_in_place(
                l.as_mut(),
                reg,
                par,
                MemStack::new(&mut mem),
                params.into(),
            )
        })?;
        let info = match result {
            Ok(info) => info,
            Err(e) => return Ok(Err(e)),
        };
        tuning::record_regularization_count(info.dynamic_regularization_count);
        let mut d = Diag::zeros(n);
        d.copy_from(l.diagonal());
        l.diagonal_mut().fill(one());
        zero_strict_upper(&mut l);
        Ok(Ok(Self { l, d }))
    }
    #[allow(non_snake_case)]
    pub fn L(&self) -> MatRef<'_, T> {
        self.l.as_ref()
    }
    #[allow(non_snake_case)]
    pub fn D(&self) -> DiagRef<'_, T> {
        self.d.as_ref()
    }
    fn solve_with(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        let mut mem = MemBuffer::new(ldlt::solve::solve_in_place_scratch::<T>(
            self.l.nrows(),
            rhs.ncols(),
            par.shape(),
        ));
        ldlt::solve::solve_in_place_with_conj(
            self.L(),
            self.D(),
            conj,
            rhs,
            par,
            MemStack::new(&mut mem),
        );
    }
}

impl<T> ShapeCore for Llt<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.l.ncols()
    }
}
impl<T> ShapeCore for Ldlt<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.l.ncols()
    }
}

impl<T: ComplexField> SolveCore<T> for Llt<T> {
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj, rhs, par);
    }
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj.compose(Conj::Yes), rhs, par);
    }
}
impl<T: ComplexField> SolveCore<T> for Ldlt<T> {
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj, rhs, par);
    }
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj.compose(Conj::Yes), rhs, par);
    }
}

impl<T: ComplexField> DenseSolveCore<T> for Llt<T> {
    fn reconstruct(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(llt::reconstruct::reconstruct_scratch::<T>(n, par.shape()));
        llt::reconstruct::reconstruct(out.as_mut(), self.L(), par, MemStack::new(&mut mem));
        make_self_adjoint(out.as_mut());
        out
    }
    fn inverse(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(llt::inverse::inverse_scratch::<T>(n, par.shape()));
        llt::inverse::inverse(out.as_mut(), self.L(), par, MemStack::new(&mut mem));
        make_self_adjoint(out.as_mut());
        out
    }
}
impl<T: ComplexField> DenseSolveCore<T> for Ldlt<T> {
    fn reconstruct(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(ldlt::reconstruct::reconstruct_scratch::<T>(n, par.shape()));
        ldlt::reconstruct::reconstruct(
            out.as_mut(),
            self.L(),
            self.D(),
            par,
            MemStack::new(&mut mem),
        );
        make_self_adjoint(out.as_mut());
        out
    }
    fn inverse(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(ldlt::inverse::inverse_scratch::<T>(n, par.shape()));
        ldlt::inverse::inverse(
            out.as_mut(),
            self.L(),
            self.D(),
            par,
            MemStack::new(&mut mem),
        );
        make_self_adjoint(out.as_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    fn spd(n: usize) -> Mat<f64> {
        Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 1.0
            } else {
                1.0 / (1.0 + i as f64 + j as f64)
            }
        })
    }

    #[test]
    fn defaults_match_faer_bitwise() {
        for n in [0, 1, 5, 70, 200] {
            let a = spd(n);
            for side in [Side::Lower, Side::Upper] {
                let ours = Llt::new(a.as_ref(), side, Par::Seq).unwrap();
                let theirs = a.llt(side, Par::Seq).unwrap();
                assert_eq!(ours.L(), theirs.L());
                let ours = Ldlt::new(a.as_ref(), side, Par::Seq).unwrap().unwrap();
                let theirs = a.ldlt(side, Par::Seq).unwrap();
                assert_eq!(ours.L(), theirs.L());
                assert_eq!(ours.D(), theirs.D());
            }
        }
        let c = Mat::from_fn(6, 6, |i, j| {
            faer::c64::new(
                spd(6)[(i, j)],
                if i == j {
                    0.0
                } else {
                    0.1 * (i as f64 - j as f64)
                },
            )
        });
        let ours = Llt::new(c.as_ref(), Side::Lower, Par::Seq).unwrap();
        assert_eq!(ours.L(), c.llt(Side::Lower, Par::Seq).unwrap().L());
        let b = Mat::from_fn(6, 2, |i, j| faer::c64::new(i as f64, j as f64));
        let x = ours.solve(&b, Par::Seq);
        assert!((&c * &x - &b).norm_max() < 1e-12);
    }

    #[test]
    fn small_tuning_values_terminate_and_agree() {
        let a = spd(37);
        let reference = a.llt(Side::Lower, Par::Seq).unwrap();
        tuning::set_selfadjoint_tuning(&[1, 1, 1, 1, 0, 0], &[1, 1, 1, 1, 0, 0]).unwrap();
        let ours = Llt::new(a.as_ref(), Side::Lower, Par::Seq).unwrap();
        assert!((ours.L() - reference.L()).norm_max() < 1e-12);
        let ours = Ldlt::new(a.as_ref(), Side::Lower, Par::Seq)
            .unwrap()
            .unwrap();
        assert!((ours.reconstruct(Par::Seq) - &a).norm_max() < 1e-12);
    }

    #[test]
    fn llt_regularization_counts_replaced_pivots() {
        // Indefinite: the second pivot is negative.
        let a = faer::mat![[1.0, 2.0], [2.0, 1.0f64]];
        assert!(Llt::new(a.as_ref(), Side::Lower, Par::Seq).is_err());
        tuning::set_selfadjoint_call_options(0, 1.0, 1e-8, &[], false).unwrap();
        let f = Llt::new(a.as_ref(), Side::Lower, Par::Seq).unwrap();
        assert_eq!(tuning::selfadjoint_regularization_count(), 1);
        assert_eq!(f.L()[(1, 1)], 1.0);
        // faer's own regularized LLᴴ breaks here (blocked, negative first
        // pivot); the factor must equal A off the diagonal.
        for n in [2, 64, 200] {
            let a = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    if i == 0 {
                        -3.0
                    } else {
                        4.0
                    }
                } else {
                    0.5
                }
            });
            let f = Llt::new(a.as_ref(), Side::Lower, Par::Seq).unwrap();
            assert_eq!(tuning::selfadjoint_regularization_count(), 1);
            assert_eq!(f.L()[(0, 0)], 1.0);
            let e = f.reconstruct(Par::Seq) - &a;
            for j in 0..n {
                for i in 0..n {
                    let expected = if (i, j) == (0, 0) { 4.0 } else { 0.0 };
                    assert!((e[(i, j)] - expected as f64).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn ldlt_regularization_count_only_tracks_positive_signs() {
        // faer increments the count only for pivots with expected sign +1:
        // pivots corrected towards -1, or replaced without signs, are not
        // counted (see docs/reproducers/faer-ldlt-regularization-count.rs).
        let a = faer::mat![[0.0, 1.0], [1.0, 0.0f64]];
        assert!(Ldlt::new(a.as_ref(), Side::Lower, Par::Seq)
            .unwrap()
            .is_err());
        tuning::set_selfadjoint_call_options(0, 1.0, 1e-8, &[], false).unwrap();
        let f = Ldlt::new(a.as_ref(), Side::Lower, Par::Seq)
            .unwrap()
            .unwrap();
        assert_eq!(f.D()[0], 1.0);
        assert_eq!(tuning::selfadjoint_regularization_count(), 0);
        tuning::set_selfadjoint_call_options(0, 1.0, 1e-8, &[1, 255], true).unwrap();
        let f = Ldlt::new(a.as_ref(), Side::Lower, Par::Seq)
            .unwrap()
            .unwrap();
        assert_eq!(f.D()[0], 1.0);
        assert_eq!(tuning::selfadjoint_regularization_count(), 1);
        tuning::set_selfadjoint_call_options(0, 1.0, 1e-8, &[1], true).unwrap();
        assert!(Ldlt::new(a.as_ref(), Side::Lower, Par::Seq).is_err());
    }
}
