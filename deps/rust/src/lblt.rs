//! Retained pivoted block LDLᴴ factor with a selectable faer pivoting strategy.
//!
//! faer's public `solvers::Lblt::new` always uses the default `PartialDiag`
//! strategy. Julia's `bunchkaufman(A, rook)` also exposes rook pivoting, which
//! faer implements in `linalg::cholesky::lblt::factor`. This owner calls that
//! low-level factorization with explicit `LbltParams` and exposes the same
//! accessors and solver traits as `solvers::Lblt`.
use faer::diag::{Diag, DiagRef};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt::{
    factor::{self, LbltParams, PivotingStrategy},
    inverse, reconstruct, solve,
};
use faer::linalg::solvers::{DenseSolveCore, ShapeCore, SolveCore};
use faer::perm::{Perm, PermRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Auto, Conj, Mat, MatMut, MatRef, Par, Side};

#[derive(Clone, Debug)]
pub struct Lblt<T> {
    l: Mat<T>,
    diag: Diag<T>,
    subdiag: Diag<T>,
    perm: Perm<usize>,
    rook: bool,
}

/// faer's automatic block LDLᴴ parameters with the pivoting strategy Julia's
/// `bunchkaufman(A, rook)` selects: `Rook`, or faer's default `PartialDiag`.
pub(crate) fn params<T: ComplexField>(rook: bool) -> LbltParams {
    let mut params = <LbltParams as Auto<T>>::auto();
    if rook {
        params.pivoting = PivotingStrategy::Rook;
    }
    params
}

impl<T: ComplexField> Lblt<T> {
    /// Factor the `side` triangle of self-adjoint `a`; `rook` selects faer's
    /// rook strategy, otherwise faer's default partial-diagonal strategy.
    pub fn new(a: MatRef<'_, T>, side: Side, par: Par<'_>, rook: bool) -> Self {
        assert!(a.nrows() == a.ncols());
        let n = a.nrows();
        let mut l = Mat::zeros(n, n);
        match side {
            Side::Lower => l.copy_from_triangular_lower(a),
            Side::Upper => l.copy_from_triangular_lower(a.adjoint()),
        }
        let params = params::<T>(rook);
        let mut diag = Diag::zeros(n);
        let mut subdiag = Diag::zeros(n);
        let mut fwd = vec![0usize; n];
        let mut bwd = vec![0usize; n];
        let mut mem = MemBuffer::new(factor::cholesky_in_place_scratch::<usize, T>(
            n,
            par.shape(),
            params.into(),
        ));
        factor::cholesky_in_place(
            l.as_mut(),
            subdiag.as_mut(),
            &mut fwd,
            &mut bwd,
            par,
            MemStack::new(&mut mem),
            params.into(),
        );
        diag.copy_from(l.diagonal());
        for j in 0..n {
            l[(j, j)] = one();
            for i in 0..j {
                l[(i, j)] = zero();
            }
        }
        let perm = Perm::new_checked(fwd.into_boxed_slice(), bwd.into_boxed_slice(), n);
        Self {
            l,
            diag,
            subdiag,
            perm,
            rook,
        }
    }
    #[allow(non_snake_case)]
    pub fn L(&self) -> MatRef<'_, T> {
        self.l.as_ref()
    }
    #[allow(non_snake_case)]
    pub fn B_diag(&self) -> DiagRef<'_, T> {
        self.diag.as_ref()
    }
    #[allow(non_snake_case)]
    pub fn B_subdiag(&self) -> DiagRef<'_, T> {
        self.subdiag.as_ref()
    }
    #[allow(non_snake_case)]
    pub fn P(&self) -> PermRef<'_, usize> {
        self.perm.as_ref()
    }
    pub fn rook(&self) -> bool {
        self.rook
    }
    fn solve_with(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        let n = self.l.nrows();
        let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, T>(
            n,
            rhs.ncols(),
            par.shape(),
        ));
        solve::solve_in_place_with_conj(
            self.L(),
            self.B_diag(),
            self.B_subdiag(),
            conj,
            self.P(),
            rhs,
            par,
            MemStack::new(&mut mem),
        );
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

impl<T> ShapeCore for Lblt<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.l.ncols()
    }
}

impl<T: ComplexField> SolveCore<T> for Lblt<T> {
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj, rhs, par);
    }
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, par: Par<'_>) {
        self.solve_with(conj.compose(Conj::Yes), rhs, par);
    }
}

impl<T: ComplexField> DenseSolveCore<T> for Lblt<T> {
    fn reconstruct(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(reconstruct::reconstruct_scratch::<usize, T>(n, par.shape()));
        reconstruct::reconstruct(
            out.as_mut(),
            self.L(),
            self.B_diag(),
            self.B_subdiag(),
            self.P(),
            par,
            MemStack::new(&mut mem),
        );
        make_self_adjoint(out.as_mut());
        out
    }
    fn inverse(&self, par: Par<'_>) -> Mat<T> {
        let n = self.l.nrows();
        let mut out = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(inverse::inverse_scratch::<usize, T>(n, par.shape()));
        inverse::inverse(
            out.as_mut(),
            self.L(),
            self.B_diag(),
            self.B_subdiag(),
            self.P(),
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

    fn indefinite() -> Mat<f64> {
        Mat::from_fn(6, 6, |i, j| {
            if i == j {
                [1e-8, -3.0, 2.0, 0.0, -1.0, 4.0][i]
            } else {
                1.0 / (1.0 + i as f64 + j as f64)
            }
        })
    }

    #[test]
    fn default_strategy_matches_faer_lblt() {
        let a = indefinite();
        for side in [Side::Lower, Side::Upper] {
            let ours = Lblt::new(a.as_ref(), side, Par::Seq, false);
            let faer = a.lblt(side, Par::Seq);
            assert_eq!(ours.P().arrays().0, faer.P().arrays().0);
            assert!((ours.L() - faer.L()).norm_max() == 0.0);
            assert!(!ours.rook());
        }
    }

    #[test]
    fn rook_strategy_reconstructs_and_solves() {
        let a = indefinite();
        let f = Lblt::new(a.as_ref(), Side::Lower, Par::Seq, true);
        assert!(f.rook());
        assert!((f.reconstruct(Par::Seq) - &a).norm_max() < 1e-10);
        let x = f.solve(Mat::<f64>::identity(6, 6), Par::Seq);
        assert!((&a * &x - Mat::<f64>::identity(6, 6)).norm_max() < 1e-8);
        assert!((f.inverse(Par::Seq) - &x).norm_max() < 1e-8);
    }
}
