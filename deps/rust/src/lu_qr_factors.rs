//! Retained LU/QR factors and caller controls for faer's dense LU/QR kernels.
//!
//! faer's public `solvers::{PartialPivLu, FullPivLu, Qr, ColPivQr}::new` always
//! pass the default `PartialPivLuParams`/`FullPivLuParams`/`QrParams`/
//! `ColPivQrParams`, and the solver types cannot be assembled from an existing
//! factor. Like `crate::cholesky_factors`, these owners call the low-level
//! `*_in_place` kernels with the parameters of the current Julia
//! `with_lu_tuning`/`with_qr_tuning` scope and expose the accessors and solver
//! traits of the faer types they replace. With default settings the factors
//! are bit-identical to faer's (same kernels, block sizes and parameters).
//!
//! Only parameters that change blocking or parallel scheduling are exposed:
//! `PartialPivLuParams.{recursion_threshold, par_threshold}`,
//! `FullPivLuParams.par_threshold`, `QrParams.blocking_threshold` and
//! `ColPivQrParams.par_threshold`. The LU rank-truncation fields (`max_rank`,
//! `abs_tol`, `rel_tol`) stop the factorization early, and faer's solve,
//! inverse and reconstruct routines must not be given such a factor, so they
//! stay at their defaults. `PartialPivLuParams.block_size` and
//! `ColPivQrParams.blocking_threshold` are not read by the kernels at the
//! pinned revision.
// faer's accessor names (`L`, `Q_basis`, `compute_thin_Q`, ...).
#![allow(non_snake_case)]
use crate::guarded;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::full_pivoting::factor::FullPivLuParams;
use faer::linalg::lu::partial_pivoting::factor::PartialPivLuParams;
use faer::linalg::qr::col_pivoting::factor::ColPivQrParams;
use faer::linalg::qr::no_pivoting::factor::QrParams;
use faer::linalg::solvers::{DenseSolveCore, ShapeCore, SolveCore, SolveLstsqCore};
use faer::linalg::{householder as hh, lu, qr};
use faer::perm::{Perm, PermRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::utils::cost::ParThreshold;
use faer::{Auto, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

const TUNING_LEN: usize = 5;

/// `None` keeps faer's default, in `set_lu_qr_tuning` order.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Tuning {
    partial_recursion: Option<usize>,
    partial_par: Option<usize>,
    full_par: Option<usize>,
    qr_blocking: Option<usize>,
    pivoted_qr_par: Option<usize>,
}
impl Tuning {
    fn all(&self) -> [Option<usize>; TUNING_LEN] {
        [
            self.partial_recursion,
            self.partial_par,
            self.full_par,
            self.qr_blocking,
            self.pivoted_qr_par,
        ]
    }
}
const NO_TUNING: Tuning = Tuning {
    partial_recursion: None,
    partial_par: None,
    full_par: None,
    qr_blocking: None,
    pivoted_qr_par: None,
};

// Julia serializes every native call, so the state is process-wide. Rust unit
// tests run on parallel threads; each test thread has its own state there.
#[cfg(not(test))]
mod state {
    use super::*;
    use std::sync::Mutex;
    static TUNING: Mutex<Tuning> = Mutex::new(NO_TUNING);
    pub(super) fn tuning() -> Tuning {
        *TUNING.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(super) fn set_tuning(t: Tuning) {
        *TUNING.lock().unwrap_or_else(|e| e.into_inner()) = t;
    }
}
#[cfg(test)]
mod state {
    use super::*;
    use std::cell::Cell;
    thread_local! {
        static TUNING: Cell<Tuning> = const { Cell::new(NO_TUNING) };
    }
    pub(super) fn tuning() -> Tuning {
        TUNING.with(|t| t.get())
    }
    pub(super) fn set_tuning(t: Tuning) {
        TUNING.with(|x| x.set(t));
    }
}
use state::tuning;

pub(crate) fn partial_piv_lu_params<T: ComplexField>() -> PartialPivLuParams {
    let t = tuning();
    let mut p = <PartialPivLuParams as Auto<T>>::auto();
    if let Some(r) = t.partial_recursion {
        p.recursion_threshold = r;
    }
    if let Some(v) = t.partial_par {
        p.par_threshold = ParThreshold::from(v);
    }
    p
}
pub(crate) fn full_piv_lu_params<T: ComplexField>() -> FullPivLuParams {
    let t = tuning();
    let mut p = <FullPivLuParams as Auto<T>>::auto();
    if let Some(v) = t.full_par {
        p.par_threshold = ParThreshold::from(v);
    }
    p
}
pub(crate) fn qr_params<T: ComplexField>() -> QrParams {
    let t = tuning();
    let mut p = <QrParams as Auto<T>>::auto();
    if let Some(b) = t.qr_blocking {
        p.blocking_threshold = b;
    }
    p
}
pub(crate) fn col_piv_qr_params<T: ComplexField>() -> ColPivQrParams {
    let t = tuning();
    let mut p = <ColPivQrParams as Auto<T>>::auto();
    if let Some(v) = t.pivoted_qr_par {
        p.par_threshold = ParThreshold::from(v);
    }
    p
}

/// `values` holds the five settings and `set` marks which are overridden, in
/// the order partial-pivot LU recursion threshold, partial-pivot LU parallel
/// floor, full-pivot LU parallel floor, QR blocking threshold and
/// column-pivoted QR parallel floor (floors are `FaerParThreshold` values).
#[julia]
pub fn set_lu_qr_tuning(values: &[usize], set: &[u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != TUNING_LEN || set.len() != TUNING_LEN {
            return Err("LU/QR tuning needs five values and flags".into());
        }
        let pick = |i: usize| (set[i] != 0).then_some(values[i]);
        // faer's recursive LU never advances with a zero recursion threshold.
        if pick(0) == Some(0) {
            return Err("partial-pivot LU recursion threshold must be positive".into());
        }
        state::set_tuning(Tuning {
            partial_recursion: pick(0),
            partial_par: pick(1),
            full_par: pick(2),
            qr_blocking: pick(3),
            pivoted_qr_par: pick(4),
        });
        Ok(())
    })
}
/// Writes the active settings and their flags in `set_lu_qr_tuning` order.
#[julia]
pub fn lu_qr_tuning_state(values: &mut [usize], set: &mut [u8]) -> Result<(), String> {
    guarded(|| {
        if values.len() != TUNING_LEN || set.len() != TUNING_LEN {
            return Err("LU/QR tuning needs five values and flags".into());
        }
        for (i, v) in tuning().all().into_iter().enumerate() {
            values[i] = v.unwrap_or(0);
            set[i] = v.is_some() as u8;
        }
        Ok(())
    })
}

#[derive(Clone, Debug)]
pub struct PartialPivLu<T> {
    l: Mat<T>,
    u: Mat<T>,
    p: Perm<usize>,
}
#[derive(Clone, Debug)]
pub struct FullPivLu<T> {
    l: Mat<T>,
    u: Mat<T>,
    p: Perm<usize>,
    q: Perm<usize>,
}
#[derive(Clone, Debug)]
pub struct Qr<T> {
    basis: Mat<T>,
    coeff: Mat<T>,
    r: Mat<T>,
}
#[derive(Clone, Debug)]
pub struct ColPivQr<T> {
    basis: Mat<T>,
    coeff: Mat<T>,
    r: Mat<T>,
    p: Perm<usize>,
}

/// faer's `split_LU`: the packed factor becomes its strict lower part with a
/// unit diagonal and its upper trapezoid.
fn split_lu<T: ComplexField>(lu: Mat<T>) -> (Mat<T>, Mat<T>) {
    let (m, n) = lu.shape();
    let size = m.min(n);
    if m >= n {
        let mut l = lu;
        let mut u = Mat::zeros(size, size);
        u.copy_from_triangular_upper(l.get(..size, ..size));
        for j in 0..n {
            for i in 0..j.min(m) {
                l[(i, j)] = zero();
            }
            l[(j, j)] = one();
        }
        (l, u)
    } else {
        let mut u = lu;
        let mut l = Mat::zeros(size, size);
        l.copy_from_strict_triangular_lower(u.get(..size, ..size));
        for j in 0..n {
            for i in j + 1..m {
                u[(i, j)] = zero();
            }
        }
        for i in 0..size {
            l[(i, i)] = one();
        }
        (l, u)
    }
}
fn perm(fwd: Vec<usize>, bwd: Vec<usize>) -> Perm<usize> {
    let n = fwd.len();
    Perm::new_checked(fwd.into_boxed_slice(), bwd.into_boxed_slice(), n)
}
fn stack(req: faer::dyn_stack::StackReq) -> MemBuffer {
    MemBuffer::new(req)
}

impl<T: ComplexField> PartialPivLu<T> {
    pub fn new(a: MatRef<'_, T>, mut par: Par<'_>) -> Self {
        let mut f = a.to_owned();
        let (m, n) = f.shape();
        let (mut fwd, mut bwd) = (vec![0usize; m], vec![0usize; m]);
        let params = partial_piv_lu_params::<T>();
        let mut mem = stack(
            lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(
                m,
                n,
                par.shape(),
                params.into(),
            ),
        );
        lu::partial_pivoting::factor::lu_in_place(
            f.as_mut(),
            &mut fwd,
            &mut bwd,
            par.reborrow(),
            MemStack::new(&mut mem),
            params.into(),
        );
        let (l, u) = split_lu(f);
        Self {
            l,
            u,
            p: perm(fwd, bwd),
        }
    }
    pub fn L(&self) -> MatRef<'_, T> {
        self.l.as_ref()
    }
    pub fn U(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }
    pub fn P(&self) -> PermRef<'_, usize> {
        self.p.as_ref()
    }
}
impl<T: ComplexField> FullPivLu<T> {
    pub fn new(a: MatRef<'_, T>, mut par: Par<'_>) -> Self {
        let mut f = a.to_owned();
        let (m, n) = f.shape();
        let (mut rf, mut rb) = (vec![0usize; m], vec![0usize; m]);
        let (mut cf, mut cb) = (vec![0usize; n], vec![0usize; n]);
        let params = full_piv_lu_params::<T>();
        let mut mem = stack(lu::full_pivoting::factor::lu_in_place_scratch::<usize, T>(
            m,
            n,
            par.shape(),
            params.into(),
        ));
        lu::full_pivoting::factor::lu_in_place(
            f.as_mut(),
            &mut rf,
            &mut rb,
            &mut cf,
            &mut cb,
            par.reborrow(),
            MemStack::new(&mut mem),
            params.into(),
        );
        let (l, u) = split_lu(f);
        Self {
            l,
            u,
            p: perm(rf, rb),
            q: perm(cf, cb),
        }
    }
    pub fn L(&self) -> MatRef<'_, T> {
        self.l.as_ref()
    }
    pub fn U(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }
    pub fn P(&self) -> PermRef<'_, usize> {
        self.p.as_ref()
    }
    pub fn Q(&self) -> PermRef<'_, usize> {
        self.q.as_ref()
    }
}
impl<T: ComplexField> Qr<T> {
    pub fn new(a: MatRef<'_, T>, mut par: Par<'_>) -> Self {
        let mut f = a.to_owned();
        let (m, n) = f.shape();
        let bs = qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
        let mut coeff = Mat::zeros(bs, m.min(n));
        let params = qr_params::<T>();
        let mut mem = stack(qr::no_pivoting::factor::qr_in_place_scratch::<T>(
            m,
            n,
            bs,
            par.shape(),
            params.into(),
        ));
        qr::no_pivoting::factor::qr_in_place(
            f.as_mut(),
            coeff.as_mut(),
            par.reborrow(),
            MemStack::new(&mut mem),
            params.into(),
        );
        let (basis, r) = split_lu(f);
        Self { basis, coeff, r }
    }
}
impl<T: ComplexField> ColPivQr<T> {
    pub fn new(a: MatRef<'_, T>, mut par: Par<'_>) -> Self {
        let mut f = a.to_owned();
        let (m, n) = f.shape();
        let (mut fwd, mut bwd) = (vec![0usize; n], vec![0usize; n]);
        let bs = qr::no_pivoting::factor::recommended_block_size::<T>(m, n);
        let mut coeff = Mat::zeros(bs, m.min(n));
        let params = col_piv_qr_params::<T>();
        let mut mem = stack(qr::col_pivoting::factor::qr_in_place_scratch::<usize, T>(
            m,
            n,
            bs,
            par.shape(),
            params.into(),
        ));
        qr::col_pivoting::factor::qr_in_place(
            f.as_mut(),
            coeff.as_mut(),
            &mut fwd,
            &mut bwd,
            par.reborrow(),
            MemStack::new(&mut mem),
            params.into(),
        );
        let (basis, r) = split_lu(f);
        Self {
            basis,
            coeff,
            r,
            p: perm(fwd, bwd),
        }
    }
    pub fn P(&self) -> PermRef<'_, usize> {
        self.p.as_ref()
    }
}

/// The Householder accessors shared by both QR owners (faer's names).
macro_rules! householder_accessors {
    ($ty:ident) => {
        impl<T: ComplexField> $ty<T> {
            pub fn Q_basis(&self) -> MatRef<'_, T> {
                self.basis.as_ref()
            }
            pub fn Q_coeff(&self) -> MatRef<'_, T> {
                self.coeff.as_ref()
            }
            pub fn R(&self) -> MatRef<'_, T> {
                self.r.as_ref()
            }
            pub fn thin_R(&self) -> MatRef<'_, T> {
                let size = self.nrows().min(self.ncols());
                self.r.get(..size, ..)
            }
            fn q_columns(&self, cols: usize, par: Par<'_>) -> Mat<T> {
                let m = self.nrows();
                let mut q = Mat::identity(m, cols);
                let mut mem = stack(
                    hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                        m,
                        self.coeff.nrows(),
                        cols,
                    ),
                );
                hh::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    self.Q_basis(),
                    self.Q_coeff(),
                    Conj::No,
                    q.as_mut(),
                    par,
                    MemStack::new(&mut mem),
                );
                q
            }
            pub fn compute_Q(&self, par: Par<'_>) -> Mat<T> {
                self.q_columns(self.nrows(), par)
            }
            pub fn compute_thin_Q(&self, par: Par<'_>) -> Mat<T> {
                self.q_columns(self.nrows().min(self.ncols()), par)
            }
        }
        impl<T: ComplexField> ShapeCore for $ty<T> {
            fn nrows(&self) -> usize {
                self.basis.nrows()
            }
            fn ncols(&self) -> usize {
                self.r.ncols()
            }
        }
    };
}
householder_accessors!(Qr);
householder_accessors!(ColPivQr);

impl<T: ComplexField> ShapeCore for PartialPivLu<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.u.ncols()
    }
}
impl<T: ComplexField> ShapeCore for FullPivLu<T> {
    fn nrows(&self) -> usize {
        self.l.nrows()
    }
    fn ncols(&self) -> usize {
        self.u.ncols()
    }
}

impl<T: ComplexField> SolveCore<T> for PartialPivLu<T> {
    #[track_caller]
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use lu::partial_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.nrows() == rhs.nrows());
        let req = s::solve_in_place_scratch::<usize, T>(self.nrows(), rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_in_place_with_conj(
            self.L(),
            self.U(),
            self.P(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
    #[track_caller]
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use lu::partial_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.ncols() == rhs.nrows());
        let req =
            s::solve_transpose_in_place_scratch::<usize, T>(self.nrows(), rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_transpose_in_place_with_conj(
            self.L(),
            self.U(),
            self.P(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> DenseSolveCore<T> for PartialPivLu<T> {
    fn reconstruct(&self, mut par: Par<'_>) -> Mat<T> {
        use lu::partial_pivoting::reconstruct as r;
        let (m, n) = (self.nrows(), self.ncols());
        let mut out = Mat::zeros(m, n);
        let mut mem = stack(r::reconstruct_scratch::<usize, T>(m, n, par.shape()));
        let stack = MemStack::new(&mut mem);
        r::reconstruct(
            out.as_mut(),
            self.L(),
            self.U(),
            self.P(),
            par.reborrow(),
            stack,
        );
        out
    }
    #[track_caller]
    fn inverse(&self, mut par: Par<'_>) -> Mat<T> {
        use lu::partial_pivoting::inverse as i;
        assert!(self.nrows() == self.ncols());
        let n = self.ncols();
        let mut out = Mat::zeros(n, n);
        let mut mem = stack(i::inverse_scratch::<usize, T>(n, par.shape()));
        let stack = MemStack::new(&mut mem);
        i::inverse(
            out.as_mut(),
            self.L(),
            self.U(),
            self.P(),
            par.reborrow(),
            stack,
        );
        out
    }
}
impl<T: ComplexField> SolveCore<T> for FullPivLu<T> {
    #[track_caller]
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use lu::full_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.nrows() == rhs.nrows());
        let req = s::solve_in_place_scratch::<usize, T>(self.nrows(), rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_in_place_with_conj(
            self.L(),
            self.U(),
            self.P(),
            self.Q(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
    #[track_caller]
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use lu::full_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.ncols() == rhs.nrows());
        let req =
            s::solve_transpose_in_place_scratch::<usize, T>(self.nrows(), rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_transpose_in_place_with_conj(
            self.L(),
            self.U(),
            self.P(),
            self.Q(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> DenseSolveCore<T> for FullPivLu<T> {
    fn reconstruct(&self, mut par: Par<'_>) -> Mat<T> {
        use lu::full_pivoting::reconstruct as r;
        let (m, n) = (self.nrows(), self.ncols());
        let mut out = Mat::zeros(m, n);
        let mut mem = stack(r::reconstruct_scratch::<usize, T>(m, n, par.shape()));
        let stack = MemStack::new(&mut mem);
        r::reconstruct(
            out.as_mut(),
            self.L(),
            self.U(),
            self.P(),
            self.Q(),
            par.reborrow(),
            stack,
        );
        out
    }
    #[track_caller]
    fn inverse(&self, mut par: Par<'_>) -> Mat<T> {
        use lu::full_pivoting::inverse as i;
        assert!(self.nrows() == self.ncols());
        let n = self.ncols();
        let mut out = Mat::zeros(n, n);
        let mut mem = stack(i::inverse_scratch::<usize, T>(n, par.shape()));
        let stack = MemStack::new(&mut mem);
        i::inverse(
            out.as_mut(),
            self.L(),
            self.U(),
            self.P(),
            self.Q(),
            par.reborrow(),
            stack,
        );
        out
    }
}

impl<T: ComplexField> SolveCore<T> for Qr<T> {
    #[track_caller]
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::no_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.nrows() == rhs.nrows());
        let (n, bs) = (self.nrows(), self.coeff.nrows());
        let mut mem = stack(s::solve_in_place_scratch::<T>(
            n,
            bs,
            rhs.ncols(),
            par.shape(),
        ));
        let stack = MemStack::new(&mut mem);
        s::solve_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
    #[track_caller]
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::no_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.ncols() == rhs.nrows());
        let (n, bs) = (self.nrows(), self.coeff.nrows());
        let req = s::solve_transpose_in_place_scratch::<T>(n, bs, rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_transpose_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> SolveLstsqCore<T> for Qr<T> {
    #[track_caller]
    fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::no_pivoting::solve as s;
        assert!(self.nrows() == rhs.nrows() && self.nrows() >= self.ncols());
        let (m, n, bs) = (self.nrows(), self.ncols(), self.coeff.nrows());
        let req = s::solve_lstsq_in_place_scratch::<T>(m, n, bs, rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_lstsq_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> DenseSolveCore<T> for Qr<T> {
    fn reconstruct(&self, mut par: Par<'_>) -> Mat<T> {
        use qr::no_pivoting::reconstruct as r;
        let (m, n, bs) = (self.nrows(), self.ncols(), self.coeff.nrows());
        let mut out = Mat::zeros(m, n);
        let mut mem = stack(r::reconstruct_scratch::<T>(m, n, bs, par.shape()));
        let stack = MemStack::new(&mut mem);
        r::reconstruct(
            out.as_mut(),
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            par.reborrow(),
            stack,
        );
        out
    }
    #[track_caller]
    fn inverse(&self, mut par: Par<'_>) -> Mat<T> {
        use qr::no_pivoting::inverse as i;
        assert!(self.nrows() == self.ncols());
        let (n, bs) = (self.ncols(), self.coeff.nrows());
        let mut out = Mat::zeros(n, n);
        let mut mem = stack(i::inverse_scratch::<T>(n, bs, par.shape()));
        let stack = MemStack::new(&mut mem);
        i::inverse(
            out.as_mut(),
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            par.reborrow(),
            stack,
        );
        out
    }
}

impl<T: ComplexField> SolveCore<T> for ColPivQr<T> {
    #[track_caller]
    fn solve_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::col_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.nrows() == rhs.nrows());
        let (n, bs) = (self.nrows(), self.coeff.nrows());
        let req = s::solve_in_place_scratch::<usize, T>(n, bs, rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            self.P(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
    #[track_caller]
    fn solve_transpose_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::col_pivoting::solve as s;
        assert!(self.nrows() == self.ncols() && self.ncols() == rhs.nrows());
        let (n, bs) = (self.nrows(), self.coeff.nrows());
        let req = s::solve_transpose_in_place_scratch::<usize, T>(n, bs, rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_transpose_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            self.P(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> SolveLstsqCore<T> for ColPivQr<T> {
    #[track_caller]
    fn solve_lstsq_in_place_with_conj(&self, conj: Conj, rhs: MatMut<'_, T>, mut par: Par<'_>) {
        use qr::col_pivoting::solve as s;
        assert!(self.nrows() == rhs.nrows() && self.nrows() >= self.ncols());
        let (m, n, bs) = (self.nrows(), self.ncols(), self.coeff.nrows());
        let req = s::solve_lstsq_in_place_scratch::<usize, T>(m, n, bs, rhs.ncols(), par.shape());
        let mut mem = stack(req);
        let stack = MemStack::new(&mut mem);
        s::solve_lstsq_in_place_with_conj(
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            self.P(),
            conj,
            rhs,
            par.reborrow(),
            stack,
        );
    }
}
impl<T: ComplexField> DenseSolveCore<T> for ColPivQr<T> {
    fn reconstruct(&self, mut par: Par<'_>) -> Mat<T> {
        use qr::col_pivoting::reconstruct as r;
        let (m, n, bs) = (self.nrows(), self.ncols(), self.coeff.nrows());
        let mut out = Mat::zeros(m, n);
        let mut mem = stack(r::reconstruct_scratch::<usize, T>(m, n, bs, par.shape()));
        let stack = MemStack::new(&mut mem);
        r::reconstruct(
            out.as_mut(),
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            self.P(),
            par.reborrow(),
            stack,
        );
        out
    }
    #[track_caller]
    fn inverse(&self, mut par: Par<'_>) -> Mat<T> {
        use qr::col_pivoting::inverse as i;
        assert!(self.nrows() == self.ncols());
        let (n, bs) = (self.ncols(), self.coeff.nrows());
        let mut out = Mat::zeros(n, n);
        let mut mem = stack(i::inverse_scratch::<usize, T>(n, bs, par.shape()));
        let stack = MemStack::new(&mut mem);
        i::inverse(
            out.as_mut(),
            self.Q_basis(),
            self.Q_coeff(),
            self.R(),
            self.P(),
            par.reborrow(),
            stack,
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::{self as fs, Solve, SolveLstsq};

    fn sample(m: usize, n: usize) -> Mat<f64> {
        Mat::from_fn(m, n, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 + if i == j { 5.0 } else { 0.0 }
        })
    }
    fn perm_eq(a: PermRef<'_, usize>, b: PermRef<'_, usize>) -> bool {
        a.arrays() == b.arrays()
    }

    #[test]
    fn defaults_are_bit_identical_to_faer() {
        for (m, n) in [(0, 0), (1, 1), (6, 6), (9, 5), (5, 9), (70, 70), (90, 60)] {
            let a = sample(m, n);
            let (x, y) = (
                PartialPivLu::new(a.as_ref(), Par::Seq),
                fs::PartialPivLu::new(a.as_ref(), Par::Seq),
            );
            assert!(x.L() == y.L() && x.U() == y.U() && perm_eq(x.P(), y.P()));
            let (x, y) = (
                FullPivLu::new(a.as_ref(), Par::Seq),
                fs::FullPivLu::new(a.as_ref(), Par::Seq),
            );
            assert!(x.L() == y.L() && x.U() == y.U());
            assert!(perm_eq(x.P(), y.P()) && perm_eq(x.Q(), y.Q()));
            let (x, y) = (
                Qr::new(a.as_ref(), Par::Seq),
                fs::Qr::new(a.as_ref(), Par::Seq),
            );
            assert!(x.Q_basis() == y.Q_basis() && x.Q_coeff() == y.Q_coeff() && x.R() == y.R());
            assert!(x.compute_Q(Par::Seq) == y.compute_Q(Par::Seq));
            let (x, y) = (
                ColPivQr::new(a.as_ref(), Par::Seq),
                fs::ColPivQr::new(a.as_ref(), Par::Seq),
            );
            assert!(x.Q_basis() == y.Q_basis() && x.Q_coeff() == y.Q_coeff() && x.R() == y.R());
            assert!(perm_eq(x.P(), y.P()));
            assert!(x.compute_thin_Q(Par::Seq) == y.compute_thin_Q(Par::Seq));
            if m == n {
                let b = sample(n, 2);
                let lu = PartialPivLu::new(a.as_ref(), Par::Seq);
                assert!(
                    lu.solve(b.as_ref(), Par::Seq)
                        == fs::PartialPivLu::new(a.as_ref(), Par::Seq).solve(b.as_ref(), Par::Seq)
                );
                assert!(
                    x.solve_transpose(b.as_ref(), Par::Seq)
                        == y.solve_transpose(b.as_ref(), Par::Seq)
                );
                assert!(x.inverse(Par::Seq) == y.inverse(Par::Seq));
            }
            if m >= n {
                let b = sample(m, 2);
                assert!(x.solve_lstsq(b.as_ref(), Par::Seq) == y.solve_lstsq(b.as_ref(), Par::Seq));
                assert!(x.reconstruct(Par::Seq) == y.reconstruct(Par::Seq));
            }
        }
    }

    #[test]
    fn tuning_overrides_apply_and_reset() {
        assert!(set_lu_qr_tuning(&[0; 5], &[1, 0, 0, 0, 0]).is_err());
        assert!(set_lu_qr_tuning(&[0; 4], &[0; 4]).is_err());
        set_lu_qr_tuning(&[3, usize::MAX, 5, 0, usize::MAX], &[1; 5]).unwrap();
        let p = partial_piv_lu_params::<f64>();
        assert_eq!(p.recursion_threshold, 3);
        assert_eq!(p.par_threshold, ParThreshold::Never);
        assert_eq!(
            full_piv_lu_params::<f64>().par_threshold,
            ParThreshold::from(5)
        );
        assert_eq!(qr_params::<f64>().blocking_threshold, 0);
        assert_eq!(
            col_piv_qr_params::<f64>().par_threshold,
            ParThreshold::Never
        );
        let (mut v, mut s) = ([9usize; 5], [9u8; 5]);
        lu_qr_tuning_state(&mut v, &mut s).unwrap();
        assert_eq!((v, s), ([3, usize::MAX, 5, 0, usize::MAX], [1; 5]));
        // Tuned factors factor the same matrix up to rounding.
        let a = sample(90, 70);
        let lu = PartialPivLu::new(a.as_ref(), Par::Seq);
        let qr = Qr::new(a.as_ref(), Par::Seq);
        let pqr = ColPivQr::new(a.as_ref(), Par::Seq);
        for r in [
            lu.reconstruct(Par::Seq),
            qr.reconstruct(Par::Seq),
            pqr.reconstruct(Par::Seq),
        ] {
            assert!((&r - &a).norm_max() < 1e-10 * a.norm_max());
        }
        set_lu_qr_tuning(&[0; 5], &[0; 5]).unwrap();
        assert_eq!(qr_params::<f64>().blocking_threshold, 48 * 48);
        assert_eq!(
            partial_piv_lu_params::<f64>().recursion_threshold,
            <PartialPivLuParams as Auto<f64>>::auto().recursion_threshold
        );
    }
}
