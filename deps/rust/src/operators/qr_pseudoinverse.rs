//! Rectangular/truncated QR actions from saved reflectors and the SVD of R.
use super::{Base, Data, InverseSpectral, Operator, OwnedLinearOperator};
use crate::pseudoinverse::ToleranceReal;
use crate::typed::{Factor, FactorData, TypedFactor};
use crate::{guarded, teams::OwnedTeam, workspace::OwnedScratch, OwnedFactor};
use faer::dyn_stack::{MemStack, StackReq};
use faer::mat::AsMatMut;
use faer::matrix_free::{BiLinOp, LinOp};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
#[derive(Clone)]
pub(super) struct QrPseudoInverse<T> {
    basis: Mat<T>,
    coeff: Mat<T>,
    middle: InverseSpectral<T>,
    permutation: faer::perm::Perm<usize>,
}
impl<T> std::fmt::Debug for QrPseudoInverse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QrPseudoInverse")
    }
}
impl<T: ComplexField> QrPseudoInverse<T> {
    /// Construction temporaries: singular values and thin singular vectors of
    /// the saved `k x n` R plus faer's SVD workspace. Retained truncated
    /// factors, Householder data and the permutation are operator output.
    pub(super) fn construction_scratch(m: usize, n: usize, par: ParShape) -> StackReq {
        let k = m.min(n);
        if k == 0 {
            return StackReq::EMPTY;
        }
        StackReq::all_of(&[
            faer::linalg::temp_mat_scratch::<T>(k, 1),
            faer::linalg::temp_mat_scratch::<T>(k, k),
            faer::linalg::temp_mat_scratch::<T>(n, k),
            faer::linalg::svd::svd_scratch::<T>(
                k,
                n,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                faer::linalg::svd::ComputeSvdVectors::Thin,
                par,
                Default::default(),
            ),
        ])
    }
    #[cfg(test)]
    fn new(
        b: MatRef<'_, T>,
        c: MatRef<'_, T>,
        r: MatRef<'_, T>,
        p: Option<faer::perm::PermRef<'_, usize>>,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
    ) -> Result<Self, String>
    where
        T::Real: ToleranceReal,
    {
        let mut memory =
            faer::dyn_stack::MemBuffer::new(Self::construction_scratch(m, n, ParShape::Seq));
        Self::new_in(
            b,
            c,
            r,
            p,
            atol,
            rtol,
            relative_f32,
            m,
            n,
            Par::Seq,
            MemStack::new(&mut memory),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_in(
        b: MatRef<'_, T>,
        c: MatRef<'_, T>,
        r: MatRef<'_, T>,
        p: Option<faer::perm::PermRef<'_, usize>>,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<Self, String>
    where
        T::Real: ToleranceReal,
    {
        let k = m.min(n);
        if b.shape() != (m, k)
            || r.shape() != (k, n)
            || c.ncols() != k
            || (k > 0 && c.nrows() == 0)
            || p.is_some_and(|p| p.len() != n)
        {
            return Err("saved QR dimensions do not match requested metadata".into());
        }
        if atol.is_nan() || rtol.is_nan() || atol < 0.0 || rtol < 0.0 {
            return Err("tolerances must be nonnegative and not NaN".into());
        }
        crate::typed::finite(b)?;
        // faer uses infinity as the inverse-Householder coefficient for an
        // identity reflector. It is valid; NaNs are not.
        for j in 0..c.ncols() {
            for i in 0..c.nrows() {
                if is_nan(&c[(i, j)]) {
                    return Err("QR Householder coefficients must not be NaN".into());
                }
            }
        }
        crate::typed::finite(r)?;
        let middle = Self::truncated_svd(r, atol, rtol, relative_f32, par, stack)?;
        let permutation = if let Some(p) = p {
            let (a, b) = p.arrays();
            faer::perm::Perm::new_checked(
                a.to_vec().into_boxed_slice(),
                b.to_vec().into_boxed_slice(),
                n,
            )
        } else {
            let a: Vec<_> = (0..n).collect();
            faer::perm::Perm::new_checked(a.clone().into_boxed_slice(), a.into_boxed_slice(), n)
        };
        Ok(Self {
            basis: b.to_owned(),
            coeff: c.to_owned(),
            middle,
            permutation,
        })
    }
    /// SVD of R into caller scratch; only the retained singular columns and
    /// their inverse weights are copied out as operator state.
    fn truncated_svd(
        r: MatRef<'_, T>,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<InverseSpectral<T>, String>
    where
        T::Real: ToleranceReal,
    {
        let (k, n) = r.shape();
        if k == 0 {
            return Ok(InverseSpectral {
                u: Mat::zeros(0, 0),
                v: Mat::zeros(n, 0),
                weights: Vec::new(),
                zero: true,
            });
        }
        let (mut s, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(k, 1, stack);
        let (mut u, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(k, k, stack);
        let (mut v, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, k, stack);
        let mut s = s.as_mat_mut();
        let (mut u, mut v) = (u.as_mat_mut(), v.as_mat_mut());
        faer::linalg::svd::svd(
            r,
            s.rb_mut().col_mut(0).as_diagonal_mut(),
            Some(u.rb_mut()),
            Some(v.rb_mut()),
            par,
            stack,
            Default::default(),
        )
        .map_err(|e| format!("QR R SVD: {e:?}"))?;
        let (s, u, v) = (s.rb(), u.rb(), v.rb());
        let cutoff = s.norm_max().cutoff(atol, rtol, relative_f32);
        // faer returns nonincreasing singular values: the retained part is a
        // prefix, so truncated columns are never stored.
        let kept = (0..k).take_while(|&i| abs(&s[(i, 0)]) > cutoff).count();
        debug_assert!((kept..k).all(|i| !(abs(&s[(i, 0)]) > cutoff)));
        Ok(InverseSpectral {
            u: u.subcols(0, kept).to_owned(),
            v: v.subcols(0, kept).to_owned(),
            weights: (0..kept)
                .map(|i| from_real::<T>(&recip(&real(&s[(i, 0)]))))
                .collect(),
            zero: kept == 0,
        })
    }
    pub(super) fn map_values<U: ComplexField>(&self, cast: impl Fn(&T) -> U) -> QrPseudoInverse<U> {
        let convert = |a: &Mat<T>| Mat::from_fn(a.nrows(), a.ncols(), |i, j| cast(&a[(i, j)]));
        QrPseudoInverse {
            basis: convert(&self.basis),
            coeff: convert(&self.coeff),
            middle: InverseSpectral {
                u: convert(&self.middle.u),
                v: convert(&self.middle.v),
                weights: self.middle.weights.iter().map(cast).collect(),
                zero: self.middle.zero,
            },
            permutation: self.permutation.clone(),
        }
    }
    fn scratch(&self, rhs: usize, p: ParShape) -> StackReq {
        let (m, n) = (self.basis.nrows(), self.middle.nrows());
        if self.middle.zero || m.min(n) == 0 || rhs == 0 {
            return StackReq::EMPTY;
        }
        let normal=faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m,self.coeff.nrows(),rhs);
        let adjoint=faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(m,self.coeff.nrows(),rhs);
        StackReq::all_of(&[
            faer::linalg::temp_mat_scratch::<T>(m, rhs),
            faer::linalg::temp_mat_scratch::<T>(n, rhs),
            StackReq::any_of(&[normal, adjoint, self.middle.apply_scratch(rhs, p)]),
        ])
    }
    fn run<const TR: bool, const C: bool>(
        &self,
        mut out: MatMut<'_, T>,
        b: MatRef<'_, T>,
        mut par: Par<'_>,
        stack: &mut MemStack,
    ) {
        let (m, n, k) = (self.basis.nrows(), self.middle.nrows(), self.basis.ncols());
        let rhs = b.ncols();
        assert_eq!(b.nrows(), if TR { n } else { m });
        assert_eq!(out.shape(), (if TR { m } else { n }, rhs));
        if self.middle.zero || k == 0 || rhs == 0 {
            out.fill(zero::<T>());
            return;
        }
        let (mut work, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(m, rhs, stack);
        let (mut permuted, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, rhs, stack);
        let mut work = work.as_mat_mut();
        let mut permuted = permuted.as_mat_mut();
        let (p, inv) = self.permutation.as_ref().arrays();
        if TR {
            for j in 0..rhs {
                for i in 0..n {
                    permuted[(i, j)] = copy(&b[(p[i], j)]);
                }
            }
            self.middle.run(
                work.rb_mut().subrows_mut(0, k),
                permuted.rb(),
                par.reborrow(),
                stack,
                true,
                C,
            );
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(self.basis.as_ref(),self.coeff.as_ref(),if C {Conj::No}else{Conj::Yes},work.rb_mut(),par,stack);
            out.copy_from(work);
        } else {
            work.copy_from(b);
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(self.basis.as_ref(),self.coeff.as_ref(),if C {Conj::No}else{Conj::Yes},work.rb_mut(),par.reborrow(),stack);
            self.middle.run(
                permuted.rb_mut(),
                work.rb().subrows(0, k),
                par,
                stack,
                false,
                C,
            );
            for j in 0..rhs {
                for i in 0..n {
                    out[(i, j)] = copy(&permuted[(inv[i], j)]);
                }
            }
        }
    }
}
impl<T: ComplexField> LinOp<T> for QrPseudoInverse<T> {
    fn nrows(&self) -> usize {
        self.middle.nrows()
    }
    fn ncols(&self) -> usize {
        self.basis.nrows()
    }
    fn apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.scratch(k, p)
    }
    fn apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, false>(o, b, p, s)
    }
    fn conj_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<false, true>(o, b, p, s)
    }
}
impl<T: ComplexField> BiLinOp<T> for QrPseudoInverse<T> {
    fn transpose_apply_scratch(&self, k: usize, p: ParShape) -> StackReq {
        self.scratch(k, p)
    }
    fn transpose_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, false>(o, b, p, s)
    }
    fn adjoint_apply(&self, o: MatMut<'_, T>, b: MatRef<'_, T>, p: Par<'_>, s: &mut MemStack) {
        self.run::<true, true>(o, b, p, s)
    }
}
fn construction_requirement(m: usize, n: usize, tag: i32, p: ParShape) -> Result<StackReq, String> {
    Ok(match tag {
        0 => QrPseudoInverse::<f32>::construction_scratch(m, n, p),
        1 => QrPseudoInverse::<f64>::construction_scratch(m, n, p),
        2 => QrPseudoInverse::<faer::c32>::construction_scratch(m, n, p),
        3 => QrPseudoInverse::<faer::c64>::construction_scratch(m, n, p),
        _ => return Err("unknown QR pseudoinverse scalar tag".into()),
    })
}
fn write_requirement(req: StackReq, out: *mut usize) {
    // SAFETY: Julia passes a live two-element UInt buffer.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
}
/// Query construction scratch for the saved-QR pseudoinverse action.
#[julia]
pub fn operator_qr_pseudoinverse_requirements(
    m: usize,
    n: usize,
    tag: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| {
        write_requirement(construction_requirement(m, n, tag, ParShape::Seq)?, out);
        Ok(())
    })
}
#[julia]
pub fn operator_qr_pseudoinverse_requirements_team(
    m: usize,
    n: usize,
    tag: i32,
    out: *mut usize,
    team: &OwnedTeam,
) -> Result<(), String> {
    guarded(|| {
        let req = team.execute(|p| construction_requirement(m, n, tag, p.shape()))??;
        write_requirement(req, out);
        Ok(())
    })
}
#[allow(clippy::too_many_arguments)]
fn build<T: ComplexField>(
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    r: MatRef<'_, T>,
    p: Option<faer::perm::PermRef<'_, usize>>,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    m: usize,
    n: usize,
    scratch: &mut OwnedScratch,
    par: Par<'_>,
) -> Result<Operator<T>, String>
where
    T::Real: ToleranceReal,
{
    let stack = scratch.stack(QrPseudoInverse::<T>::construction_scratch(
        m,
        n,
        par.shape(),
    ))?;
    let saved = QrPseudoInverse::new_in(b, c, r, p, atol, rtol, relative_f32, m, n, par, stack)?;
    Ok(Operator::new(Base::QrPseudo(Box::new(saved))))
}
#[allow(clippy::too_many_arguments)]
fn build_f64(
    f: &OwnedFactor,
    scratch: &mut OwnedScratch,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    m: usize,
    n: usize,
    tag: i32,
    par: Par<'_>,
) -> Result<Data, String> {
    if tag != 1 {
        return Err("QR factor native scalar does not match metadata".into());
    }
    let make = |b, c, r, p| build(b, c, r, p, atol, rtol, relative_f32, m, n, scratch, par);
    Ok(Data::F64(match f.get()? {
        crate::DenseFactor::Qr(f) => make(f.Q_basis(), f.Q_coeff(), f.R(), None)?,
        crate::DenseFactor::PivotQr(f) => make(f.Q_basis(), f.Q_coeff(), f.R(), Some(f.P()))?,
        _ => return Err("QR pseudoinverse requires saved QR factors".into()),
    }))
}
#[allow(clippy::too_many_arguments)]
fn build_typed(
    f: &TypedFactor,
    scratch: &mut OwnedScratch,
    atol: f64,
    rtol: f64,
    relative_f32: bool,
    m: usize,
    n: usize,
    tag: i32,
    par: Par<'_>,
) -> Result<Data, String> {
    f.solve_ready()?;
    macro_rules! make {
        ($f:expr,$p:expr,$variant:ident,$tag:expr) => {{
            if tag != $tag {
                return Err("QR factor native scalar does not match metadata".into());
            }
            Data::$variant(build(
                $f.Q_basis(),
                $f.Q_coeff(),
                $f.R(),
                $p,
                atol,
                rtol,
                relative_f32,
                m,
                n,
                scratch,
                par,
            )?)
        }};
    }
    Ok(match f.data()? {
        FactorData::F32(Factor::Qr(f)) => make!(f, None, F32, 0),
        FactorData::F32(Factor::PivotQr(f)) => make!(f, Some(f.P()), F32, 0),
        FactorData::C32(Factor::Qr(f)) => make!(f, None, C32, 2),
        FactorData::C32(Factor::PivotQr(f)) => make!(f, Some(f.P()), C32, 2),
        FactorData::C64(Factor::Qr(f)) => make!(f, None, C64, 3),
        FactorData::C64(Factor::PivotQr(f)) => make!(f, Some(f.P()), C64, 3),
        _ => return Err("QR pseudoinverse requires matching saved QR factors".into()),
    })
}
#[julia]
impl OwnedLinearOperator {
    /// Build the saved-QR pseudoinverse action using caller construction scratch.
    #[julia]
    pub fn operator_qr_pseudoinverse(
        &mut self,
        f: &OwnedFactor,
        scratch: &mut OwnedScratch,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let data = build_f64(f, scratch, atol, rtol, relative_f32, m, n, tag, Par::Seq)?;
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn operator_qr_pseudoinverse_team(
        &mut self,
        f: &OwnedFactor,
        scratch: &mut OwnedScratch,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            let data =
                team.execute(|p| build_f64(f, scratch, atol, rtol, relative_f32, m, n, tag, p))??;
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn operator_qr_pseudoinverse_typed(
        &mut self,
        f: &TypedFactor,
        scratch: &mut OwnedScratch,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let data = build_typed(f, scratch, atol, rtol, relative_f32, m, n, tag, Par::Seq)?;
            self.data = Some(data);
            Ok(())
        })
    }
    #[julia]
    pub fn operator_qr_pseudoinverse_typed_team(
        &mut self,
        f: &TypedFactor,
        scratch: &mut OwnedScratch,
        atol: f64,
        rtol: f64,
        relative_f32: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            let data =
                team.execute(|p| build_typed(f, scratch, atol, rtol, relative_f32, m, n, tag, p))??;
            self.data = Some(data);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn exercise<T: ComplexField, const TR: bool, const C: bool>(
        op: &QrPseudoInverse<T>,
        inverse: MatRef<'_, T>,
        team: &crate::teams::OwnedTeam,
    ) {
        let action = if TR {
            if C {
                inverse.adjoint().to_owned()
            } else {
                inverse.transpose().to_owned()
            }
        } else if C {
            inverse.conjugate().to_owned()
        } else {
            inverse.to_owned()
        };
        let b = Mat::from_fn(action.ncols(), 2, |i, j| {
            from_f64::<T>((i + 2 * j + 1) as f64)
        });
        let mut expected = Mat::zeros(action.nrows(), 2);
        faer::linalg::matmul::matmul(
            expected.as_mut(),
            faer::Accum::Replace,
            action.as_ref(),
            b.as_ref(),
            one::<T>(),
            Par::Seq,
        );
        let req = op.scratch(2, ParShape::Seq);
        let mut memory = faer::dyn_stack::MemBuffer::new(req);
        let mut out = Mat::zeros(action.nrows(), 2);
        let mut run = || {
            op.run::<TR, C>(
                out.as_mut(),
                b.as_ref(),
                Par::Seq,
                MemStack::new(&mut memory),
            )
        };
        run();
        assert_eq!(
            crate::allocation_probe::count(|| {
                run();
                run();
            }),
            0,
            "saved QR action allocated on the calling thread"
        );
        drop(run);
        let mut parallel = || {
            team.execute(|p| {
                op.run::<TR, C>(out.as_mut(), b.as_ref(), p, MemStack::new(&mut memory))
            })
            .unwrap()
        };
        parallel();
        assert_eq!(
            crate::allocation_probe::count(|| {
                parallel();
                parallel();
            }),
            0,
            "saved QR team action allocated on the calling thread"
        );
        drop(parallel);
        let scale = expected.norm_max();
        let scale = if scale > one::<T::Real>() {
            scale
        } else {
            one::<T::Real>()
        };
        let bound = from_f64::<T::Real>(500.0 * op.basis.nrows().max(op.middle.nrows()) as f64)
            * eps::<T::Real>()
            * scale;
        assert!((&out - &expected).norm_max() < bound);
    }
    fn check<T: ComplexField>()
    where
        T::Real: ToleranceReal,
    {
        let mut team = crate::teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for (m, n) in [(7, 3), (3, 7)] {
            for deficient in [false, true] {
                for pivoted in [false, true] {
                    let mut a = Mat::from_fn(m, n, |i, j| {
                        from_f64::<T>(if i == j { 4.0 + j as f64 } else { 0.03 })
                    });
                    if deficient {
                        if m > n {
                            for i in 0..m {
                                a[(i, n - 1)] = copy(&a[(i, 0)]);
                            }
                        } else {
                            for j in 0..n {
                                a[(m - 1, j)] = copy(&a[(0, j)]);
                            }
                        }
                    }
                    let saved = if pivoted {
                        let f = a.as_ref().col_piv_qr(Par::Seq);
                        QrPseudoInverse::new(
                            f.Q_basis(),
                            f.Q_coeff(),
                            f.R(),
                            Some(f.P()),
                            0.0,
                            1e-5,
                            false,
                            m,
                            n,
                        )
                        .unwrap()
                    } else {
                        let f = a.as_ref().qr(Par::Seq);
                        QrPseudoInverse::new(
                            f.Q_basis(),
                            f.Q_coeff(),
                            f.R(),
                            None,
                            0.0,
                            1e-5,
                            false,
                            m,
                            n,
                        )
                        .unwrap()
                    };
                    let inverse = a
                        .as_ref()
                        .thin_svd(Par::Seq)
                        .unwrap()
                        .pseudoinverse(Par::Seq);
                    exercise::<T, false, false>(&saved, inverse.as_ref(), &team);
                    exercise::<T, false, true>(&saved, inverse.as_ref(), &team);
                    exercise::<T, true, false>(&saved, inverse.as_ref(), &team);
                    exercise::<T, true, true>(&saved, inverse.as_ref(), &team);
                }
            }
        }
        team.team_close().unwrap();
    }
    #[test]
    fn f32_saved_qr_actions() {
        check::<f32>();
    }
    #[test]
    fn f64_saved_qr_actions() {
        check::<f64>();
    }
    #[test]
    fn c32_saved_qr_actions() {
        check::<faer::c32>();
    }
    #[test]
    fn c64_saved_qr_actions() {
        check::<faer::c64>();
    }
    // Retained construction output: Householder basis/coefficients, the two
    // permutation arrays, truncated U/V and the inverse weights.
    const RETAINED_BUFFERS: usize = 7;
    fn svd_kernel_allocations<T: ComplexField>(r: MatRef<'_, T>) -> usize {
        let (k, n) = r.shape();
        let thin = faer::linalg::svd::ComputeSvdVectors::Thin;
        let req = faer::linalg::svd::svd_scratch::<T>(
            k,
            n,
            thin,
            thin,
            ParShape::Seq,
            Default::default(),
        );
        let mut memory = faer::dyn_stack::MemBuffer::new(req);
        let (mut s, mut u, mut v) = (Mat::zeros(k, 1), Mat::zeros(k, k), Mat::zeros(n, k));
        let mut run = || {
            faer::linalg::svd::svd(
                r,
                s.as_mut().col_mut(0).as_diagonal_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                Par::Seq,
                MemStack::new(&mut memory),
                Default::default(),
            )
            .unwrap()
        };
        run();
        crate::allocation_probe::count(run)
    }
    fn construction<T: ComplexField>(value: fn(f64) -> T)
    where
        T::Real: ToleranceReal,
    {
        let mut team = crate::teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for (m, n) in [(7, 3), (3, 7), (48, 32), (32, 48)] {
            for pivoted in [false, true] {
                let mut a = Mat::from_fn(m, n, |i, j| {
                    value(if i == j { 4.0 + j as f64 } else { 0.03 })
                });
                // Rank deficient: the truncated factors keep fewer columns.
                if m > n {
                    for i in 0..m {
                        a[(i, n - 1)] = copy(&a[(i, 0)]);
                    }
                } else {
                    for j in 0..n {
                        a[(m - 1, j)] = copy(&a[(0, j)]);
                    }
                }
                let (qr, pqr) = (a.as_ref().qr(Par::Seq), a.as_ref().col_piv_qr(Par::Seq));
                let (b, c, r, p) = if pivoted {
                    (pqr.Q_basis(), pqr.Q_coeff(), pqr.R(), Some(pqr.P()))
                } else {
                    (qr.Q_basis(), qr.Q_coeff(), qr.R(), None)
                };
                // Attribution: faer's SVD kernel alone, with every output and
                // its workspace preallocated. Above small sizes its only
                // allocations are matrix-product packing buffers
                // (t4a-faer upstream-reports/gemm-001); at 7x3/3x7 it is 0.
                let kernel = svd_kernel_allocations(r);
                if m.max(n) < 8 {
                    assert_eq!(kernel, 0);
                }
                let req = QrPseudoInverse::<T>::construction_scratch(m, n, ParShape::Seq);
                let mut memory = faer::dyn_stack::MemBuffer::new(req);
                let mut saved = None;
                let count = crate::allocation_probe::count(|| {
                    saved = Some(
                        QrPseudoInverse::new_in(
                            b,
                            c,
                            r,
                            p,
                            0.0,
                            1e-5,
                            false,
                            m,
                            n,
                            Par::Seq,
                            MemStack::new(&mut memory),
                        )
                        .unwrap(),
                    )
                });
                assert!(
                    count <= RETAINED_BUFFERS + kernel,
                    "QR pseudoinverse construction allocated {count} buffers for {m}x{n} \
                     (SVD kernel alone: {kernel})"
                );
                let saved = saved.unwrap();
                assert_eq!(saved.middle.weights.len(), m.min(n) - 1);
                assert_eq!(saved.middle.u.ncols(), m.min(n) - 1);
                assert_eq!(saved.middle.v.ncols(), m.min(n) - 1);
                let inverse = a
                    .as_ref()
                    .thin_svd(Par::Seq)
                    .unwrap()
                    .pseudoinverse(Par::Seq);
                exercise::<T, false, false>(&saved, inverse.as_ref(), &team);
                exercise::<T, true, true>(&saved, inverse.as_ref(), &team);
                // Team construction: the declared team requirement suffices
                // and the result matches the sequential build.
                let parallel = team
                    .execute(|par| {
                        let req = QrPseudoInverse::<T>::construction_scratch(m, n, par.shape());
                        let mut memory = faer::dyn_stack::MemBuffer::new(req);
                        QrPseudoInverse::new_in(
                            b,
                            c,
                            r,
                            p,
                            0.0,
                            1e-5,
                            false,
                            m,
                            n,
                            par,
                            MemStack::new(&mut memory),
                        )
                        .unwrap()
                    })
                    .unwrap();
                assert_eq!(parallel.middle.weights.len(), m.min(n) - 1);
                exercise::<T, false, true>(&parallel, inverse.as_ref(), &team);
            }
        }
        team.team_close().unwrap();
    }
    #[test]
    fn f32_construction_uses_caller_scratch() {
        construction::<f32>(|x| x as f32);
    }
    #[test]
    fn f64_construction_uses_caller_scratch() {
        construction::<f64>(|x| x);
    }
    #[test]
    fn c32_construction_uses_caller_scratch() {
        construction::<faer::c32>(|x| faer::c32::new(x as f32, 0.25 * x as f32));
    }
    #[test]
    fn c64_construction_uses_caller_scratch() {
        construction::<faer::c64>(|x| faer::c64::new(x, 0.25 * x));
    }
    #[test]
    fn qr_constructor_rejects_native_metadata_before_replacing_operator() {
        let a = crate::OwnedMatrix {
            matrix: Mat::identity(4, 3),
        };
        let mut f = OwnedFactor::new();
        f.initialize(&a, 2, true, 4, 3, 1).unwrap();
        let mut out = OwnedLinearOperator::new();
        let mut info = [0usize; 2];
        operator_qr_pseudoinverse_requirements(4, 3, 1, info.as_mut_ptr()).unwrap();
        assert!(info[0] > 0);
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0] - 1, info[1]).unwrap();
        assert!(out
            .operator_qr_pseudoinverse(&f, &mut scratch, 0.0, 1e-12, false, 4, 3, 1)
            .is_err());
        assert!(out.data().is_err());
        scratch.scratch_reserve(info[0], info[1]).unwrap();
        assert!(out
            .operator_qr_pseudoinverse(&f, &mut scratch, 0.0, 1e-12, false, 4, 3, 0)
            .is_err());
        assert!(out.data().is_err());
        assert!(out
            .operator_qr_pseudoinverse(&f, &mut scratch, 0.0, 1e-12, false, 5, 3, 1)
            .is_err());
        assert!(out.data().is_err());
        out.operator_qr_pseudoinverse(&f, &mut scratch, 0.0, 1e-12, false, 4, 3, 1)
            .unwrap();
        assert!(out
            .operator_qr_pseudoinverse(&f, &mut scratch, -1.0, 1e-12, false, 4, 3, 1)
            .is_err());
        match out.data().unwrap() {
            Data::F64(a) => assert_eq!(a.shape(), (3, 4)),
            _ => panic!("unexpected scalar"),
        }
    }
}

#[cfg(test)]
mod truncation_tests {
    use super::*;
    #[test]
    fn discarded_spectral_row_cannot_propagate_infinity() {
        let v = 0.5f64.sqrt();
        let op = InverseSpectral {
            u: Mat::from_fn(2, 2, |i, j| if i == 1 && j == 1 { -v } else { v }),
            v: Mat::identity(2, 2),
            weights: vec![0.0, 1.0],
            zero: false,
        };
        let b = Mat::from_fn(2, 1, |_, _| f64::MAX);
        let mut out = Mat::zeros(2, 1);
        let mut memory = faer::dyn_stack::MemBuffer::new(op.apply_scratch(1, ParShape::Seq));
        op.apply(
            out.as_mut(),
            b.as_ref(),
            Par::Seq,
            MemStack::new(&mut memory),
        );
        assert_eq!(out[(0, 0)], 0.0);
        // The retained coordinate is an exact cancellation mathematically;
        // FMA targets may retain epsilon-scaled roundoff instead of zero.
        assert!(out[(1, 0)].abs() <= 4.0 * f64::EPSILON * f64::MAX);
    }
}
