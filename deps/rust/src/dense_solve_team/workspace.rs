//! Workspace-backed retained-factor solves. Results alone own a new allocation.
use super::{validate_owned, validate_typed};
use crate::typed::{Factor, FactorData, MatrixData, TypedFactor, TypedMatrix};
use crate::{guarded, DenseFactor, OwnedFactor, OwnedMatrix};
use crate::{teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{
    self, householder as hh,
    solvers::{self, ShapeCore},
    svd,
};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

enum Saved<'a, T: ComplexField> {
    Lu(&'a solvers::PartialPivLu<T>),
    FullLu(&'a solvers::FullPivLu<T>),
    Qr(&'a solvers::Qr<T>),
    PivotQr(&'a solvers::ColPivQr<T>),
    Llt(&'a solvers::Llt<T>),
    Ldlt(&'a solvers::Ldlt<T>),
    Lblt(&'a crate::lblt::Lblt<T>),
    Svd(&'a solvers::Svd<T>),
}
macro_rules! dispatch_saved {
    ($f:expr,$x:ident=>$body:expr) => {
        match $f {
            Saved::Lu($x) => $body,
            Saved::FullLu($x) => $body,
            Saved::Qr($x) => $body,
            Saved::PivotQr($x) => $body,
            Saved::Llt($x) => $body,
            Saved::Ldlt($x) => $body,
            Saved::Lblt($x) => $body,
            Saved::Svd($x) => $body,
        }
    };
}
impl<'a, T: ComplexField> From<&'a Factor<T>> for Saved<'a, T> {
    fn from(f: &'a Factor<T>) -> Self {
        match f {
            Factor::Lu(x) => Self::Lu(x),
            Factor::FullLu(x) => Self::FullLu(x),
            Factor::Qr(x) => Self::Qr(x),
            Factor::PivotQr(x) => Self::PivotQr(x),
            Factor::Llt(x) => Self::Llt(x),
            Factor::Ldlt(x) => Self::Ldlt(x),
            Factor::Lblt(x) => Self::Lblt(x),
            Factor::Svd(x) => Self::Svd(x),
        }
    }
}
fn owned(f: &OwnedFactor) -> Result<Saved<'_, f64>, String> {
    if f.info != 0 {
        return Err("singular factorization".into());
    }
    Ok(match f.get()? {
        DenseFactor::Lu(x) => Saved::Lu(x),
        DenseFactor::FullLu(x) => Saved::FullLu(x),
        DenseFactor::Qr(x) => Saved::Qr(x),
        DenseFactor::PivotQr(x) => Saved::PivotQr(x),
        DenseFactor::Llt(x) => Saved::Llt(x),
        DenseFactor::Ldlt(x) => Saved::Ldlt(x),
        DenseFactor::Lblt(x) => Saved::Lblt(x),
        DenseFactor::Svd(x) => Saved::Svd(x),
    })
}
fn qr_parts<'a, T: ComplexField>(
    f: &Saved<'a, T>,
) -> Option<(
    MatRef<'a, T>,
    MatRef<'a, T>,
    MatRef<'a, T>,
    Option<faer::perm::PermRef<'a, usize>>,
)> {
    match f {
        Saved::Qr(x) => Some((x.Q_basis(), x.Q_coeff(), x.thin_R(), None)),
        Saved::PivotQr(x) => Some((x.Q_basis(), x.Q_coeff(), x.thin_R(), Some(x.P()))),
        _ => None,
    }
}
fn deficient<T: ComplexField>(r: MatRef<'_, T>, precision: f64) -> bool {
    let tol = from_f64::<T::Real>(precision)
        * from_f64::<T::Real>(r.nrows().max(r.ncols()) as f64)
        * r.norm_max();
    (0..r.nrows().min(r.ncols())).any(|i| abs(&r[(i, i)]) <= tol)
}
fn spectral_qr<T: ComplexField>(f: &Saved<'_, T>, t: bool, precision: f64) -> bool {
    qr_parts(f).is_some_and(|(b, _, r, _)| {
        b.nrows() < r.ncols() || deficient(r, precision) || (t && b.nrows() != r.ncols())
    })
}
fn qr_req<T: ComplexField>(m: usize, n: usize, bs: usize, columns: usize, p: ParShape) -> StackReq {
    let k = m.min(n);
    StackReq::all_of(&[
  linalg::temp_mat_scratch::<T>(k,k),linalg::temp_mat_scratch::<T>(n,k),linalg::temp_mat_scratch::<T>(k,1),
  linalg::temp_mat_scratch::<T>(m,columns),linalg::temp_mat_scratch::<T>(n,columns),linalg::temp_mat_scratch::<T>(k,columns),
  svd::svd_scratch::<T>(k,n,svd::ComputeSvdVectors::Thin,svd::ComputeSvdVectors::Thin,p,Default::default()).or(
    hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m,bs,columns).or(hh::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(m,bs,columns)))])
}
fn lu_rect_req<T: ComplexField>(
    f: &Saved<'_, T>,
    m: usize,
    n: usize,
    columns: usize,
    p: ParShape,
) -> StackReq {
    let k = m.min(n);
    let reconstruct = if matches!(f, Saved::FullLu(_)) {
        linalg::lu::full_pivoting::reconstruct::reconstruct_scratch::<usize, T>(m, n, p)
    } else {
        linalg::lu::partial_pivoting::reconstruct::reconstruct_scratch::<usize, T>(m, n, p)
    };
    StackReq::all_of(&[
        linalg::temp_mat_scratch::<T>(m, n),
        linalg::temp_mat_scratch::<T>(m, k),
        linalg::temp_mat_scratch::<T>(n, k),
        linalg::temp_mat_scratch::<T>(k, 1),
        linalg::temp_mat_scratch::<T>(k, columns),
        reconstruct.or(svd::svd_scratch::<T>(
            m,
            n,
            svd::ComputeSvdVectors::Thin,
            svd::ComputeSvdVectors::Thin,
            p,
            Default::default(),
        )),
    ])
}
fn requirement_precision<T: ComplexField>(
    f: &Saved<'_, T>,
    columns: usize,
    op: i32,
    p: ParShape,
    precision: f64,
) -> Result<StackReq, String> {
    if !precision.is_finite() || precision < 0.0 {
        return Err("invalid solve source precision".into());
    }
    if !(0..4).contains(&op) {
        return Err("unknown factor solve transform".into());
    }
    let (m, n) = (
        dispatch_saved!(f,x=>x.nrows()),
        dispatch_saved!(f,x=>x.ncols()),
    );
    if m.min(n) == 0 || columns == 0 {
        return Ok(StackReq::EMPTY);
    }
    if m != n && matches!(f, Saved::Lu(_) | Saved::FullLu(_)) {
        return Ok(lu_rect_req::<T>(f, m, n, columns, p));
    }
    if m != n && !matches!(f, Saved::Qr(_) | Saved::PivotQr(_) | Saved::Svd(_)) {
        return Err("rectangular solve requires QR or SVD".into());
    }
    let t = op == 1 || op == 2;
    if spectral_qr(f, t, precision) {
        let (_, c, _, _) = qr_parts(f).unwrap();
        return Ok(qr_req::<T>(m, n, c.nrows(), columns, p));
    }
    Ok(match f {
        Saved::Svd(_) => linalg::temp_mat_scratch::<T>(m.min(n), columns),
        Saved::Lu(_) => {
            if t {
                linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(
                    n, columns, p,
                )
            } else {
                linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, T>(
                    n, columns, p,
                )
            }
        }
        Saved::FullLu(_) => {
            if t {
                linalg::lu::full_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(
                    n, columns, p,
                )
            } else {
                linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T>(n, columns, p)
            }
        }
        Saved::Llt(_) => linalg::cholesky::llt::solve::solve_in_place_scratch::<T>(n, columns, p),
        Saved::Ldlt(_) => linalg::cholesky::ldlt::solve::solve_in_place_scratch::<T>(n, columns, p),
        Saved::Lblt(_) => {
            linalg::cholesky::lblt::solve::solve_in_place_scratch::<usize, T>(n, columns, p)
        }
        Saved::Qr(x) => {
            if m != n {
                linalg::temp_mat_scratch::<T>(m, columns).and(
                    linalg::qr::no_pivoting::solve::solve_lstsq_in_place_scratch::<T>(
                        m,
                        n,
                        x.Q_coeff().nrows(),
                        columns,
                        p,
                    ),
                )
            } else if t {
                linalg::qr::no_pivoting::solve::solve_transpose_in_place_scratch::<T>(
                    n,
                    x.Q_coeff().nrows(),
                    columns,
                    p,
                )
            } else {
                linalg::qr::no_pivoting::solve::solve_in_place_scratch::<T>(
                    n,
                    x.Q_coeff().nrows(),
                    columns,
                    p,
                )
            }
        }
        Saved::PivotQr(x) => {
            if m != n {
                linalg::temp_mat_scratch::<T>(m, columns).and(
                    linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch::<usize, T>(
                        m,
                        n,
                        x.Q_coeff().nrows(),
                        columns,
                        p,
                    ),
                )
            } else if t {
                linalg::qr::col_pivoting::solve::solve_transpose_in_place_scratch::<usize, T>(
                    n,
                    x.Q_coeff().nrows(),
                    columns,
                    p,
                )
            } else {
                linalg::qr::col_pivoting::solve::solve_in_place_scratch::<usize, T>(
                    n,
                    x.Q_coeff().nrows(),
                    columns,
                    p,
                )
            }
        }
    })
}
// A retained SVD's U/V, or the SVD of retained QR's R, act without an explicit pseudoinverse.
fn spectral<T: ComplexField>(
    mut out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    u: MatRef<'_, T>,
    v: MatRef<'_, T>,
    s: faer::diag::DiagRef<'_, T>,
    mut projection: MatMut<'_, T>,
    original_order: usize,
    t: bool,
    c: bool,
    mut p: Par,
    precision: f64,
) {
    let tol = from_f64::<T::Real>(precision)
        * from_f64::<T::Real>(original_order as f64)
        * s.column_vector().norm_max();
    if !(0..s.dim()).any(|i| abs(&s[i]) > tol) {
        out.fill(zero::<T>());
        return;
    }
    let yes = if c { Conj::Yes } else { Conj::No };
    let no = yes.compose(Conj::Yes);
    linalg::matmul::matmul_with_conj(
        projection.rb_mut(),
        Accum::Replace,
        if t { v.transpose() } else { u.transpose() },
        if t { yes } else { no },
        b,
        Conj::No,
        one::<T>(),
        p.reborrow(),
    );
    for j in 0..b.ncols() {
        for i in 0..s.dim() {
            let val = real(&s[i]);
            projection[(i, j)] = if abs(&val) > tol {
                mul_real(&projection[(i, j)], &recip(&val))
            } else {
                zero::<T>()
            };
        }
    }
    linalg::matmul::matmul_with_conj(
        out,
        Accum::Replace,
        if t { u } else { v },
        if t { no } else { yes },
        projection.rb(),
        Conj::No,
        one::<T>(),
        p,
    );
}
fn qr_solve<T: ComplexField>(
    mut out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    basis: MatRef<'_, T>,
    coeff: MatRef<'_, T>,
    r: MatRef<'_, T>,
    perm: Option<faer::perm::PermRef<'_, usize>>,
    op: i32,
    mut p: Par,
    stack: &mut MemStack,
    precision: f64,
) -> Result<(), String> {
    let (m, n, k, columns) = (
        basis.nrows(),
        r.ncols(),
        basis.nrows().min(r.ncols()),
        b.ncols(),
    );
    let (mut u, stack) = linalg::temp_mat_zeroed::<T, _, _>(k, k, stack);
    let (mut v, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, k, stack);
    let (mut s, stack) = linalg::temp_mat_zeroed::<T, _, _>(k, 1, stack);
    let (mut work, stack) = linalg::temp_mat_zeroed::<T, _, _>(m, columns, stack);
    let (mut permuted, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, columns, stack);
    let (mut proj, stack) = linalg::temp_mat_zeroed::<T, _, _>(k, columns, stack);
    svd::svd(
        r,
        s.as_mat_mut().col_mut(0).as_diagonal_mut(),
        Some(u.as_mat_mut()),
        Some(v.as_mat_mut()),
        p.reborrow(),
        stack,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let t = op == 1 || op == 2;
    let c = op == 2 || op == 3;
    let diag = s.as_mat_ref().col(0).as_diagonal();
    let tol = from_f64::<T::Real>(precision)
        * from_f64::<T::Real>(m.max(n) as f64)
        * diag.column_vector().norm_max();
    if !(0..k).any(|i| abs(&diag[i]) > tol) {
        out.fill(zero::<T>());
        return Ok(());
    }
    let mut work = work.as_mat_mut();
    let mut permuted = permuted.as_mat_mut();
    if t {
        for j in 0..columns {
            for i in 0..n {
                permuted[(i, j)] = copy(&b[(perm.map_or(i, |p| p.arrays().0[i]), j)]);
            }
        }
        spectral(
            work.rb_mut().subrows_mut(0, k),
            permuted.rb(),
            u.as_mat_ref(),
            v.as_mat_ref(),
            diag,
            proj.as_mat_mut(),
            m.max(n),
            true,
            c,
            p.reborrow(),
            precision,
        );
        hh::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            basis,
            coeff,
            if c { Conj::No } else { Conj::Yes },
            work.rb_mut(),
            p,
            stack,
        );
        out.copy_from(work);
    } else {
        work.copy_from(b);
        hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            basis,
            coeff,
            if c { Conj::No } else { Conj::Yes },
            work.rb_mut(),
            p.reborrow(),
            stack,
        );
        spectral(
            permuted.rb_mut(),
            work.rb().subrows(0, k),
            u.as_mat_ref(),
            v.as_mat_ref(),
            diag,
            proj.as_mat_mut(),
            m.max(n),
            false,
            c,
            p,
            precision,
        );
        for j in 0..columns {
            for i in 0..n {
                out[(i, j)] = copy(&permuted[(perm.map_or(i, |p| p.arrays().1[i]), j)]);
            }
        }
    }
    Ok(())
}

fn lu_rect_solve<T: ComplexField>(
    f: &Saved<'_, T>,
    out: MatMut<'_, T>,
    b: MatRef<'_, T>,
    m: usize,
    n: usize,
    op: i32,
    precision: f64,
    mut par: Par,
    stack: &mut MemStack,
) -> Result<(), String> {
    let k = m.min(n);
    let (mut a, stack) = linalg::temp_mat_zeroed::<T, _, _>(m, n, stack);
    let (mut u, stack) = linalg::temp_mat_zeroed::<T, _, _>(m, k, stack);
    let (mut v, stack) = linalg::temp_mat_zeroed::<T, _, _>(n, k, stack);
    let (mut s, stack) = linalg::temp_mat_zeroed::<T, _, _>(k, 1, stack);
    let (mut proj, stack) = linalg::temp_mat_zeroed::<T, _, _>(k, b.ncols(), stack);
    match f {
        Saved::Lu(x) => linalg::lu::partial_pivoting::reconstruct::reconstruct(
            a.as_mat_mut(),
            x.L(),
            x.U(),
            x.P(),
            par.reborrow(),
            stack,
        ),
        Saved::FullLu(x) => linalg::lu::full_pivoting::reconstruct::reconstruct(
            a.as_mat_mut(),
            x.L(),
            x.U(),
            x.P(),
            x.Q(),
            par.reborrow(),
            stack,
        ),
        _ => unreachable!(),
    }
    svd::svd(
        a.as_mat_ref(),
        s.as_mat_mut().col_mut(0).as_diagonal_mut(),
        Some(u.as_mat_mut()),
        Some(v.as_mat_mut()),
        par.reborrow(),
        stack,
        Default::default(),
    )
    .map_err(|e| format!("{e:?}"))?;
    spectral(
        out,
        b,
        u.as_mat_ref(),
        v.as_mat_ref(),
        s.as_mat_ref().col(0).as_diagonal(),
        proj.as_mat_mut(),
        m.max(n),
        op == 1 || op == 2,
        op == 2 || op == 3,
        par,
        precision,
    );
    Ok(())
}
fn run_precision<T: ComplexField>(
    f: &Saved<'_, T>,
    b: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    op: i32,
    columns: usize,
    p: Par,
    precision: f64,
) -> Result<Mat<T>, String> {
    let (m, n) = (
        dispatch_saved!(f,x=>x.nrows()),
        dispatch_saved!(f,x=>x.ncols()),
    );
    let t = op == 1 || op == 2;
    let c = op == 2 || op == 3;
    if b.shape() != (if t { n } else { m }, columns) {
        return Err("factor RHS native dimensions do not match requested metadata".into());
    }
    if !b.is_all_finite() {
        return Err("solve RHS must be finite".into());
    }
    let stack = scratch.stack(requirement_precision(f, columns, op, p.shape(), precision)?)?;
    let mut out = Mat::zeros(if t { m } else { n }, columns);
    if m.min(n) == 0 || columns == 0 {
        return Ok(out);
    }
    if m != n && matches!(f, Saved::Lu(_) | Saved::FullLu(_)) {
        lu_rect_solve(f, out.as_mut(), b, m, n, op, precision, p, stack)?;
        return Ok(out);
    }
    if spectral_qr(f, t, precision) {
        let (basis, coeff, r, perm) = qr_parts(f).unwrap();
        qr_solve(
            out.as_mut(),
            b,
            basis,
            coeff,
            r,
            perm,
            op,
            p,
            stack,
            precision,
        )?;
        return Ok(out);
    }
    if let Saved::Svd(x) = f {
        let (mut proj, _) = linalg::temp_mat_zeroed::<T, _, _>(m.min(n), columns, stack);
        spectral(
            out.as_mut(),
            b,
            x.U().subcols(0, m.min(n)),
            x.V().subcols(0, m.min(n)),
            x.S(),
            proj.as_mat_mut(),
            m.max(n),
            t,
            c,
            p,
            precision,
        );
        return Ok(out);
    }
    if m != n {
        let (mut work, stack) = linalg::temp_mat_zeroed::<T, _, _>(m, columns, stack);
        work.as_mat_mut().copy_from(b);
        match f {
            Saved::Qr(x) => linalg::qr::no_pivoting::solve::solve_lstsq_in_place_with_conj(
                x.Q_basis(),
                x.Q_coeff(),
                x.R(),
                if c { Conj::Yes } else { Conj::No },
                work.as_mat_mut(),
                p,
                stack,
            ),
            Saved::PivotQr(x) => linalg::qr::col_pivoting::solve::solve_lstsq_in_place_with_conj(
                x.Q_basis(),
                x.Q_coeff(),
                x.R(),
                x.P(),
                if c { Conj::Yes } else { Conj::No },
                work.as_mat_mut(),
                p,
                stack,
            ),
            _ => return Err("rectangular solve requires QR or SVD".into()),
        }
        out.as_mut().copy_from(work.as_mat_ref().subrows(0, n));
        return Ok(out);
    }
    out.as_mut().copy_from(b);
    let conj = if c { Conj::Yes } else { Conj::No };
    match f {
        Saved::Lu(x) => {
            if t {
                linalg::lu::partial_pivoting::solve::solve_transpose_in_place_with_conj(
                    x.L(),
                    x.U(),
                    x.P(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            } else {
                linalg::lu::partial_pivoting::solve::solve_in_place_with_conj(
                    x.L(),
                    x.U(),
                    x.P(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            }
        }
        Saved::FullLu(x) => {
            if t {
                linalg::lu::full_pivoting::solve::solve_transpose_in_place_with_conj(
                    x.L(),
                    x.U(),
                    x.P(),
                    x.Q(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            } else {
                linalg::lu::full_pivoting::solve::solve_in_place_with_conj(
                    x.L(),
                    x.U(),
                    x.P(),
                    x.Q(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            }
        }
        Saved::Qr(x) => {
            if t {
                linalg::qr::no_pivoting::solve::solve_transpose_in_place_with_conj(
                    x.Q_basis(),
                    x.Q_coeff(),
                    x.R(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            } else {
                linalg::qr::no_pivoting::solve::solve_in_place_with_conj(
                    x.Q_basis(),
                    x.Q_coeff(),
                    x.R(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            }
        }
        Saved::PivotQr(x) => {
            if t {
                linalg::qr::col_pivoting::solve::solve_transpose_in_place_with_conj(
                    x.Q_basis(),
                    x.Q_coeff(),
                    x.R(),
                    x.P(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            } else {
                linalg::qr::col_pivoting::solve::solve_in_place_with_conj(
                    x.Q_basis(),
                    x.Q_coeff(),
                    x.R(),
                    x.P(),
                    conj,
                    out.as_mut(),
                    p,
                    stack,
                )
            }
        }

        Saved::Llt(x) => linalg::cholesky::llt::solve::solve_in_place_with_conj(
            x.L(),
            if t { conj.compose(Conj::Yes) } else { conj },
            out.as_mut(),
            p,
            stack,
        ),
        Saved::Ldlt(x) => linalg::cholesky::ldlt::solve::solve_in_place_with_conj(
            x.L(),
            x.D(),
            if t { conj.compose(Conj::Yes) } else { conj },
            out.as_mut(),
            p,
            stack,
        ),
        Saved::Lblt(x) => linalg::cholesky::lblt::solve::solve_in_place_with_conj(
            x.L(),
            x.B_diag(),
            x.B_subdiag(),
            if t { conj.compose(Conj::Yes) } else { conj },
            x.P(),
            out.as_mut(),
            p,
            stack,
        ),
        Saved::Svd(_) => unreachable!(),
    }
    Ok(out)
}
fn write_req(req: StackReq, out: *mut usize) -> Result<(), String> {
    if out.is_null()
        || (out as usize) % std::mem::align_of::<usize>() != 0
        || (out as usize)
            .checked_add(2 * std::mem::size_of::<usize>())
            .is_none()
    {
        return Err("query requires aligned two-entry usize buffer".into());
    }
    // SAFETY: caller preserves a two-entry writable UInt buffer; address/span checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
impl OwnedFactor {
    #[julia]
    pub fn factor_solve_requirements(
        &self,
        columns: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate_owned(self, m, n, tag, kind)?;
            let f = owned(self)?;
            write_req(
                requirement_precision(&f, columns, op, par.shape(), precision)?,
                out,
            )
        })
    }
}
#[julia]
impl OwnedFactor {
    #[julia]
    pub fn factor_solve_requirements_team(
        &self,
        columns: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate_owned(self, m, n, tag, kind)?;
                let f = owned(self)?;
                write_req(
                    requirement_precision(&f, columns, op, par.shape(), precision)?,
                    out,
                )
            })?
        })
    }
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn factor_solve_requirements(
        &self,
        columns: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate_typed(self, m, n, tag, kind)?;
            self.solve_ready()?;
            let req = match self.data()? {
                FactorData::F32(f) => {
                    requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                }
                FactorData::F64(f) => {
                    requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                }
                FactorData::C32(f) => {
                    requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                }
                FactorData::C64(f) => {
                    requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                }
            };
            write_req(req, out)
        })
    }
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn factor_solve_requirements_team(
        &self,
        columns: usize,
        op: i32,
        m: usize,
        n: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        out: *mut usize,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate_typed(self, m, n, tag, kind)?;
                self.solve_ready()?;
                let req = match self.data()? {
                    FactorData::F32(f) => {
                        requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                    }
                    FactorData::F64(f) => {
                        requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                    }
                    FactorData::C32(f) => {
                        requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                    }
                    FactorData::C64(f) => {
                        requirement_precision(&Saved::from(f), columns, op, par.shape(), precision)?
                    }
                };
                write_req(req, out)
            })?
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn factor_solve_workspace(
        &mut self,
        f: &OwnedFactor,
        rhs: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        precision: f64,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate_owned(f, m, n, tag, kind)?;
            self.matrix = run_precision(
                &owned(f)?,
                rhs.matrix.as_ref(),
                scratch,
                op,
                columns,
                par,
                precision,
            )?;
            Ok(())
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn factor_solve_workspace_team(
        &mut self,
        f: &OwnedFactor,
        rhs: &OwnedMatrix,
        scratch: &mut OwnedScratch,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate_owned(f, m, n, tag, kind)?;
                self.matrix = run_precision(
                    &owned(f)?,
                    rhs.matrix.as_ref(),
                    scratch,
                    op,
                    columns,
                    par,
                    precision,
                )?;
                Ok(())
            })?
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn factor_solve_workspace(
        &mut self,
        f: &TypedFactor,
        rhs: &TypedMatrix,
        scratch: &mut OwnedScratch,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        precision: f64,
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
            validate_typed(f, m, n, tag, kind)?;
            f.solve_ready()?;
            self.data = Some(match (f.data()?, rhs.data()?) {
                (FactorData::F32(f), MatrixData::F32(b)) => MatrixData::F32(run_precision(
                    &Saved::from(f),
                    b.as_ref(),
                    scratch,
                    op,
                    columns,
                    par,
                    precision,
                )?),
                (FactorData::F64(f), MatrixData::F64(b)) => MatrixData::F64(run_precision(
                    &Saved::from(f),
                    b.as_ref(),
                    scratch,
                    op,
                    columns,
                    par,
                    precision,
                )?),
                (FactorData::C32(f), MatrixData::C32(b)) => MatrixData::C32(run_precision(
                    &Saved::from(f),
                    b.as_ref(),
                    scratch,
                    op,
                    columns,
                    par,
                    precision,
                )?),
                (FactorData::C64(f), MatrixData::C64(b)) => MatrixData::C64(run_precision(
                    &Saved::from(f),
                    b.as_ref(),
                    scratch,
                    op,
                    columns,
                    par,
                    precision,
                )?),
                _ => return Err("factor/RHS native scalar types must match".into()),
            });
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn factor_solve_workspace_team(
        &mut self,
        f: &TypedFactor,
        rhs: &TypedMatrix,
        scratch: &mut OwnedScratch,
        op: i32,
        m: usize,
        n: usize,
        columns: usize,
        tag: i32,
        kind: i32,
        precision: f64,
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
                validate_typed(f, m, n, tag, kind)?;
                f.solve_ready()?;
                self.data = Some(match (f.data()?, rhs.data()?) {
                    (FactorData::F32(f), MatrixData::F32(b)) => MatrixData::F32(run_precision(
                        &Saved::from(f),
                        b.as_ref(),
                        scratch,
                        op,
                        columns,
                        par,
                        precision,
                    )?),
                    (FactorData::F64(f), MatrixData::F64(b)) => MatrixData::F64(run_precision(
                        &Saved::from(f),
                        b.as_ref(),
                        scratch,
                        op,
                        columns,
                        par,
                        precision,
                    )?),
                    (FactorData::C32(f), MatrixData::C32(b)) => MatrixData::C32(run_precision(
                        &Saved::from(f),
                        b.as_ref(),
                        scratch,
                        op,
                        columns,
                        par,
                        precision,
                    )?),
                    (FactorData::C64(f), MatrixData::C64(b)) => MatrixData::C64(run_precision(
                        &Saved::from(f),
                        b.as_ref(),
                        scratch,
                        op,
                        columns,
                        par,
                        precision,
                    )?),
                    _ => return Err("factor/RHS native scalar types must match".into()),
                });
                Ok(())
            })?
        })
    }
}
#[cfg(test)]
fn requirement<T: ComplexField>(
    f: &Saved<'_, T>,
    columns: usize,
    op: i32,
    p: ParShape,
) -> Result<StackReq, String> {
    requirement_precision(
        f,
        columns,
        op,
        p,
        if eps::<T::Real>() > from_f64::<T::Real>(1e-10) {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        },
    )
}
#[cfg(test)]
fn run<T: ComplexField>(
    f: &Saved<'_, T>,
    b: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    op: i32,
    columns: usize,
    p: Par,
) -> Result<Mat<T>, String> {
    run_precision(
        f,
        b,
        scratch,
        op,
        columns,
        p,
        if eps::<T::Real>() > from_f64::<T::Real>(1e-10) {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        },
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::{c32, c64, Side};
    fn exercise<T: ComplexField>() {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for kind in 0..8 {
            let a = Mat::from_fn(5, 5, |i, j| {
                from_f64::<T>(if i == j { 6.0 + i as f64 } else { 0.1 })
            });
            let f = match kind {
                0 => Factor::Lu(a.partial_piv_lu(Par::Seq)),
                1 => Factor::FullLu(a.full_piv_lu(Par::Seq)),
                2 => Factor::Qr(a.qr(Par::Seq)),
                3 => Factor::PivotQr(a.col_piv_qr(Par::Seq)),
                4 => Factor::Llt(a.llt(Side::Lower, Par::Seq).unwrap()),
                5 => Factor::Ldlt(a.ldlt(Side::Lower, Par::Seq).unwrap()),
                6 => Factor::Lblt(crate::lblt::Lblt::new(
                    a.as_ref(),
                    Side::Lower,
                    Par::Seq,
                    false,
                )),
                _ => Factor::Svd(a.thin_svd(Par::Seq).unwrap()),
            };
            check(&Saved::from(&f), a.as_ref(), &team);
        }
        for (m, n, rankdef) in [
            (7, 3, false),
            (3, 7, false),
            (7, 3, true),
            (3, 7, true),
            (0, 3, false),
            (3, 0, false),
        ] {
            let a = Mat::from_fn(m, n, |i, j| {
                if rankdef && j == n - 1 {
                    from_f64::<T>(if i == 0 { 4.0 } else { 0.1 })
                } else {
                    from_f64::<T>(if i == j { 4.0 + i as f64 } else { 0.1 })
                }
            });
            if !rankdef && m.min(n) > 0 {
                for f in [
                    Factor::Lu(a.partial_piv_lu(Par::Seq)),
                    Factor::FullLu(a.full_piv_lu(Par::Seq)),
                ] {
                    check(&Saved::from(&f), a.as_ref(), &team);
                }
            }
            for pivot in [false, true] {
                let f = if pivot {
                    Factor::PivotQr(a.col_piv_qr(Par::Seq))
                } else {
                    Factor::Qr(a.qr(Par::Seq))
                };
                check(&Saved::from(&f), a.as_ref(), &team);
            }
        }
    }
    fn check<T: ComplexField>(f: &Saved<'_, T>, a: MatRef<'_, T>, team: &OwnedTeam) {
        for op in 0..4 {
            let t = op == 1 || op == 2;
            let c = op == 2 || op == 3;
            let effective = Mat::from_fn(
                if t { a.ncols() } else { a.nrows() },
                if t { a.nrows() } else { a.ncols() },
                |i, j| {
                    let x = if t { &a[(j, i)] } else { &a[(i, j)] };
                    if c {
                        conj(x)
                    } else {
                        copy(x)
                    }
                },
            );
            let b = Mat::from_fn(effective.nrows(), 2, |i, j| {
                from_f64::<T>((1 + i + 2 * j) as f64)
            });
            let gold = if a.nrows().min(a.ncols()) == 0 {
                Mat::zeros(effective.ncols(), 2)
            } else {
                &effective
                    .thin_svd(Par::Seq)
                    .unwrap()
                    .pseudoinverse(Par::Seq)
                    * &b
            };
            for threaded in [false, true] {
                let req = if threaded {
                    team.execute(|p| requirement(f, 2, op, p.shape()))
                        .unwrap()
                        .unwrap()
                } else {
                    requirement(f, 2, op, ParShape::Seq).unwrap()
                };
                let mut scratch = OwnedScratch::new();
                scratch
                    .scratch_reserve(req.size_bytes(), req.align_bytes())
                    .unwrap();
                let out = if threaded {
                    team.execute(|p| run(f, b.as_ref(), &mut scratch, op, 2, p))
                        .unwrap()
                        .unwrap()
                } else {
                    run(f, b.as_ref(), &mut scratch, op, 2, Par::Seq).unwrap()
                };
                let magnitude = gold.norm_max();
                let scale = if magnitude > one::<T::Real>() {
                    magnitude
                } else {
                    one::<T::Real>()
                };
                assert!(
                    (&out - &gold).norm_max()
                        <= from_f64::<T::Real>(10000.0) * eps::<T::Real>() * scale
                );
                if req.size_bytes() > 0 {
                    let mut short = OwnedScratch::new();
                    short
                        .scratch_reserve(req.size_bytes() - 1, req.align_bytes())
                        .unwrap();
                    assert!(run(f, b.as_ref(), &mut short, op, 2, Par::Seq).is_err());
                }
            }
        }
    }
    #[test]
    fn source_precision_changes_workspace_and_discards_spurious_rank() {
        let a: Mat<f64> = Mat::from_fn(3, 2, |i, j| {
            if i == j {
                if i == 0 {
                    1.0
                } else {
                    1e-8
                }
            } else {
                0.0
            }
        });
        let f = Factor::Qr(a.qr(Par::Seq));
        let saved = Saved::from(&f);
        let precision = f32::EPSILON as f64;
        let req = requirement_precision(&saved, 1, 0, ParShape::Seq, precision).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let b = Mat::full(3, 1, 1.0);
        let x = run_precision(&saved, b.as_ref(), &mut scratch, 0, 1, Par::Seq, precision).unwrap();
        assert!((x[(0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(x[(1, 0)], 0.0);
        assert!(requirement_precision(&saved, 1, 0, ParShape::Seq, f64::NAN).is_err());
    }
    #[test]
    fn workspace_f32() {
        exercise::<f32>();
    }
    #[test]
    fn workspace_f64() {
        exercise::<f64>();
    }
    #[test]
    fn workspace_c32() {
        exercise::<c32>();
    }
    #[test]
    fn workspace_c64() {
        exercise::<c64>();
    }
    #[test]
    fn owned_boundary_preserves_output_on_bad_metadata_and_closed_resources() {
        let a = OwnedMatrix {
            matrix: Mat::identity(4, 4),
        };
        let mut f = OwnedFactor::new();
        f.initialize(&a, 0, true, 4, 4, 1).unwrap();
        let mut out = OwnedMatrix {
            matrix: Mat::identity(4, 4),
        };
        let mut info = [0; 2];
        f.factor_solve_requirements(4, 0, 4, 4, 1, 0, f64::EPSILON, info.as_mut_ptr())
            .unwrap();
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(info[0], info[1]).unwrap();
        out.factor_solve_workspace(&f, &a, &mut scratch, 0, 4, 4, 4, 1, 0, f64::EPSILON)
            .unwrap();
        for (op, m, n, k, tag, kind) in [
            (8, 4, 4, 4, 1, 0),
            (0, 5, 4, 4, 1, 0),
            (0, 4, 4, 5, 1, 0),
            (0, 4, 4, 4, 2, 0),
            (0, 4, 4, 4, 1, 2),
        ] {
            assert!(out
                .factor_solve_workspace(&f, &a, &mut scratch, op, m, n, k, tag, kind, f64::EPSILON)
                .is_err());
            assert_eq!(out.matrix, a.matrix);
        }
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.team_close().unwrap();
        assert!(out
            .factor_solve_workspace_team(
                &f,
                &a,
                &mut scratch,
                0,
                4,
                4,
                4,
                1,
                0,
                f64::EPSILON,
                &team
            )
            .is_err());
        assert_eq!(out.matrix, a.matrix);
        scratch.scratch_close().unwrap();
        assert!(out
            .factor_solve_workspace(&f, &a, &mut scratch, 0, 4, 4, 4, 1, 0, f64::EPSILON)
            .is_err());
        assert_eq!(out.matrix, a.matrix);
    }
}
// Check array owners before Julia-side promotion/conversion touches raw buffers.
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn factor_rhs_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        layout: i32,
    ) -> Result<(), String> {
        guarded(|| {
            if (self.matrix.nrows(), self.matrix.ncols(), 1, 3) == (m, n, tag, layout) {
                Ok(())
            } else {
                Err("dense solve array native metadata mismatch".into())
            }
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn factor_rhs_validate(
        &self,
        m: usize,
        n: usize,
        tag: i32,
        layout: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let (rows, cols, t) = match self.data()? {
                MatrixData::F32(a) => (a.nrows(), a.ncols(), 0),
                MatrixData::F64(a) => (a.nrows(), a.ncols(), 1),
                MatrixData::C32(a) => (a.nrows(), a.ncols(), 2),
                MatrixData::C64(a) => (a.nrows(), a.ncols(), 3),
            };
            if (rows, cols, t, 3) == (m, n, tag, layout) {
                Ok(())
            } else {
                Err("dense solve array native metadata mismatch".into())
            }
        })
    }
}
