//! Rank decisions and rank-truncated minimum-norm solves for column-pivoted QR.
//! faer's `ColPivQr` exposes no rank or tolerance, so both follow Julia's
//! `QRPivoted` rule on the stored R diagonal: rank is the first index with
//! `|R[i,i]| <= max(atol, rtol*|R[1,1]|)`. The truncated solve uses a complete
//! orthogonal decomposition of the leading rank rows, like LAPACK `xgelsy`.
use crate::lu_qr_factors::ColPivQr;
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{Factor, FactorData, MatrixData, TypedFactor, TypedMatrix};
use crate::{guarded, DenseFactor, OwnedFactor, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_transpose_on_the_left_in_place_scratch,
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, Mat, MatRef, Par};
use rustcall_julia_macros::julia;

fn rank<T: ComplexField>(f: &ColPivQr<T>, atol: f64, rtol: f64) -> usize {
    rank_of(f.thin_R(), atol, rtol)
}
/// Rank of a stored R (only its diagonal is read) with Julia's `QRPivoted` rule.
pub(crate) fn rank_of<T: ComplexField>(r: MatRef<'_, T>, atol: f64, rtol: f64) -> usize {
    let k = r.nrows().min(r.ncols());
    if k == 0 {
        return 0;
    }
    let first = abs(&r[(0, 0)]);
    let scaled = mul(&from_f64::<T::Real>(rtol), &first);
    let atol = from_f64::<T::Real>(atol);
    let tol = if atol > scaled { atol } else { scaled };
    (0..k).find(|&i| abs(&r[(i, i)]) <= tol).unwrap_or(k)
}
/// What the truncated solve reads from a column-pivoted QR: the m×min(m,n)
/// Householder basis, its block coefficients, the min(m,n)×n R (only its upper
/// triangle is read, so packed storage works) and the forward column permutation.
pub(crate) struct PivotedQrParts<'a, T> {
    pub basis: MatRef<'a, T>,
    pub coeff: MatRef<'a, T>,
    pub r: MatRef<'a, T>,
    pub forward: &'a [usize],
}
fn truncated_solve<T: ComplexField>(
    f: &ColPivQr<T>,
    b: MatRef<'_, T>,
    r: usize,
    par: Par<'_>,
) -> Result<Mat<T>, String> {
    let parts = PivotedQrParts {
        basis: f.Q_basis(),
        coeff: f.Q_coeff(),
        r: f.thin_R(),
        forward: f.P().arrays().0,
    };
    truncated_solve_parts(parts, b, r, par)
}
/// Minimum-norm solution of the problem truncated to the leading `r` rows of
/// R. The caller chooses `r` with `rank`, so every tolerance form shares one cutoff.
pub(crate) fn truncated_solve_parts<T: ComplexField>(
    f: PivotedQrParts<'_, T>,
    b: MatRef<'_, T>,
    r: usize,
    mut par: Par<'_>,
) -> Result<Mat<T>, String> {
    let (m, n) = (f.basis.nrows(), f.r.ncols());
    if f.forward.len() != n || f.r.nrows() != m.min(n) {
        return Err("column-pivoted QR parts do not match".into());
    }
    if b.nrows() != m {
        return Err("solve dimensions mismatch".into());
    }
    if r > m.min(n) {
        return Err("truncation rank exceeds min(size(F))".into());
    }
    crate::typed::finite(b)?;
    let k = b.ncols();
    let mut x = Mat::<T>::zeros(n, k);
    if r == 0 || k == 0 {
        return Ok(x);
    }
    let mut c = b.to_owned();
    let (basis, coeff) = (f.basis, f.coeff);
    let req = apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
        m,
        coeff.nrows(),
        k,
    );
    let mut buffer = MemBuffer::try_new(req).map_err(|_| "scratch allocation failed")?;
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
        basis,
        coeff,
        Conj::Yes,
        c.as_mut(),
        par.reborrow(),
        MemStack::new(&mut buffer),
    );
    let r1 = Mat::from_fn(
        r,
        n,
        |i, j| {
            if i <= j {
                copy(&f.r[(i, j)])
            } else {
                zero()
            }
        },
    );
    let r1 = r1.as_ref();
    let mut y = c.as_ref().subrows(0, r).to_owned();
    let z = if r == n {
        solve_upper_triangular_in_place(r1.subcols(0, r), y.as_mut(), par.reborrow());
        y
    } else {
        // R1 = T^H Z^H with R1^H = Z T, so the minimum-norm y is Z T^-H c1.
        let cod = r1.adjoint().to_owned().qr(par.reborrow());
        let t = cod.thin_R();
        solve_lower_triangular_in_place(t.adjoint(), y.as_mut(), par.reborrow());
        cod.compute_thin_Q(par.reborrow()) * y
    };
    let forward = f.forward;
    for j in 0..k {
        for i in 0..n {
            x[(forward[i], j)] = copy(&z[(i, j)]);
        }
    }
    Ok(x)
}
fn owned(f: &OwnedFactor) -> Result<&ColPivQr<f64>, String> {
    match f.get()? {
        DenseFactor::PivotQr(x) => Ok(x),
        _ => Err("not a column-pivoted QR factorization".into()),
    }
}
macro_rules! typed_qr {
    ($f:expr, $x:ident => $body:expr) => {
        match $f.data()? {
            FactorData::F32(Factor::PivotQr($x)) => $body,
            FactorData::F64(Factor::PivotQr($x)) => $body,
            FactorData::C32(Factor::PivotQr($x)) => $body,
            FactorData::C64(Factor::PivotQr($x)) => $body,
            _ => Err("not a column-pivoted QR factorization".into()),
        }
    };
}

#[julia]
impl OwnedFactor {
    #[julia]
    pub fn pivoted_qr_rank(&self, atol: f64, rtol: f64) -> Result<usize, String> {
        guarded(|| Ok(rank(owned(self)?, atol, rtol)))
    }
}
#[julia]
impl TypedFactor {
    #[julia]
    pub fn pivoted_qr_rank(&self, atol: f64, rtol: f64) -> Result<usize, String> {
        guarded(|| typed_qr!(self, x => Ok(rank(x, atol, rtol))))
    }
}
#[julia]
impl OwnedMatrix {
    /// Writes the solution truncated to the leading `rank` rows of R into `self`.
    #[julia]
    pub fn pivoted_qr_truncated_solve(
        &mut self,
        f: &OwnedFactor,
        rhs: &OwnedMatrix,
        rank: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = with_team(team, |par| {
                truncated_solve(owned(f)?, rhs.matrix.as_ref(), rank, par)
            })?;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn pivoted_qr_truncated_solve(
        &mut self,
        f: &TypedFactor,
        rhs: &TypedMatrix,
        rank: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            macro_rules! go {
                ($x:expr, $V:ident) => {{
                    let MatrixData::$V(b) = rhs.data()? else {
                        return Err("factor and right-hand side scalar types must match".into());
                    };
                    let x = with_team(team, |par| truncated_solve($x, b.as_ref(), rank, par))?;
                    self.data = Some(MatrixData::$V(x));
                    Ok(())
                }};
            }
            match f.data()? {
                FactorData::F32(Factor::PivotQr(x)) => go!(x, F32),
                FactorData::F64(Factor::PivotQr(x)) => go!(x, F64),
                FactorData::C32(Factor::PivotQr(x)) => go!(x, C32),
                FactorData::C64(Factor::PivotQr(x)) => go!(x, C64),
                _ => Err("not a column-pivoted QR factorization".into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_and_minimum_norm_truncation() {
        // Rank two: the second column is twice the first.
        let a = Mat::from_fn(4, 3, |i, j| {
            let base = [1.0, 2.0, 1.0, 0.0][i];
            match j {
                0 => base,
                1 => 2.0 * base,
                _ => [3.0, 6.0, 1.0, 1.0][i],
            }
        });
        let f = ColPivQr::new(a.as_ref(), Par::Seq);
        assert_eq!(rank(&f, 0.0, 3.0 * f64::EPSILON), 2);
        assert_eq!(rank(&f, 1e3, 0.0), 0);
        let b = Mat::from_fn(4, 1, |i, _| (i + 1) as f64);
        let r = rank(&f, 0.0, 1e-10);
        assert_eq!(r, 2);
        let x = truncated_solve(&f, b.as_ref(), r, Par::Seq).unwrap();
        // The minimum-norm solution satisfies the normal equations and lies in
        // the row space: x1 and x2 are proportional to (1, 2).
        let res = &a * &x - &b;
        let g = a.transpose() * res;
        assert!(g.norm_max() < 1e-10);
        assert!((x[(1, 0)] - 2.0 * x[(0, 0)]).abs() < 1e-10);
        assert!(truncated_solve(&f, b.as_ref(), 4, Par::Seq).is_err());
        let mut bad = b.clone();
        bad[(0, 0)] = f64::INFINITY;
        assert!(truncated_solve(&f, bad.as_ref(), r, Par::Seq).is_err());
    }
}
