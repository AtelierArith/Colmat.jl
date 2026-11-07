//! Dense factorizations, spectral decompositions and retained-factor solves
//! that read their input (or write their right-hand side) through a view
//! descriptor: native owners and views, and borrowed Julia strided buffers
//! (descriptor kind 4). The factor still owns its numerical storage; only the
//! intermediate input snapshot is avoided. Each `MatRef`/`MatMut` lives for
//! one call while Julia roots the owner and descriptor.
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{self, Factor, FactorData, TypedFactor};
use crate::views::{destination, source, Borrowed, BorrowedMut, Spec};
use crate::{guarded, numerically_singular_r, DenseFactor, OwnedFactor};
use faer::linalg::solvers::{ShapeCore, SolveCore};
use faer::traits::ComplexField;
use faer::{Conj, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

fn tag_of(a: &Borrowed<'_>) -> i32 {
    match a {
        Borrowed::F32(_) => 0,
        Borrowed::F64(_) => 1,
        Borrowed::C32(_) => 2,
        Borrowed::C64(_) => 3,
    }
}
fn shape_of(a: &Borrowed<'_>) -> (usize, usize) {
    match a {
        Borrowed::F32(x) => x.shape(),
        Borrowed::F64(x) => x.shape(),
        Borrowed::C32(x) => x.shape(),
        Borrowed::C64(x) => x.shape(),
    }
}
/// The unconjugated input named by `p`/`desc`, checked against the Julia
/// metadata. Conjugated views have no in-place faer reading here; Julia
/// snapshots them first.
///
/// # Safety
/// `p`/`desc` name a live owner or rooted Julia buffer for this call.
unsafe fn input<'a>(
    p: *const u8,
    desc: *const isize,
    m: usize,
    n: usize,
    tag: i32,
) -> Result<Borrowed<'a>, String> {
    let s = Spec::read(desc)?;
    if s.conjugate {
        return Err("factor inputs cannot be conjugated views".into());
    }
    let a = source(p, s)?;
    if tag_of(&a) != tag || shape_of(&a) != (m, n) {
        return Err(
            "factor input native scalar type/dimensions do not match requested metadata".into(),
        );
    }
    Ok(a)
}

#[julia]
impl OwnedFactor {
    /// Factor the Float64 view named by `p`/`desc` without first copying it
    /// into an owned matrix.
    #[julia]
    pub fn initialize_view(
        &mut self,
        p: *const u8,
        desc: *const isize,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let a = match unsafe { input(p, desc, m, n, tag)? } {
                Borrowed::F64(a) => a,
                _ => return Err("Float64 factors require a Float64 input".into()),
            };
            with_team(team, |par| self.initialize_ref(a, kind, lower, par))
        })
    }
    /// Solve in place in the destination view named by `p`/`desc` with a
    /// square, numerically nonsingular factor. Returns `false` without
    /// touching the destination when the factor needs the rank-revealing or
    /// least-squares path (rectangular, rank-deficient QR, SVD).
    #[julia]
    pub fn solve_in_place_view(
        &self,
        p: *mut u8,
        desc: *const isize,
        op: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<bool, String> {
        guarded(|| {
            if self.info != 0 {
                return Err("singular factorization".into());
            }
            let f = self.get()?;
            let square = dispatch_factor!(f, x => x.nrows() == x.ncols());
            let deficient = match f {
                DenseFactor::Qr(x) => numerically_singular_r(x.thin_R()),
                DenseFactor::PivotQr(x) => numerically_singular_r(x.thin_R()),
                DenseFactor::Svd(_) => true,
                _ => false,
            };
            if !square || deficient {
                return Ok(false);
            }
            let n = dispatch_factor!(f, x => x.nrows());
            let rhs = match unsafe { destination(p, Spec::read(desc)?)? } {
                BorrowedMut::F64(b) => b,
                _ => return Err("Float64 factors require a Float64 right-hand side".into()),
            };
            check_rhs(rhs.as_ref(), n)?;
            with_team(
                team,
                |par| dispatch_factor!(f, x => in_place(x, rhs, op, par)),
            )?;
            Ok(true)
        })
    }
}

#[julia]
impl TypedFactor {
    /// Factor the view named by `p`/`desc` (any native scalar) without first
    /// copying it into an owned matrix.
    #[julia]
    pub fn initialize_view(
        &mut self,
        p: *const u8,
        desc: *const isize,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let a = unsafe { input(p, desc, m, n, tag)? };
            with_team(team, |par| self.initialize_ref(a, kind, lower, par))
        })
    }
    /// Typed counterpart of `OwnedFactor::solve_in_place_view`.
    #[julia]
    pub fn solve_in_place_view(
        &self,
        p: *mut u8,
        desc: *const isize,
        op: i32,
        team: Option<&OwnedTeam>,
    ) -> Result<bool, String> {
        guarded(|| {
            self.solve_ready()?;
            let rhs = unsafe { destination(p, Spec::read(desc)?)? };
            match (self.data()?, rhs) {
                (FactorData::F32(f), BorrowedMut::F32(b)) => typed_in_place(f, b, op, team),
                (FactorData::F64(f), BorrowedMut::F64(b)) => typed_in_place(f, b, op, team),
                (FactorData::C32(f), BorrowedMut::C32(b)) => typed_in_place(f, b, op, team),
                (FactorData::C64(f), BorrowedMut::C64(b)) => typed_in_place(f, b, op, team),
                _ => Err("factor and right-hand side native scalar types must match".into()),
            }
        })
    }
}

fn check_rhs<T: ComplexField>(b: MatRef<'_, T>, n: usize) -> Result<(), String> {
    if b.nrows() != n {
        return Err("solve dimensions mismatch".into());
    }
    typed::finite(b)
}
/// op 0: A x = b, 1: Aᵀ x = b, 2: Aᴴ x = b, 3: conj(A) x = b.
fn in_place<T: ComplexField, F: SolveCore<T>>(
    f: &F,
    rhs: MatMut<'_, T>,
    op: i32,
    par: Par<'_>,
) -> Result<(), String> {
    match op {
        0 => f.solve_in_place_with_conj(Conj::No, rhs, par),
        1 => f.solve_transpose_in_place_with_conj(Conj::No, rhs, par),
        2 => f.solve_transpose_in_place_with_conj(Conj::Yes, rhs, par),
        3 => f.solve_in_place_with_conj(Conj::Yes, rhs, par),
        _ => return Err("unknown solve transform".into()),
    }
    Ok(())
}
fn typed_in_place<T: ComplexField>(
    f: &Factor<T>,
    rhs: MatMut<'_, T>,
    op: i32,
    team: Option<&OwnedTeam>,
) -> Result<bool, String> {
    macro_rules! each {
        ($x:ident => $body:expr) => {
            match f {
                Factor::Lu($x) => $body,
                Factor::FullLu($x) => $body,
                Factor::Qr($x) => $body,
                Factor::PivotQr($x) => $body,
                Factor::Llt($x) => $body,
                Factor::Ldlt($x) => $body,
                Factor::Lblt($x) => $body,
                Factor::Svd($x) => $body,
            }
        };
    }
    let square = each!(x => x.nrows() == x.ncols());
    let deficient = match f {
        Factor::Qr(x) => typed::rank_deficient(x.thin_R()),
        Factor::PivotQr(x) => typed::rank_deficient(x.thin_R()),
        Factor::Svd(_) => true,
        _ => false,
    };
    if !square || deficient {
        return Ok(false);
    }
    check_rhs(rhs.as_ref(), each!(x => x.nrows()))?;
    with_team(team, |par| each!(x => in_place(x, rhs, op, par)))?;
    Ok(true)
}

/// SVD of the view named by `p`/`desc` into Julia buffers (see
/// `TypedMatrix::svd_buffers`), read in place.
#[julia]
pub fn view_svd_buffers(
    p: *const u8,
    desc: *const isize,
    u: *mut u8,
    s: *mut u8,
    v: *mut u8,
    full: bool,
    values_only: bool,
) -> Result<(), String> {
    guarded(|| unsafe {
        let spec = Spec::read(desc)?;
        if spec.conjugate {
            return Err("SVD inputs cannot be conjugated views".into());
        }
        match source(p, spec)? {
            Borrowed::F32(a) => typed::svd_output(a, u, s, v, full, values_only),
            Borrowed::F64(a) => typed::svd_output(a, u, s, v, full, values_only),
            Borrowed::C32(a) => typed::svd_output(a, u, s, v, full, values_only),
            Borrowed::C64(a) => typed::svd_output(a, u, s, v, full, values_only),
        }
    })
}
/// Eigendecomposition of the view named by `p`/`desc` into Julia buffers
/// (see `TypedMatrix::eigen_typed_buffers`), read in place.
#[julia]
pub fn view_eigen_buffers(
    p: *const u8,
    desc: *const isize,
    values: *mut u8,
    vectors: *mut u8,
    compute_vectors: bool,
    hermitian: bool,
    lower: bool,
) -> Result<(), String> {
    guarded(|| unsafe {
        let spec = Spec::read(desc)?;
        if spec.conjugate {
            return Err("eigen inputs cannot be conjugated views".into());
        }
        macro_rules! go {
            ($a:expr) => {
                typed::eigen_output($a, values, vectors, compute_vectors, hermitian, lower)
            };
        }
        match source(p, spec)? {
            Borrowed::F32(a) => go!(a),
            Borrowed::F64(a) => go!(a),
            Borrowed::C32(a) => go!(a),
            Borrowed::C64(a) => go!(a),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn raw_desc(m: usize, n: usize, rs: isize, cs: isize, tag: i32) -> Vec<isize> {
        vec![
            4,
            0,
            1,
            m as isize,
            0,
            1,
            n as isize,
            0,
            0,
            m as isize,
            n as isize,
            rs,
            cs,
            tag as isize,
        ]
    }

    #[test]
    fn borrowed_lu_matches_owned_and_solves_in_place() {
        let a = Mat::<f64>::from_fn(4, 4, |i, j| if i == j { 5.0 } else { (i + 2 * j) as f64 });
        let d = raw_desc(4, 4, a.row_stride(), a.col_stride(), 1);
        let mut f = OwnedFactor::new();
        let info = f
            .initialize_view(a.as_ptr().cast(), d.as_ptr(), 0, true, 4, 4, 1, None)
            .unwrap();
        assert_eq!(info, 0);
        let mut b = Mat::<f64>::from_fn(4, 2, |i, j| (i + j) as f64);
        let expected = Mat::<f64>::from_fn(4, 2, |i, j| (i + j) as f64);
        let e = raw_desc(4, 2, b.row_stride(), b.col_stride(), 1);
        assert!(f
            .solve_in_place_view(b.as_ptr_mut().cast(), e.as_ptr(), 0, None)
            .unwrap());
        assert!((&a * &b - &expected).norm_max() < 1e-12);
        // Mismatched metadata leaves the factor untouched.
        assert!(f
            .initialize_view(a.as_ptr().cast(), d.as_ptr(), 0, true, 4, 4, 0, None)
            .is_err());
    }

    #[test]
    fn rank_deficient_qr_declines_in_place_solve() {
        let a = Mat::<f64>::zeros(3, 3);
        let d = raw_desc(3, 3, a.row_stride(), a.col_stride(), 1);
        let mut f = TypedFactor::new();
        f.initialize_view(a.as_ptr().cast(), d.as_ptr(), 2, true, 3, 3, 1, None)
            .unwrap();
        let mut b = Mat::<f64>::ones(3, 1);
        let e = raw_desc(3, 1, b.row_stride(), b.col_stride(), 1);
        assert!(!f
            .solve_in_place_view(b.as_ptr_mut().cast(), e.as_ptr(), 0, None)
            .unwrap());
        assert_eq!(b, Mat::<f64>::ones(3, 1));
    }
}
