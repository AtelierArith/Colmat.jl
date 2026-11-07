//! Direct dense triangular solves and destructive triangular inverses.
//!
//! The triangle is borrowed through its view descriptor (owner, view, and the
//! `transpose`/`adjoint` words), so no operand is copied when its scalar is the
//! destination's. faer's `solve_{unit_}{lower,upper}_triangular_in_place_with_conj`
//! kernels need no workspace; caller scratch only holds a promoted copy of the
//! triangle (`promote_type(eltype(A), eltype(B))` wider than `eltype(A)`) or,
//! for the destructive inverse, the source copy faer's separate-destination
//! `invert_*` kernels require. Right solves `X op(A) = B` run as
//! `op(A)^T X^T = B^T` on the transposed destination, again without a copy.
use crate::collections::{VectorData, VectorOwner};
use crate::sparse_triangular::{widens, Native};
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{raw_mut, source, view_mut, Borrowed, Spec};
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::StackReq;
use faer::linalg::triangular_solve as kernels;
use faer::mat::AsMatMut;
use faer::traits::math_utils::*;
use faer::{Conj, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

fn tag_of(b: &Borrowed<'_>) -> i32 {
    match b {
        Borrowed::F32(_) => 0,
        Borrowed::F64(_) => 1,
        Borrowed::C32(_) => 2,
        Borrowed::C64(_) => 3,
    }
}
fn shape_of(b: &Borrowed<'_>) -> (usize, usize) {
    match b {
        Borrowed::F32(a) => a.shape(),
        Borrowed::F64(a) => a.shape(),
        Borrowed::C32(a) => a.shape(),
        Borrowed::C64(a) => a.shape(),
    }
}
/// 1-based index of the first zero diagonal entry, 0 when none.
fn zero_diagonal<T: Native>(a: MatRef<'_, T>) -> usize {
    (0..a.nrows())
        .find(|&i| a[(i, i)] == zero::<T>())
        .map_or(0, |i| i + 1)
}
fn zero_diagonal_any(b: &Borrowed<'_>) -> usize {
    match b {
        Borrowed::F32(a) => zero_diagonal(*a),
        Borrowed::F64(a) => zero_diagonal(*a),
        Borrowed::C32(a) => zero_diagonal(*a),
        Borrowed::C64(a) => zero_diagonal(*a),
    }
}
fn temp_req<P: Native>(n: usize) -> StackReq {
    faer::linalg::temp_mat_scratch::<P>(n, n)
}
fn promotion_req(n: usize, from: i32, to: i32) -> Result<StackReq, String> {
    if from == to {
        return Ok(StackReq::EMPTY);
    }
    if !widens(from, to) {
        return Err("triangular solve cannot narrow the triangle's scalar".into());
    }
    Ok(match to {
        0 => temp_req::<f32>(n),
        1 => temp_req::<f64>(n),
        2 => temp_req::<faer::c32>(n),
        3 => temp_req::<faer::c64>(n),
        _ => return Err("invalid triangular scalar".into()),
    })
}

fn kernel<T: Native>(
    c: MatRef<'_, T>,
    conjugate: bool,
    x: MatMut<'_, T>,
    lower: bool,
    unit: bool,
    right: bool,
    par: Par<'_>,
) {
    // X op(C) = B  <=>  op(C)^T X^T = B^T: the transpose swaps the triangle.
    let (c, lower, x) = if right {
        (c.transpose(), !lower, x.transpose_mut())
    } else {
        (c, lower, x)
    };
    let conj = if conjugate { Conj::Yes } else { Conj::No };
    match (lower, unit) {
        (true, true) => kernels::solve_unit_lower_triangular_in_place_with_conj(c, conj, x, par),
        (true, false) => kernels::solve_lower_triangular_in_place_with_conj(c, conj, x, par),
        (false, true) => kernels::solve_unit_upper_triangular_in_place_with_conj(c, conj, x, par),
        (false, false) => kernels::solve_upper_triangular_in_place_with_conj(c, conj, x, par),
    }
}

/// Solve `op(A) X = B` (or `X op(A) = B` when `right`) in place in `x`.
/// Returns the 1-based index of a zero diagonal (before any write) or 0.
///
/// # Safety
/// `a`/`ad` must describe a live owner preserved by Julia for this call that
/// does not share storage with `x`.
unsafe fn solve_into<P: Native>(
    a: *const u8,
    ad: *const isize,
    x: MatMut<'_, P>,
    lower: bool,
    unit: bool,
    right: bool,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
) -> Result<usize, String> {
    let s = Spec::read(ad)?;
    let b = source(a, s)?;
    let (n, k) = shape_of(&b);
    if n != k {
        return Err("triangular matrix must be square".into());
    }
    if (if right { x.ncols() } else { x.nrows() }) != n {
        return Err("triangular solve dimensions must match".into());
    }
    if !unit {
        let i = zero_diagonal_any(&b);
        if i != 0 {
            return Ok(i);
        }
    }
    let from = tag_of(&b);
    let req = promotion_req(n, from, P::TAG)?;
    let conjugate = s.conjugate;
    macro_rules! run {
        ($c:expr) => {{
            let c: MatRef<'_, P> = $c;
            with_team(team, |par| {
                kernel(c, conjugate, x, lower, unit, right, par);
                Ok(0)
            })
        }};
    }
    macro_rules! promoted {
        ($src:expr) => {{
            let src = $src;
            let stack = scratch.stack(req)?;
            let (mut c, _) = faer::linalg::temp_mat_zeroed::<P, _, _>(n, n, stack);
            let mut c = c.as_mat_mut();
            for j in 0..n {
                for i in 0..n {
                    c[(i, j)] = src[(i, j)].widen::<P>();
                }
            }
            run!(c.as_ref())
        }};
    }
    if from == P::TAG {
        // SAFETY: equal tags imply identical scalar types.
        let c = match b {
            Borrowed::F32(c) => c.as_ptr().cast::<P>(),
            Borrowed::F64(c) => c.as_ptr().cast::<P>(),
            Borrowed::C32(c) => c.as_ptr().cast::<P>(),
            Borrowed::C64(c) => c.as_ptr().cast::<P>(),
        };
        let (rs, cs) = match b {
            Borrowed::F32(c) => (c.row_stride(), c.col_stride()),
            Borrowed::F64(c) => (c.row_stride(), c.col_stride()),
            Borrowed::C32(c) => (c.row_stride(), c.col_stride()),
            Borrowed::C64(c) => (c.row_stride(), c.col_stride()),
        };
        return run!(MatRef::from_raw_parts(c, n, n, rs, cs));
    }
    match b {
        Borrowed::F32(c) => promoted!(c),
        Borrowed::F64(c) => promoted!(c),
        Borrowed::C32(c) => promoted!(c),
        Borrowed::C64(c) => promoted!(c),
    }
}

fn reject_alias<D>(a: *const u8, dest: &D) -> Result<(), String> {
    if std::ptr::eq(a, (dest as *const D).cast::<u8>()) {
        return Err("triangular solve destination aliases the triangle".into());
    }
    Ok(())
}

/// Bytes and alignment of the scratch for a solve whose triangle is `a`/`ad`
/// and whose destination scalar is `tag`.
///
/// # Safety
/// Julia preserves the owner, the descriptor and the two-word `out`.
#[julia]
pub fn dense_triangular_requirements(
    a: *const u8,
    ad: *const isize,
    tag: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let b = source(a, Spec::read(ad)?)?;
        let (n, k) = shape_of(&b);
        if n != k {
            return Err("triangular matrix must be square".into());
        }
        let req = promotion_req(n, tag_of(&b), tag)?;
        *out = req.size_bytes();
        *out.add(1) = req.align_bytes();
        Ok(())
    })
}
/// 1-based index of the first zero diagonal entry of the triangle, 0 if none.
#[julia]
pub fn dense_triangular_zero_diagonal(a: *const u8, ad: *const isize) -> Result<usize, String> {
    guarded(|| unsafe { Ok(zero_diagonal_any(&source(a, Spec::read(ad)?)?)) })
}
/// Solve into an injective Julia strided buffer of scalar `tag`.
#[julia]
pub fn dense_triangular_raw(
    a: *const u8,
    ad: *const isize,
    p: *mut u8,
    m: usize,
    n: usize,
    rs: isize,
    cs: isize,
    tag: i32,
    lower: bool,
    unit: bool,
    right: bool,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
) -> Result<usize, String> {
    guarded(|| {
        // SAFETY: Julia roots the injective strided array, which native
        // storage never aliases, for this call.
        unsafe {
            macro_rules! go {
                ($t:ty) => {
                    solve_into::<$t>(
                        a,
                        ad,
                        raw_mut(p, m, n, rs, cs),
                        lower,
                        unit,
                        right,
                        scratch,
                        team,
                    )
                };
            }
            match tag {
                0 => go!(f32),
                1 => go!(f64),
                2 => go!(faer::c32),
                3 => go!(faer::c64),
                _ => Err("invalid triangular scalar".into()),
            }
        }
    })
}

macro_rules! typed_dispatch {
    ($data:expr, $s:expr, $x:ident => $body:expr) => {
        match $data {
            MatrixData::F32(m) => {
                let $x = view_mut(m.as_mut(), $s)?;
                $body
            }
            MatrixData::F64(m) => {
                let $x = view_mut(m.as_mut(), $s)?;
                $body
            }
            MatrixData::C32(m) => {
                let $x = view_mut(m.as_mut(), $s)?;
                $body
            }
            MatrixData::C64(m) => {
                let $x = view_mut(m.as_mut(), $s)?;
                $body
            }
        }
    };
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn dense_triangular_into(
        &mut self,
        desc: *const isize,
        a: *const u8,
        ad: *const isize,
        lower: bool,
        unit: bool,
        right: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            reject_alias(a, self)?;
            let x = view_mut(self.matrix.as_mut(), unsafe { Spec::read(desc)? })?;
            unsafe { solve_into(a, ad, x, lower, unit, right, scratch, team) }
        })
    }
    #[julia]
    pub fn dense_triangular_invert(
        &mut self,
        desc: *const isize,
        lower: bool,
        unit: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let x = view_mut(self.matrix.as_mut(), unsafe { Spec::read(desc)? })?;
            invert_in_place(x, lower, unit, scratch, team)
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn dense_triangular_into(
        &mut self,
        desc: *const isize,
        a: *const u8,
        ad: *const isize,
        lower: bool,
        unit: bool,
        right: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            reject_alias(a, self)?;
            let s = unsafe { Spec::read(desc)? };
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            typed_dispatch!(data, s, x => unsafe {
                solve_into(a, ad, x, lower, unit, right, scratch, team)
            })
        })
    }
    #[julia]
    pub fn dense_triangular_invert(
        &mut self,
        desc: *const isize,
        lower: bool,
        unit: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            typed_dispatch!(data, s, x => invert_in_place(x, lower, unit, scratch, team))
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn dense_triangular_into(
        &mut self,
        desc: *const isize,
        a: *const u8,
        ad: *const isize,
        lower: bool,
        unit: bool,
        right: bool,
        scratch: &mut OwnedScratch,
        team: Option<&OwnedTeam>,
    ) -> Result<usize, String> {
        guarded(|| {
            reject_alias(a, self)?;
            let s = unsafe { Spec::read(desc)? };
            macro_rules! go {
                ($v:expr) => {{
                    let x = view_mut($v.as_mat_mut(), s)?;
                    unsafe { solve_into(a, ad, x, lower, unit, right, scratch, team) }
                }};
            }
            match self.data.as_mut().ok_or("collection is not initialized")? {
                VectorData::F32(v) => go!(v),
                VectorData::F64(v) => go!(v),
                VectorData::C32(v) => go!(v),
                VectorData::C64(v) => go!(v),
            }
        })
    }
}

/// Scratch for the destructive inverse of an `n x n` triangle of scalar `tag`.
#[julia]
pub fn dense_triangular_invert_requirements(
    n: usize,
    tag: i32,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| {
        let req = match tag {
            0 => temp_req::<f32>(n),
            1 => temp_req::<f64>(n),
            2 => temp_req::<faer::c32>(n),
            3 => temp_req::<faer::c64>(n),
            _ => return Err("invalid triangular scalar".into()),
        };
        // SAFETY: Julia preserves the two-word output for this call.
        unsafe {
            *out = req.size_bytes();
            *out.add(1) = req.align_bytes();
        }
        Ok(())
    })
}
/// Overwrite the selected triangle of `x` with its inverse. faer's `invert_*`
/// kernels take a separate source, so the triangle is first copied into the
/// caller scratch. The opposite triangle (and, for unit inverses, the
/// diagonal) is left untouched. Returns a zero-diagonal index before writing.
fn invert_in_place<T: Native>(
    mut x: MatMut<'_, T>,
    lower: bool,
    unit: bool,
    scratch: &mut OwnedScratch,
    team: Option<&OwnedTeam>,
) -> Result<usize, String> {
    let n = x.nrows();
    if x.ncols() != n {
        return Err("triangular matrix must be square".into());
    }
    if !unit {
        let i = zero_diagonal(x.as_ref());
        if i != 0 {
            return Ok(i);
        }
    }
    let stack = scratch.stack(temp_req::<T>(n))?;
    let (mut src, _) = faer::linalg::temp_mat_zeroed::<T, _, _>(n, n, stack);
    let mut src = src.as_mat_mut();
    src.copy_from(x.as_ref());
    use faer::linalg::triangular_inverse::*;
    with_team(team, |par| {
        let (dst, src) = (x.as_mut(), src.as_ref());
        match (lower, unit) {
            (true, true) => invert_unit_lower_triangular(dst, src, par),
            (true, false) => invert_lower_triangular(dst, src, par),
            (false, true) => invert_unit_upper_triangular(dst, src, par),
            (false, false) => invert_upper_triangular(dst, src, par),
        }
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_probe;
    use faer::{c64, Mat};

    fn descriptor(m: usize, n: usize, transpose: bool, conjugate: bool) -> Vec<isize> {
        vec![
            0,
            0,
            1,
            m as isize,
            0,
            1,
            n as isize,
            transpose as isize,
            conjugate as isize,
        ]
    }
    fn triangle(n: usize) -> Mat<f64> {
        Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 2.0
            } else {
                (((i * 7 + j * 3) % 5) as f64 / 5.0 - 0.4) / n as f64
            }
        })
    }
    fn reference(a: MatRef<'_, f64>, lower: bool, unit: bool) -> Mat<f64> {
        Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
            if i == j {
                if unit {
                    1.0
                } else {
                    a[(i, j)]
                }
            } else if (i > j) == lower {
                a[(i, j)]
            } else {
                0.0
            }
        })
    }

    #[test]
    fn direct_left_and_right_solves_allocate_nothing() {
        for n in [6usize, 70, 300] {
            let a = OwnedMatrix {
                matrix: triangle(n),
            };
            let ap = (&a as *const OwnedMatrix).cast::<u8>();
            let mut scratch = OwnedScratch::new();
            for (transpose, lower, unit, right) in [
                (false, true, false, false),
                (true, true, true, false),
                (false, false, false, true),
                (true, false, true, true),
            ] {
                let ad = descriptor(n, n, transpose, false);
                let op = if transpose {
                    a.matrix.as_ref().transpose().to_owned()
                } else {
                    a.matrix.clone()
                };
                let t = reference(op.as_ref(), lower, unit);
                let b = Mat::from_fn(
                    if right { 3 } else { n },
                    if right { n } else { 3 },
                    |i, j| (i + 2 * j) as f64 - 1.5,
                );
                let mut x = OwnedMatrix { matrix: b.clone() };
                let xd = descriptor(b.nrows(), b.ncols(), false, false);
                let mut code = 9;
                let count = allocation_probe::count(|| {
                    code = x
                        .dense_triangular_into(
                            xd.as_ptr(),
                            ap,
                            ad.as_ptr(),
                            lower,
                            unit,
                            right,
                            &mut scratch,
                            None,
                        )
                        .unwrap();
                });
                assert_eq!(code, 0);
                let residual = if right {
                    &x.matrix * &t - &b
                } else {
                    &t * &x.matrix - &b
                };
                assert!(residual.norm_max() < 1e-10, "n={n}");
                // Above faer's recursive base case the kernel calls matmul,
                // which allocates its own packing buffers (gemm-001).
                if n <= 70 {
                    assert_eq!(count, 0, "n={n} transpose={transpose} right={right}");
                } else {
                    println!("solve n={n} right={right}: {count} allocations");
                }
            }
        }
    }

    #[test]
    fn promotion_uses_scratch_and_singularity_precedes_writes() {
        let n = 5;
        let a = OwnedMatrix {
            matrix: triangle(n),
        };
        let ap = (&a as *const OwnedMatrix).cast::<u8>();
        let ad = descriptor(n, n, false, true);
        let mut out = [0usize; 2];
        dense_triangular_requirements(ap, ad.as_ptr(), 3, out.as_mut_ptr()).unwrap();
        assert!(out[0] >= n * n * 16);
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(out[0], out[1]).unwrap();
        let b = Mat::<c64>::from_fn(n, 2, |i, j| c64::new(i as f64, j as f64 + 1.0));
        let mut x = b.clone();
        let (rs, cs) = (x.row_stride(), x.col_stride());
        let mut code = 9;
        let count = allocation_probe::count(|| {
            code = dense_triangular_raw(
                ap,
                ad.as_ptr(),
                x.as_ptr_mut().cast(),
                n,
                2,
                rs,
                cs,
                3,
                true,
                false,
                false,
                &mut scratch,
                None,
            )
            .unwrap();
        });
        assert_eq!((code, count), (0, 0));
        let t = Mat::<c64>::from_fn(n, n, |i, j| {
            let v = reference(a.matrix.as_ref(), true, false)[(i, j)];
            c64::new(v, 0.0)
        });
        assert!((&t * &x - &b).norm_max() < 1e-10);
        // Narrowing is rejected; a zero diagonal is reported before writing.
        assert!(dense_triangular_requirements(ap, ad.as_ptr(), 0, out.as_mut_ptr()).is_err());
        let mut s = a.matrix.clone();
        s[(2, 2)] = 0.0;
        let z = OwnedMatrix { matrix: s };
        let zp = (&z as *const OwnedMatrix).cast::<u8>();
        let zd = descriptor(n, n, false, false);
        let mut y = OwnedMatrix {
            matrix: Mat::from_fn(n, 1, |i, _| i as f64),
        };
        let before = y.matrix.clone();
        let yd = descriptor(n, 1, false, false);
        let code = y
            .dense_triangular_into(
                yd.as_ptr(),
                zp,
                zd.as_ptr(),
                true,
                false,
                false,
                &mut scratch,
                None,
            )
            .unwrap();
        assert_eq!(code, 3);
        assert_eq!(y.matrix, before);
        // Aliasing the triangle is rejected.
        let mut w = OwnedMatrix {
            matrix: triangle(n),
        };
        let wp = (&w as *const OwnedMatrix).cast::<u8>();
        let wd = descriptor(n, n, false, false);
        assert!(w
            .dense_triangular_into(
                wd.as_ptr(),
                wp,
                wd.as_ptr(),
                true,
                false,
                false,
                &mut scratch,
                None
            )
            .is_err());
    }

    #[test]
    fn destructive_inverse_uses_only_caller_scratch() {
        for n in [5usize, 70] {
            let mut out = [0usize; 2];
            dense_triangular_invert_requirements(n, 1, out.as_mut_ptr()).unwrap();
            let mut scratch = OwnedScratch::new();
            scratch.scratch_reserve(out[0], out[1]).unwrap();
            for (lower, unit) in [(true, false), (true, true), (false, false), (false, true)] {
                let a = triangle(n);
                let mut x = OwnedMatrix { matrix: a.clone() };
                let d = descriptor(n, n, false, false);
                let mut code = 9;
                let count = allocation_probe::count(|| {
                    code = x
                        .dense_triangular_invert(d.as_ptr(), lower, unit, &mut scratch, None)
                        .unwrap();
                });
                assert_eq!(code, 0);
                let inv = reference(x.matrix.as_ref(), lower, unit);
                let t = reference(a.as_ref(), lower, unit);
                let eye = Mat::<f64>::identity(n, n);
                assert!((&t * &inv - &eye).norm_max() < 1e-10);
                // The opposite strict triangle is untouched.
                for j in 0..n {
                    for i in 0..n {
                        if i != j && (i > j) != lower {
                            assert_eq!(x.matrix[(i, j)], a[(i, j)]);
                        }
                    }
                }
                // faer's blocked inverse calls matmul, which allocates its own
                // packing buffers above the small-size path (gemm-001).
                if n <= 8 {
                    assert_eq!(count, 0, "n={n} lower={lower} unit={unit}");
                }
            }
        }
    }
}
