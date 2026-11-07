//! Direct sparse triangular solves into native or borrowed Julia destinations.
//!
//! faer's `sp_solve_{unit_}{lower,upper}_triangular_in_place` kernels trust the
//! stored layout: the diagonal must be the first (lower) or last (upper) entry
//! of every column and no entry may lie in the other half. A stored owner that
//! already has that layout is borrowed as is; any other owner (both halves
//! stored, missing unit diagonal) or a promoted scalar type is staged in the
//! caller's scratch, never in a heap allocation. CSR storage, `transpose` and
//! `adjoint` are mapped onto the column-major kernels and their `transpose`
//! variants with a conjugation flag, so no operand is copied.
use crate::collections::{VectorData, VectorOwner};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::Spec;
use crate::workspace::OwnedScratch;
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemStack, StackReq};
use faer::sparse::linalg::triangular_solve as kernels;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Conj, MatMut, Par};
use rustcall_julia_macros::julia;

/// Exact scalar conversions between the four native scalar types.
pub(crate) trait Native: ComplexField + Copy {
    const TAG: i32;
    fn widen<W: Native>(self) -> W;
    fn from_parts(re: f64, im: f64) -> Self;
}
macro_rules! real_native {
    ($t:ty, $tag:expr) => {
        impl Native for $t {
            const TAG: i32 = $tag;
            fn widen<W: Native>(self) -> W {
                W::from_parts(self as f64, 0.0)
            }
            fn from_parts(re: f64, _: f64) -> Self {
                re as $t
            }
        }
    };
}
macro_rules! complex_native {
    ($t:ty, $r:ty, $tag:expr) => {
        impl Native for $t {
            const TAG: i32 = $tag;
            fn widen<W: Native>(self) -> W {
                W::from_parts(self.re as f64, self.im as f64)
            }
            fn from_parts(re: f64, im: f64) -> Self {
                <$t>::new(re as $r, im as $r)
            }
        }
    };
}
real_native!(f32, 0);
real_native!(f64, 1);
complex_native!(faer::c32, f32, 2);
complex_native!(faer::c64, f64, 3);

/// Lossless promotion pairs, as produced by Julia's `promote_type`.
pub(crate) fn widens(from: i32, to: i32) -> bool {
    from == to || matches!((from, to), (0, 1) | (0, 2) | (0, 3) | (1, 3) | (2, 3))
}

/// Logical solve `op(A) x = b` with `op` 0 (none), 1 (transpose), 2 (adjoint)
/// and the triangle `lower` of `op(A)`, rewritten on the column-major ref `C`
/// of the storage: `op(A) = conj?(C)` or `conj?(C)^T`.
#[derive(Clone, Copy)]
struct Plan {
    transpose: bool,
    conjugate: bool,
    /// Triangle of `C` the kernel reads.
    lower: bool,
    unit: bool,
}
fn column_major<T>(a: &Storage<T>) -> (SparseColMatRef<'_, usize, T>, bool) {
    match a {
        Storage::Csc(a) => (a.as_ref(), false),
        Storage::Csr(a) => (a.as_ref().transpose(), true),
    }
}
fn plan(stored_transpose: bool, lower: bool, unit: bool, op: i32) -> Result<Plan, String> {
    if !(0..=2).contains(&op) {
        return Err("unknown sparse triangular operation".into());
    }
    let transpose = stored_transpose != (op != 0);
    Ok(Plan {
        transpose,
        conjugate: op == 2,
        lower: if transpose { !lower } else { lower },
        unit,
    })
}
/// The diagonal position of column `j` of `C` when it has faer's layout.
fn conforming_column(rows: &[usize], j: usize, lower: bool) -> Option<usize> {
    if lower {
        (rows.first() == Some(&j) && rows[1..].iter().all(|&i| i > j)).then_some(0)
    } else {
        let last = rows.len().checked_sub(1)?;
        (rows[last] == j && rows[..last].iter().all(|&i| i < j)).then_some(last)
    }
}
fn conforms<T>(c: SparseColMatRef<'_, usize, T>, lower: bool) -> bool {
    let rows = c.symbolic().row_idx();
    (0..c.ncols()).all(|j| conforming_column(&rows[c.symbolic().col_range(j)], j, lower).is_some())
}
/// First one-based index whose diagonal is structurally or numerically zero.
fn zero_diagonal<T: ComplexField>(c: SparseColMatRef<'_, usize, T>) -> usize {
    let rows = c.symbolic().row_idx();
    for j in 0..c.ncols() {
        let range = c.symbolic().col_range(j);
        let diagonal = rows[range.clone()]
            .iter()
            .position(|&i| i == j)
            .map(|k| &c.val()[range.start + k]);
        if !matches!(diagonal, Some(d) if *d != zero::<T>()) {
            return j + 1;
        }
    }
    0
}
/// Entries of the normalized triangle: the kept half plus one diagonal each.
fn normalized_capacity<T>(c: SparseColMatRef<'_, usize, T>) -> Result<usize, String> {
    c.symbolic()
        .compute_nnz()
        .checked_add(c.ncols())
        .ok_or_else(|| "sparse triangular size overflow".into())
}
fn requirement<P: Native, T>(
    c: SparseColMatRef<'_, usize, T>,
    plan: Plan,
    same: bool,
) -> Result<StackReq, String> {
    if conforms(c, plan.lower) {
        return Ok(if same {
            StackReq::EMPTY
        } else {
            StackReq::new::<P>(c.val().len())
        });
    }
    let cap = normalized_capacity(c)?;
    Ok(StackReq::all_of(&[
        StackReq::new::<usize>(c.ncols() + 1),
        StackReq::new::<usize>(cap),
        StackReq::new::<P>(cap),
    ]))
}
fn kernel<T: ComplexField>(c: SparseColMatRef<'_, usize, T>, plan: Plan, x: MatMut<'_, T>) {
    let conj = if plan.conjugate { Conj::Yes } else { Conj::No };
    // faer's sparse triangular kernels ignore `par`; they run sequentially.
    let par = Par::Seq;
    match (plan.transpose, plan.lower, plan.unit) {
        (false, true, false) => kernels::solve_lower_triangular_in_place(c, conj, x, par),
        (false, true, true) => kernels::solve_unit_lower_triangular_in_place(c, conj, x, par),
        (false, false, false) => kernels::solve_upper_triangular_in_place(c, conj, x, par),
        (false, false, true) => kernels::solve_unit_upper_triangular_in_place(c, conj, x, par),
        (true, true, false) => kernels::solve_lower_triangular_transpose_in_place(c, conj, x, par),
        (true, true, true) => {
            kernels::solve_unit_lower_triangular_transpose_in_place(c, conj, x, par)
        }
        (true, false, false) => kernels::solve_upper_triangular_transpose_in_place(c, conj, x, par),
        (true, false, true) => {
            kernels::solve_unit_upper_triangular_transpose_in_place(c, conj, x, par)
        }
    }
}
/// Solve in place. Returns the one-based index of a zero diagonal (nothing is
/// written then) or 0. All validation precedes the first write to `x`.
fn solve<S: Native, P: Native>(
    a: &Storage<S>,
    x: MatMut<'_, P>,
    lower: bool,
    unit: bool,
    op: i32,
    stack: &mut MemStack,
) -> Result<usize, String> {
    if !widens(S::TAG, P::TAG) {
        return Err("sparse triangular destination must hold the promoted scalar type".into());
    }
    let (c, stored_transpose) = column_major(a);
    let n = c.nrows();
    if c.ncols() != n || x.nrows() != n {
        return Err("sparse triangular solve dimensions must match".into());
    }
    if !unit {
        let i = zero_diagonal(c);
        if i != 0 {
            return Ok(i);
        }
    }
    let plan = plan(stored_transpose, lower, unit, op)?;
    let same = S::TAG == P::TAG;
    if conforms(c, plan.lower) {
        if same {
            // SAFETY: S and P are the same scalar type; reinterpret in place.
            let val =
                unsafe { std::slice::from_raw_parts(c.val().as_ptr().cast::<P>(), c.val().len()) };
            kernel(SparseColMatRef::new(c.symbolic(), val), plan, x);
        } else {
            let (val, _) = stack.make_with(c.val().len(), |k| c.val()[k].widen::<P>());
            kernel(SparseColMatRef::new(c.symbolic(), &*val), plan, x);
        }
        return Ok(0);
    }
    let cap = normalized_capacity(c)?;
    let (mut ptr, stack) = stack.make_with(n + 1, |_| 0usize);
    let (mut idx, stack) = stack.make_with(cap, |_| 0usize);
    let (mut val, _) = stack.make_with(cap, |_| zero::<P>());
    let rows = c.symbolic().row_idx();
    let mut k = 0;
    for j in 0..n {
        let range = c.symbolic().col_range(j);
        let diagonal = rows[range.clone()]
            .iter()
            .position(|&i| i == j)
            .map(|p| c.val()[range.start + p].widen::<P>());
        // Unit kernels skip the stored value, but still expect an entry.
        let diagonal = if unit {
            one::<P>()
        } else {
            diagonal.ok_or("missing diagonal")?
        };
        if plan.lower {
            idx[k] = j;
            val[k] = diagonal;
            k += 1;
        }
        for p in range {
            let i = rows[p];
            if (plan.lower && i > j) || (!plan.lower && i < j) {
                idx[k] = i;
                val[k] = c.val()[p].widen::<P>();
                k += 1;
            }
        }
        if !plan.lower {
            idx[k] = j;
            val[k] = diagonal;
            k += 1;
        }
        ptr[j + 1] = k;
    }
    // SAFETY: column pointers are nondecreasing, end at `k`, and every row
    // index is below `n`; rows within a column need not be sorted.
    let symbolic = unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, &ptr, None, &idx[..k]) };
    kernel(SparseColMatRef::new(symbolic, &val[..k]), plan, x);
    Ok(0)
}

macro_rules! with_storage {
    ($data:expr, $a:ident => $body:expr) => {
        match $data {
            SparseData::F32($a) => $body,
            SparseData::F64($a) => $body,
            SparseData::C32($a) => $body,
            SparseData::C64($a) => $body,
        }
    };
}
fn stack_for<'s, S: Native, P: Native>(
    a: &Storage<S>,
    lower: bool,
    unit: bool,
    op: i32,
    scratch: &'s mut OwnedScratch,
) -> Result<&'s mut MemStack, String> {
    let (c, stored_transpose) = column_major(a);
    let req = requirement::<P, S>(
        c,
        plan(stored_transpose, lower, unit, op)?,
        S::TAG == P::TAG,
    )?;
    scratch.stack(req)
}
fn solve_any<P: Native>(
    sparse: &SparseOwner,
    x: MatMut<'_, P>,
    lower: bool,
    unit: bool,
    op: i32,
    scratch: &mut OwnedScratch,
) -> Result<usize, String> {
    with_storage!(sparse.data()?, a => {
        let stack = stack_for::<_, P>(a, lower, unit, op, scratch)?;
        solve(a, x, lower, unit, op, stack)
    })
}

/// Bounds-checked mutable borrow of a view descriptor over `base`.
pub(crate) fn view_mut<'a, T>(base: MatMut<'a, T>, s: Spec) -> Result<MatMut<'a, T>, String> {
    if s.conjugate {
        return Err("triangular solve destination cannot be conjugated".into());
    }
    let axis = |first: isize, step: isize, len: usize, cap: usize| -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let last = isize::try_from(len - 1)
            .ok()
            .and_then(|l| step.checked_mul(l))
            .and_then(|x| first.checked_add(x))
            .ok_or("view index overflow")?;
        if first < 0 || last < 0 || first as usize >= cap || last as usize >= cap {
            return Err("view is out of bounds".into());
        }
        Ok(())
    };
    axis(s.r, s.rs, s.m, base.nrows())?;
    axis(s.c, s.cs, s.n, base.ncols())?;
    if (s.rs == 0 && s.m > 1) || (s.cs == 0 && s.n > 1) {
        return Err("triangular solve destination must not repeat elements".into());
    }
    let (rs, cs) = (base.row_stride(), base.col_stride());
    let offset = if s.m == 0 || s.n == 0 {
        0
    } else {
        rs.checked_mul(s.r)
            .and_then(|x| cs.checked_mul(s.c).and_then(|y| x.checked_add(y)))
            .ok_or("view offset overflow")?
    };
    let srs = rs.checked_mul(s.rs).ok_or("view stride overflow")?;
    let scs = cs.checked_mul(s.cs).ok_or("view stride overflow")?;
    // SAFETY: every selected index is inside the owner and the map is
    // injective, so this is a genuine exclusive borrow of `base`.
    let v =
        unsafe { MatMut::from_raw_parts_mut(base.as_ptr_mut().offset(offset), s.m, s.n, srs, scs) };
    Ok(if s.transpose { v.transpose_mut() } else { v })
}

#[julia]
impl SparseOwner {
    /// Write `(bytes, alignment)` of the scratch a solve with destination
    /// scalar `tag` needs; zero when the stored layout is borrowed directly.
    #[julia]
    pub fn sparse_triangular_requirements(
        &self,
        lower: bool,
        unit: bool,
        op: i32,
        tag: i32,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| {
            fn query<S: Native, P: Native>(
                a: &Storage<S>,
                lower: bool,
                unit: bool,
                op: i32,
            ) -> Result<StackReq, String> {
                if !widens(S::TAG, P::TAG) {
                    return Err(
                        "sparse triangular destination must hold the promoted scalar type".into(),
                    );
                }
                let (c, t) = column_major(a);
                requirement::<P, S>(c, plan(t, lower, unit, op)?, S::TAG == P::TAG)
            }
            let req = with_storage!(self.data()?, a => match tag {
                0 => query::<_, f32>(a, lower, unit, op)?,
                1 => query::<_, f64>(a, lower, unit, op)?,
                2 => query::<_, faer::c32>(a, lower, unit, op)?,
                3 => query::<_, faer::c64>(a, lower, unit, op)?,
                _ => return Err("invalid sparse triangular scalar".into()),
            });
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
    /// Solve into a borrowed Julia strided buffer of scalar `tag`.
    #[julia]
    pub fn sparse_triangular_raw(
        &self,
        p: *mut u8,
        m: usize,
        n: usize,
        rs: isize,
        cs: isize,
        tag: i32,
        lower: bool,
        unit: bool,
        op: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| {
            unsafe fn raw<'a, T>(
                p: *mut u8,
                m: usize,
                n: usize,
                rs: isize,
                cs: isize,
            ) -> MatMut<'a, T> {
                let ptr = if m == 0 || n == 0 {
                    std::ptr::NonNull::<T>::dangling().as_ptr()
                } else {
                    p.cast::<T>()
                };
                MatMut::from_raw_parts_mut(ptr, m, n, rs, cs)
            }
            // SAFETY: Julia roots the injective strided array for this call.
            unsafe {
                match tag {
                    0 => solve_any::<f32>(self, raw(p, m, n, rs, cs), lower, unit, op, scratch),
                    1 => solve_any::<f64>(self, raw(p, m, n, rs, cs), lower, unit, op, scratch),
                    2 => {
                        solve_any::<faer::c32>(self, raw(p, m, n, rs, cs), lower, unit, op, scratch)
                    }
                    3 => {
                        solve_any::<faer::c64>(self, raw(p, m, n, rs, cs), lower, unit, op, scratch)
                    }
                    _ => Err("invalid sparse triangular scalar".into()),
                }
            }
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_triangular_into(
        &mut self,
        desc: *const isize,
        sparse: &SparseOwner,
        lower: bool,
        unit: bool,
        op: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| {
            let x = view_mut(self.matrix.as_mut(), unsafe { Spec::read(desc)? })?;
            solve_any(sparse, x, lower, unit, op, scratch)
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_triangular_into(
        &mut self,
        desc: *const isize,
        sparse: &SparseOwner,
        lower: bool,
        unit: bool,
        op: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            match self.data.as_mut().ok_or("matrix is not initialized")? {
                MatrixData::F32(a) => {
                    solve_any(sparse, view_mut(a.as_mut(), s)?, lower, unit, op, scratch)
                }
                MatrixData::F64(a) => {
                    solve_any(sparse, view_mut(a.as_mut(), s)?, lower, unit, op, scratch)
                }
                MatrixData::C32(a) => {
                    solve_any(sparse, view_mut(a.as_mut(), s)?, lower, unit, op, scratch)
                }
                MatrixData::C64(a) => {
                    solve_any(sparse, view_mut(a.as_mut(), s)?, lower, unit, op, scratch)
                }
            }
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn sparse_triangular_into(
        &mut self,
        desc: *const isize,
        sparse: &SparseOwner,
        lower: bool,
        unit: bool,
        op: i32,
        scratch: &mut OwnedScratch,
    ) -> Result<usize, String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            match self.data.as_mut().ok_or("collection is not initialized")? {
                VectorData::F32(a) => solve_any(
                    sparse,
                    view_mut(a.as_mat_mut(), s)?,
                    lower,
                    unit,
                    op,
                    scratch,
                ),
                VectorData::F64(a) => solve_any(
                    sparse,
                    view_mut(a.as_mat_mut(), s)?,
                    lower,
                    unit,
                    op,
                    scratch,
                ),
                VectorData::C32(a) => solve_any(
                    sparse,
                    view_mut(a.as_mat_mut(), s)?,
                    lower,
                    unit,
                    op,
                    scratch,
                ),
                VectorData::C64(a) => solve_any(
                    sparse,
                    view_mut(a.as_mat_mut(), s)?,
                    lower,
                    unit,
                    op,
                    scratch,
                ),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
    use faer::{c64, Mat};

    fn entries(n: usize, full: bool) -> Vec<Triplet<usize, usize, c64>> {
        let mut e = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let keep =
                    i == j || (i > j && (i + j) % 2 == 0) || (full && i < j && (i * j) % 3 == 1);
                if keep {
                    let d = if i == j { 6.0 } else { 0.0 };
                    e.push(Triplet {
                        row: i,
                        col: j,
                        val: c64::new(d + (i as f64) - 0.5 * j as f64, 0.25 * (i + 2 * j) as f64),
                    });
                }
            }
        }
        e
    }
    fn storage(n: usize, full: bool, csr: bool) -> Storage<c64> {
        let e = entries(n, full);
        if csr {
            Storage::Csr(SparseRowMat::try_new_from_triplets(n, n, &e).unwrap())
        } else {
            Storage::Csc(SparseColMat::try_new_from_triplets(n, n, &e).unwrap())
        }
    }
    /// Dense triangle of `op(A)` as the Julia wrappers define it.
    fn reference(a: &Storage<c64>, lower: bool, unit: bool, op: i32) -> Mat<c64> {
        let d = match a {
            Storage::Csc(a) => a.to_dense(),
            Storage::Csr(a) => a.to_dense(),
        };
        let n = d.nrows();
        Mat::from_fn(n, n, |i, j| {
            let v = match op {
                0 => d[(i, j)],
                1 => d[(j, i)],
                _ => d[(j, i)].conj(),
            };
            if i == j {
                if unit {
                    c64::new(1.0, 0.0)
                } else {
                    v
                }
            } else if (i > j) == lower {
                v
            } else {
                c64::new(0.0, 0.0)
            }
        })
    }
    fn scratch_for(a: &Storage<c64>, lower: bool, unit: bool, op: i32) -> OwnedScratch {
        let (c, t) = column_major(a);
        let req = requirement::<c64, c64>(c, plan(t, lower, unit, op).unwrap(), true).unwrap();
        let mut s = OwnedScratch::new();
        s.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        s
    }

    #[test]
    fn every_storage_operation_and_triangle_matches_dense_without_allocating() {
        let n = 7;
        for full in [false, true] {
            for csr in [false, true] {
                let a = storage(n, full, csr);
                for op in 0..3 {
                    for lower in [false, true] {
                        for unit in [false, true] {
                            let b = Mat::<c64>::from_fn(n, 3, |i, j| {
                                c64::new(i as f64 - 1.0, j as f64)
                            });
                            let mut x = b.clone();
                            let mut scratch = scratch_for(&a, lower, unit, op);
                            let stack =
                                stack_for::<c64, c64>(&a, lower, unit, op, &mut scratch).unwrap();
                            let mut code = usize::MAX;
                            let count = crate::allocation_probe::count(|| {
                                code = solve(&a, x.as_mut(), lower, unit, op, stack).unwrap();
                            });
                            assert_eq!(code, 0);
                            assert_eq!(
                                count, 0,
                                "full={full} csr={csr} op={op} lower={lower} unit={unit}"
                            );
                            let residual = &reference(&a, lower, unit, op) * &x - &b;
                            assert!(
                                residual.norm_max() < 1e-10,
                                "full={full} csr={csr} op={op} lower={lower} unit={unit}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn conforming_storage_needs_no_scratch_and_promotion_stages_values_only() {
        let a = storage(6, false, false);
        let (c, t) = column_major(&a);
        let p = plan(t, true, false, 0).unwrap();
        assert_eq!(requirement::<c64, c64>(c, p, true).unwrap().size_bytes(), 0);
        // The stored lower triangle read as an upper (transposed) solve still conforms.
        assert_eq!(
            requirement::<c64, c64>(c, plan(t, false, true, 1).unwrap(), true)
                .unwrap()
                .size_bytes(),
            0
        );
        // Upper solve of a lower-stored owner must normalize.
        assert!(
            requirement::<c64, c64>(c, plan(t, false, false, 0).unwrap(), true)
                .unwrap()
                .size_bytes()
                > 0
        );
        // Real sparse, complex destination: only converted values are staged.
        let e: Vec<_> = entries(6, false)
            .into_iter()
            .map(|e| Triplet {
                row: e.row,
                col: e.col,
                val: e.val.re,
            })
            .collect();
        let real = Storage::Csc(SparseColMat::try_new_from_triplets(6, 6, &e).unwrap());
        let (rc, rt) = column_major(&real);
        let req = requirement::<c64, f64>(rc, plan(rt, true, false, 0).unwrap(), false).unwrap();
        assert!(req.size_bytes() >= rc.val().len() * std::mem::size_of::<c64>());
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let b = Mat::<c64>::from_fn(6, 2, |i, j| c64::new(1.0 + i as f64, j as f64));
        let mut x = b.clone();
        let stack = stack_for::<f64, c64>(&real, true, false, 0, &mut scratch).unwrap();
        let count = crate::allocation_probe::count(|| {
            assert_eq!(solve(&real, x.as_mut(), true, false, 0, stack).unwrap(), 0);
        });
        assert_eq!(count, 0);
        let promoted = Storage::Csc(
            SparseColMat::try_new_from_triplets(
                6,
                6,
                &e.iter()
                    .map(|e| Triplet {
                        row: e.row,
                        col: e.col,
                        val: c64::new(e.val, 0.0),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        assert!((&reference(&promoted, true, false, 0) * &x - &b).norm_max() < 1e-10);
    }

    #[test]
    fn failures_leave_the_destination_untouched() {
        // Zero stored diagonal at index 3 and missing diagonal at index 5.
        let e = vec![
            Triplet {
                row: 0,
                col: 0,
                val: 1.0f64,
            },
            Triplet {
                row: 1,
                col: 1,
                val: 1.0,
            },
            Triplet {
                row: 2,
                col: 2,
                val: 0.0,
            },
            Triplet {
                row: 3,
                col: 0,
                val: 2.0,
            },
            Triplet {
                row: 3,
                col: 3,
                val: 1.0,
            },
        ];
        let a = Storage::Csc(SparseColMat::try_new_from_triplets(5, 5, &e).unwrap());
        let mut scratch = OwnedScratch::new();
        scratch.scratch_reserve(1 << 12, 64).unwrap();
        let b = Mat::<f64>::from_fn(5, 2, |i, j| (i + 7 * j) as f64);
        let mut x = b.clone();
        let stack = scratch.stack(StackReq::EMPTY).unwrap();
        assert_eq!(solve(&a, x.as_mut(), true, false, 0, stack).unwrap(), 3);
        assert_eq!(x, b);
        assert!(solve(&a, x.as_mut().get_mut(0..4, ..), true, false, 0, stack).is_err());
        assert_eq!(x, b);
        // Narrowing destinations are rejected before any write.
        let mut narrow = Mat::<f32>::zeros(5, 1);
        assert!(solve::<f64, f32>(&a, narrow.as_mut(), true, true, 0, stack).is_err());
        // Undersized scratch is rejected by the stack request.
        let full = storage(5, true, false);
        let mut small = OwnedScratch::new();
        small.scratch_reserve(0, 1).unwrap();
        assert!(stack_for::<c64, c64>(&full, true, false, 0, &mut small).is_err());
    }

    #[test]
    fn exported_entry_points_do_not_allocate_with_caller_scratch() {
        let n = 9;
        for full in [false, true] {
            let sparse = SparseOwner {
                data: Some(SparseData::C64(storage(n, full, false))),
            };
            let mut req = [0usize; 2];
            sparse
                .sparse_triangular_requirements(true, false, 0, 3, req.as_mut_ptr())
                .unwrap();
            assert_eq!(req[0] == 0, !full);
            let mut scratch = OwnedScratch::new();
            scratch.scratch_reserve(req[0], req[1]).unwrap();
            let b = Mat::<c64>::from_fn(n, 2, |i, j| c64::new(i as f64, 1.0 - j as f64));
            let gold = reference(
                sparse
                    .data()
                    .map(|d| match d {
                        SparseData::C64(a) => a,
                        _ => unreachable!(),
                    })
                    .unwrap(),
                true,
                false,
                0,
            );
            // Native owner through a column-subrange view descriptor.
            let mut owner = TypedMatrix {
                data: Some(MatrixData::C64(Mat::from_fn(n, 4, |i, j| {
                    if (1..3).contains(&j) {
                        b[(i, j - 1)]
                    } else {
                        c64::new(0.0, 0.0)
                    }
                }))),
            };
            let desc: [isize; 9] = [1, 0, 1, n as isize, 1, 1, 2, 0, 0];
            let mut code = usize::MAX;
            let count = crate::allocation_probe::count(|| {
                code = owner
                    .sparse_triangular_into(desc.as_ptr(), &sparse, true, false, 0, &mut scratch)
                    .unwrap();
            });
            assert_eq!((code, count), (0, 0), "full={full}");
            let Some(MatrixData::C64(x)) = &owner.data else {
                unreachable!()
            };
            assert!((&gold * x.as_ref().subcols(1, 2) - &b).norm_max() < 1e-10);
            // Borrowed Julia buffer with a reversed row stride.
            let mut buffer = Mat::<c64>::from_fn(n, 2, |i, j| b[(n - 1 - i, j)]);
            let (rs, cs) = (buffer.row_stride(), buffer.col_stride());
            let last = unsafe { buffer.as_ptr_mut().offset((n as isize - 1) * rs) };
            let count = crate::allocation_probe::count(|| {
                code = sparse
                    .sparse_triangular_raw(
                        last.cast(),
                        n,
                        2,
                        -rs,
                        cs,
                        3,
                        true,
                        false,
                        0,
                        &mut scratch,
                    )
                    .unwrap();
            });
            assert_eq!((code, count), (0, 0), "full={full}");
            let x = Mat::<c64>::from_fn(n, 2, |i, j| buffer[(n - 1 - i, j)]);
            assert!((&gold * &x - &b).norm_max() < 1e-10);
        }
    }
}
