//! Per-call faer views of a Julia `SparseMatrixCSC`.
//!
//! faer's `SymbolicSparseColMatRef` is always 0-based, so the Julia layer
//! converts `colptr`/`rowval` into reused 0-based `usize` buffers before each
//! call; the stored values are read in place from Julia's `nzval`. Every
//! pointer is valid only for the duration of one synchronous call (Julia roots
//! the parent and the buffers) and is never retained.
use crate::sparse::{fixed_update_columns, SparseData, SparseOwner, Storage};
use crate::sparsity::{Pattern, SparsityOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{source, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::sparse::linalg::matmul::{
    dense_sparse_matmul, sparse_dense_matmul, DenseSparseMatMul, SparseDenseMatMul,
};
use faer::sparse::{
    SparseColMat, SparseColMatMut, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef,
};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

/// Raw CSC parts: `n + 1` column pointers, `k` row indices and `k` values.
#[derive(Clone, Copy)]
pub(crate) struct Raw {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) ptr: *const usize,
    pub(crate) idx: *const usize,
    pub(crate) val: *const u8,
    pub(crate) k: usize,
}

// SAFETY: `ptr` addresses `n + 1` and `idx` `k` initialized `usize` values
// that Julia keeps alive for this call. The structure is fully validated here,
// so faer's unchecked invariants hold.
pub(crate) unsafe fn symbolic<'a>(r: Raw) -> Result<SymbolicSparseColMatRef<'a, usize>, String> {
    let len = r.n.checked_add(1).ok_or("column pointer overflow")?;
    if r.ptr.is_null() {
        return Err("missing column pointers".into());
    }
    let ptr = std::slice::from_raw_parts(r.ptr, len);
    let idx: &[usize] = if r.k == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(r.idx, r.k)
    };
    if r.m > isize::MAX as usize || r.n > isize::MAX as usize {
        return Err("sparse dimensions are too large".into());
    }
    if ptr[0] != 0 || ptr[r.n] != r.k || ptr.windows(2).any(|w| w[0] > w[1]) {
        return Err("invalid compressed pointers".into());
    }
    for j in 0..r.n {
        let col = &idx[ptr[j]..ptr[j + 1]];
        if col.iter().any(|&i| i >= r.m) || col.windows(2).any(|w| w[0] >= w[1]) {
            return Err("CSC row indices must be in bounds, sorted and unique".into());
        }
    }
    Ok(SymbolicSparseColMatRef::new_unchecked(
        r.m, r.n, ptr, None, idx,
    ))
}

// SAFETY: as `symbolic`, and `val` addresses `k` initialized `T` values.
pub(crate) unsafe fn numeric<'a, T>(r: Raw) -> Result<SparseColMatRef<'a, usize, T>, String> {
    let s = symbolic(r)?;
    let val: &[T] = if r.k == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(r.val.cast::<T>(), r.k)
    };
    Ok(SparseColMatRef::new(s, val))
}

// SAFETY: as `symbolic`, and `val` addresses `k` initialized `T` values that
// nothing else reads or writes during this call (Julia checks aliasing).
pub(crate) unsafe fn numeric_mut<'a, T>(
    r: Raw,
    val: *mut u8,
) -> Result<SparseColMatMut<'a, usize, T>, String> {
    let s = symbolic(r)?;
    let val: &mut [T] = if r.k == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(val.cast::<T>(), r.k)
    };
    Ok(SparseColMatMut::new(s, val))
}

fn owned<T: Clone>(a: SparseColMatRef<'_, usize, T>) -> SparseColMat<usize, T> {
    let s = a.symbolic();
    // SAFETY: copied from a validated symbolic view.
    let symbolic = unsafe {
        SymbolicSparseColMat::new_unchecked(
            s.nrows(),
            s.ncols(),
            s.col_ptr().to_vec(),
            None,
            s.row_idx().to_vec(),
        )
    };
    SparseColMat::new(symbolic, a.val().to_vec())
}

/// Stored operand orientation: the CSC matrix itself, its transpose (the same
/// arrays read as a CSR `SparseRowMatRef`), its adjoint or its conjugate (the
/// adjoint of a CSR owner, whose column-major view is its transpose).
pub(crate) const PLAIN: i32 = 0;
pub(crate) const TRANSPOSE: i32 = 1;
pub(crate) const ADJOINT: i32 = 2;
pub(crate) const CONJUGATE: i32 = 3;

fn apply<T: ComplexField, S: SparseDenseMatMul<T> + DenseSparseMatMul<T>>(
    out: MatMut<'_, T>,
    accum: Accum,
    a: S,
    b: MatRef<'_, T>,
    conjugate: bool,
    alpha: T,
    left: bool,
) {
    match (left, conjugate) {
        (true, false) => sparse_dense_matmul(out, accum, a, b, alpha, Par::Seq),
        (true, true) => sparse_dense_matmul(out, accum, a, b.conjugate(), alpha, Par::Seq),
        (false, false) => dense_sparse_matmul(out, accum, b, a, alpha, Par::Seq),
        (false, true) => dense_sparse_matmul(out, accum, b.conjugate(), a, alpha, Par::Seq),
    }
}

/// `out = alpha * op + beta * out` (`beta == None` replaces `out`), where `op`
/// is `op(a) * b` (`left`) or `b * op(a)` with `op(a)` selected by `trans`
/// (`a`, `aᵀ` or `aᴴ`, all read in place); a conjugated `b` is read in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn product<T: ComplexField>(
    mut out: MatMut<'_, T>,
    a: SparseColMatRef<'_, usize, T>,
    trans: i32,
    b: MatRef<'_, T>,
    conjugate: bool,
    alpha: T,
    beta: Option<T>,
    left: bool,
) -> Result<(), String> {
    let (m, n) = match trans {
        PLAIN | CONJUGATE => (a.nrows(), a.ncols()),
        TRANSPOSE | ADJOINT => (a.ncols(), a.nrows()),
        _ => return Err("unknown sparse operand orientation".into()),
    };
    let shape = if left {
        (n == b.nrows()).then(|| (m, b.ncols()))
    } else {
        (b.ncols() == m).then(|| (b.nrows(), n))
    };
    if shape != Some(out.shape()) {
        return Err("sparse/dense product dimensions mismatch".into());
    }
    let accum = match beta {
        None => Accum::Replace,
        Some(beta) => {
            if beta != one::<T>() {
                for j in 0..out.ncols() {
                    for i in 0..out.nrows() {
                        out[(i, j)] = mul(&beta, &out[(i, j)]);
                    }
                }
            }
            Accum::Add
        }
    };
    match trans {
        PLAIN => apply(out, accum, a, b, conjugate, alpha, left),
        TRANSPOSE => apply(out, accum, a.transpose(), b, conjugate, alpha, left),
        ADJOINT => apply(out, accum, a.adjoint(), b, conjugate, alpha, left),
        _ => apply(out, accum, a.conjugate(), b, conjugate, alpha, left),
    }
    Ok(())
}

pub(crate) fn new_product<T: ComplexField>(
    a: SparseColMatRef<'_, usize, T>,
    trans: i32,
    b: MatRef<'_, T>,
    conjugate: bool,
    left: bool,
) -> Result<Mat<T>, String> {
    let (m, n) = if trans == PLAIN || trans == CONJUGATE {
        (a.nrows(), a.ncols())
    } else {
        (a.ncols(), a.nrows())
    };
    let (rows, cols) = if left { (m, b.ncols()) } else { (b.nrows(), n) };
    let mut out = Mat::zeros(rows, cols);
    product(out.as_mut(), a, trans, b, conjugate, one::<T>(), None, left)?;
    Ok(out)
}

// SAFETY: `p`/`desc` describe a live dense source for this call.
pub(crate) unsafe fn dense<'a>(
    p: *const u8,
    desc: *const isize,
) -> Result<(Borrowed<'a>, bool), String> {
    let s = Spec::read(desc)?;
    Ok((source(p, s)?, s.conjugate))
}

// SAFETY: alpha/beta address one `T` each; beta is ignored unless `accumulate`.
pub(crate) unsafe fn scalars<T: ComplexField>(
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
) -> (T, Option<T>) {
    let alpha = copy(&*alpha.cast::<T>());
    let beta = accumulate.then(|| copy(&*beta.cast::<T>()));
    (alpha, beta)
}

pub(crate) const MISMATCH: &str = "sparse/dense scalar types must match";

#[julia]
impl SparsityOwner {
    /// Own the validated pattern of a borrowed Julia CSC matrix.
    #[julia]
    pub fn sparsity_from_borrowed(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        k: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val: std::ptr::null(),
                k,
            };
            let s = unsafe { symbolic(raw)? };
            // SAFETY: copied from a validated symbolic view.
            self.data = Some(Pattern::Csc(unsafe {
                SymbolicSparseColMat::new_unchecked(
                    m,
                    n,
                    s.col_ptr().to_vec(),
                    None,
                    s.row_idx().to_vec(),
                )
            }));
            Ok(())
        })
    }
}

#[julia]
impl SparseOwner {
    /// Own one copy of a borrowed Julia CSC matrix (pattern and values).
    #[julia]
    pub fn sparse_from_borrowed(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            self.data = Some(match tag {
                0 => SparseData::F32(Storage::Csc(owned(numeric(raw)?))),
                1 => SparseData::F64(Storage::Csc(owned(numeric(raw)?))),
                2 => SparseData::C32(Storage::Csc(owned(numeric(raw)?))),
                3 => SparseData::C64(Storage::Csc(owned(numeric(raw)?))),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
    /// Write `[m, n, col_ptr, row_idx, values, nnz]` of a CSC owner into `out`
    /// (six words) so it can be read as a `Raw` operand next to borrowed
    /// operands. The addresses stay valid while the owner is rooted and not
    /// mutated; Julia uses them only inside one call under its matrix lock.
    #[julia]
    pub fn sparse_csc_raw_parts(&self, out: *mut usize) -> Result<bool, String> {
        guarded(|| {
            fn parts<T>(a: &Storage<T>) -> Option<[usize; 6]> {
                match a {
                    Storage::Csc(a) => {
                        let s = a.symbolic();
                        Some([
                            s.nrows(),
                            s.ncols(),
                            s.col_ptr().as_ptr() as usize,
                            s.row_idx().as_ptr() as usize,
                            a.val().as_ptr() as usize,
                            a.val().len(),
                        ])
                    }
                    Storage::Csr(_) => None,
                }
            }
            let words = match self.data()? {
                SparseData::F32(a) => parts(a),
                SparseData::F64(a) => parts(a),
                SparseData::C32(a) => parts(a),
                SparseData::C64(a) => parts(a),
            };
            let Some(words) = words else {
                return Ok(false);
            };
            // SAFETY: Julia passes a buffer of six words.
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), out, 6) };
            Ok(true)
        })
    }
    /// Fixed-pattern update of a borrowed Julia CSC destination of scalar
    /// `tag` from the CSC owners `self` (left) and `rhs`; see
    /// `sparse_fixed_update`. `Ok(false)` leaves the destination unchanged.
    #[julia]
    pub fn sparse_fixed_update_into_borrowed(
        &self,
        rhs: &SparseOwner,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *mut u8,
        k: usize,
        tag: i32,
        op: i32,
    ) -> Result<bool, String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val: val.cast_const(),
                k,
            };
            fn columns<T>(a: &Storage<T>) -> Result<SparseColMatRef<'_, usize, T>, String> {
                match a {
                    Storage::Csc(a) => Ok(a.as_ref()),
                    Storage::Csr(_) => Err("fixed sparse update storage formats must match".into()),
                }
            }
            match (tag, self.data()?, rhs.data()?) {
                (0, SparseData::F32(a), SparseData::F32(b)) => {
                    fixed_update_columns(numeric_mut(raw, val)?, columns(a)?, columns(b)?, op)
                }
                (1, SparseData::F64(a), SparseData::F64(b)) => {
                    fixed_update_columns(numeric_mut(raw, val)?, columns(a)?, columns(b)?, op)
                }
                (2, SparseData::C32(a), SparseData::C32(b)) => {
                    fixed_update_columns(numeric_mut(raw, val)?, columns(a)?, columns(b)?, op)
                }
                (3, SparseData::C64(a), SparseData::C64(b)) => {
                    fixed_update_columns(numeric_mut(raw, val)?, columns(a)?, columns(b)?, op)
                }
                _ => Err("fixed sparse update scalar types must match".into()),
            }
        })
    }
}

#[julia]
impl OwnedMatrix {
    /// New `Float64` product of a borrowed CSC matrix and a dense source.
    #[julia]
    pub fn borrowed_sparse_product(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        trans: i32,
        p: *const u8,
        desc: *const isize,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            match (tag, dense(p, desc)?) {
                (1, (Borrowed::F64(b), c)) => {
                    self.matrix = new_product(numeric(raw)?, trans, b, c, left)?
                }
                _ => return Err(MISMATCH.into()),
            }
            Ok(())
        })
    }
    /// In-place `self = alpha * op + beta * self` for a borrowed CSC matrix.
    #[julia]
    pub fn borrowed_sparse_mul(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        trans: i32,
        p: *const u8,
        desc: *const isize,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            match (tag, dense(p, desc)?) {
                (1, (Borrowed::F64(b), c)) => {
                    let (alpha, beta) = scalars::<f64>(alpha, beta, accumulate);
                    product(
                        self.matrix.as_mut(),
                        numeric(raw)?,
                        trans,
                        b,
                        c,
                        alpha,
                        beta,
                        left,
                    )
                }
                _ => Err(MISMATCH.into()),
            }
        })
    }
}

macro_rules! typed_dispatch {
    ($tag:expr, $b:expr, $x:ident, $c:ident => $body:expr) => {
        match ($tag, $b) {
            (0, (Borrowed::F32($x), $c)) => MatrixData::F32($body),
            (1, (Borrowed::F64($x), $c)) => MatrixData::F64($body),
            (2, (Borrowed::C32($x), $c)) => MatrixData::C32($body),
            (3, (Borrowed::C64($x), $c)) => MatrixData::C64($body),
            _ => return Err(MISMATCH.into()),
        }
    };
}

#[julia]
impl TypedMatrix {
    /// New product of a borrowed CSC matrix and a dense source.
    #[julia]
    pub fn borrowed_sparse_product(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        trans: i32,
        p: *const u8,
        desc: *const isize,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            self.data = Some(typed_dispatch!(tag, dense(p, desc)?, b, c =>
                new_product(numeric(raw)?, trans, b, c, left)?));
            Ok(())
        })
    }
    /// In-place `self = alpha * op + beta * self` for a borrowed CSC matrix.
    #[julia]
    pub fn borrowed_sparse_mul(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        trans: i32,
        p: *const u8,
        desc: *const isize,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let raw = Raw {
                m,
                n,
                ptr,
                idx,
                val,
                k,
            };
            let b = dense(p, desc)?;
            match (
                self.data.as_mut().ok_or("matrix is not initialized")?,
                tag,
                b,
            ) {
                (MatrixData::F32(out), 0, (Borrowed::F32(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, trans, b, c, x, y, left)
                }
                (MatrixData::F64(out), 1, (Borrowed::F64(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, trans, b, c, x, y, left)
                }
                (MatrixData::C32(out), 2, (Borrowed::C32(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, trans, b, c, x, y, left)
                }
                (MatrixData::C64(out), 3, (Borrowed::C64(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, trans, b, c, x, y, left)
                }
                _ => Err(MISMATCH.into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(m: usize, n: usize, ptr: &[usize], idx: &[usize], val: &[f64]) -> Raw {
        Raw {
            m,
            n,
            ptr: ptr.as_ptr(),
            idx: idx.as_ptr(),
            val: val.as_ptr().cast(),
            k: idx.len(),
        }
    }

    #[test]
    fn borrowed_views_validate_structure() {
        let val = [1.0, 2.0];
        unsafe {
            assert!(numeric::<f64>(raw(2, 2, &[0, 1, 2], &[0, 1], &val)).is_ok());
            assert!(numeric::<f64>(raw(2, 2, &[0, 2, 2], &[1, 0], &val)).is_err());
            assert!(numeric::<f64>(raw(2, 2, &[0, 2, 2], &[0, 0], &val)).is_err());
            assert!(numeric::<f64>(raw(2, 2, &[0, 1, 2], &[0, 2], &val)).is_err());
            assert!(numeric::<f64>(raw(2, 2, &[1, 1, 2], &[0, 1], &val)).is_err());
            assert!(numeric::<f64>(raw(3, 0, &[0], &[], &[])).is_ok());
        }
    }

    #[test]
    fn borrowed_products_do_not_allocate_beyond_the_output() {
        let (ptr, idx, val) = ([0usize, 1, 3], [0usize, 0, 1], [2.0, 3.0, 4.0]);
        let b = Mat::<f64>::from_fn(2, 3, |i, j| (i + 2 * j) as f64);
        let mut out = Mat::<f64>::ones(2, 3);
        let r = raw(2, 2, &ptr, &idx, &val);
        let count = crate::allocation_probe::count(|| unsafe {
            product(
                out.as_mut(),
                numeric(r).unwrap(),
                PLAIN,
                b.as_ref(),
                false,
                2.0,
                Some(0.5),
                true,
            )
            .unwrap();
        });
        assert_eq!(count, 0);
        let a = Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 3.0], [0.0, 4.0]][i][j]);
        let expected = (&a * &b) * 2.0 + Mat::<f64>::ones(2, 3) * 0.5;
        assert!((out - expected).norm_max() < 1e-12);
    }

    #[test]
    fn transposed_and_adjoint_products_read_in_place() {
        use faer::c64;
        // a = [2 3; 0 4+i] stored as CSC.
        let (ptr, idx) = ([0usize, 1, 3], [0usize, 0, 1]);
        let val = [c64::new(2.0, 0.0), c64::new(3.0, -1.0), c64::new(4.0, 1.0)];
        let r = Raw {
            m: 2,
            n: 2,
            ptr: ptr.as_ptr(),
            idx: idx.as_ptr(),
            val: val.as_ptr().cast(),
            k: 3,
        };
        let a = Mat::<c64>::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => val[0],
            (0, 1) => val[1],
            (1, 1) => val[2],
            _ => c64::new(0.0, 0.0),
        });
        let b = Mat::<c64>::from_fn(2, 3, |i, j| c64::new((i + 2 * j) as f64, i as f64 - 1.0));
        let bt = b.transpose().to_owned();
        for (trans, op) in [
            (TRANSPOSE, a.transpose().to_owned()),
            (ADJOINT, a.adjoint().to_owned()),
        ] {
            let mut out = Mat::<c64>::zeros(2, 3);
            let count = crate::allocation_probe::count(|| unsafe {
                let s = numeric(r).unwrap();
                product(
                    out.as_mut(),
                    s,
                    trans,
                    b.as_ref(),
                    false,
                    c64::new(1.0, 0.0),
                    None,
                    true,
                )
                .unwrap();
            });
            assert_eq!(count, 0);
            assert!((&out - &op * &b).norm_max() < 1e-12);
            // Right products read a conjugated dense operand in place too.
            let mut right = Mat::<c64>::zeros(3, 2);
            let count = crate::allocation_probe::count(|| unsafe {
                let s = numeric(r).unwrap();
                product(
                    right.as_mut(),
                    s,
                    trans,
                    bt.as_ref(),
                    true,
                    c64::new(1.0, 0.0),
                    None,
                    false,
                )
                .unwrap();
            });
            assert_eq!(count, 0);
            let expected = bt.conjugate().to_owned() * &op;
            assert!((&right - &expected).norm_max() < 1e-12);
        }
        let mut out = Mat::<c64>::zeros(3, 3);
        assert!(unsafe {
            product(
                out.as_mut(),
                numeric(r).unwrap(),
                3,
                b.as_ref(),
                false,
                c64::new(1.0, 0.0),
                None,
                true,
            )
        }
        .is_err());
    }

    #[test]
    fn borrowed_destinations_take_fixed_updates_in_place() {
        let (ptr, idx) = ([0usize, 2, 3], [0usize, 1, 1]);
        let mut val = [1.0, 2.0, 3.0];
        let r = Raw {
            m: 2,
            n: 2,
            ptr: ptr.as_ptr(),
            idx: idx.as_ptr(),
            val: val.as_ptr().cast(),
            k: 3,
        };
        let src = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            2,
            &[
                faer::sparse::Triplet::new(1, 0, 5.0),
                faer::sparse::Triplet::new(1, 1, 7.0),
            ],
        )
        .unwrap();
        let p = val.as_mut_ptr().cast::<u8>();
        let ok = unsafe {
            fixed_update_columns(
                numeric_mut::<f64>(r, p).unwrap(),
                src.as_ref(),
                src.as_ref(),
                0,
            )
        };
        assert_eq!(ok, Ok(true));
        assert_eq!(val, [1.0, 7.0, 10.0]);
        let outside = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            2,
            &[faer::sparse::Triplet::new(0, 1, 1.0)],
        )
        .unwrap();
        let p = val.as_mut_ptr().cast::<u8>();
        let ok = unsafe {
            fixed_update_columns(
                numeric_mut::<f64>(r, p).unwrap(),
                outside.as_ref(),
                outside.as_ref(),
                1,
            )
        };
        assert_eq!(ok, Ok(false));
        assert_eq!(val, [1.0, 7.0, 10.0]);
    }
}
