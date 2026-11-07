//! Per-call faer views of a Julia `SparseMatrixCSC`.
//!
//! faer's `SymbolicSparseColMatRef` is always 0-based, so the Julia layer
//! converts `colptr`/`rowval` into reused 0-based `usize` buffers before each
//! call; the stored values are read in place from Julia's `nzval`. Every
//! pointer is valid only for the duration of one synchronous call (Julia roots
//! the parent and the buffers) and is never retained.
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::sparsity::{Pattern, SparsityOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{source, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

/// Raw CSC parts: `n + 1` column pointers, `k` row indices and `k` values.
#[derive(Clone, Copy)]
struct Raw {
    m: usize,
    n: usize,
    ptr: *const usize,
    idx: *const usize,
    val: *const u8,
    k: usize,
}

// SAFETY: `ptr` addresses `n + 1` and `idx` `k` initialized `usize` values
// that Julia keeps alive for this call. The structure is fully validated here,
// so faer's unchecked invariants hold.
unsafe fn symbolic<'a>(r: Raw) -> Result<SymbolicSparseColMatRef<'a, usize>, String> {
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
unsafe fn numeric<'a, T>(r: Raw) -> Result<SparseColMatRef<'a, usize, T>, String> {
    let s = symbolic(r)?;
    let val: &[T] = if r.k == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(r.val.cast::<T>(), r.k)
    };
    Ok(SparseColMatRef::new(s, val))
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

/// `out = alpha * op + beta * out` (`beta == None` replaces `out`), where `op`
/// is `a * b` (`left`) or `b * a`; a conjugated `b` is read in place.
fn product<T: ComplexField>(
    mut out: MatMut<'_, T>,
    a: SparseColMatRef<'_, usize, T>,
    b: MatRef<'_, T>,
    conjugate: bool,
    alpha: T,
    beta: Option<T>,
    left: bool,
) -> Result<(), String> {
    let (m, n) = (a.nrows(), a.ncols());
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
    use faer::sparse::linalg::matmul::{dense_sparse_matmul, sparse_dense_matmul};
    match (left, conjugate) {
        (true, false) => sparse_dense_matmul(out, accum, a, b, alpha, Par::Seq),
        (true, true) => sparse_dense_matmul(out, accum, a, b.conjugate(), alpha, Par::Seq),
        (false, false) => dense_sparse_matmul(out, accum, b, a, alpha, Par::Seq),
        (false, true) => dense_sparse_matmul(out, accum, b.conjugate(), a, alpha, Par::Seq),
    }
    Ok(())
}

fn new_product<T: ComplexField>(
    a: SparseColMatRef<'_, usize, T>,
    b: MatRef<'_, T>,
    conjugate: bool,
    left: bool,
) -> Result<Mat<T>, String> {
    let (rows, cols) = if left {
        (a.nrows(), b.ncols())
    } else {
        (b.nrows(), a.ncols())
    };
    let mut out = Mat::zeros(rows, cols);
    product(out.as_mut(), a, b, conjugate, one::<T>(), None, left)?;
    Ok(out)
}

// SAFETY: `p`/`desc` describe a live dense source for this call.
unsafe fn dense<'a>(p: *const u8, desc: *const isize) -> Result<(Borrowed<'a>, bool), String> {
    let s = Spec::read(desc)?;
    Ok((source(p, s)?, s.conjugate))
}

// SAFETY: alpha/beta address one `T` each; beta is ignored unless `accumulate`.
unsafe fn scalars<T: ComplexField>(
    alpha: *const u8,
    beta: *const u8,
    accumulate: bool,
) -> (T, Option<T>) {
    let alpha = copy(&*alpha.cast::<T>());
    let beta = accumulate.then(|| copy(&*beta.cast::<T>()));
    (alpha, beta)
}

const MISMATCH: &str = "sparse/dense scalar types must match";

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
                (1, (Borrowed::F64(b), c)) => self.matrix = new_product(numeric(raw)?, b, c, left)?,
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
                    product(self.matrix.as_mut(), numeric(raw)?, b, c, alpha, beta, left)
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
                new_product(numeric(raw)?, b, c, left)?));
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
                    product(out.as_mut(), numeric(raw)?, b, c, x, y, left)
                }
                (MatrixData::F64(out), 1, (Borrowed::F64(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, b, c, x, y, left)
                }
                (MatrixData::C32(out), 2, (Borrowed::C32(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, b, c, x, y, left)
                }
                (MatrixData::C64(out), 3, (Borrowed::C64(b), c)) => {
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), numeric(raw)?, b, c, x, y, left)
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
}
