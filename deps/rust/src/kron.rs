//! Kronecker products read their operands in place through view descriptors.
//!
//! Dense operands are any `views::source` (native owners, strided and reversed
//! views, transposes, borrowed Julia arrays); faer `linalg::kron::kron` writes
//! straight into the new owner or the destination view. Sparse products build
//! the CSC (or, through transposed operands, CSR) structure directly. Borrowed
//! refs live only inside one synchronous call.
use crate::borrowed_sparse::{numeric, Raw};
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{source, write_view, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::linalg::kron::kron;
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Mat, MatMut, MatRef};
use rustcall_julia_macros::julia;

fn kron_shape(a: (usize, usize), b: (usize, usize)) -> Result<(usize, usize), String> {
    Ok((
        a.0.checked_mul(b.0).ok_or("Kronecker size overflow")?,
        a.1.checked_mul(b.1).ok_or("Kronecker size overflow")?,
    ))
}

/// `dst = op(a) ⊗ op(b)`, `op` conjugating when requested; read in place.
fn kron_into<T: ComplexField>(
    dst: MatMut<'_, T>,
    a: MatRef<'_, T>,
    ca: bool,
    b: MatRef<'_, T>,
    cb: bool,
) -> Result<(), String> {
    if kron_shape(a.shape(), b.shape())? != dst.shape() {
        return Err("Kronecker destination dimensions must match".into());
    }
    match (ca, cb) {
        (false, false) => kron(dst, a, b),
        (true, false) => kron(dst, a.conjugate(), b),
        (false, true) => kron(dst, a, b.conjugate()),
        (true, true) => kron(dst, a.conjugate(), b.conjugate()),
    }
    Ok(())
}

fn kron_owned<T: ComplexField>(
    a: MatRef<'_, T>,
    ca: bool,
    b: MatRef<'_, T>,
    cb: bool,
) -> Result<Mat<T>, String> {
    let (m, n) = kron_shape(a.shape(), b.shape())?;
    let mut out = Mat::zeros(m, n);
    kron_into(out.as_mut(), a, ca, b, cb)?;
    Ok(out)
}

/// A vector owner of `kind` holding the column-major values of the product.
fn kron_vector<T: ComplexField>(
    kind: i32,
    a: MatRef<'_, T>,
    ca: bool,
    b: MatRef<'_, T>,
    cb: bool,
) -> Result<VectorStore<T>, String> {
    let (m, n) = kron_shape(a.shape(), b.shape())?;
    if m != 1 && n != 1 {
        return Err("a vector Kronecker product needs one unit dimension".into());
    }
    let mut out = VectorStore::new(kind, m * n, |_| zero::<T>())?;
    {
        let mut dst = out.as_mat_mut();
        if dst.shape() != (m, n) {
            dst = dst.transpose_mut();
        }
        kron_into(dst, a, ca, b, cb)?;
    }
    Ok(out)
}

unsafe fn kron_data(
    left: *const u8,
    ls: Spec,
    right: *const u8,
    rs: Spec,
) -> Result<MatrixData, String> {
    Ok(match (source(left, ls)?, source(right, rs)?) {
        (Borrowed::F32(a), Borrowed::F32(b)) => {
            MatrixData::F32(kron_owned(a, ls.conjugate, b, rs.conjugate)?)
        }
        (Borrowed::F64(a), Borrowed::F64(b)) => {
            MatrixData::F64(kron_owned(a, ls.conjugate, b, rs.conjugate)?)
        }
        (Borrowed::C32(a), Borrowed::C32(b)) => {
            MatrixData::C32(kron_owned(a, ls.conjugate, b, rs.conjugate)?)
        }
        (Borrowed::C64(a), Borrowed::C64(b)) => {
            MatrixData::C64(kron_owned(a, ls.conjugate, b, rs.conjugate)?)
        }
        _ => return Err("Kronecker scalar types must match".into()),
    })
}

/// Writes `op(a) ⊗ op(b)` into the view `s` of `dst`. Injective views are
/// written in place; repeated indices go through an ordered staged write.
fn kron_view<T: ComplexField>(
    dst: MatMut<'_, T>,
    s: Spec,
    a: MatRef<'_, T>,
    ca: bool,
    b: MatRef<'_, T>,
    cb: bool,
) -> Result<(), String> {
    s.check(dst.nrows(), dst.ncols())?;
    let expected = if s.transpose { (s.n, s.m) } else { (s.m, s.n) };
    if kron_shape(a.shape(), b.shape())? != expected {
        return Err("Kronecker destination dimensions must match".into());
    }
    if s.m == 0 || s.n == 0 {
        return Ok(());
    }
    if (s.rs == 0 && s.m > 1) || (s.cs == 0 && s.n > 1) {
        let staged = kron_owned(a, ca, b, cb)?;
        return write_view(dst, s, staged.as_ref(), s.conjugate);
    }
    let rs = dst
        .row_stride()
        .checked_mul(s.rs)
        .ok_or("view stride overflow")?;
    let cs = dst
        .col_stride()
        .checked_mul(s.cs)
        .ok_or("view stride overflow")?;
    let offset = dst
        .row_stride()
        .checked_mul(s.r)
        .and_then(|x| {
            dst.col_stride()
                .checked_mul(s.c)
                .and_then(|y| x.checked_add(y))
        })
        .ok_or("view offset overflow")?;
    // SAFETY: the checked, injective index map selects distinct initialized
    // elements of the uniquely borrowed destination.
    let mut v =
        unsafe { MatMut::from_raw_parts_mut(dst.as_ptr_mut().offset(offset), s.m, s.n, rs, cs) };
    if s.transpose {
        v = v.transpose_mut();
    }
    // A conjugated destination view stores conj(op(a) ⊗ op(b)).
    kron_into(v, a, ca != s.conjugate, b, cb != s.conjugate)
}

// ---------------------------------------------------------------- sparse ---

/// One operand of a sparse Kronecker product, accessible by columns.
enum Major<'a, T> {
    Sparse(SparseColMatRef<'a, usize, T>),
    Owned(SparseColMat<usize, T>),
    Dense(MatRef<'a, T>),
}
impl<T: ComplexField> Major<'_, T> {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Sparse(a) => (a.nrows(), a.ncols()),
            Self::Owned(a) => (a.nrows(), a.ncols()),
            Self::Dense(a) => (a.nrows(), a.ncols()),
        }
    }
    fn sparse(&self) -> Option<SparseColMatRef<'_, usize, T>> {
        match self {
            Self::Sparse(a) => Some(*a),
            Self::Owned(a) => Some(a.as_ref()),
            Self::Dense(_) => None,
        }
    }
    /// Entries of column `j`; dense zeros are not stored, as SparseArrays
    /// sparsifies a dense Kronecker operand.
    fn count(&self, j: usize) -> usize {
        match (self.sparse(), self) {
            (Some(a), _) => a.symbolic().col_range(j).len(),
            (None, Self::Dense(a)) => (0..a.nrows()).filter(|&i| a[(i, j)] != zero::<T>()).count(),
            _ => unreachable!(),
        }
    }
    fn each(&self, j: usize, conjugate: bool, mut f: impl FnMut(usize, T)) {
        let op = |x: &T| if conjugate { conj(x) } else { copy(x) };
        match (self.sparse(), self) {
            (Some(a), _) => {
                let range = a.symbolic().col_range(j);
                let rows = &a.symbolic().row_idx()[range.clone()];
                for (&i, x) in rows.iter().zip(&a.val()[range]) {
                    f(i, op(x));
                }
            }
            (None, Self::Dense(a)) => {
                for i in 0..a.nrows() {
                    let x = &a[(i, j)];
                    if *x != zero::<T>() {
                        f(i, op(x));
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// CSC of `op(a) ⊗ op(b)`. Output column `ja * nb + jb` holds rows
/// `ia * mb + ib`, so sorted operand columns give sorted output columns.
fn sparse_kron_csc<T: ComplexField>(
    a: &Major<'_, T>,
    ca: bool,
    b: &Major<'_, T>,
    cb: bool,
) -> Result<SparseColMat<usize, T>, String> {
    let ((ma, na), (mb, nb)) = (a.shape(), b.shape());
    let (m, n) = kron_shape((ma, na), (mb, nb))?;
    let counts_a: Vec<usize> = (0..na).map(|j| a.count(j)).collect();
    let counts_b: Vec<usize> = (0..nb).map(|j| b.count(j)).collect();
    let mut ptr = Vec::with_capacity(n.checked_add(1).ok_or("Kronecker size overflow")?);
    ptr.push(0usize);
    let mut total = 0usize;
    for &x in &counts_a {
        for &y in &counts_b {
            total = x
                .checked_mul(y)
                .and_then(|k| total.checked_add(k))
                .ok_or("Kronecker nonzero count overflow")?;
            ptr.push(total);
        }
    }
    let mut idx = Vec::with_capacity(total);
    let mut val = Vec::with_capacity(total);
    for ja in 0..na {
        if counts_a[ja] == 0 {
            continue;
        }
        for jb in 0..nb {
            if counts_b[jb] == 0 {
                continue;
            }
            a.each(ja, ca, |ia, x| {
                b.each(jb, cb, |ib, y| {
                    idx.push(ia * mb + ib);
                    val.push(mul(&x, &y));
                })
            });
        }
    }
    Ok(SparseColMat::new(
        SymbolicSparseColMat::new_checked(m, n, ptr, None, idx),
        val,
    ))
}

fn storage_major<T: ComplexField>(s: &Storage<T>, transpose: bool) -> Result<Major<'_, T>, String> {
    let e = |e| format!("{e:?}");
    Ok(match (s, transpose) {
        (Storage::Csc(a), false) => Major::Sparse(a.as_ref()),
        (Storage::Csr(a), true) => Major::Sparse(a.as_ref().transpose()),
        (Storage::Csc(a), true) => Major::Owned(a.as_ref().transpose().to_col_major().map_err(e)?),
        (Storage::Csr(a), false) => Major::Owned(a.as_ref().to_col_major().map_err(e)?),
    })
}
fn csc_major<T: ComplexField>(
    a: SparseColMatRef<'_, usize, T>,
    transpose: bool,
) -> Result<Major<'_, T>, String> {
    Ok(if transpose {
        Major::Owned(a.transpose().to_col_major().map_err(|e| format!("{e:?}"))?)
    } else {
        Major::Sparse(a)
    })
}

fn dense_major<T>(a: MatRef<'_, T>, flip: bool) -> Major<'_, T> {
    Major::Dense(if flip { a.transpose() } else { a })
}

enum AnyMajor<'a> {
    F32(Major<'a, f32>),
    F64(Major<'a, f64>),
    C32(Major<'a, faer::c32>),
    C64(Major<'a, faer::c64>),
}

/// Operand descriptor: `[0, transpose, conjugate]` for a native sparse owner
/// `p`; `[1, transpose, conjugate, m, n, colptr, rowval, nzval, nnz, tag]` for
/// a borrowed Julia CSC (0-based index buffers rooted by Julia); `[2, view
/// descriptor...]` for a dense source `p`. `flip` transposes the operand, used
/// to build CSR output as the CSC of the transposed product.
// SAFETY: pointers follow the descriptor contract above for this call.
unsafe fn sparse_operand<'a>(
    p: *const u8,
    desc: *const isize,
    flip: bool,
) -> Result<(AnyMajor<'a>, bool), String> {
    let kind = *desc;
    if kind == 2 {
        let s = Spec::read(desc.add(1))?;
        return Ok((
            match source(p, s)? {
                Borrowed::F32(a) => AnyMajor::F32(dense_major(a, flip)),
                Borrowed::F64(a) => AnyMajor::F64(dense_major(a, flip)),
                Borrowed::C32(a) => AnyMajor::C32(dense_major(a, flip)),
                Borrowed::C64(a) => AnyMajor::C64(dense_major(a, flip)),
            },
            s.conjugate,
        ));
    }
    let d = std::slice::from_raw_parts(desc, 3);
    let transpose = (d[1] != 0) != flip;
    let conjugate = d[2] != 0;
    let major = match kind {
        0 => match (&*p.cast::<SparseOwner>()).data()? {
            SparseData::F32(s) => AnyMajor::F32(storage_major(s, transpose)?),
            SparseData::F64(s) => AnyMajor::F64(storage_major(s, transpose)?),
            SparseData::C32(s) => AnyMajor::C32(storage_major(s, transpose)?),
            SparseData::C64(s) => AnyMajor::C64(storage_major(s, transpose)?),
        },
        1 => {
            let r = std::slice::from_raw_parts(desc.add(3), 7);
            let raw = Raw {
                m: usize::try_from(r[0]).map_err(|_| "negative sparse rows")?,
                n: usize::try_from(r[1]).map_err(|_| "negative sparse columns")?,
                ptr: r[2] as *const usize,
                idx: r[3] as *const usize,
                val: r[4] as *const u8,
                k: usize::try_from(r[5]).map_err(|_| "negative sparse nnz")?,
            };
            match r[6] {
                0 => AnyMajor::F32(csc_major(numeric(raw)?, transpose)?),
                1 => AnyMajor::F64(csc_major(numeric(raw)?, transpose)?),
                2 => AnyMajor::C32(csc_major(numeric(raw)?, transpose)?),
                3 => AnyMajor::C64(csc_major(numeric(raw)?, transpose)?),
                _ => return Err("unknown borrowed sparse scalar".into()),
            }
        }
        _ => return Err("unknown sparse Kronecker operand".into()),
    };
    Ok((major, conjugate))
}

fn finish<T: ComplexField>(c: SparseColMat<usize, T>, csr: bool) -> Storage<T> {
    if csr {
        Storage::Csr(c.into_transpose())
    } else {
        Storage::Csc(c)
    }
}

#[julia]
impl SparseOwner {
    /// Sparse `a ⊗ b` in CSC (or CSR with `csr`), where at least one operand is
    /// sparse; see `sparse_operand` for the descriptors.
    #[julia]
    pub fn sparse_kron(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let (a, ca) = sparse_operand(left, ld, csr)?;
            let (b, cb) = sparse_operand(right, rd, csr)?;
            // The CSR product is the CSC of `aᵀ ⊗ bᵀ` (the operands are flipped).
            self.data = Some(match (a, b) {
                (AnyMajor::F32(a), AnyMajor::F32(b)) => {
                    SparseData::F32(finish(sparse_kron_csc(&a, ca, &b, cb)?, csr))
                }
                (AnyMajor::F64(a), AnyMajor::F64(b)) => {
                    SparseData::F64(finish(sparse_kron_csc(&a, ca, &b, cb)?, csr))
                }
                (AnyMajor::C32(a), AnyMajor::C32(b)) => {
                    SparseData::C32(finish(sparse_kron_csc(&a, ca, &b, cb)?, csr))
                }
                (AnyMajor::C64(a), AnyMajor::C64(b)) => {
                    SparseData::C64(finish(sparse_kron_csc(&a, ca, &b, cb)?, csr))
                }
                _ => return Err("Kronecker scalar types must match".into()),
            });
            Ok(())
        })
    }
}

// ----------------------------------------------------------------- dense ---

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn view_kron(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = match unsafe { kron_data(left, Spec::read(ld)?, right, Spec::read(rd)?)? }
            {
                MatrixData::F64(a) => a,
                _ => return Err("Float64 output requires Float64 operands".into()),
            };
            Ok(())
        })
    }
    /// Writes `a ⊗ b` into the view `desc` of `self`. Julia snapshots any
    /// operand sharing this owner before the call.
    #[julia]
    pub fn view_kron_into(
        &mut self,
        desc: *const isize,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let (s, ls, rs) = (Spec::read(desc)?, Spec::read(ld)?, Spec::read(rd)?);
            match (source(left, ls)?, source(right, rs)?) {
                (Borrowed::F64(a), Borrowed::F64(b)) => {
                    kron_view(self.matrix.as_mut(), s, a, ls.conjugate, b, rs.conjugate)
                }
                _ => Err("Kronecker scalar types must match the destination".into()),
            }
        })
    }
}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn view_kron(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { kron_data(left, Spec::read(ld)?, right, Spec::read(rd)?)? });
            Ok(())
        })
    }
    #[julia]
    pub fn view_kron_into(
        &mut self,
        desc: *const isize,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let (s, ls, rs) = (Spec::read(desc)?, Spec::read(ld)?, Spec::read(rd)?);
            let (ca, cb) = (ls.conjugate, rs.conjugate);
            match (
                self.data.as_mut().ok_or("matrix is not initialized")?,
                source(left, ls)?,
                source(right, rs)?,
            ) {
                (MatrixData::F32(d), Borrowed::F32(a), Borrowed::F32(b)) => {
                    kron_view(d.as_mut(), s, a, ca, b, cb)
                }
                (MatrixData::F64(d), Borrowed::F64(a), Borrowed::F64(b)) => {
                    kron_view(d.as_mut(), s, a, ca, b, cb)
                }
                (MatrixData::C32(d), Borrowed::C32(a), Borrowed::C32(b)) => {
                    kron_view(d.as_mut(), s, a, ca, b, cb)
                }
                (MatrixData::C64(d), Borrowed::C64(a), Borrowed::C64(b)) => {
                    kron_view(d.as_mut(), s, a, ca, b, cb)
                }
                _ => Err("Kronecker scalar types must match the destination".into()),
            }
        })
    }
}

#[julia]
impl VectorOwner {
    /// A `kind` collection (column or row) holding `a ⊗ b`, which has one unit
    /// dimension, in column-major order.
    #[julia]
    pub fn vector_view_kron(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let (ls, rs) = (Spec::read(ld)?, Spec::read(rd)?);
            let (ca, cb) = (ls.conjugate, rs.conjugate);
            self.data = Some(match (source(left, ls)?, source(right, rs)?) {
                (Borrowed::F32(a), Borrowed::F32(b)) => {
                    VectorData::F32(kron_vector(kind, a, ca, b, cb)?)
                }
                (Borrowed::F64(a), Borrowed::F64(b)) => {
                    VectorData::F64(kron_vector(kind, a, ca, b, cb)?)
                }
                (Borrowed::C32(a), Borrowed::C32(b)) => {
                    VectorData::C32(kron_vector(kind, a, ca, b, cb)?)
                }
                (Borrowed::C64(a), Borrowed::C64(b)) => {
                    VectorData::C64(kron_vector(kind, a, ca, b, cb)?)
                }
                _ => return Err("Kronecker scalar types must match".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn view_kron_into(
        &mut self,
        desc: *const isize,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let (s, ls, rs) = (Spec::read(desc)?, Spec::read(ld)?, Spec::read(rd)?);
            let (ca, cb) = (ls.conjugate, rs.conjugate);
            match (
                self.data.as_mut().ok_or("collection is not initialized")?,
                source(left, ls)?,
                source(right, rs)?,
            ) {
                (VectorData::F32(d), Borrowed::F32(a), Borrowed::F32(b)) => {
                    kron_view(d.as_mat_mut(), s, a, ca, b, cb)
                }
                (VectorData::F64(d), Borrowed::F64(a), Borrowed::F64(b)) => {
                    kron_view(d.as_mat_mut(), s, a, ca, b, cb)
                }
                (VectorData::C32(d), Borrowed::C32(a), Borrowed::C32(b)) => {
                    kron_view(d.as_mat_mut(), s, a, ca, b, cb)
                }
                (VectorData::C64(d), Borrowed::C64(a), Borrowed::C64(b)) => {
                    kron_view(d.as_mat_mut(), s, a, ca, b, cb)
                }
                _ => Err("Kronecker scalar types must match the destination".into()),
            }
        })
    }
}
