//! Products of native sparse owners read in place under `op ∈ {A, Aᵀ, Aᴴ}`.
//!
//! Every owner has one zero-copy column-major view: a CSC owner is itself and
//! a CSR owner is read as its transpose. `op(A)` is therefore either
//! column-major (that view is `op(A)`) or row-major (the view is `op(A)ᵀ`),
//! possibly conjugated through faer's `Conjugate` types without copying.
//!
//! * Sparse×dense: faer's `sparse_dense_matmul`/`dense_sparse_matmul` read
//!   both orientations, so `op(A)` is never copied. They take no `MemStack`
//!   and run sequentially in this faer pin, so there is no workspace or team
//!   to supply; the in-place call writes straight into the destination.
//! * Sparse×sparse: faer only multiplies column-major operands. Two
//!   column-major operands give `C` in CSC; two row-major operands give
//!   `Cᵀ = op(B)ᵀ op(A)ᵀ` in CSC, i.e. `C` in CSR, both without copying an
//!   operand. Mixed orientations convert the right operand once to the
//!   orientation of the left one.
use crate::borrowed_sparse::{
    dense, new_product, product, scalars, ADJOINT, CONJUGATE, MISMATCH, PLAIN, TRANSPOSE,
};
use crate::sparse::{SparseData, SparseOwner, Storage};
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::Borrowed;
use crate::{guarded, OwnedMatrix};
use faer::sparse::linalg::matmul::sparse_sparse_matmul;
use faer::sparse::{SparseColMat, SparseColMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::Par;
use rustcall_julia_macros::julia;

/// `op(A)` as a column-major view: `view` is `op(A)` when `col_major`, else
/// `op(A)ᵀ`; `conj` conjugates the view.
struct Oriented<'a, T> {
    view: SparseColMatRef<'a, usize, T>,
    conj: bool,
    col_major: bool,
}
impl<T> Oriented<'_, T> {
    fn shape(&self) -> (usize, usize) {
        let (m, n) = (self.view.nrows(), self.view.ncols());
        if self.col_major {
            (m, n)
        } else {
            (n, m)
        }
    }
}
fn column_major<T>(a: &Storage<T>) -> (SparseColMatRef<'_, usize, T>, bool) {
    match a {
        Storage::Csc(a) => (a.as_ref(), false),
        Storage::Csr(a) => (a.as_ref().transpose(), true),
    }
}
fn oriented<T>(a: &Storage<T>, op: i32) -> Result<Oriented<'_, T>, String> {
    let (view, csr) = column_major(a);
    let (col_major, conj) = match op {
        PLAIN => (!csr, false),
        TRANSPOSE => (csr, false),
        ADJOINT => (csr, true),
        _ => return Err("unknown sparse operand orientation".into()),
    };
    Ok(Oriented {
        view,
        conj,
        col_major,
    })
}
/// Column-major view and `borrowed_sparse::product` orientation code of `op(A)`.
pub(crate) fn dense_operand<T>(
    a: &Storage<T>,
    op: i32,
) -> Result<(SparseColMatRef<'_, usize, T>, i32), String> {
    if !matches!(op, PLAIN | TRANSPOSE | ADJOINT) {
        return Err("unknown sparse operand orientation".into());
    }
    let (view, csr) = column_major(a);
    let trans = match (csr, op) {
        (false, op) => op,
        (true, PLAIN) => TRANSPOSE,
        (true, TRANSPOSE) => PLAIN,
        (true, _) => CONJUGATE,
    };
    Ok((view, trans))
}

fn multiply<T: ComplexField>(
    l: SparseColMatRef<'_, usize, T>,
    lc: bool,
    r: SparseColMatRef<'_, usize, T>,
    rc: bool,
) -> Result<SparseColMat<usize, T>, String> {
    let one = one::<T>();
    match (lc, rc) {
        (false, false) => sparse_sparse_matmul(l, r, one, Par::Seq),
        (true, false) => sparse_sparse_matmul(l.conjugate(), r, one, Par::Seq),
        (false, true) => sparse_sparse_matmul(l, r.conjugate(), one, Par::Seq),
        (true, true) => sparse_sparse_matmul(l.conjugate(), r.conjugate(), one, Par::Seq),
    }
    .map_err(|e| format!("{e:?}"))
}
/// The transpose of `x` (conjugated when `conj`) as an owned CSC matrix.
fn transposed<T: ComplexField>(x: &Oriented<'_, T>) -> Result<SparseColMat<usize, T>, String> {
    let t = x.view.transpose();
    if x.conj {
        t.conjugate().to_col_major()
    } else {
        t.to_col_major()
    }
    .map_err(|e| format!("{e:?}"))
}
/// `op(A) * op(B)`; the result orientation follows `op(A)`.
fn oriented_product<T: ComplexField>(
    a: &Storage<T>,
    aop: i32,
    b: &Storage<T>,
    bop: i32,
) -> Result<Storage<T>, String> {
    let (a, b) = (oriented(a, aop)?, oriented(b, bop)?);
    if a.shape().1 != b.shape().0 {
        return Err("sparse operation dimensions must match".into());
    }
    let converted = if a.col_major != b.col_major {
        Some(transposed(&b)?)
    } else {
        None
    };
    let (bv, bc) = match &converted {
        Some(c) => (c.as_ref(), false),
        None => (b.view, b.conj),
    };
    Ok(if a.col_major {
        Storage::Csc(multiply(a.view, a.conj, bv, bc)?)
    } else {
        Storage::Csr(multiply(bv, bc, a.view, a.conj)?.into_transpose())
    })
}

#[julia]
impl SparseOwner {
    /// `self = op(lhs) * op(rhs)` with each operand read in place; returns
    /// whether the result is CSR.
    #[julia]
    pub fn sparse_oriented_product(
        &mut self,
        lhs: &SparseOwner,
        lop: i32,
        rhs: &SparseOwner,
        rop: i32,
    ) -> Result<bool, String> {
        guarded(|| {
            let data = match (lhs.data()?, rhs.data()?) {
                (SparseData::F32(a), SparseData::F32(b)) => {
                    SparseData::F32(oriented_product(a, lop, b, rop)?)
                }
                (SparseData::F64(a), SparseData::F64(b)) => {
                    SparseData::F64(oriented_product(a, lop, b, rop)?)
                }
                (SparseData::C32(a), SparseData::C32(b)) => {
                    SparseData::C32(oriented_product(a, lop, b, rop)?)
                }
                (SparseData::C64(a), SparseData::C64(b)) => {
                    SparseData::C64(oriented_product(a, lop, b, rop)?)
                }
                _ => return Err("sparse scalar types must match".into()),
            };
            let csr = matches!(
                &data,
                SparseData::F32(Storage::Csr(_))
                    | SparseData::F64(Storage::Csr(_))
                    | SparseData::C32(Storage::Csr(_))
                    | SparseData::C64(Storage::Csr(_))
            );
            self.data = Some(data);
            Ok(csr)
        })
    }
}

#[julia]
impl OwnedMatrix {
    /// New `Float64` product `op(S) * B` (`left`) or `B * op(S)`.
    #[julia]
    pub fn sparse_owner_product(
        &mut self,
        sparse: &SparseOwner,
        op: i32,
        p: *const u8,
        desc: *const isize,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            match (sparse.data()?, dense(p, desc)?) {
                (SparseData::F64(a), (Borrowed::F64(b), c)) => {
                    let (a, trans) = dense_operand(a, op)?;
                    self.matrix = new_product(a, trans, b, c, left)?;
                    Ok(())
                }
                _ => Err(MISMATCH.into()),
            }
        })
    }
    /// In place `self = alpha * op(S) * B + beta * self` (or `B * op(S)`).
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_owner_mul(
        &mut self,
        sparse: &SparseOwner,
        op: i32,
        p: *const u8,
        desc: *const isize,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            match (sparse.data()?, dense(p, desc)?) {
                (SparseData::F64(a), (Borrowed::F64(b), c)) => {
                    let (a, trans) = dense_operand(a, op)?;
                    let (x, y) = scalars::<f64>(alpha, beta, accumulate);
                    product(self.matrix.as_mut(), a, trans, b, c, x, y, left)
                }
                _ => Err(MISMATCH.into()),
            }
        })
    }
}

#[julia]
impl TypedMatrix {
    /// New product `op(S) * B` (`left`) or `B * op(S)`.
    #[julia]
    pub fn sparse_owner_product(
        &mut self,
        sparse: &SparseOwner,
        op: i32,
        p: *const u8,
        desc: *const isize,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let out = match (sparse.data()?, dense(p, desc)?) {
                (SparseData::F32(a), (Borrowed::F32(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    MatrixData::F32(new_product(a, t, b, c, left)?)
                }
                (SparseData::F64(a), (Borrowed::F64(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    MatrixData::F64(new_product(a, t, b, c, left)?)
                }
                (SparseData::C32(a), (Borrowed::C32(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    MatrixData::C32(new_product(a, t, b, c, left)?)
                }
                (SparseData::C64(a), (Borrowed::C64(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    MatrixData::C64(new_product(a, t, b, c, left)?)
                }
                _ => return Err(MISMATCH.into()),
            };
            self.data = Some(out);
            Ok(())
        })
    }
    /// In place `self = alpha * op(S) * B + beta * self` (or `B * op(S)`).
    #[julia]
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_owner_mul(
        &mut self,
        sparse: &SparseOwner,
        op: i32,
        p: *const u8,
        desc: *const isize,
        alpha: *const u8,
        beta: *const u8,
        accumulate: bool,
        left: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let out = self.data.as_mut().ok_or("matrix is not initialized")?;
            match (out, sparse.data()?, dense(p, desc)?) {
                (MatrixData::F32(out), SparseData::F32(a), (Borrowed::F32(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), a, t, b, c, x, y, left)
                }
                (MatrixData::F64(out), SparseData::F64(a), (Borrowed::F64(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), a, t, b, c, x, y, left)
                }
                (MatrixData::C32(out), SparseData::C32(a), (Borrowed::C32(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), a, t, b, c, x, y, left)
                }
                (MatrixData::C64(out), SparseData::C64(a), (Borrowed::C64(b), c)) => {
                    let (a, t) = dense_operand(a, op)?;
                    let (x, y) = scalars(alpha, beta, accumulate);
                    product(out.as_mut(), a, t, b, c, x, y, left)
                }
                _ => Err(MISMATCH.into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;
    use faer::{c64, Mat};

    fn sample(m: usize, n: usize, shift: usize) -> SparseColMat<usize, c64> {
        let mut t = vec![];
        for j in 0..n {
            for i in 0..m {
                if (i * 3 + j * 5 + shift) % 4 == 0 {
                    t.push(Triplet::new(
                        i,
                        j,
                        c64::new((i + 1) as f64, (j as f64) - 0.5 * shift as f64),
                    ));
                }
            }
        }
        SparseColMat::try_new_from_triplets(m, n, &t).unwrap()
    }
    fn dense_op(a: &Mat<c64>, op: i32) -> Mat<c64> {
        match op {
            PLAIN => a.clone(),
            TRANSPOSE => a.transpose().to_owned(),
            _ => a.adjoint().to_owned(),
        }
    }
    fn to_dense(s: &Storage<c64>) -> Mat<c64> {
        match s {
            Storage::Csc(a) => a.to_dense(),
            Storage::Csr(a) => a.to_dense(),
        }
    }
    fn close(a: &Mat<c64>, b: &Mat<c64>) -> bool {
        a.shape() == b.shape()
            && (0..a.nrows())
                .all(|i| (0..a.ncols()).all(|j| (a[(i, j)] - b[(i, j)]).norm() < 1e-12))
    }

    #[test]
    fn oriented_sparse_products_match_dense_products() {
        let a = sample(4, 4, 0);
        let b = sample(4, 4, 1);
        for (acsr, bcsr) in [(false, false), (false, true), (true, false), (true, true)] {
            let sa = if acsr {
                Storage::Csr(a.to_row_major().unwrap())
            } else {
                Storage::Csc(a.clone())
            };
            let sb = if bcsr {
                Storage::Csr(b.to_row_major().unwrap())
            } else {
                Storage::Csc(b.clone())
            };
            for aop in [PLAIN, TRANSPOSE, ADJOINT] {
                for bop in [PLAIN, TRANSPOSE, ADJOINT] {
                    let c = oriented_product(&sa, aop, &sb, bop).unwrap();
                    let left_col_major = oriented(&sa, aop).unwrap().col_major;
                    assert_eq!(matches!(c, Storage::Csc(_)), left_col_major);
                    let want = dense_op(&a.to_dense(), aop) * dense_op(&b.to_dense(), bop);
                    assert!(close(&to_dense(&c), &want), "{acsr} {bcsr} {aop} {bop}");
                }
            }
        }
        let sa = Storage::Csc(sample(3, 4, 0));
        assert!(oriented_product(&sa, PLAIN, &sa, PLAIN).is_err());
        assert!(oriented_product(&sa, 7, &sa, PLAIN).is_err());
    }

    #[test]
    fn same_orientation_products_allocate_only_the_result() {
        // Both row-major: the only allocations are faer's symbolic/numeric
        // product buffers, the same as a plain CSC×CSC product.
        let a = sample(30, 30, 0);
        let b = sample(30, 30, 1);
        let plain = (Storage::Csc(a.clone()), Storage::Csc(b.clone()));
        let base = crate::allocation_probe::count(|| {
            oriented_product(&plain.0, PLAIN, &plain.1, PLAIN).unwrap();
        });
        let adj = crate::allocation_probe::count(|| {
            oriented_product(&plain.0, ADJOINT, &plain.1, TRANSPOSE).unwrap();
        });
        assert_eq!(base, adj);
        let mixed = crate::allocation_probe::count(|| {
            oriented_product(&plain.0, ADJOINT, &plain.1, PLAIN).unwrap();
        });
        assert!(mixed > base);
    }

    #[test]
    fn oriented_dense_products_read_csr_owners_in_place() {
        let a = sample(4, 3, 0);
        let d = a.to_dense();
        let b = Mat::<c64>::from_fn(4, 2, |i, j| c64::new(i as f64, 1.0 + j as f64));
        for csr in [false, true] {
            let s = if csr {
                Storage::Csr(a.to_row_major().unwrap())
            } else {
                Storage::Csc(a.clone())
            };
            for op in [TRANSPOSE, ADJOINT] {
                let (view, trans) = dense_operand(&s, op).unwrap();
                let mut out = Mat::<c64>::zeros(3, 2);
                let count = crate::allocation_probe::count(|| {
                    product(
                        out.as_mut(),
                        view,
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
                assert!(close(&out, &(dense_op(&d, op) * &b)));
            }
        }
        assert!(dense_operand(&Storage::Csc(a), 5).is_err());
    }
}
