//! Structural primitives of the native multi-operand sparse broadcast.
//!
//! Julia applies the broadcast function to value buffers; Rust only compares
//! patterns, builds the union pattern (faer's `union_symbolic`) and aligns
//! each operand's stored values onto it, all on the physical compressed
//! arrays of owners sharing one storage format.
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner, Storage};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::traits::{math_utils::*, ComplexField};
use rustcall_julia_macros::julia;

macro_rules! each {
    ($data:expr, $a:ident => $body:expr) => {
        match $data {
            SparseData::F32($a) => $body,
            SparseData::F64($a) => $body,
            SparseData::C32($a) => $body,
            SparseData::C64($a) => $body,
        }
    };
}

/// Physical compressed pattern: `(csr, shape, pointers, indices)`; its
/// column-major view is the matrix itself (CSC) or its transpose (CSR).
fn physical<T>(a: &Storage<T>) -> (bool, SymbolicSparseColMatRef<'_, usize>) {
    match a {
        Storage::Csc(a) => (false, a.symbolic()),
        Storage::Csr(a) => (true, a.symbolic().transpose()),
    }
}
fn same(
    a: (bool, SymbolicSparseColMatRef<'_, usize>),
    b: (bool, SymbolicSparseColMatRef<'_, usize>),
) -> bool {
    let (x, y) = (a.1, b.1);
    a.0 == b.0
        && x.nrows() == y.nrows()
        && x.ncols() == y.ncols()
        && x.col_ptr() == y.col_ptr()
        && x.row_idx() == y.row_idx()
}
fn zeros<T: ComplexField>(s: SymbolicSparseColMat<usize>, csr: bool) -> Storage<T> {
    let k = s.row_idx().len();
    let c = SparseColMat::new(s, vec![zero::<T>(); k]);
    if csr {
        Storage::Csr(c.into_transpose())
    } else {
        Storage::Csc(c)
    }
}
/// Write `src`'s values at the positions of `target` (which must contain
/// `src`'s pattern), zero elsewhere. Nothing is written on error.
fn align<T: ComplexField>(
    target: SymbolicSparseColMatRef<'_, usize>,
    src: SymbolicSparseColMatRef<'_, usize>,
    vals: &[T],
    out: &mut [T],
) -> Result<(), String> {
    let (tp, ti, sp, si) = (
        target.col_ptr(),
        target.row_idx(),
        src.col_ptr(),
        src.row_idx(),
    );
    // Validate containment first so a failure leaves `out` untouched.
    for j in 0..src.ncols() {
        let mut q = tp[j];
        for &i in &si[sp[j]..sp[j + 1]] {
            while q < tp[j + 1] && ti[q] < i {
                q += 1;
            }
            if q == tp[j + 1] || ti[q] != i {
                return Err("broadcast operand pattern is not contained in the target".into());
            }
        }
    }
    for j in 0..src.ncols() {
        let mut p = sp[j];
        for q in tp[j]..tp[j + 1] {
            out[q] = if p < sp[j + 1] && si[p] == ti[q] {
                p += 1;
                copy(&vals[p - 1])
            } else {
                zero()
            };
        }
    }
    Ok(())
}

#[julia]
impl SparseOwner {
    /// Whether `self` and `other` (any scalar types) store the same physical
    /// pattern in the same format.
    #[julia]
    pub fn sparse_same_structure(&self, other: &SparseOwner) -> Result<bool, String> {
        guarded(|| {
            let a = each!(self.data()?, x => physical(x));
            let b = each!(other.data()?, x => physical(x));
            Ok(same(a, b))
        })
    }
    /// Zero-valued owner of scalar `tag` whose pattern is the union of the
    /// patterns of `a` and `b` (same shape and storage format).
    #[julia]
    pub fn sparse_union_from(
        &mut self,
        a: &SparseOwner,
        b: &SparseOwner,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let (acsr, x) = each!(a.data()?, s => physical(s));
            let (bcsr, y) = each!(b.data()?, s => physical(s));
            if acsr != bcsr || x.nrows() != y.nrows() || x.ncols() != y.ncols() {
                return Err("broadcast operands must share shape and storage format".into());
            }
            let s = faer::sparse::ops::union_symbolic(x, y).map_err(|e| format!("{e:?}"))?;
            self.data = Some(match tag {
                0 => SparseData::F32(zeros(s, acsr)),
                1 => SparseData::F64(zeros(s, acsr)),
                2 => SparseData::C32(zeros(s, acsr)),
                3 => SparseData::C64(zeros(s, acsr)),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
    /// Write `self`'s stored values aligned on `target`'s pattern into `out`
    /// (`len` values of `self`'s scalar type); `target` must contain `self`.
    #[julia]
    pub fn sparse_aligned_values(
        &self,
        target: &SparseOwner,
        out: *mut u8,
        len: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let (tcsr, t) = each!(target.data()?, s => physical(s));
            fn run<T: ComplexField>(
                a: &Storage<T>,
                tcsr: bool,
                t: SymbolicSparseColMatRef<'_, usize>,
                out: *mut u8,
                len: usize,
            ) -> Result<(), String> {
                let (csr, s) = physical(a);
                if csr != tcsr || s.nrows() != t.nrows() || s.ncols() != t.ncols() {
                    return Err("broadcast operands must share shape and storage format".into());
                }
                if len != t.row_idx().len() {
                    return Err("aligned value buffer length mismatch".into());
                }
                let vals = match a {
                    Storage::Csc(a) => a.val(),
                    Storage::Csr(a) => a.val(),
                };
                // SAFETY: Julia passes `len` writable values of `T`.
                let out: &mut [T] = if len == 0 {
                    &mut []
                } else {
                    unsafe { std::slice::from_raw_parts_mut(out.cast::<T>(), len) }
                };
                align(t, s, vals, out)
            }
            each!(self.data()?, a => run(a, tcsr, t, out, len))
        })
    }
    /// Own the transpose of `p`'s pattern (the other storage format, same
    /// physical arrays) with a caller value buffer of scalar `tag`.
    #[julia]
    pub fn sparse_from_values_transposed(
        &mut self,
        p: &SparseOwner,
        val: *const u8,
        len: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            unsafe fn attach<U: ComplexField + Copy>(
                csr: bool,
                s: SymbolicSparseColMatRef<'_, usize>,
                val: *const u8,
                len: usize,
            ) -> Result<Storage<U>, String> {
                if s.row_idx().len() != len {
                    return Err("stored-value buffer length mismatch".into());
                }
                let vals = if len == 0 {
                    vec![]
                } else {
                    std::slice::from_raw_parts(val.cast::<U>(), len).to_vec()
                };
                let c = SparseColMat::new(s.to_owned().map_err(|e| format!("{e:?}"))?, vals);
                // `c` is the column-major view of `p`; the transpose of `p`
                // reads the same arrays in the other format.
                Ok(if csr {
                    // p is CSR, c = pᵀ in CSC: pᵀ is exactly c.
                    Storage::Csc(c)
                } else {
                    // p is CSC, c = p: pᵀ is c read as CSR.
                    Storage::Csr(c.into_transpose())
                })
            }
            let (csr, s) = each!(p.data()?, x => physical(x));
            self.data = Some(unsafe {
                match tag {
                    0 => SparseData::F32(attach(csr, s, val, len)?),
                    1 => SparseData::F64(attach(csr, s, val, len)?),
                    2 => SparseData::C32(attach(csr, s, val, len)?),
                    3 => SparseData::C64(attach(csr, s, val, len)?),
                    _ => return Err("unsupported sparse scalar".into()),
                }
            });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::Triplet;

    fn owner(m: usize, n: usize, t: &[(usize, usize, f64)], csr: bool) -> SparseOwner {
        let t: Vec<_> = t.iter().map(|&(i, j, v)| Triplet::new(i, j, v)).collect();
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(m, n, &t).unwrap();
        SparseOwner {
            data: Some(SparseData::F64(if csr {
                Storage::Csr(a.to_row_major().unwrap())
            } else {
                Storage::Csc(a)
            })),
        }
    }

    #[test]
    fn union_and_alignment_follow_the_physical_pattern() {
        for csr in [false, true] {
            let a = owner(3, 3, &[(0, 0, 1.0), (2, 1, 2.0)], csr);
            let b = owner(3, 3, &[(1, 0, 3.0), (2, 1, 4.0), (0, 2, 5.0)], csr);
            assert!(!a.sparse_same_structure(&b).unwrap());
            assert!(a.sparse_same_structure(&a).unwrap());
            let mut u = SparseOwner::new();
            u.sparse_union_from(&a, &b, 1).unwrap();
            let k = u.sparse_nnz().unwrap();
            assert_eq!(k, 4);
            let mut va = vec![f64::NAN; k];
            let mut vb = vec![f64::NAN; k];
            a.sparse_aligned_values(&u, va.as_mut_ptr().cast(), k)
                .unwrap();
            b.sparse_aligned_values(&u, vb.as_mut_ptr().cast(), k)
                .unwrap();
            let sum: Vec<f64> = va.iter().zip(&vb).map(|(x, y)| x + y).collect();
            let mut c = SparseOwner::new();
            c.sparse_from_values(&u, sum.as_ptr().cast(), k, 1).unwrap();
            let mut d = vec![0.0; 9];
            c.sparse_dense_values_for_test(&mut d);
            assert_eq!(d, vec![1.0, 3.0, 0.0, 0.0, 0.0, 6.0, 5.0, 0.0, 0.0]);
            // `b` is not contained in `a`: rejected without writing.
            let mut bad = vec![7.0; 2];
            assert!(b
                .sparse_aligned_values(&a, bad.as_mut_ptr().cast(), 2)
                .is_err());
            assert_eq!(bad, vec![7.0; 2]);
            let other = owner(3, 3, &[(0, 0, 1.0)], !csr);
            assert!(u.sparse_union_from(&a, &other, 1).is_err());
        }
    }

    #[test]
    fn transposed_values_flip_the_format() {
        let a = owner(2, 3, &[(0, 1, 1.0), (1, 2, 2.0)], false);
        let v = [10.0, 20.0];
        let mut t = SparseOwner::new();
        t.sparse_from_values_transposed(&a, v.as_ptr().cast(), 2, 1)
            .unwrap();
        let mut d = vec![0.0; 6];
        t.sparse_dense_values_for_test(&mut d);
        // 3×2 column-major dense of aᵀ with the new values.
        assert_eq!(d, vec![0.0, 10.0, 0.0, 0.0, 0.0, 20.0]);
        let mut back = SparseOwner::new();
        back.sparse_from_values_transposed(&t, v.as_ptr().cast(), 2, 1)
            .unwrap();
        let mut e = vec![0.0; 6];
        back.sparse_dense_values_for_test(&mut e);
        assert_eq!(e, vec![0.0, 0.0, 10.0, 0.0, 0.0, 20.0]);
    }

    impl SparseOwner {
        fn sparse_dense_values_for_test(&self, out: &mut [f64]) {
            let Some(SparseData::F64(s)) = &self.data else {
                unreachable!()
            };
            let d = match s {
                Storage::Csc(a) => a.to_dense(),
                Storage::Csr(a) => a.to_dense(),
            };
            for j in 0..d.ncols() {
                for i in 0..d.nrows() {
                    out[i + j * d.nrows()] = d[(i, j)];
                }
            }
        }
    }
}
