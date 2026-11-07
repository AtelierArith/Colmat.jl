//! Owned faer permutations and validated gather/scatter operations.
//!
//! Dense in-place updates borrow the selected owner range (a whole owner or a
//! strided/reversed view) as one faer `MatMut` for a single call and run
//! faer's in-place kernels on it, using a caller-owned `OwnedScratch` when one
//! is supplied.
use crate::collections::{VectorData, VectorOwner};
use crate::guarded;
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{borrowed, source, Borrowed, Spec};
use crate::workspace::OwnedScratch;
use crate::OwnedMatrix;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::perm::{self, Perm, PermRef};
use faer::traits::ComplexField;
use faer::{Mat, MatMut, MatRef};
use rustcall_julia_macros::julia;

fn checked(forward: Vec<usize>) -> Result<Perm<usize>, String> {
    let n = forward.len();
    let mut inverse = vec![usize::MAX; n];
    for (i, &j) in forward.iter().enumerate() {
        if j >= n || inverse[j] != usize::MAX {
            return Err("indices must be a permutation of 1:length(p)".into());
        }
        inverse[j] = i;
    }
    Ok(Perm::new_checked(
        forward.into_boxed_slice(),
        inverse.into_boxed_slice(),
        n,
    ))
}
#[julia]
pub struct OwnedPermutation {
    pub(crate) data: Option<Perm<usize>>,
}
impl OwnedPermutation {
    pub(crate) fn data(&self) -> Result<&Perm<usize>, String> {
        self.data
            .as_ref()
            .ok_or("permutation is not initialized".into())
    }
}
#[julia]
impl OwnedPermutation {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn permutation_load(&mut self, p: *const usize, n: usize) -> Result<(), String> {
        guarded(|| {
            let p = if n == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(p, n) }
            };
            let v = p
                .iter()
                .map(|&i| {
                    i.checked_sub(1)
                        .ok_or_else(|| "permutation indices must be positive".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.data = Some(checked(v)?);
            Ok(())
        })
    }
    #[julia]
    pub fn permutation_indices(
        &self,
        out: *mut usize,
        n: usize,
        inverse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let p = self.data()?;
            if p.len() != n {
                return Err("permutation length mismatch".into());
            }
            let a = if inverse { p.arrays().1 } else { p.arrays().0 };
            for (i, &v) in a.iter().enumerate() {
                unsafe { out.add(i).write(v + 1) }
            }
            Ok(())
        })
    }
    #[julia]
    pub fn permutation_element(&self, i: usize) -> Result<usize, String> {
        guarded(|| {
            self.data()?
                .arrays()
                .0
                .get(i)
                .map(|&j| j + 1)
                .ok_or("permutation index out of bounds".into())
        })
    }
    #[julia]
    pub fn permutation_clone(&mut self, p: &OwnedPermutation, inverse: bool) -> Result<(), String> {
        guarded(|| {
            let p = p.data()?;
            let q = p.clone();
            self.data = Some(if inverse { q.into_inverse() } else { q });
            Ok(())
        })
    }
    #[julia]
    pub fn permutation_compose(
        &mut self,
        left: &OwnedPermutation,
        right: &OwnedPermutation,
    ) -> Result<(), String> {
        guarded(|| {
            let p = left.data()?;
            let q = right.data()?;
            if p.len() != q.len() {
                return Err("permutation lengths must match".into());
            }
            self.data = Some(checked(
                q.arrays().0.iter().map(|&i| p.arrays().0[i]).collect(),
            )?);
            Ok(())
        })
    }
}
fn check_shape<T>(a: MatRef<'_, T>, p: PermRef<'_, usize>, columns: bool) -> Result<(), String> {
    if p.len() != if columns { a.ncols() } else { a.nrows() } {
        return Err("permutation size does not match matrix axis".into());
    }
    Ok(())
}
fn permuted<T: ComplexField>(
    a: MatRef<'_, T>,
    p: PermRef<'_, usize>,
    columns: bool,
) -> Result<Mat<T>, String> {
    check_shape(a, p, columns)?;
    let mut out = Mat::zeros(a.nrows(), a.ncols());
    if columns {
        perm::permute_cols(out.as_mut(), a, p)
    } else {
        perm::permute_rows(out.as_mut(), a, p)
    }
    Ok(out)
}
/// Borrow an injective view selection mutably for one call. Repeated indices
/// (a zero step over more than one element) would alias a `MatMut`, so they
/// are rejected; Julia stages those updates instead.
fn borrowed_mut<T>(a: MatMut<'_, T>, s: Spec) -> Result<MatMut<'_, T>, String> {
    if (s.rs == 0 && s.m > 1) || (s.cs == 0 && s.n > 1) {
        return Err("repeated view indices cannot be borrowed mutably".into());
    }
    let (offset, m, n, rs, cs) = {
        let view = borrowed(a.as_ref(), s)?;
        // SAFETY: `borrowed` offsets the owner's pointer in its allocation.
        let offset = unsafe { view.as_ptr().offset_from(a.as_ptr()) };
        let (rs, cs) = (view.row_stride(), view.col_stride());
        (offset, view.nrows(), view.ncols(), rs, cs)
    };
    // SAFETY: `borrowed` validated every selected index against the owner and
    // the nonzero steps make the selection injective. The pointer derives from
    // the exclusive owner borrow and does not outlive this call.
    Ok(unsafe { MatMut::from_raw_parts_mut(a.as_ptr_mut().offset(offset), m, n, rs, cs) })
}
fn requirement<T: ComplexField>(m: usize, n: usize, columns: bool) -> StackReq {
    if columns {
        perm::permute_cols_in_place_scratch::<usize, T>(m, n)
    } else {
        perm::permute_rows_in_place_scratch::<usize, T>(m, n)
    }
}
/// Exact faer workspace for an in-place row or column permutation of an
/// `m x n` matrix of the tagged scalar.
#[julia]
pub fn permutation_requirements(
    m: usize,
    n: usize,
    tag: i32,
    columns: bool,
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| {
        let req = match tag {
            0 => requirement::<f32>(m, n, columns),
            1 => requirement::<f64>(m, n, columns),
            2 => requirement::<faer::c32>(m, n, columns),
            3 => requirement::<faer::c64>(m, n, columns),
            _ => return Err("invalid permutation scalar".into()),
        };
        unsafe {
            out.write(req.size_bytes());
            out.add(1).write(req.align_bytes());
        }
        Ok(())
    })
}
/// Permute the selected rows or columns in place. Without caller scratch a
/// call-local buffer of the same faer requirement is allocated.
fn apply_view<T: ComplexField>(
    a: MatMut<'_, T>,
    s: Spec,
    p: PermRef<'_, usize>,
    columns: bool,
    inverse: bool,
    scratch: Option<&mut OwnedScratch>,
) -> Result<(), String> {
    let a = borrowed_mut(a, s)?;
    check_shape(a.as_ref(), p, columns)?;
    let req = requirement::<T>(a.nrows(), a.ncols(), columns);
    let mut memory = None;
    let stack =
        match scratch {
            Some(work) => work.stack(req)?,
            None => MemStack::new(memory.insert(
                MemBuffer::try_new(req).map_err(|e| format!("permutation scratch: {e:?}"))?,
            )),
        };
    let p = if inverse { p.inverse() } else { p };
    if columns {
        perm::permute_cols_in_place(a, p, stack)
    } else {
        perm::permute_rows_in_place(a, p, stack)
    }
    Ok(())
}
fn swap<T>(a: MatMut<'_, T>, i: usize, j: usize, columns: bool) -> Result<(), String> {
    let n = if columns { a.ncols() } else { a.nrows() };
    if i >= n || j >= n {
        return Err("swap index out of bounds".into());
    }
    if columns {
        perm::swap_cols_idx(a, i, j)
    } else {
        perm::swap_rows_idx(a, i, j)
    }
    Ok(())
}
fn permutation(p: &OwnedPermutation) -> Result<PermRef<'_, usize>, String> {
    Ok(p.data()?.as_ref())
}
/// Gather a borrowed source selection into a new owner.
// SAFETY: Julia preserves the source owner and its nine-word descriptor.
unsafe fn permuted_source(
    src: *const u8,
    desc: *const isize,
    p: &OwnedPermutation,
    columns: bool,
) -> Result<MatrixData, String> {
    let s = Spec::read(desc)?;
    if s.conjugate {
        return Err("conjugated permutation sources are not supported".into());
    }
    let p = permutation(p)?;
    Ok(match source(src, s)? {
        Borrowed::F32(a) => MatrixData::F32(permuted(a, p, columns)?),
        Borrowed::F64(a) => MatrixData::F64(permuted(a, p, columns)?),
        Borrowed::C32(a) => MatrixData::C32(permuted(a, p, columns)?),
        Borrowed::C64(a) => MatrixData::C64(permuted(a, p, columns)?),
    })
}
macro_rules! mapped {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            MatrixData::F32($a) => MatrixData::F32($body),
            MatrixData::F64($a) => MatrixData::F64($body),
            MatrixData::C32($a) => MatrixData::C32($body),
            MatrixData::C64($a) => MatrixData::C64($body),
        }
    };
}
macro_rules! dispatch {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            MatrixData::F32($a) => $body,
            MatrixData::F64($a) => $body,
            MatrixData::C32($a) => $body,
            MatrixData::C64($a) => $body,
        }
    };
}
macro_rules! vector_dispatch {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            VectorData::F32($a) => $body,
            VectorData::F64($a) => $body,
            VectorData::C32($a) => $body,
            VectorData::C64($a) => $body,
        }
    };
}
// Native entry points shared by every dense owner: `desc` selects the whole
// owner or a strided/reversed view of it, borrowed for this call only.
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn permute_view_storage(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            apply_view(
                self.matrix.as_mut(),
                s,
                permutation(p)?,
                columns,
                inverse,
                None,
            )
        })
    }
    #[julia]
    pub fn permute_view_workspace(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let p = permutation(p)?;
            apply_view(self.matrix.as_mut(), s, p, columns, inverse, Some(scratch))
        })
    }
    #[julia]
    pub fn swap_view_storage(
        &mut self,
        desc: *const isize,
        i: usize,
        j: usize,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            swap(borrowed_mut(self.matrix.as_mut(), s)?, i, j, columns)
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn permute_view_storage(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let p = permutation(p)?;
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(data, a => apply_view(a.as_mut(), s, p, columns, inverse, None))
        })
    }
    #[julia]
    pub fn permute_view_workspace(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let p = permutation(p)?;
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(data, a => apply_view(a.as_mut(), s, p, columns, inverse, Some(scratch)))
        })
    }
    #[julia]
    pub fn swap_view_storage(
        &mut self,
        desc: *const isize,
        i: usize,
        j: usize,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let data = self.data.as_mut().ok_or("matrix is not initialized")?;
            dispatch!(data, a => swap(borrowed_mut(a.as_mut(), s)?, i, j, columns))
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn permute_view_storage(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let p = permutation(p)?;
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            vector_dispatch!(data, a => apply_view(a.as_mat_mut(), s, p, columns, inverse, None))
        })
    }
    #[julia]
    pub fn permute_view_workspace(
        &mut self,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
        inverse: bool,
        scratch: &mut OwnedScratch,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let p = permutation(p)?;
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            vector_dispatch!(data, a => apply_view(a.as_mat_mut(), s, p, columns, inverse, Some(scratch)))
        })
    }
    #[julia]
    pub fn swap_view_storage(
        &mut self,
        desc: *const isize,
        i: usize,
        j: usize,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            vector_dispatch!(data, a => swap(borrowed_mut(a.as_mat_mut(), s)?, i, j, columns))
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn permute_from(
        &mut self,
        source: &OwnedMatrix,
        p: &OwnedPermutation,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = permuted(source.matrix.as_ref(), permutation(p)?, columns)?;
            Ok(())
        })
    }
    #[julia]
    pub fn permute_from_view(
        &mut self,
        src: *const u8,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = match unsafe { permuted_source(src, desc, p, columns)? } {
                MatrixData::F64(a) => a,
                _ => return Err("Float64 output requires a Float64 view".into()),
            };
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn permute_from(
        &mut self,
        source: &TypedMatrix,
        p: &OwnedPermutation,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            let p = permutation(p)?;
            self.data = Some(mapped!(a,x=>permuted(x.as_ref(),p,columns)?));
            Ok(())
        })
    }
    #[julia]
    pub fn permute_from_view(
        &mut self,
        src: *const u8,
        desc: *const isize,
        p: &OwnedPermutation,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { permuted_source(src, desc, p, columns)? });
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn spec(r: isize, rs: isize, m: usize, c: isize, cs: isize, n: usize) -> Spec {
        let words = [0, r, rs, m as isize, c, cs, n as isize, 0, 0];
        // SAFETY: a nine-word descriptor of an owned (kind 0) matrix.
        unsafe { Spec::read(words.as_ptr()) }.unwrap()
    }
    #[test]
    fn reversed_view_permutes_in_caller_scratch_without_allocating() {
        let mut a = Mat::<f64>::from_fn(6, 5, |i, j| (10 * i + j) as f64);
        let expected_source = a.clone();
        // Rows 5,3,1 and columns 4,2,0 of the owner.
        let s = spec(5, -2, 3, 4, -2, 3);
        let p = checked(vec![2, 0, 1]).unwrap();
        let req = requirement::<f64>(3, 3, false);
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        for inverse in [false, true] {
            let count = crate::allocation_probe::count(|| {
                apply_view(
                    a.as_mut(),
                    s,
                    p.as_ref(),
                    false,
                    inverse,
                    Some(&mut scratch),
                )
                .unwrap()
            });
            assert_eq!(count, 0);
        }
        // A permutation followed by its inverse restores the owner.
        assert_eq!(a, expected_source);
        apply_view(a.as_mut(), s, p.as_ref(), false, false, Some(&mut scratch)).unwrap();
        for (k, &src) in [2usize, 0, 1].iter().enumerate() {
            for j in 0..5 {
                // Only the selected columns 4, 2, 0 move.
                let from = if j % 2 == 0 { 5 - 2 * src } else { 5 - 2 * k };
                assert_eq!(a[(5 - 2 * k, j)], expected_source[(from, j)]);
            }
        }
        // Unselected rows are untouched.
        for i in [0usize, 2, 4] {
            assert_eq!(a.row(i), expected_source.row(i));
        }
    }
    #[test]
    fn insufficient_scratch_and_repeated_views_fail_before_writing() {
        let mut a = Mat::<f64>::from_fn(4, 4, |i, j| (i + 4 * j) as f64);
        let before = a.clone();
        let p = checked(vec![1, 0, 3, 2]).unwrap();
        let mut empty = OwnedScratch::new();
        empty.scratch_reserve(0, 1).unwrap();
        assert!(apply_view(
            a.as_mut(),
            spec(0, 1, 4, 0, 1, 4),
            p.as_ref(),
            true,
            false,
            Some(&mut empty)
        )
        .is_err());
        let q = checked(vec![2, 0, 1]).unwrap();
        assert!(apply_view(
            a.as_mut(),
            spec(1, 0, 3, 0, 1, 4),
            q.as_ref(),
            false,
            false,
            None
        )
        .is_err());
        assert!(apply_view(
            a.as_mut(),
            spec(0, 1, 4, 0, 1, 4),
            q.as_ref(),
            false,
            false,
            None
        )
        .is_err());
        assert!(swap(
            borrowed_mut(a.as_mut(), spec(0, 1, 4, 0, 1, 4)).unwrap(),
            0,
            4,
            false
        )
        .is_err());
        assert_eq!(a, before);
    }
}
