//! Owned faer permutations and validated gather/scatter operations.
use crate::collections::{VectorData, VectorOwner};
use crate::guarded;
use crate::typed::{MatrixData, TypedMatrix};
use crate::OwnedMatrix;
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
fn apply<T: ComplexField>(
    a: MatMut<'_, T>,
    p: PermRef<'_, usize>,
    columns: bool,
) -> Result<(), String> {
    check_shape(a.as_ref(), p, columns)?;
    let req = if columns {
        perm::permute_cols_in_place_scratch::<usize, T>(a.nrows(), a.ncols())
    } else {
        perm::permute_rows_in_place_scratch::<usize, T>(a.nrows(), a.ncols())
    };
    let mut memory = faer::dyn_stack::MemBuffer::new(req);
    let stack = faer::dyn_stack::MemStack::new(&mut memory);
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
    pub fn permute_storage(&mut self, p: &OwnedPermutation, columns: bool) -> Result<(), String> {
        guarded(|| apply(self.matrix.as_mut(), permutation(p)?, columns))
    }
    #[julia]
    pub fn swap_storage(&mut self, i: usize, j: usize, columns: bool) -> Result<(), String> {
        guarded(|| swap(self.matrix.as_mut(), i, j, columns))
    }
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
    pub fn permute_storage(&mut self, p: &OwnedPermutation, columns: bool) -> Result<(), String> {
        guarded(|| {
            let p = permutation(p)?;
            dispatch!(self.data.as_mut().ok_or("matrix is not initialized")?,a=>apply(a.as_mut(),p,columns))
        })
    }
    #[julia]
    pub fn swap_storage(&mut self, i: usize, j: usize, columns: bool) -> Result<(), String> {
        guarded(
            || dispatch!(self.data.as_mut().ok_or("matrix is not initialized")?,a=>swap(a.as_mut(),i,j,columns)),
        )
    }
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
#[julia]
impl VectorOwner {
    #[julia]
    pub fn permute_storage(&mut self, p: &OwnedPermutation, columns: bool) -> Result<(), String> {
        guarded(|| {
            let p = permutation(p)?;
            vector_dispatch!(self.data.as_mut().ok_or("collection is not initialized")?,a=>apply(a.as_mat_mut(),p,columns))
        })
    }
    #[julia]
    pub fn swap_storage(&mut self, i: usize, j: usize, columns: bool) -> Result<(), String> {
        guarded(
            || vector_dispatch!(self.data.as_mut().ok_or("collection is not initialized")?,a=>swap(a.as_mat_mut(),i,j,columns)),
        )
    }
}
