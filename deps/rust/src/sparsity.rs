//! Numeric-value-free sparse patterns, independent of numeric owners.
use crate::guarded;
use crate::sparse::{SparseData, SparseOwner, Storage};
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, SymbolicSparseRowMat};
use faer::traits::ComplexField;
use rustcall_julia_macros::julia;
#[derive(Clone)]
pub(crate) enum Pattern {
    Csc(SymbolicSparseColMat<usize>),
    Csr(SymbolicSparseRowMat<usize>),
}
impl Pattern {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Csc(a) => (a.nrows(), a.ncols()),
            Self::Csr(a) => (a.nrows(), a.ncols()),
        }
    }
    fn pointers(&self) -> &[usize] {
        match self {
            Self::Csc(a) => a.col_ptr(),
            Self::Csr(a) => a.row_ptr(),
        }
    }
    fn indices(&self) -> &[usize] {
        match self {
            Self::Csc(a) => a.row_idx(),
            Self::Csr(a) => a.col_idx(),
        }
    }
    pub(crate) fn csc(&self) -> Result<SymbolicSparseColMat<usize>, String> {
        match self {
            Self::Csc(a) => Ok(a.clone()),
            Self::Csr(a) => a.to_col_major().map_err(|e| format!("{e:?}")),
        }
    }
    fn row_major(&self) -> Result<SymbolicSparseRowMat<usize>, String> {
        match self {
            Self::Csr(a) => Ok(a.clone()),
            Self::Csc(a) => a.to_row_major().map_err(|e| format!("{e:?}")),
        }
    }
    fn converted(&self, csr: bool) -> Result<Self, String> {
        if csr {
            Ok(Self::Csr(self.row_major()?))
        } else {
            Ok(Self::Csc(self.csc()?))
        }
    }
    fn transpose(&self) -> Self {
        match self {
            Self::Csc(a) => Self::Csr(a.clone().into_transpose()),
            Self::Csr(a) => Self::Csc(a.clone().into_transpose()),
        }
    }
    fn contains(&self, i: usize, j: usize) -> Result<bool, String> {
        let (m, n) = self.shape();
        if i >= m || j >= n {
            return Err("pattern index out of bounds".into());
        }
        let (outer, inner) = if matches!(self, Self::Csr(_)) {
            (i, j)
        } else {
            (j, i)
        };
        Ok(
            self.indices()[self.pointers()[outer]..self.pointers()[outer + 1]]
                .binary_search(&inner)
                .is_ok(),
        )
    }
}
fn pattern<T: ComplexField>(a: &Storage<T>) -> Result<Pattern, String> {
    match a {
        Storage::Csc(a) => Ok(Pattern::Csc(
            a.symbolic().to_owned().map_err(|e| format!("{e:?}"))?,
        )),
        Storage::Csr(a) => Ok(Pattern::Csr(
            a.symbolic().to_owned().map_err(|e| format!("{e:?}"))?,
        )),
    }
}
unsafe fn numeric<T: ComplexField + Copy>(
    p: &Pattern,
    values: *const u8,
    k: usize,
    csr: bool,
) -> Result<Storage<T>, String> {
    if k != p.indices().len() {
        return Err("pattern and numeric value lengths must match".into());
    }
    let values = if k == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(values.cast::<T>(), k).to_vec()
    };
    let a = match p {
        Pattern::Csc(s) => Storage::Csc(SparseColMat::new(s.clone(), values)),
        Pattern::Csr(s) => Storage::Csr(SparseRowMat::new(s.clone(), values)),
    };
    a.converted(csr)
}
#[julia]
pub struct SparsityOwner {
    pub(crate) data: Option<Pattern>,
}
impl SparsityOwner {
    pub(crate) fn data(&self) -> Result<&Pattern, String> {
        self.data
            .as_ref()
            .ok_or("sparsity pattern is not initialized".into())
    }
}
#[julia]
impl SparsityOwner {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn sparsity_from_sparse(&mut self, source: *const u8) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match unsafe { &*source.cast::<SparseOwner>() }.data()? {
                SparseData::F32(a) => pattern(a)?,
                SparseData::F64(a) => pattern(a)?,
                SparseData::C32(a) => pattern(a)?,
                SparseData::C64(a) => pattern(a)?,
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparsity_clone_from(
        &mut self,
        source: *const u8,
        csr: bool,
        transpose: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let p = unsafe { &*source.cast::<SparsityOwner>() }.data()?;
            self.data = Some(if transpose {
                p.transpose().converted(csr)?
            } else {
                p.converted(csr)?
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparsity_nnz(&self) -> Result<usize, String> {
        guarded(|| Ok(self.data()?.indices().len()))
    }
    #[julia]
    pub fn sparsity_element(&self, i: usize, j: usize) -> Result<bool, String> {
        guarded(|| self.data()?.contains(i, j))
    }
    #[julia]
    pub fn sparsity_export(
        &self,
        p: *mut usize,
        i: *mut usize,
        pn: usize,
        k: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            if a.pointers().len() != pn || a.indices().len() != k {
                return Err("sparsity output buffer sizes must match".into());
            }
            for (j, &v) in a.pointers().iter().enumerate() {
                unsafe { p.add(j).write(v + 1) }
            }
            for (j, &v) in a.indices().iter().enumerate() {
                unsafe { i.add(j).write(v + 1) }
            }
            Ok(())
        })
    }
    #[julia]
    pub fn sparsity_union_from(
        &mut self,
        left: *const u8,
        right: *const u8,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = unsafe { &*left.cast::<SparsityOwner>() }.data()?;
            let b = unsafe { &*right.cast::<SparsityOwner>() }.data()?;
            if a.shape() != b.shape() {
                return Err("pattern union shapes must match".into());
            }
            self.data = Some(if csr {
                let a = a.row_major()?;
                let b = b.row_major()?;
                Pattern::Csr(
                    faer::sparse::ops::union_symbolic(
                        a.as_ref().transpose(),
                        b.as_ref().transpose(),
                    )
                    .map_err(|e| format!("{e:?}"))?
                    .into_transpose(),
                )
            } else {
                let a = a.csc()?;
                let b = b.csc()?;
                Pattern::Csc(
                    faer::sparse::ops::union_symbolic(a.as_ref(), b.as_ref())
                        .map_err(|e| format!("{e:?}"))?,
                )
            });
            Ok(())
        })
    }
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_from_pattern(
        &mut self,
        p: *const u8,
        values: *const u8,
        k: usize,
        tag: i32,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let p = (&*p.cast::<SparsityOwner>()).data()?;
            self.data = Some(match tag {
                0 => SparseData::F32(numeric(p, values, k, csr)?),
                1 => SparseData::F64(numeric(p, values, k, csr)?),
                2 => SparseData::C32(numeric(p, values, k, csr)?),
                3 => SparseData::C64(numeric(p, values, k, csr)?),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
}

#[julia]
impl SparsityOwner {
    #[julia]
    pub fn sparsity_permute(
        &mut self,
        source: *const u8,
        permutation: *const u8,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = unsafe { &*source.cast::<Self>() }.data()?;
            let p =
                unsafe { &*permutation.cast::<crate::permutations::OwnedPermutation>() }.data()?;
            let (m, n) = a.shape();
            if p.len() != if columns { n } else { m } {
                return Err("pattern permutation length mismatch".into());
            }
            let inverse = p.arrays().1;
            let csr = matches!(a, Pattern::Csr(_));
            let mut entries = Vec::with_capacity(a.indices().len());
            for outer in 0..a.pointers().len() - 1 {
                for &inner in &a.indices()[a.pointers()[outer]..a.pointers()[outer + 1]] {
                    let (mut row, mut col) = if csr { (outer, inner) } else { (inner, outer) };
                    if columns {
                        col = inverse[col];
                    } else {
                        row = inverse[row];
                    }
                    entries.push(faer::sparse::Pair { row, col });
                }
            }
            self.data = Some(if csr {
                Pattern::Csr(
                    SymbolicSparseRowMat::try_new_from_indices(m, n, &entries)
                        .map_err(|e| format!("{e:?}"))?
                        .0,
                )
            } else {
                Pattern::Csc(
                    SymbolicSparseColMat::try_new_from_indices(m, n, &entries)
                        .map_err(|e| format!("{e:?}"))?
                        .0,
                )
            });
            Ok(())
        })
    }
}

#[julia]
impl SparsityOwner {
    #[julia]
    pub fn sparsity_selfadjoint(&mut self, source: *const u8, lower: bool) -> Result<(), String> {
        guarded(|| {
            let a = unsafe { &*source.cast::<Self>() }.data()?;
            let (m, n) = a.shape();
            if m != n {
                return Err("self-adjoint pattern requires square dimensions".into());
            }
            let csr = matches!(a, Pattern::Csr(_));
            let mut entries = Vec::new();
            for outer in 0..a.pointers().len() - 1 {
                for &inner in &a.indices()[a.pointers()[outer]..a.pointers()[outer + 1]] {
                    let (row, col) = if csr { (outer, inner) } else { (inner, outer) };
                    if if lower { row < col } else { row > col } {
                        continue;
                    }
                    entries.push(faer::sparse::Pair { row, col });
                    if row != col {
                        entries.push(faer::sparse::Pair { row: col, col: row });
                    }
                }
            }
            self.data = Some(if csr {
                Pattern::Csr(
                    SymbolicSparseRowMat::try_new_from_indices(m, n, &entries)
                        .map_err(|e| format!("{e:?}"))?
                        .0,
                )
            } else {
                Pattern::Csc(
                    SymbolicSparseColMat::try_new_from_indices(m, n, &entries)
                        .map_err(|e| format!("{e:?}"))?
                        .0,
                )
            });
            Ok(())
        })
    }
}
