//! Native sparse owners; dense conversion is explicit.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::sparse::{SparseColMat, SparseRowMat, SymbolicSparseColMat, Triplet};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, ColRef, Mat, MatRef, Par};
use rustcall_julia_macros::julia;
#[derive(Clone)]
pub(crate) enum Storage<T> {
    Csc(SparseColMat<usize, T>),
    Csr(SparseRowMat<usize, T>),
}
impl<T: ComplexField> Storage<T> {
    pub(crate) fn shape(&self) -> (usize, usize) {
        match self {
            Self::Csc(a) => (a.nrows(), a.ncols()),
            Self::Csr(a) => (a.nrows(), a.ncols()),
        }
    }
    fn csr(&self) -> bool {
        matches!(self, Self::Csr(_))
    }
    fn values(&self) -> &[T] {
        match self {
            Self::Csc(a) => a.val(),
            Self::Csr(a) => a.val(),
        }
    }
    fn values_mut(&mut self) -> &mut [T] {
        match self {
            Self::Csc(a) => a.val_mut(),
            Self::Csr(a) => a.val_mut(),
        }
    }
    fn pointers(&self) -> &[usize] {
        match self {
            Self::Csc(a) => a.symbolic().col_ptr(),
            Self::Csr(a) => a.symbolic().row_ptr(),
        }
    }
    fn indices(&self) -> &[usize] {
        match self {
            Self::Csc(a) => a.symbolic().row_idx(),
            Self::Csr(a) => a.symbolic().col_idx(),
        }
    }
    pub(crate) fn csc(&self) -> Result<SparseColMat<usize, T>, String> {
        match self {
            Self::Csc(a) => Ok(a.clone()),
            Self::Csr(a) => a.to_col_major().map_err(|e| format!("{e:?}")),
        }
    }
    fn row_major(&self) -> Result<SparseRowMat<usize, T>, String> {
        match self {
            Self::Csr(a) => Ok(a.clone()),
            Self::Csc(a) => a.to_row_major().map_err(|e| format!("{e:?}")),
        }
    }
    pub(crate) fn converted(&self, csr: bool) -> Result<Self, String> {
        if csr {
            Ok(Self::Csr(self.row_major()?))
        } else {
            Ok(Self::Csc(self.csc()?))
        }
    }
    fn from_triplets(
        m: usize,
        n: usize,
        entries: &[Triplet<usize, usize, T>],
        csr: bool,
    ) -> Result<Self, String> {
        if csr {
            Ok(Self::Csr(
                SparseRowMat::try_new_from_triplets(m, n, entries).map_err(|e| format!("{e:?}"))?,
            ))
        } else {
            Ok(Self::Csc(
                SparseColMat::try_new_from_triplets(m, n, entries).map_err(|e| format!("{e:?}"))?,
            ))
        }
    }
    fn triplets(&self) -> Vec<Triplet<usize, usize, T>> {
        match self {
            Self::Csc(a) => a
                .as_ref()
                .triplet_iter()
                .map(|e| Triplet {
                    row: e.row,
                    col: e.col,
                    val: copy(e.val),
                })
                .collect(),
            Self::Csr(a) => a
                .as_ref()
                .triplet_iter()
                .map(|e| Triplet {
                    row: e.row,
                    col: e.col,
                    val: copy(e.val),
                })
                .collect(),
        }
    }
    fn get(&self, i: usize, j: usize) -> Result<T, String> {
        let (m, n) = self.shape();
        if i >= m || j >= n {
            return Err("sparse index out of bounds".into());
        }
        Ok(match self {
            Self::Csc(a) => a.get(i, j),
            Self::Csr(a) => a.get(i, j),
        }
        .map(copy)
        .unwrap_or_else(zero::<T>))
    }
    fn set(&mut self, i: usize, j: usize, value: T) -> Result<bool, String> {
        let (m, n) = self.shape();
        let csr = self.csr();
        if i >= m || j >= n {
            return Err("sparse index out of bounds".into());
        }
        if let Some(slot) = match self {
            Self::Csc(a) => a.get_mut(i, j),
            Self::Csr(a) => a.get_mut(i, j),
        } {
            *slot = value;
            return Ok(false);
        }
        if value == zero::<T>() {
            return Ok(false);
        }
        let mut entries = self.triplets();
        entries.push(Triplet {
            row: i,
            col: j,
            val: value,
        });
        *self = Self::from_triplets(m, n, &entries, csr)?;
        Ok(true)
    }
    fn dropzeros(&mut self) -> Result<bool, String> {
        let mut entries = self.triplets();
        let old = entries.len();
        entries.retain(|e| e.val != zero::<T>());
        if entries.len() == old {
            return Ok(false);
        }
        let (m, n) = self.shape();
        *self = Self::from_triplets(m, n, &entries, self.csr())?;
        Ok(true)
    }
    fn transpose(&self, conjugated: bool) -> Self {
        let mut out = match self {
            Self::Csc(a) => Self::Csr(a.clone().into_transpose()),
            Self::Csr(a) => Self::Csc(a.clone().into_transpose()),
        };
        if conjugated {
            for x in out.values_mut() {
                *x = conj(x);
            }
        }
        out
    }
    fn dense(&self) -> Mat<T> {
        match self {
            Self::Csc(a) => a.to_dense(),
            Self::Csr(a) => a.to_dense(),
        }
    }
}
#[derive(Clone)]
pub(crate) enum SparseData {
    F32(Storage<f32>),
    F64(Storage<f64>),
    C32(Storage<faer::c32>),
    C64(Storage<faer::c64>),
}
macro_rules! dispatch {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            SparseData::F32($a) => $body,
            SparseData::F64($a) => $body,
            SparseData::C32($a) => $body,
            SparseData::C64($a) => $body,
        }
    };
}
macro_rules! mapped {
    ($data:expr,$a:ident=>$body:expr) => {
        match $data {
            SparseData::F32($a) => SparseData::F32($body),
            SparseData::F64($a) => SparseData::F64($body),
            SparseData::C32($a) => SparseData::C32($body),
            SparseData::C64($a) => SparseData::C64($body),
        }
    };
}
fn arithmetic<T: ComplexField>(
    a: &Storage<T>,
    b: &Storage<T>,
    op: i32,
    csr: bool,
) -> Result<Storage<T>, String> {
    let ash = a.shape();
    let bsh = b.shape();
    if (op == 2 && ash.1 != bsh.0) || (op != 2 && ash != bsh) {
        return Err("sparse operation dimensions must match".into());
    }
    fn calculate<T: ComplexField>(
        a: faer::sparse::SparseColMatRef<'_, usize, T>,
        b: faer::sparse::SparseColMatRef<'_, usize, T>,
        op: i32,
    ) -> Result<SparseColMat<usize, T>, String> {
        match op {
            0 => faer::sparse::ops::add(a, b),
            1 => faer::sparse::ops::sub(a, b),
            2 => faer::sparse::linalg::matmul::sparse_sparse_matmul(a, b, one::<T>(), Par::Seq),
            _ => return Err("unknown sparse operation".into()),
        }
        .map_err(|e| format!("{e:?}"))
    }
    if csr {
        let a = a.row_major()?;
        let b = b.row_major()?;
        let c = if op == 2 {
            calculate(b.as_ref().transpose(), a.as_ref().transpose(), op)?
        } else {
            calculate(a.as_ref().transpose(), b.as_ref().transpose(), op)?
        };
        Ok(Storage::Csr(c.into_transpose()))
    } else {
        let a = a.csc()?;
        let b = b.csc()?;
        Ok(Storage::Csc(calculate(a.as_ref(), b.as_ref(), op)?))
    }
}
unsafe fn load_csc<T: ComplexField + Copy>(
    m: usize,
    n: usize,
    ptr: *const usize,
    idx: *const usize,
    val: *const u8,
    k: usize,
    csr: bool,
) -> Result<Storage<T>, String> {
    let p = std::slice::from_raw_parts(ptr, n.checked_add(1).ok_or("column pointer overflow")?)
        .to_vec();
    let rows = if k == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(idx, k).to_vec()
    };
    let vals = if k == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(val.cast::<T>(), k).to_vec()
    };
    if p[0] != 0 || p[n] != k || p.windows(2).any(|x| x[0] > x[1]) {
        return Err("invalid compressed pointers".into());
    }
    for j in 0..n {
        if p[j + 1] > k
            || rows[p[j]..p[j + 1]].iter().any(|&i| i >= m)
            || rows[p[j]..p[j + 1]].windows(2).any(|w| w[0] >= w[1])
        {
            return Err("CSC row indices must be in bounds, sorted and unique".into());
        }
    }
    let a = Storage::Csc(SparseColMat::new(
        SymbolicSparseColMat::new_checked(m, n, p, None, rows),
        vals,
    ));
    a.converted(csr)
}
unsafe fn load_triplets<T: ComplexField + Copy>(
    m: usize,
    n: usize,
    rows: *const usize,
    cols: *const usize,
    val: *const u8,
    k: usize,
    csr: bool,
) -> Result<Storage<T>, String> {
    let mut entries = Vec::with_capacity(k);
    for i in 0..k {
        let r = *rows.add(i);
        let c = *cols.add(i);
        if r >= m || c >= n {
            return Err("triplet index out of bounds".into());
        }
        entries.push(Triplet {
            row: r,
            col: c,
            val: *val.cast::<T>().add(i),
        });
    }
    Storage::from_triplets(m, n, &entries, csr)
}
unsafe fn export<T: ComplexField>(
    a: &Storage<T>,
    ptr: *mut usize,
    idx: *mut usize,
    val: *mut u8,
    pn: usize,
    k: usize,
) -> Result<(), String> {
    if a.pointers().len() != pn || a.values().len() != k {
        return Err("sparse output buffer size mismatch".into());
    }
    for (i, &x) in a.pointers().iter().enumerate() {
        ptr.add(i).write(x + 1);
    }
    for (i, &x) in a.indices().iter().enumerate() {
        idx.add(i).write(x + 1);
    }
    for (i, x) in a.values().iter().enumerate() {
        val.cast::<T>().add(i).write(copy(x));
    }
    Ok(())
}
fn sum_value<T: ComplexField>(a: &Storage<T>) -> T {
    a.values().iter().fold(zero::<T>(), |s, x| add(&s, x))
}
unsafe fn values_buffer<T: ComplexField>(
    a: &Storage<T>,
    out: *mut u8,
    len: usize,
) -> Result<(), String> {
    if a.values().len() != len {
        return Err("stored-value buffer length mismatch".into());
    }
    for (i, x) in a.values().iter().enumerate() {
        out.cast::<T>().add(i).write(copy(x));
    }
    Ok(())
}
unsafe fn write_scalar<T: ComplexField>(value: T, out: *mut u8) {
    out.cast::<T>().write(value);
}
unsafe fn write_value<T: ComplexField>(
    a: &Storage<T>,
    i: usize,
    out: *mut u8,
) -> Result<(), String> {
    let x = a
        .values()
        .get(i)
        .ok_or("stored-value index out of bounds")?;
    write_scalar(copy(x), out);
    Ok(())
}
unsafe fn set_value<T: ComplexField>(
    a: &mut Storage<T>,
    i: usize,
    value: *const u8,
) -> Result<(), String> {
    let x = a
        .values_mut()
        .get_mut(i)
        .ok_or("stored-value index out of bounds")?;
    *x = copy(&*value.cast::<T>());
    Ok(())
}
unsafe fn scale<T: ComplexField>(a: &Storage<T>, value: *const u8) -> Storage<T> {
    let mut out = a.clone();
    let v = &*value.cast::<T>();
    for x in out.values_mut() {
        *x = mul(x, v);
    }
    out
}
fn dense_product<T: ComplexField>(
    a: &Storage<T>,
    b: MatRef<'_, T>,
    left_sparse: bool,
) -> Result<Mat<T>, String> {
    let (m, n) = a.shape();
    let mut out;
    if left_sparse {
        if n != b.nrows() {
            return Err("sparse/dense product dimensions mismatch".into());
        }
        out = Mat::zeros(m, b.ncols());
        match a {
            Storage::Csc(a) => faer::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut(),
                Accum::Replace,
                a.as_ref(),
                b,
                one::<T>(),
                Par::Seq,
            ),
            Storage::Csr(a) => faer::sparse::linalg::matmul::sparse_dense_matmul(
                out.as_mut(),
                Accum::Replace,
                a.as_ref(),
                b,
                one::<T>(),
                Par::Seq,
            ),
        }
    } else {
        if b.ncols() != m {
            return Err("dense/sparse product dimensions mismatch".into());
        }
        out = Mat::zeros(b.nrows(), n);
        match a {
            Storage::Csc(a) => faer::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut(),
                Accum::Replace,
                b,
                a.as_ref(),
                one::<T>(),
                Par::Seq,
            ),
            Storage::Csr(a) => faer::sparse::linalg::matmul::dense_sparse_matmul(
                out.as_mut(),
                Accum::Replace,
                b,
                a.as_ref(),
                one::<T>(),
                Par::Seq,
            ),
        }
    }
    Ok(out)
}
#[julia]
pub struct SparseOwner {
    pub(crate) data: Option<SparseData>,
}
impl SparseOwner {
    pub(crate) fn data(&self) -> Result<&SparseData, String> {
        // Lazy: an eager message would allocate on every borrow.
        self.data
            .as_ref()
            .ok_or_else(|| "sparse matrix is not initialized".into())
    }
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn sparse_load_csc(
        &mut self,
        m: usize,
        n: usize,
        ptr: *const usize,
        idx: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.data = Some(match tag {
                0 => SparseData::F32(load_csc(m, n, ptr, idx, val, k, csr)?),
                1 => SparseData::F64(load_csc(m, n, ptr, idx, val, k, csr)?),
                2 => SparseData::C32(load_csc(m, n, ptr, idx, val, k, csr)?),
                3 => SparseData::C64(load_csc(m, n, ptr, idx, val, k, csr)?),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_load_triplets(
        &mut self,
        m: usize,
        n: usize,
        rows: *const usize,
        cols: *const usize,
        val: *const u8,
        k: usize,
        tag: i32,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.data = Some(match tag {
                0 => SparseData::F32(load_triplets(m, n, rows, cols, val, k, csr)?),
                1 => SparseData::F64(load_triplets(m, n, rows, cols, val, k, csr)?),
                2 => SparseData::C32(load_triplets(m, n, rows, cols, val, k, csr)?),
                3 => SparseData::C64(load_triplets(m, n, rows, cols, val, k, csr)?),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_nnz(&self) -> Result<usize, String> {
        guarded(|| Ok(dispatch!(self.data()?,a=>a.values().len())))
    }
    #[julia]
    pub fn sparse_values_buffer(&self, out: *mut u8, len: usize) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,a=>values_buffer(a,out,len)) })
    }
    #[julia]
    pub fn sparse_export(
        &self,
        p: *mut usize,
        i: *mut usize,
        v: *mut u8,
        pn: usize,
        k: usize,
    ) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,a=>export(a,p,i,v,pn,k)) })
    }
    #[julia]
    pub fn sparse_clone_from(
        &mut self,
        p: &SparseOwner,
        csr: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = p.data()?;
            self.data = Some(
                mapped!(a,x=>{let mut out=if transpose{x.transpose(conjugate)}else{x.clone()};if conjugate&&!transpose{for v in out.values_mut(){*v=conj(v);}}out.converted(csr)?}),
            );
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_arithmetic(
        &mut self,
        left: &SparseOwner,
        right: &SparseOwner,
        op: i32,
        csr: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = left.data()?;
            let b = right.data()?;
            self.data = Some(match (a, b) {
                (SparseData::F32(a), SparseData::F32(b)) => {
                    SparseData::F32(arithmetic(a, b, op, csr)?)
                }
                (SparseData::F64(a), SparseData::F64(b)) => {
                    SparseData::F64(arithmetic(a, b, op, csr)?)
                }
                (SparseData::C32(a), SparseData::C32(b)) => {
                    SparseData::C32(arithmetic(a, b, op, csr)?)
                }
                (SparseData::C64(a), SparseData::C64(b)) => {
                    SparseData::C64(arithmetic(a, b, op, csr)?)
                }
                _ => return Err("sparse scalar types must match".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_scale(&mut self, p: &SparseOwner, value: *const u8) -> Result<(), String> {
        guarded(|| unsafe {
            let a = (p).data()?;
            self.data = Some(mapped!(a,x=>scale(x,value)));
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_element(&self, i: usize, j: usize, out: *mut u8) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,a=>write_scalar(a.get(i,j)?,out));
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_set(&mut self, i: usize, j: usize, value: *const u8) -> Result<bool, String> {
        guarded(|| unsafe {
            match self
                .data
                .as_mut()
                .ok_or("sparse matrix is not initialized")?
            {
                SparseData::F32(a) => a.set(i, j, *value.cast::<f32>()),
                SparseData::F64(a) => a.set(i, j, *value.cast::<f64>()),
                SparseData::C32(a) => a.set(i, j, *value.cast::<faer::c32>()),
                SparseData::C64(a) => a.set(i, j, *value.cast::<faer::c64>()),
            }
        })
    }
    #[julia]
    pub fn sparse_value(&self, i: usize, out: *mut u8) -> Result<(), String> {
        guarded(|| unsafe { dispatch!(self.data()?,a=>write_value(a,i,out)) })
    }
    #[julia]
    pub fn sparse_set_value(&mut self, i: usize, value: *const u8) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data.as_mut().ok_or("sparse matrix is not initialized")?,a=>set_value(a,i,value))
        })
    }
    #[julia]
    pub fn sparse_fill_values(&mut self, value: *const u8) -> Result<(), String> {
        guarded(|| unsafe {
            macro_rules! fill {
                ($a:expr,$t:ty) => {
                    for x in $a.values_mut() {
                        *x = *value.cast::<$t>();
                    }
                };
            }
            match self
                .data
                .as_mut()
                .ok_or("sparse matrix is not initialized")?
            {
                SparseData::F32(a) => fill!(a, f32),
                SparseData::F64(a) => fill!(a, f64),
                SparseData::C32(a) => fill!(a, faer::c32),
                SparseData::C64(a) => fill!(a, faer::c64),
            }
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_dropzeros(&mut self) -> Result<bool, String> {
        guarded(
            || dispatch!(self.data.as_mut().ok_or("sparse matrix is not initialized")?,a=>a.dropzeros()),
        )
    }
    #[julia]
    pub fn sparse_sum(&self, out: *mut u8) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,a=>write_scalar(sum_value(a),out));
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_norm(&self, out: *mut u8, code: i32) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,a=>crate::typed::norm_output(ColRef::from_slice(a.values()).as_mat(),out,code))
        })
    }
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sparse_dense_from(&mut self, p: &SparseOwner) -> Result<(), String> {
        guarded(|| {
            match p.data()? {
                SparseData::F64(a) => self.matrix = a.dense(),
                _ => return Err("sparse scalar type mismatch".into()),
            }
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_product_from(
        &mut self,
        sparse: &SparseOwner,
        dense: &OwnedMatrix,
        left_sparse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = dense;
            match sparse.data()? {
                SparseData::F64(a) => {
                    self.matrix = dense_product(a, b.matrix.as_ref(), left_sparse)?
                }
                _ => return Err("sparse scalar type mismatch".into()),
            }
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sparse_dense_from(&mut self, p: &SparseOwner) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match p.data()? {
                SparseData::F32(a) => MatrixData::F32(a.dense()),
                SparseData::F64(a) => MatrixData::F64(a.dense()),
                SparseData::C32(a) => MatrixData::C32(a.dense()),
                SparseData::C64(a) => MatrixData::C64(a.dense()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sparse_product_from(
        &mut self,
        sparse: &SparseOwner,
        dense: &TypedMatrix,
        left_sparse: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = sparse.data()?;
            let b = dense.data()?;
            self.data = Some(match (a, b) {
                (SparseData::F32(a), MatrixData::F32(b)) => {
                    MatrixData::F32(dense_product(a, b.as_ref(), left_sparse)?)
                }
                (SparseData::F64(a), MatrixData::F64(b)) => {
                    MatrixData::F64(dense_product(a, b.as_ref(), left_sparse)?)
                }
                (SparseData::C32(a), MatrixData::C32(b)) => {
                    MatrixData::C32(dense_product(a, b.as_ref(), left_sparse)?)
                }
                (SparseData::C64(a), MatrixData::C64(b)) => {
                    MatrixData::C64(dense_product(a, b.as_ref(), left_sparse)?)
                }
                _ => return Err("sparse/dense scalar types must match".into()),
            });
            Ok(())
        })
    }
}

fn same_pattern<T: ComplexField>(a: &Storage<T>, b: &Storage<T>) -> bool {
    a.shape() == b.shape()
        && a.csr() == b.csr()
        && a.pointers() == b.pointers()
        && a.indices() == b.indices()
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_assign(&mut self, source: &SparseOwner, csr: bool) -> Result<bool, String> {
        guarded(|| {
            let source = source.data()?;
            let next = mapped!(source,a=>a.converted(csr)?);
            let changed = match (self.data()?, &next) {
                (SparseData::F32(a), SparseData::F32(b)) => !same_pattern(a, b),
                (SparseData::F64(a), SparseData::F64(b)) => !same_pattern(a, b),
                (SparseData::C32(a), SparseData::C32(b)) => !same_pattern(a, b),
                (SparseData::C64(a), SparseData::C64(b)) => !same_pattern(a, b),
                _ => return Err("sparse assignment scalar types must match".into()),
            };
            self.data = Some(next);
            Ok(changed)
        })
    }
}

#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_zero_diagonal(&self) -> Result<usize, String> {
        guarded(|| {
            fn first_zero<T: ComplexField>(a: &Storage<T>) -> Result<usize, String> {
                let (m, n) = a.shape();
                for i in 0..m.min(n) {
                    if a.get(i, i)? == zero::<T>() {
                        return Ok(i + 1);
                    }
                }
                Ok(0)
            }
            match self.data()? {
                SparseData::F32(a) => first_zero(a),
                SparseData::F64(a) => first_zero(a),
                SparseData::C32(a) => first_zero(a),
                SparseData::C64(a) => first_zero(a),
            }
        })
    }
}

#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_is_selfadjoint(&self, hermitian: bool) -> Result<bool, String> {
        guarded(|| {
            fn check<T: ComplexField>(a: &Storage<T>, hermitian: bool) -> Result<bool, String> {
                let (m, n) = a.shape();
                if m != n {
                    return Ok(false);
                }
                for e in a.triplets() {
                    let value = a.get(e.col, e.row)?;
                    if e.val != if hermitian { conj(&value) } else { value } {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            match self.data()? {
                SparseData::F32(a) => check(a, hermitian),
                SparseData::F64(a) => check(a, hermitian),
                SparseData::C32(a) => check(a, hermitian),
                SparseData::C64(a) => check(a, hermitian),
            }
        })
    }
}

#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_permute(
        &mut self,
        source: &SparseOwner,
        permutation: &crate::permutations::OwnedPermutation,
        columns: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            let p = permutation.data()?;
            fn apply<T: ComplexField>(
                a: &Storage<T>,
                p: faer::perm::PermRef<'_, usize>,
                columns: bool,
            ) -> Result<Storage<T>, String> {
                let (m, n) = a.shape();
                if p.len() != if columns { n } else { m } {
                    return Err("sparse permutation length mismatch".into());
                }
                let mut entries = a.triplets();
                let inverse = p.arrays().1;
                for e in &mut entries {
                    if columns {
                        e.col = inverse[e.col];
                    } else {
                        e.row = inverse[e.row];
                    }
                }
                Storage::from_triplets(m, n, &entries, a.csr())
            }
            self.data = Some(match a {
                SparseData::F32(a) => SparseData::F32(apply(a, p.as_ref(), columns)?),
                SparseData::F64(a) => SparseData::F64(apply(a, p.as_ref(), columns)?),
                SparseData::C32(a) => SparseData::C32(apply(a, p.as_ref(), columns)?),
                SparseData::C64(a) => SparseData::C64(apply(a, p.as_ref(), columns)?),
            });
            Ok(())
        })
    }
}

fn pattern_subset(
    dst: faer::sparse::SymbolicSparseColMatRef<'_, usize>,
    src: faer::sparse::SymbolicSparseColMatRef<'_, usize>,
) -> bool {
    for col in 0..src.ncols() {
        let target = &dst.row_idx()[dst.col_ptr()[col]..dst.col_ptr()[col + 1]];
        let mut pos = 0;
        for &index in &src.row_idx()[src.col_ptr()[col]..src.col_ptr()[col + 1]] {
            while pos < target.len() && target[pos] < index {
                pos += 1;
            }
            if pos == target.len() || target[pos] != index {
                return false;
            }
        }
    }
    true
}
/// Fixed-pattern `dst (+|-)= lhs` (`op` 0/1) or `dst = lhs (+|-) rhs` (2/3) on
/// column-major views; `Ok(false)` (nothing written) when a source position is
/// missing from `dst`. `dst` may be an owner or a borrowed Julia CSC matrix.
pub(crate) fn fixed_update_columns<T: ComplexField>(
    mut dst: faer::sparse::SparseColMatMut<'_, usize, T>,
    lhs: faer::sparse::SparseColMatRef<'_, usize, T>,
    rhs: faer::sparse::SparseColMatRef<'_, usize, T>,
    op: i32,
) -> Result<bool, String> {
    use faer::reborrow::ReborrowMut;
    if !(0..=3).contains(&op) {
        return Err("unknown fixed sparse update".into());
    }
    let shape = (dst.nrows(), dst.ncols());
    if shape != (lhs.nrows(), lhs.ncols()) || shape != (rhs.nrows(), rhs.ncols()) {
        return Err("fixed sparse update dimensions must match".into());
    }
    // faer's kernels may mutate before detecting a missing index. Validate all
    // stored positions first, including positions whose numeric value is zero.
    if !pattern_subset(dst.symbolic(), lhs.symbolic())
        || (op >= 2 && !pattern_subset(dst.symbolic(), rhs.symbolic()))
    {
        return Ok(false);
    }
    // faer's ternary kernel currently omits trailing destination positions
    // after both sources end. Independent input snapshots let us initialize
    // all stored destination values without affecting aliased source values.
    if op >= 2 {
        for value in dst.rb_mut().val_mut() {
            *value = zero::<T>();
        }
    }
    match op {
        0 => faer::sparse::ops::add_assign(dst, lhs),
        1 => faer::sparse::ops::sub_assign(dst, lhs),
        2 => faer::sparse::ops::add_into(dst, lhs, rhs),
        _ => faer::sparse::ops::sub_into(dst, lhs, rhs),
    }
    Ok(true)
}
fn fixed_update<T: ComplexField>(
    dst: &mut Storage<T>,
    lhs: &Storage<T>,
    rhs: &Storage<T>,
    op: i32,
) -> Result<bool, String> {
    match (dst, lhs, rhs) {
        (Storage::Csc(d), Storage::Csc(a), Storage::Csc(b)) => {
            fixed_update_columns(d.as_dyn_mut(), a.as_ref(), b.as_ref(), op)
        }
        (Storage::Csr(d), Storage::Csr(a), Storage::Csr(b)) => fixed_update_columns(
            d.transpose_mut(),
            a.as_ref().transpose(),
            b.as_ref().transpose(),
            op,
        ),
        _ => Err("fixed sparse update storage formats must match".into()),
    }
}
#[julia]
impl SparseOwner {
    #[julia]
    pub fn sparse_fixed_update(
        &mut self,
        lhs: &SparseOwner,
        rhs: &SparseOwner,
        op: i32,
    ) -> Result<bool, String> {
        guarded(|| {
            let lhs = lhs.data()?;
            let rhs = rhs.data()?;
            let dst = self
                .data
                .as_mut()
                .ok_or("sparse matrix is not initialized")?;
            match (dst, lhs, rhs) {
                (SparseData::F32(d), SparseData::F32(a), SparseData::F32(b)) => {
                    fixed_update(d, a, b, op)
                }
                (SparseData::F64(d), SparseData::F64(a), SparseData::F64(b)) => {
                    fixed_update(d, a, b, op)
                }
                (SparseData::C32(d), SparseData::C32(a), SparseData::C32(b)) => {
                    fixed_update(d, a, b, op)
                }
                (SparseData::C64(d), SparseData::C64(a), SparseData::C64(b)) => {
                    fixed_update(d, a, b, op)
                }
                _ => Err("fixed sparse update scalar types must match".into()),
            }
        })
    }
}
/// Attach `len` values of scalar type `U` to an owned copy of `a`'s pattern.
unsafe fn with_values<T: ComplexField, U: ComplexField + Copy>(
    a: &Storage<T>,
    val: *const u8,
    len: usize,
) -> Result<Storage<U>, String> {
    if a.values().len() != len {
        return Err("stored-value buffer length mismatch".into());
    }
    let vals = if len == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(val.cast::<U>(), len).to_vec()
    };
    let err = |e| format!("{e:?}");
    Ok(match a {
        Storage::Csc(x) => Storage::Csc(SparseColMat::new(
            x.symbolic().to_owned().map_err(err)?,
            vals,
        )),
        Storage::Csr(x) => Storage::Csr(SparseRowMat::new(
            x.symbolic().to_owned().map_err(err)?,
            vals,
        )),
    })
}
unsafe fn write_values<T: ComplexField + Copy>(
    a: &mut Storage<T>,
    val: *const u8,
    len: usize,
) -> Result<(), String> {
    let out = a.values_mut();
    if out.len() != len {
        return Err("stored-value buffer length mismatch".into());
    }
    if len > 0 {
        out.copy_from_slice(std::slice::from_raw_parts(val.cast::<T>(), len));
    }
    Ok(())
}
#[julia]
impl SparseOwner {
    /// Own `p`'s CSC/CSR pattern with a caller value buffer of scalar `tag`.
    #[julia]
    pub fn sparse_from_values(
        &mut self,
        p: &SparseOwner,
        val: *const u8,
        len: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let a = p.data()?;
            self.data = Some(match tag {
                0 => SparseData::F32(dispatch!(a, x => with_values(x, val, len)?)),
                1 => SparseData::F64(dispatch!(a, x => with_values(x, val, len)?)),
                2 => SparseData::C32(dispatch!(a, x => with_values(x, val, len)?)),
                3 => SparseData::C64(dispatch!(a, x => with_values(x, val, len)?)),
                _ => return Err("unsupported sparse scalar".into()),
            });
            Ok(())
        })
    }
    /// Overwrite every stored value in physical order; the pattern is unchanged.
    #[julia]
    pub fn sparse_write_values(&mut self, val: *const u8, len: usize) -> Result<(), String> {
        guarded(|| unsafe {
            match self
                .data
                .as_mut()
                .ok_or("sparse matrix is not initialized")?
            {
                SparseData::F32(a) => write_values(a, val, len),
                SparseData::F64(a) => write_values(a, val, len),
                SparseData::C32(a) => write_values(a, val, len),
                SparseData::C64(a) => write_values(a, val, len),
            }
        })
    }
}
