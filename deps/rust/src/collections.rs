//! Native owned row, column and diagonal collections, sharing no Julia buffers.
use super::guarded;
use crate::{typed::TypedMatrix, OwnedMatrix};
use faer::diag::Diag;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, Mat, MatMut, MatRef, Row};
use rustcall_julia_macros::julia;

#[derive(Clone)]
pub(crate) enum VectorStore<T> {
    Column(Col<T>),
    Row(Row<T>),
    Diagonal(Diag<T>),
}
impl<T: ComplexField> VectorStore<T> {
    pub(crate) fn new(kind: i32, n: usize, mut f: impl FnMut(usize) -> T) -> Result<Self, String> {
        Ok(match kind {
            0 => Self::Column(Col::from_fn(n, f)),
            1 => Self::Row(Row::from_fn(n, f)),
            2 => {
                let mut d = Diag::zeros(n);
                {
                    let mut v = d.column_vector_mut();
                    for i in 0..n {
                        v[i] = f(i);
                    }
                }
                Self::Diagonal(d)
            }
            _ => return Err("unknown collection kind".into()),
        })
    }
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Column(x) => x.nrows(),
            Self::Row(x) => x.ncols(),
            Self::Diagonal(x) => x.dim(),
        }
    }
    pub(crate) fn kind(&self) -> i32 {
        match self {
            Self::Column(_) => 0,
            Self::Row(_) => 1,
            Self::Diagonal(_) => 2,
        }
    }
    pub(crate) fn as_mat(&self) -> MatRef<'_, T> {
        match self {
            Self::Column(x) => x.as_mat(),
            Self::Row(x) => x.as_mat(),
            Self::Diagonal(x) => x.column_vector().as_mat(),
        }
    }
    pub(crate) fn as_mat_mut(&mut self) -> MatMut<'_, T> {
        match self {
            Self::Column(x) => x.as_mat_mut(),
            Self::Row(x) => x.as_mat_mut(),
            Self::Diagonal(x) => x.column_vector_mut().as_mat_mut(),
        }
    }
    pub(crate) fn get(&self, i: usize) -> T {
        match self {
            Self::Column(x) => copy(&x[i]),
            Self::Row(x) => copy(&x[i]),
            Self::Diagonal(x) => copy(&x.column_vector()[i]),
        }
    }
    fn set(&mut self, i: usize, value: T) {
        match self {
            Self::Column(x) => x[i] = value,
            Self::Row(x) => x[i] = value,
            Self::Diagonal(x) => x.column_vector_mut()[i] = value,
        }
    }
    pub(crate) fn full_matrix(&self) -> Mat<T> {
        match self {
            Self::Diagonal(_) => Mat::from_fn(self.len(), self.len(), |i, j| {
                if i == j {
                    self.get(i)
                } else {
                    zero::<T>()
                }
            }),
            _ => self.as_mat().to_owned(),
        }
    }
    fn reserve(&mut self, n: usize) -> Result<(), String> {
        match self {
            Self::Column(x) => x.try_reserve(n),
            Self::Row(x) => x.try_reserve(n),
            Self::Diagonal(_) => {
                return Err("diagonal capacity changes are not supported by faer Diag".into())
            }
        }
        .map_err(|e| format!("{e:?}"))
    }
    unsafe fn resize(&mut self, n: usize, value: *const u8) -> Result<(), String> {
        let value = &*value.cast::<T>();
        match self {
            Self::Column(x) => x.resize_with(n, |_| copy(value)),
            Self::Row(x) => x.resize_with(n, |_| copy(value)),
            Self::Diagonal(_) => return Err("diagonal dimensions are fixed".into()),
        }
        Ok(())
    }
}
#[derive(Clone)]
pub(crate) enum VectorData {
    F32(VectorStore<f32>),
    F64(VectorStore<f64>),
    C32(VectorStore<c32>),
    C64(VectorStore<c64>),
}
macro_rules! dispatch {
    ($data:expr,$x:ident=>$body:expr) => {
        match $data {
            VectorData::F32($x) => $body,
            VectorData::F64($x) => $body,
            VectorData::C32($x) => $body,
            VectorData::C64($x) => $body,
        }
    };
}
macro_rules! mapped {
    ($data:expr,$x:ident=>$body:expr) => {
        match $data {
            VectorData::F32($x) => VectorData::F32($body),
            VectorData::F64($x) => VectorData::F64($body),
            VectorData::C32($x) => VectorData::C32($body),
            VectorData::C64($x) => VectorData::C64($body),
        }
    };
}
macro_rules! binary {
    ($a:expr,$b:expr,$x:ident,$y:ident=>$body:expr) => {
        match ($a, $b) {
            (VectorData::F32($x), VectorData::F32($y)) => VectorData::F32($body),
            (VectorData::F64($x), VectorData::F64($y)) => VectorData::F64($body),
            (VectorData::C32($x), VectorData::C32($y)) => VectorData::C32($body),
            (VectorData::C64($x), VectorData::C64($y)) => VectorData::C64($body),
            _ => return Err("collection scalar types must match".into()),
        }
    };
}
unsafe fn load<T: ComplexField + Copy>(
    p: *const u8,
    n: usize,
    kind: i32,
) -> Result<VectorStore<T>, String> {
    let v = if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p.cast::<T>(), n)
    };
    // Stored values keep Inf/NaN like Julia vectors.
    VectorStore::new(kind, n, |i| v[i])
}
fn arithmetic<T: ComplexField>(
    a: &VectorStore<T>,
    b: &VectorStore<T>,
    op: i32,
) -> Result<VectorStore<T>, String> {
    if a.len() != b.len() || a.kind() != b.kind() {
        return Err("collection dimensions and orientations must match".into());
    }
    VectorStore::new(a.kind(), a.len(), |i| match op {
        0 => add(&a.get(i), &b.get(i)),
        1 => sub(&a.get(i), &b.get(i)),
        2 => mul(&a.get(i), &b.get(i)),
        _ => unreachable!(),
    })
}
unsafe fn scaled<T: ComplexField>(
    a: &VectorStore<T>,
    value: *const u8,
) -> Result<VectorStore<T>, String> {
    let value = &*value.cast::<T>();
    VectorStore::new(a.kind(), a.len(), |i| mul(value, &a.get(i)))
}
unsafe fn scaled_real<T: ComplexField>(
    a: &VectorStore<T>,
    value: *const u8,
) -> Result<VectorStore<T>, String> {
    let value = &*value.cast::<T::Real>();
    VectorStore::new(a.kind(), a.len(), |i| mul_real(&a.get(i), value))
}
unsafe fn element<T: ComplexField>(a: &VectorStore<T>, i: usize, p: *mut u8) {
    p.cast::<T>().write(a.get(i));
}
unsafe fn set<T: ComplexField>(a: &mut VectorStore<T>, i: usize, p: *const u8) {
    a.set(i, copy(&*p.cast::<T>()));
}
unsafe fn fill<T: ComplexField>(a: &mut VectorStore<T>, p: *const u8) {
    a.as_mat_mut().fill(copy(&*p.cast::<T>()));
}
unsafe fn sum<T: ComplexField>(a: &VectorStore<T>, out: *mut u8) {
    out.cast::<T>().write(a.as_mat().sum());
}
unsafe fn inner<T: ComplexField>(
    a: &VectorStore<T>,
    b: &VectorStore<T>,
    out: *mut u8,
    conjugate: bool,
) -> Result<(), String> {
    if a.len() != b.len() {
        return Err("collection lengths must match".into());
    }
    let v = (0..a.len()).fold(zero::<T>(), |s, i| {
        add(
            &s,
            &mul(
                &if conjugate { conj(&a.get(i)) } else { a.get(i) },
                &b.get(i),
            ),
        )
    });
    out.cast::<T>().write(v);
    Ok(())
}
#[julia]
pub struct VectorOwner {
    pub(crate) data: Option<VectorData>,
}
impl VectorOwner {
    pub(crate) fn data(&self) -> Result<&VectorData, String> {
        self.data
            .as_ref()
            .ok_or("collection is not initialized".into())
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn collection_load(
        &mut self,
        p: *const u8,
        n: usize,
        tag: i32,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe {
                match tag {
                    0 => VectorData::F32(load(p, n, kind)?),
                    1 => VectorData::F64(load(p, n, kind)?),
                    2 => VectorData::C32(load(p, n, kind)?),
                    3 => VectorData::C64(load(p, n, kind)?),
                    _ => return Err("unsupported scalar type".into()),
                }
            });
            Ok(())
        })
    }
    #[julia]
    pub fn collection_zeroes(&mut self, n: usize, tag: i32, kind: i32) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match tag {
                0 => VectorData::F32(VectorStore::new(kind, n, |_| 0f32)?),
                1 => VectorData::F64(VectorStore::new(kind, n, |_| 0f64)?),
                2 => VectorData::C32(VectorStore::new(kind, n, |_| c32::new(0., 0.))?),
                3 => VectorData::C64(VectorStore::new(kind, n, |_| c64::new(0., 0.))?),
                _ => return Err("unsupported scalar type".into()),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn collection_buffer(&self, out: *mut u8, len: usize) -> Result<(), String> {
        guarded(|| {
            let data = self.data()?;
            if dispatch!(data,x=>x.len()) != len {
                return Err("output length mismatch".into());
            }
            unsafe {
                dispatch!(data,x=>super::typed::write_matrix(x.as_mat(),out));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn collection_element(&self, i: usize, out: *mut u8) -> Result<(), String> {
        guarded(|| {
            let data = self.data()?;
            if i >= dispatch!(data,x=>x.len()) {
                return Err("index out of bounds".into());
            }
            unsafe {
                dispatch!(data,x=>element(x,i,out));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn collection_set(&mut self, i: usize, value: *const u8) -> Result<(), String> {
        guarded(|| {
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            if i >= dispatch!(data,x=>x.len()) {
                return Err("index out of bounds".into());
            }
            unsafe {
                dispatch!(data,x=>set(x,i,value));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn collection_fill(&mut self, value: *const u8) -> Result<(), String> {
        guarded(|| {
            let data = self.data.as_mut().ok_or("collection is not initialized")?;
            unsafe {
                dispatch!(data,x=>fill(x,value));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn collection_copy(
        &mut self,
        source: &VectorOwner,
        kind: i32,
        conjugated: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(
                mapped!(a,x=>VectorStore::new(if kind<0{x.kind()}else{kind},x.len(),|i|if conjugated{conj(&x.get(i))}else{x.get(i)})?),
            );
            Ok(())
        })
    }
    #[julia]
    pub fn collection_arithmetic(
        &mut self,
        left: &VectorOwner,
        right: &VectorOwner,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            if !(0..=2).contains(&op) {
                return Err("unknown collection operation".into());
            }
            let a = left.data()?;
            let b = right.data()?;
            self.data = Some(binary!(a,b,x,y=>arithmetic(x,y,op)?));
            Ok(())
        })
    }
    #[julia]
    pub fn collection_scale(
        &mut self,
        source: &VectorOwner,
        value: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>unsafe{scaled(x,value)?}));
            Ok(())
        })
    }
    #[julia]
    pub fn collection_scale_real(
        &mut self,
        source: &VectorOwner,
        value: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            self.data = Some(mapped!(a,x=>unsafe{scaled_real(x,value)?}));
            Ok(())
        })
    }
    #[julia]
    pub fn collection_sum(&self, out: *mut u8) -> Result<(), String> {
        guarded(|| {
            unsafe {
                dispatch!(self.data()?,x=>sum(x,out));
            }
            Ok(())
        })
    }
    #[julia]
    pub fn collection_norm(&self, out: *mut u8, kind: i32) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data()?,x=>super::typed::norm_output(x.as_mat(),out,kind))
        })
    }
    #[julia]
    pub fn collection_dot(
        &self,
        other: &VectorOwner,
        out: *mut u8,
        conjugated: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let b = other.data()?;
            unsafe {
                match (self.data()?, b) {
                    (VectorData::F32(x), VectorData::F32(y)) => inner(x, y, out, conjugated),
                    (VectorData::F64(x), VectorData::F64(y)) => inner(x, y, out, conjugated),
                    (VectorData::C32(x), VectorData::C32(y)) => inner(x, y, out, conjugated),
                    (VectorData::C64(x), VectorData::C64(y)) => inner(x, y, out, conjugated),
                    _ => Err("collection scalar types must match".into()),
                }
            }
        })
    }
    #[julia]
    pub fn collection_reserve(&mut self, n: usize) -> Result<(), String> {
        guarded(
            || dispatch!(self.data.as_mut().ok_or("collection is not initialized")?,x=>x.reserve(n)),
        )
    }
    #[julia]
    pub fn collection_resize(&mut self, n: usize, value: *const u8) -> Result<(), String> {
        guarded(|| unsafe {
            dispatch!(self.data.as_mut().ok_or("collection is not initialized")?,x=>x.resize(n,value))
        })
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn matrix_from_collection(&mut self, source: &VectorOwner) -> Result<(), String> {
        guarded(|| {
            self.matrix = match source.data()? {
                VectorData::F64(x) => x.full_matrix(),
                _ => return Err("Float64 output requires Float64 input".into()),
            };
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn matrix_from_collection(&mut self, source: &VectorOwner) -> Result<(), String> {
        guarded(|| {
            use super::typed::MatrixData;
            self.data = Some(match source.data()? {
                VectorData::F32(x) => MatrixData::F32(x.full_matrix()),
                VectorData::F64(x) => MatrixData::F64(x.full_matrix()),
                VectorData::C32(x) => MatrixData::C32(x.full_matrix()),
                VectorData::C64(x) => MatrixData::C64(x.full_matrix()),
            });
            Ok(())
        })
    }
}
