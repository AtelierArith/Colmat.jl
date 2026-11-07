//! Borrowed refs exist only inside one synchronous FFI call. Julia retains all
//! owners and serializes calls; no Rust reference is stored across calls.
use super::collections::{VectorData, VectorOwner, VectorStore};
use super::typed::{MatrixData, TypedMatrix};
use super::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Col, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

#[derive(Clone, Copy)]
pub(crate) struct Spec {
    pub(crate) kind: i32,
    r: isize,
    rs: isize,
    m: usize,
    c: isize,
    cs: isize,
    n: usize,
    transpose: bool,
    pub(crate) conjugate: bool,
}
impl Spec {
    // SAFETY: the Julia layer preserves a nine-word descriptor for this call.
    pub(crate) unsafe fn read(p: *const isize) -> Result<Self, String> {
        let a = std::slice::from_raw_parts(p, 9);
        Ok(Self {
            kind: i32::try_from(a[0]).map_err(|_| "invalid owner kind")?,
            r: a[1],
            rs: a[2],
            m: usize::try_from(a[3]).map_err(|_| "negative row count")?,
            c: a[4],
            cs: a[5],
            n: usize::try_from(a[6]).map_err(|_| "negative column count")?,
            transpose: a[7] != 0,
            conjugate: a[8] != 0,
        })
    }
    fn check(&self, m: usize, n: usize) -> Result<(), String> {
        check_axis(self.r, self.rs, self.m, m)?;
        check_axis(self.c, self.cs, self.n, n)
    }
}
fn check_axis(first: isize, step: isize, len: usize, capacity: usize) -> Result<(), String> {
    if len == 0 {
        return Ok(());
    }
    let last = step
        .checked_mul(isize::try_from(len - 1).map_err(|_| "view size overflow")?)
        .and_then(|x| first.checked_add(x))
        .ok_or("view index overflow")?;
    if first < 0 || last < 0 || first as usize >= capacity || last as usize >= capacity {
        return Err("view is out of bounds".into());
    }
    Ok(())
}
fn borrowed<'a, T>(a: MatRef<'a, T>, s: Spec) -> Result<MatRef<'a, T>, String> {
    s.check(a.nrows(), a.ncols())?;
    let rs = a
        .row_stride()
        .checked_mul(s.rs)
        .ok_or("view stride overflow")?;
    let cs = a
        .col_stride()
        .checked_mul(s.cs)
        .ok_or("view stride overflow")?;
    let offset = if s.m == 0 || s.n == 0 {
        0
    } else {
        a.row_stride()
            .checked_mul(s.r)
            .and_then(|x| {
                a.col_stride()
                    .checked_mul(s.c)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or("view offset overflow")?
    };
    // All selected indices are in the initialized owner; immutable refs may
    // repeat indices. No pointer escapes this call's retained owner lifetime.
    let view = unsafe { MatRef::from_raw_parts(a.as_ptr().offset(offset), s.m, s.n, rs, cs) };
    Ok(if s.transpose { view.transpose() } else { view })
}
pub(crate) enum Borrowed<'a> {
    F32(MatRef<'a, f32>),
    F64(MatRef<'a, f64>),
    C32(MatRef<'a, faer::c32>),
    C64(MatRef<'a, faer::c64>),
}
pub(crate) unsafe fn source<'a>(p: *const u8, s: Spec) -> Result<Borrowed<'a>, String> {
    Ok(match s.kind {
        0 => Borrowed::F64(borrowed((&*p.cast::<OwnedMatrix>()).matrix.as_ref(), s)?),
        1 => match (&*p.cast::<TypedMatrix>()).data()? {
            MatrixData::F32(x) => Borrowed::F32(borrowed(x.as_ref(), s)?),
            MatrixData::F64(x) => Borrowed::F64(borrowed(x.as_ref(), s)?),
            MatrixData::C32(x) => Borrowed::C32(borrowed(x.as_ref(), s)?),
            MatrixData::C64(x) => Borrowed::C64(borrowed(x.as_ref(), s)?),
        },
        2 => match (&*p.cast::<VectorOwner>()).data()? {
            VectorData::F32(x) => Borrowed::F32(borrowed(x.as_mat(), s)?),
            VectorData::F64(x) => Borrowed::F64(borrowed(x.as_mat(), s)?),
            VectorData::C32(x) => Borrowed::C32(borrowed(x.as_mat(), s)?),
            VectorData::C64(x) => Borrowed::C64(borrowed(x.as_mat(), s)?),
        },
        3 => {
            // A Julia column-major buffer preserved for this call: `r` holds the
            // scalar tag, `rs`/`cs` the element strides and `m`/`n` the shape.
            if s.m != 0 && s.n != 0 && (p.is_null() || s.rs < 1 || s.cs < 1) {
                return Err("invalid borrowed Julia buffer".into());
            }
            return raw(p, s);
        }
        _ => return Err("unknown view owner".into()),
    })
}
unsafe fn raw<'a>(p: *const u8, s: Spec) -> Result<Borrowed<'a>, String> {
    unsafe fn view<'a, T>(p: *const u8, s: Spec) -> MatRef<'a, T> {
        let v = if s.m == 0 || s.n == 0 {
            let p = std::ptr::NonNull::<T>::dangling().as_ptr().cast_const();
            MatRef::from_raw_parts(p, s.m, s.n, 1, 1)
        } else {
            MatRef::from_raw_parts(p.cast::<T>(), s.m, s.n, s.rs, s.cs)
        };
        if s.transpose {
            v.transpose()
        } else {
            v
        }
    }
    Ok(match s.r {
        0 => Borrowed::F32(view(p, s)),
        1 => Borrowed::F64(view(p, s)),
        2 => Borrowed::C32(view(p, s)),
        3 => Borrowed::C64(view(p, s)),
        _ => return Err("unsupported borrowed buffer scalar".into()),
    })
}
/// A mutable, injective, untransposed range of a native owner's matrix.
pub(crate) fn borrowed_mut<'a, T>(a: MatMut<'a, T>, s: Spec) -> Result<MatMut<'a, T>, String> {
    if s.transpose || s.conjugate {
        return Err("destination views cannot be transposed or conjugated".into());
    }
    if (s.rs == 0 && s.m > 1) || (s.cs == 0 && s.n > 1) {
        return Err("destination view must not repeat indices".into());
    }
    s.check(a.nrows(), a.ncols())?;
    let rs = a
        .row_stride()
        .checked_mul(s.rs)
        .ok_or("view stride overflow")?;
    let cs = a
        .col_stride()
        .checked_mul(s.cs)
        .ok_or("view stride overflow")?;
    let offset = if s.m == 0 || s.n == 0 {
        0
    } else {
        a.row_stride()
            .checked_mul(s.r)
            .and_then(|x| {
                a.col_stride()
                    .checked_mul(s.c)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or("view offset overflow")?
    };
    // SAFETY: the checked, injective index map stays inside the owner, which
    // the caller borrows mutably for the whole call.
    Ok(unsafe { MatMut::from_raw_parts_mut(a.as_ptr_mut().offset(offset), s.m, s.n, rs, cs) })
}
fn copied<T: ComplexField>(a: MatRef<'_, T>, conjugate: bool) -> Mat<T> {
    if conjugate {
        a.conjugate().to_owned()
    } else {
        a.to_owned()
    }
}
unsafe fn snapshot(p: *const u8, s: Spec) -> Result<MatrixData, String> {
    Ok(match source(p, s)? {
        Borrowed::F32(a) => MatrixData::F32(copied(a, s.conjugate)),
        Borrowed::F64(a) => MatrixData::F64(copied(a, s.conjugate)),
        Borrowed::C32(a) => MatrixData::C32(copied(a, s.conjugate)),
        Borrowed::C64(a) => MatrixData::C64(copied(a, s.conjugate)),
    })
}
fn binary<T: ComplexField>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    ca: bool,
    cb: bool,
    op: i32,
) -> Result<Mat<T>, String> {
    if op == 2 {
        if a.ncols() != b.nrows() {
            return Err("inner dimensions must match".into());
        }
        let mut out = Mat::zeros(a.nrows(), b.ncols());
        faer::linalg::matmul::matmul_with_conj(
            out.as_mut(),
            Accum::Replace,
            a,
            if ca { Conj::Yes } else { Conj::No },
            b,
            if cb { Conj::Yes } else { Conj::No },
            one::<T>(),
            Par::Seq,
        );
        return Ok(out);
    }
    if !(0..=1).contains(&op) {
        return Err("unknown view operation".into());
    }
    if a.shape() != b.shape() {
        return Err("view shapes must match".into());
    }
    Ok(Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
        let x = if ca {
            conj(&a[(i, j)])
        } else {
            copy(&a[(i, j)])
        };
        let y = if cb {
            conj(&b[(i, j)])
        } else {
            copy(&b[(i, j)])
        };
        if op == 0 {
            add(&x, &y)
        } else {
            sub(&x, &y)
        }
    }))
}
unsafe fn binary_data(
    left: *const u8,
    ls: Spec,
    right: *const u8,
    rs: Spec,
    op: i32,
) -> Result<MatrixData, String> {
    Ok(match (source(left, ls)?, source(right, rs)?) {
        (Borrowed::F32(a), Borrowed::F32(b)) => {
            MatrixData::F32(binary(a, b, ls.conjugate, rs.conjugate, op)?)
        }
        (Borrowed::F64(a), Borrowed::F64(b)) => {
            MatrixData::F64(binary(a, b, ls.conjugate, rs.conjugate, op)?)
        }
        (Borrowed::C32(a), Borrowed::C32(b)) => {
            MatrixData::C32(binary(a, b, ls.conjugate, rs.conjugate, op)?)
        }
        (Borrowed::C64(a), Borrowed::C64(b)) => {
            MatrixData::C64(binary(a, b, ls.conjugate, rs.conjugate, op)?)
        }
        _ => return Err("view scalar types must match".into()),
    })
}
fn write_view<T: ComplexField>(
    mut a: MatMut<'_, T>,
    s: Spec,
    b: MatRef<'_, T>,
) -> Result<(), String> {
    s.check(a.nrows(), a.ncols())?;
    let expected = if s.transpose { (s.n, s.m) } else { (s.m, s.n) };
    if b.shape() != expected {
        return Err("view assignment shape mismatch".into());
    }
    if s.m == 0 || s.n == 0 {
        return Ok(());
    }
    // An injective index map can borrow a genuine faer MatMut. Repeated ranges
    // use ordered scalar writes instead, never aliased mutable Rust refs.
    if (s.rs != 0 || s.m <= 1) && (s.cs != 0 || s.n <= 1) {
        let rs = a
            .row_stride()
            .checked_mul(s.rs)
            .ok_or("view stride overflow")?;
        let cs = a
            .col_stride()
            .checked_mul(s.cs)
            .ok_or("view stride overflow")?;
        let offset = a
            .row_stride()
            .checked_mul(s.r)
            .and_then(|x| {
                a.col_stride()
                    .checked_mul(s.c)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or("view offset overflow")?;
        let mut v =
            unsafe { MatMut::from_raw_parts_mut(a.as_ptr_mut().offset(offset), s.m, s.n, rs, cs) };
        if s.transpose {
            v = v.transpose_mut();
        }
        if s.conjugate {
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    v[(i, j)] = conj(&b[(i, j)]);
                }
            }
        } else {
            v.copy_from(b);
        }
    } else {
        for j in 0..b.ncols() {
            for i in 0..b.nrows() {
                let (r, c) = if s.transpose { (j, i) } else { (i, j) };
                let value = if s.conjugate {
                    conj(&b[(i, j)])
                } else {
                    copy(&b[(i, j)])
                };
                a[(
                    (s.r + s.rs * r as isize) as usize,
                    (s.c + s.cs * c as isize) as usize,
                )] = value;
            }
        }
    }
    Ok(())
}
unsafe fn fill_view<T: ComplexField>(
    mut a: MatMut<'_, T>,
    s: Spec,
    value: *const u8,
) -> Result<(), String> {
    s.check(a.nrows(), a.ncols())?;
    let x = &*value.cast::<T>();
    let value = if s.conjugate { conj(x) } else { copy(x) };
    for j in 0..s.n {
        for i in 0..s.m {
            a[(
                (s.r + s.rs * i as isize) as usize,
                (s.c + s.cs * j as isize) as usize,
            )] = copy(&value);
        }
    }
    Ok(())
}
fn vector_copy<T: ComplexField>(
    a: MatRef<'_, T>,
    kind: i32,
    conjugate: bool,
) -> Result<VectorStore<T>, String> {
    let m = a.nrows();
    VectorStore::new(
        kind,
        m.checked_mul(a.ncols()).ok_or("vector size overflow")?,
        |i| {
            if conjugate {
                conj(&a[(i % m, i / m)])
            } else {
                copy(&a[(i % m, i / m)])
            }
        },
    )
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn view_copy(&mut self, p: *const u8, desc: *const isize) -> Result<(), String> {
        guarded(|| {
            self.matrix = match unsafe { snapshot(p, Spec::read(desc)?)? } {
                MatrixData::F64(a) => a,
                _ => return Err("Float64 output requires Float64 view".into()),
            };
            Ok(())
        })
    }
    #[julia]
    pub fn view_binary(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix =
                match unsafe { binary_data(left, Spec::read(ld)?, right, Spec::read(rd)?, op)? } {
                    MatrixData::F64(a) => a,
                    _ => return Err("Float64 output requires Float64 views".into()),
                };
            Ok(())
        })
    }
    #[julia]
    pub fn view_assign(
        &mut self,
        desc: *const isize,
        p: *const u8,
        source_desc: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { snapshot(p, Spec::read(source_desc)?)? };
            match b {
                MatrixData::F64(b) => write_view(
                    self.matrix.as_mut(),
                    unsafe { Spec::read(desc)? },
                    b.as_ref(),
                ),
                _ => Err("assignment scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn view_fill(&mut self, desc: *const isize, value: *const u8) -> Result<(), String> {
        guarded(|| unsafe { fill_view(self.matrix.as_mut(), Spec::read(desc)?, value) })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn view_copy(&mut self, p: *const u8, desc: *const isize) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { snapshot(p, Spec::read(desc)?)? });
            Ok(())
        })
    }
    #[julia]
    pub fn view_binary(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data =
                Some(unsafe { binary_data(left, Spec::read(ld)?, right, Spec::read(rd)?, op)? });
            Ok(())
        })
    }
    #[julia]
    pub fn view_assign(
        &mut self,
        desc: *const isize,
        p: *const u8,
        source_desc: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { snapshot(p, Spec::read(source_desc)?)? };
            let s = unsafe { Spec::read(desc)? };
            match (self.data.as_mut().ok_or("matrix is not initialized")?, b) {
                (MatrixData::F32(a), MatrixData::F32(b)) => write_view(a.as_mut(), s, b.as_ref()),
                (MatrixData::F64(a), MatrixData::F64(b)) => write_view(a.as_mut(), s, b.as_ref()),
                (MatrixData::C32(a), MatrixData::C32(b)) => write_view(a.as_mut(), s, b.as_ref()),
                (MatrixData::C64(a), MatrixData::C64(b)) => write_view(a.as_mut(), s, b.as_ref()),
                _ => Err("assignment scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn view_fill(&mut self, desc: *const isize, value: *const u8) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            unsafe {
                match self.data.as_mut().ok_or("matrix is not initialized")? {
                    MatrixData::F32(a) => fill_view(a.as_mut(), s, value),
                    MatrixData::F64(a) => fill_view(a.as_mut(), s, value),
                    MatrixData::C32(a) => fill_view(a.as_mut(), s, value),
                    MatrixData::C64(a) => fill_view(a.as_mut(), s, value),
                }
            }
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn vector_from_view(
        &mut self,
        p: *const u8,
        desc: *const isize,
        kind: i32,
    ) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            self.data = Some(match unsafe { source(p, s)? } {
                Borrowed::F32(a) => VectorData::F32(vector_copy(a, kind, s.conjugate)?),
                Borrowed::F64(a) => VectorData::F64(vector_copy(a, kind, s.conjugate)?),
                Borrowed::C32(a) => VectorData::C32(vector_copy(a, kind, s.conjugate)?),
                Borrowed::C64(a) => VectorData::C64(vector_copy(a, kind, s.conjugate)?),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn vector_binary_views(
        &mut self,
        left: *const u8,
        ld: *const isize,
        right: *const u8,
        rd: *const isize,
        op: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(
                match unsafe { binary_data(left, Spec::read(ld)?, right, Spec::read(rd)?, op)? } {
                    MatrixData::F32(a) => VectorData::F32(vector_copy(a.as_ref(), 0, false)?),
                    MatrixData::F64(a) => VectorData::F64(vector_copy(a.as_ref(), 0, false)?),
                    MatrixData::C32(a) => VectorData::C32(vector_copy(a.as_ref(), 0, false)?),
                    MatrixData::C64(a) => VectorData::C64(vector_copy(a.as_ref(), 0, false)?),
                },
            );
            Ok(())
        })
    }
    #[julia]
    pub fn view_fill(&mut self, desc: *const isize, value: *const u8) -> Result<(), String> {
        guarded(|| {
            let s = unsafe { Spec::read(desc)? };
            unsafe {
                match self.data.as_mut().ok_or("collection is not initialized")? {
                    VectorData::F32(a) => fill_view(a.as_mat_mut(), s, value),
                    VectorData::F64(a) => fill_view(a.as_mat_mut(), s, value),
                    VectorData::C32(a) => fill_view(a.as_mat_mut(), s, value),
                    VectorData::C64(a) => fill_view(a.as_mat_mut(), s, value),
                }
            }
        })
    }
    #[julia]
    pub fn view_assign(
        &mut self,
        desc: *const isize,
        p: *const u8,
        source_desc: *const isize,
    ) -> Result<(), String> {
        guarded(|| {
            let b = unsafe { snapshot(p, Spec::read(source_desc)?)? };
            let s = unsafe { Spec::read(desc)? };
            match (
                self.data.as_mut().ok_or("collection is not initialized")?,
                b,
            ) {
                (VectorData::F32(a), MatrixData::F32(b)) => {
                    write_view(a.as_mat_mut(), s, b.as_ref())
                }
                (VectorData::F64(a), MatrixData::F64(b)) => {
                    write_view(a.as_mat_mut(), s, b.as_ref())
                }
                (VectorData::C32(a), MatrixData::C32(b)) => {
                    write_view(a.as_mat_mut(), s, b.as_ref())
                }
                (VectorData::C64(a), MatrixData::C64(b)) => {
                    write_view(a.as_mat_mut(), s, b.as_ref())
                }
                _ => Err("assignment scalar types must match".into()),
            }
        })
    }
}
