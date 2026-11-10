//! Borrowed refs exist only inside one synchronous FFI call. Julia retains all
//! owners and serializes calls; no Rust reference is stored across calls.
use super::collections::{VectorData, VectorOwner, VectorStore};
use super::statistics::{widens, Native};
use super::typed::{MatrixData, TypedMatrix};
use super::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;

mod destination;
mod gather;
pub(crate) use destination::{
    action_into, destination, destination_span, operand, raw_mut, spec_shape, view_mut, Arm,
    BorrowedMut,
};
#[cfg(test)]
pub(crate) use destination::raw_descriptor;

#[derive(Clone, Copy)]
pub(crate) struct Spec {
    pub(crate) kind: i32,
    pub(crate) r: isize,
    pub(crate) rs: isize,
    pub(crate) m: usize,
    pub(crate) c: isize,
    pub(crate) cs: isize,
    pub(crate) n: usize,
    pub(crate) transpose: bool,
    pub(crate) conjugate: bool,
    raw: Option<RawLayout>,
}
/// Layout of a borrowed Julia strided buffer (owner kind 4). The owner pointer
/// is the address of its first element; strides are in elements and may be
/// negative. Julia roots the parent array for the duration of the call.
#[derive(Clone, Copy)]
struct RawLayout {
    m: usize,
    n: usize,
    rs: isize,
    cs: isize,
    tag: i32,
}
const RAW_KIND: i32 = 4;
impl Spec {
    // SAFETY: the Julia layer preserves a nine-word descriptor for this call,
    // followed by five raw-layout words for owner kind 4.
    pub(crate) unsafe fn read(p: *const isize) -> Result<Self, String> {
        let a = std::slice::from_raw_parts(p, 9);
        let kind = i32::try_from(a[0]).map_err(|_| "invalid owner kind")?;
        let raw = if kind == RAW_KIND {
            let b = std::slice::from_raw_parts(p.add(9), 5);
            Some(RawLayout {
                m: usize::try_from(b[0]).map_err(|_| "negative borrowed row count")?,
                n: usize::try_from(b[1]).map_err(|_| "negative borrowed column count")?,
                rs: b[2],
                cs: b[3],
                tag: i32::try_from(b[4]).map_err(|_| "invalid borrowed scalar tag")?,
            })
        } else {
            None
        };
        Ok(Self {
            raw,
            kind,
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
    /// The same selection without conjugation (for value-independent writes
    /// such as permutations).
    pub(crate) fn unconjugated(mut self) -> Self {
        self.conjugate = false;
        self
    }
    pub(crate) fn check(&self, m: usize, n: usize) -> Result<(), String> {
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
pub(crate) fn borrowed<'a, T>(a: MatRef<'a, T>, s: Spec) -> Result<MatRef<'a, T>, String> {
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
// SAFETY: `p` addresses the first element of a live Julia strided array with
// the declared layout and scalar; Julia keeps the parent rooted for the call.
unsafe fn raw_base<'a, T>(p: *const u8, raw: RawLayout) -> MatRef<'a, T> {
    let ptr = if raw.m == 0 || raw.n == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr().cast_const()
    } else {
        p.cast::<T>()
    };
    MatRef::from_raw_parts(ptr, raw.m, raw.n, raw.rs, raw.cs)
}
fn raw_layout(s: &Spec) -> Option<RawLayout> {
    s.raw.filter(|_| s.kind == RAW_KIND)
}
pub(crate) unsafe fn source<'a>(p: *const u8, s: Spec) -> Result<Borrowed<'a>, String> {
    if let Some(raw) = raw_layout(&s) {
        return Ok(match raw.tag {
            0 => Borrowed::F32(borrowed(raw_base(p, raw), s)?),
            1 => Borrowed::F64(borrowed(raw_base(p, raw), s)?),
            2 => Borrowed::C32(borrowed(raw_base(p, raw), s)?),
            3 => Borrowed::C64(borrowed(raw_base(p, raw), s)?),
            _ => return Err("unknown borrowed scalar".into()),
        });
    }
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
        _ => return Err("unknown view owner".into()),
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
/// One converted copy of a view at the wider scalar `tag`, applying the
/// descriptor's transpose and conjugation (conjugating only complex sources).
unsafe fn promoted(p: *const u8, s: Spec, tag: i32) -> Result<MatrixData, String> {
    converted(p, s, tag, false)
}
/// Copies a borrowed matrix into scalar `tag`, applying its transpose and
/// conjugation. Without `narrowing` the target must widen the source; with it
/// any real or complex precision is allowed except complex to real.
pub(crate) unsafe fn converted(
    p: *const u8,
    s: Spec,
    tag: i32,
    narrowing: bool,
) -> Result<MatrixData, String> {
    fn convert_to<S: Native, W: Native>(
        a: MatRef<'_, S>,
        conjugate: bool,
        narrowing: bool,
    ) -> Result<Mat<W>, String> {
        if S::TAG >= 2 && W::TAG < 2 {
            return Err("complex matrices cannot convert to a real scalar type".into());
        }
        if !narrowing && !widens(S::TAG, W::TAG) {
            return Err("native promotion must widen the scalar type".into());
        }
        let conjugate = conjugate && S::TAG >= 2;
        Ok(Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
            let (re, im) = a[(i, j)].parts();
            W::from_parts(re, if conjugate { -im } else { im })
        }))
    }
    fn to<S: Native>(a: MatRef<'_, S>, c: bool, tag: i32, n: bool) -> Result<MatrixData, String> {
        Ok(match tag {
            0 => MatrixData::F32(convert_to(a, c, n)?),
            1 => MatrixData::F64(convert_to(a, c, n)?),
            2 => MatrixData::C32(convert_to(a, c, n)?),
            3 => MatrixData::C64(convert_to(a, c, n)?),
            _ => return Err("unknown promotion scalar".into()),
        })
    }
    let c = s.conjugate;
    match source(p, s)? {
        Borrowed::F32(a) => to(a, c, tag, narrowing),
        Borrowed::F64(a) => to(a, c, tag, narrowing),
        Borrowed::C32(a) => to(a, c, tag, narrowing),
        Borrowed::C64(a) => to(a, c, tag, narrowing),
    }
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
/// Assignment input: Julia strided buffers are read in place; native owners,
/// which may be the destination itself, are snapshot first.
enum Input<'a, T> {
    Direct(MatRef<'a, T>, bool),
    Owned(Mat<T>),
}
/// The four native scalar arms of an assignment input.
enum AnyInput<'a> {
    F32(Input<'a, f32>),
    F64(Input<'a, f64>),
    C32(Input<'a, faer::c32>),
    C64(Input<'a, faer::c64>),
}
fn wrap_input<T: ComplexField>(b: MatRef<'_, T>, raw: bool, conjugate: bool) -> Input<'_, T> {
    if raw {
        Input::Direct(b, conjugate)
    } else {
        Input::Owned(copied(b, conjugate))
    }
}
// Called before the destination is borrowed mutably, so an owner source that
// is the destination itself is fully copied before any write.
unsafe fn any_input<'a>(p: *const u8, source_desc: *const isize) -> Result<AnyInput<'a>, String> {
    let s = Spec::read(source_desc)?;
    let raw = raw_layout(&s).is_some();
    Ok(match source(p, s)? {
        Borrowed::F32(b) => AnyInput::F32(wrap_input(b, raw, s.conjugate)),
        Borrowed::F64(b) => AnyInput::F64(wrap_input(b, raw, s.conjugate)),
        Borrowed::C32(b) => AnyInput::C32(wrap_input(b, raw, s.conjugate)),
        Borrowed::C64(b) => AnyInput::C64(wrap_input(b, raw, s.conjugate)),
    })
}
/// Byte range `[lo, hi)` touched by a matrix ref; `None` when empty.
fn span<T>(a: MatRef<'_, T>) -> Option<(usize, usize)> {
    if a.nrows() == 0 || a.ncols() == 0 {
        return None;
    }
    let size = std::mem::size_of::<T>() as isize;
    let (r, c) = (
        a.row_stride() * (a.nrows() as isize - 1),
        a.col_stride() * (a.ncols() as isize - 1),
    );
    let lo = r.min(0) + c.min(0);
    let hi = r.max(0) + c.max(0) + 1;
    let base = a.as_ptr() as isize;
    Some(((base + lo * size) as usize, (base + hi * size) as usize))
}
fn disjoint<T>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> bool {
    match (span(a), span(b)) {
        (Some((a0, a1)), Some((b0, b1))) => a1 <= b0 || b1 <= a0,
        _ => true,
    }
}
fn assign<T: ComplexField>(a: MatMut<'_, T>, s: Spec, b: Input<'_, T>) -> Result<(), String> {
    match b {
        // Julia memory never holds Rust-owned storage today; the range check
        // keeps a future aliasing borrow (for example mmap) on the copy path.
        Input::Direct(b, c) if disjoint(a.as_ref(), b) => write_view(a, s, b, s.conjugate != c),
        Input::Direct(b, c) => write_view(a, s, copied(b, c).as_ref(), s.conjugate),
        Input::Owned(b) => write_view(a, s, b.as_ref(), s.conjugate),
    }
}
pub(crate) fn write_view<T: ComplexField>(
    mut a: MatMut<'_, T>,
    s: Spec,
    b: MatRef<'_, T>,
    conjugate: bool,
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
        if conjugate {
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
                let value = if conjugate {
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
    pub fn view_promoted(
        &mut self,
        p: *const u8,
        desc: *const isize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = match unsafe { promoted(p, Spec::read(desc)?, tag)? } {
                MatrixData::F64(a) => a,
                _ => return Err("Float64 output requires the Float64 tag".into()),
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
            let b = unsafe { any_input(p, source_desc)? };
            match b {
                AnyInput::F64(b) => assign(self.matrix.as_mut(), unsafe { Spec::read(desc)? }, b),
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
    pub fn view_promoted(
        &mut self,
        p: *const u8,
        desc: *const isize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { promoted(p, Spec::read(desc)?, tag)? });
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
            let b = unsafe { any_input(p, source_desc)? };
            let s = unsafe { Spec::read(desc)? };
            match (self.data.as_mut().ok_or("matrix is not initialized")?, b) {
                (MatrixData::F32(a), AnyInput::F32(b)) => assign(a.as_mut(), s, b),
                (MatrixData::F64(a), AnyInput::F64(b)) => assign(a.as_mut(), s, b),
                (MatrixData::C32(a), AnyInput::C32(b)) => assign(a.as_mut(), s, b),
                (MatrixData::C64(a), AnyInput::C64(b)) => assign(a.as_mut(), s, b),
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
            let b = unsafe { any_input(p, source_desc)? };
            let s = unsafe { Spec::read(desc)? };
            match (
                self.data.as_mut().ok_or("collection is not initialized")?,
                b,
            ) {
                (VectorData::F32(a), AnyInput::F32(b)) => assign(a.as_mat_mut(), s, b),
                (VectorData::F64(a), AnyInput::F64(b)) => assign(a.as_mat_mut(), s, b),
                (VectorData::C32(a), AnyInput::C32(b)) => assign(a.as_mat_mut(), s, b),
                (VectorData::C64(a), AnyInput::C64(b)) => assign(a.as_mat_mut(), s, b),
                _ => Err("assignment scalar types must match".into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_desc(m: usize, n: usize, rs: isize, cs: isize, transpose: bool) -> Vec<isize> {
        vec![
            RAW_KIND as isize,
            0,
            1,
            m as isize,
            0,
            1,
            n as isize,
            transpose as isize,
            0,
            m as isize,
            n as isize,
            rs,
            cs,
            1,
        ]
    }

    #[test]
    fn raw_julia_buffers_are_read_in_place_with_negative_strides() {
        // A 3 x 2 column-major buffer read with reversed rows.
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let desc = raw_desc(3, 2, -1, 3, false);
        let first = unsafe { data.as_ptr().add(2) }.cast::<u8>();
        let s = unsafe { Spec::read(desc.as_ptr()) }.unwrap();
        let Borrowed::F64(a) = (unsafe { source(first, s) }.unwrap()) else {
            panic!("expected f64");
        };
        assert_eq!(a.shape(), (3, 2));
        assert_eq!((a[(0, 0)], a[(2, 0)], a[(0, 1)]), (3.0, 1.0, 6.0));
        assert_eq!(a.as_ptr(), first.cast::<f64>());

        // Assign its transpose into a 2 x 3 owner without a snapshot.
        let mut out = Mat::<f64>::zeros(2, 3);
        let target = [0isize, 0, 1, 2, 0, 1, 3, 0, 0];
        let ts = unsafe { Spec::read(target.as_ptr()) }.unwrap();
        let t = raw_desc(3, 2, -1, 3, true);
        let AnyInput::F64(input) = (unsafe { any_input(first, t.as_ptr()) }.unwrap()) else {
            panic!("expected f64");
        };
        assert!(matches!(input, Input::Direct(..)));
        assign(out.as_mut(), ts, input).unwrap();
        assert_eq!(out[(0, 0)], 3.0);
        assert_eq!(out[(1, 2)], 4.0);
    }

    #[test]
    fn promotion_widens_and_applies_transpose_and_conjugation() {
        // A 2 x 2 complex f32 buffer read as its adjoint and widened to c64.
        let data = [
            faer::c32::new(1.0, 2.0),
            faer::c32::new(3.0, -1.0),
            faer::c32::new(-2.0, 0.5),
            faer::c32::new(4.0, 0.0),
        ];
        let mut desc = raw_desc(2, 2, 1, 2, true);
        desc[8] = 1;
        desc[13] = 2;
        let s = unsafe { Spec::read(desc.as_ptr()) }.unwrap();
        let MatrixData::C64(a) = (unsafe { promoted(data.as_ptr().cast(), s, 3) }.unwrap()) else {
            panic!("expected c64");
        };
        assert_eq!(a[(0, 1)], faer::c64::new(3.0, 1.0));
        assert_eq!(a[(1, 0)], faer::c64::new(-2.0, -0.5));
        // Real sources keep a positive zero imaginary part under conjugation.
        let real = [1.5f32, -2.0];
        let mut desc = raw_desc(2, 1, 1, 2, false);
        desc[8] = 1;
        desc[13] = 0;
        let s = unsafe { Spec::read(desc.as_ptr()) }.unwrap();
        let MatrixData::C64(b) = (unsafe { promoted(real.as_ptr().cast(), s, 3) }.unwrap()) else {
            panic!("expected c64");
        };
        assert!(b[(1, 0)].im == 0.0 && b[(1, 0)].im.is_sign_positive());
        assert!(unsafe { promoted(real.as_ptr().cast(), s, 0) }.is_ok());
        let wide = [1.0f64];
        let mut desc = raw_desc(1, 1, 1, 1, false);
        desc[13] = 1;
        let s = unsafe { Spec::read(desc.as_ptr()) }.unwrap();
        assert!(unsafe { promoted(wide.as_ptr().cast(), s, 0) }.is_err());
    }

    #[test]
    fn empty_raw_buffers_never_dereference_the_pointer() {
        let desc = raw_desc(0, 4, 1, 0, false);
        let s = unsafe { Spec::read(desc.as_ptr()) }.unwrap();
        let Borrowed::F64(a) = (unsafe { source(std::ptr::null(), s) }.unwrap()) else {
            panic!("expected f64");
        };
        assert_eq!(a.shape(), (0, 4));
    }

    #[test]
    fn overlapping_spans_are_detected() {
        let data = [0.0f64; 12];
        let whole = MatRef::from_column_major_slice(&data, 3, 4);
        let tail = MatRef::from_column_major_slice(&data[9..], 3, 1);
        let head = MatRef::from_column_major_slice(&data[..9], 3, 3);
        assert!(!disjoint(whole, tail));
        assert!(disjoint(head, tail));
        assert!(disjoint(whole.subrows(0, 0), whole));
    }
}
