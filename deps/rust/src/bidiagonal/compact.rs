//! Shared owned bidiagonal reflectors, including rectangular basis actions.
use super::construction::{block_size, params};
use super::workspace::buffer;
use crate::teams::with_team;
use crate::views::{
    destination, destination_span, operand, source as source_view, spec_shape, Arm, Borrowed, Spec,
};
use crate::{guarded, householder_sequence, teams::OwnedTeam, workspace::OwnedScratch};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::{self, svd};
use faer::mat::{AsMatMut, AsMatRef};
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

#[derive(Clone)]
struct Core<T> {
    m: usize,
    n: usize,
    packed: Mat<T>,
    left: Mat<T>,
    right: Mat<T>,
}
impl<T: ComplexField> Core<T> {
    fn k(&self) -> usize {
        self.m.min(self.n)
    }
    fn b(&self) -> usize {
        self.left.nrows()
    }
    fn left_side(&self, u: bool) -> bool {
        u == (self.m >= self.n)
    }
    fn dim(&self, u: bool) -> usize {
        if u {
            self.m
        } else {
            self.n
        }
    }
    fn validate_cols(&self, u: bool, cols: usize) -> Result<(), String> {
        if cols != self.k() && cols != self.dim(u) {
            return Err("compact bidiagonal basis column count mismatch".into());
        }
        Ok(())
    }
    fn apply_in(
        &self,
        u: bool,
        mut a: MatMut<'_, T>,
        right: bool,
        t: bool,
        c: bool,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<(), String> {
        if (if right { a.ncols() } else { a.nrows() }) != self.dim(u) {
            return Err("compact bidiagonal action dimensions mismatch".into());
        }
        if self.k() == 0 {
            return Ok(());
        }
        if self.left_side(u) {
            householder_sequence::apply_views_in(
                self.packed.as_ref(),
                self.left.as_ref(),
                a,
                right,
                t,
                c,
                par,
                stack,
            )
        } else if self.k() > 1 {
            let k = self.k() - 1;
            let target = if right {
                a.rb_mut().subcols_mut(1, k)
            } else {
                a.rb_mut().subrows_mut(1, k)
            };
            householder_sequence::apply_views_in(
                self.packed.as_ref().submatrix(0, 1, k, k).transpose(),
                self.right.as_ref(),
                target,
                right,
                t,
                !c,
                par,
                stack,
            )
        } else {
            Ok(())
        }
    }
}
#[derive(Clone)]
enum Data {
    F32(Core<f32>),
    F64(Core<f64>),
    C32(Core<c32>),
    C64(Core<c64>),
}
#[julia]
pub struct OwnedCompactBidiagonal {
    data: Data,
}
fn construction_req<T: ComplexField>(
    m: usize,
    n: usize,
    bs: usize,
    par: ParShape,
    threshold: usize,
) -> Result<StackReq, String> {
    if m > isize::MAX as usize || n > isize::MAX as usize {
        return Err("compact bidiagonal dimensions overflow".into());
    }
    let k = m.min(n);
    block_size::<T>(m.max(n), k, bs)?;
    Ok(if k == 0 {
        StackReq::EMPTY
    } else {
        svd::bidiag::bidiag_in_place_scratch::<T>(
            m.max(n),
            k,
            if threshold == usize::MAX {
                ParShape::Seq
            } else {
                par
            },
            params::<T>(threshold),
        )
    })
}
fn apply_req<T: ComplexField>(
    m: usize,
    n: usize,
    b: usize,
    u: bool,
    cols: usize,
    extent: usize,
    right: bool,
    t: bool,
) -> Result<StackReq, String> {
    let k = m.min(n);
    let dim = if u { m } else { n };
    if cols != k && cols != dim {
        return Err("compact bidiagonal basis column count mismatch".into());
    }
    dim.checked_mul(extent)
        .ok_or("compact bidiagonal action size overflow")?;
    if cols == 0 {
        return Ok(StackReq::EMPTY);
    }
    let left = u == (m >= n);
    let hh = if k == 0 {
        StackReq::EMPTY
    } else if left {
        householder_sequence::layout_requirement::<T>(dim, k, b, extent, right, t)
    } else {
        householder_sequence::layout_requirement::<T>(k - 1, k - 1, b, extent, right, t)
    };
    Ok((if right {
        linalg::temp_mat_scratch::<T>(extent, dim)
    } else {
        linalg::temp_mat_scratch::<T>(dim, extent)
    })
    .and(hh))
}
fn no_overlap(a: (usize, usize), b: (usize, usize)) -> Result<(), String> {
    if a.0 < a.1 && b.0 < b.1 && a.0 < b.1 && b.0 < a.1 {
        return Err("compact bidiagonal numeric buffers overlap".into());
    }
    Ok(())
}
unsafe fn build<T: ComplexField>(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    d: *mut u8,
    e: *mut u8,
    bs: usize,
    mut par: Par,
    threshold: usize,
) -> Result<Core<T>, String> {
    let k = m.min(n);
    let rows = m.max(n);
    let nn = m.checked_mul(n).ok_or("compact bidiagonal size overflow")?;
    let spans = [
        buffer::<T>(input, nn)?,
        buffer::<T>(d, k)?,
        buffer::<T>(e, k.saturating_sub(1))?,
    ];
    for i in 0..3 {
        for j in i + 1..3 {
            no_overlap(spans[i], spans[j])?;
        }
    }
    if threshold == usize::MAX {
        par = Par::Seq;
    }
    let stack = scratch.stack(construction_req::<T>(m, n, bs, par.shape(), threshold)?)?;
    let b = block_size::<T>(rows, k, bs)?.max(1);
    let ptr = if nn == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr().cast_const()
    } else {
        input.cast::<T>()
    };
    // SAFETY: preserved contiguous column-major input; scalar span checked above.
    let a = MatRef::from_raw_parts(ptr, m, n, 1, m as isize);
    crate::typed::finite(a)?;
    let mut core = Core {
        m,
        n,
        packed: Mat::from_fn(rows, k, |i, j| {
            if m < n {
                conj(&a[(j, i)])
            } else {
                copy(&a[(i, j)])
            }
        }),
        left: Mat::zeros(b, k),
        right: Mat::zeros(b, k.saturating_sub(1)),
    };
    if k == 0 {
        return Ok(core);
    }
    let scaling = crate::schur::scale_in(core.packed.as_mut());
    svd::bidiag::bidiag_in_place(
        core.packed.as_mut(),
        core.left.as_mut(),
        core.right.as_mut(),
        par.reborrow(),
        stack,
        params::<T>(threshold),
    );
    crate::typed::finite(core.packed.as_ref())?;
    for f in [&core.left, &core.right] {
        for j in 0..f.ncols() {
            for i in 0..=j % b {
                let x = &f[(i, j)];
                if i == j % b {
                    if imag(x) != zero::<T::Real>() || !(real(x) > zero::<T::Real>()) {
                        return Err("invalid compact bidiagonal denominator".into());
                    }
                } else if !is_finite(x) {
                    return Err("nonfinite compact bidiagonal factor".into());
                }
            }
        }
    }
    let inv = scaling.map(|s| recip(&s));
    let coefficient = |i: usize, j: usize| {
        let x = &core.packed[(i, j)];
        let x = if let Some(ref scale) = inv {
            mul_real(x, scale)
        } else {
            copy(x)
        };
        if m < n {
            conj(&x)
        } else {
            x
        }
    };
    for i in 0..k {
        if !is_finite(&coefficient(i, i)) || (i + 1 < k && !is_finite(&coefficient(i, i + 1))) {
            return Err("compact bidiagonalization overflowed the working type".into());
        }
    }
    // SAFETY: disjoint, preserved output bands are written only after validation.
    for i in 0..k {
        d.cast::<T>().add(i).write(coefficient(i, i));
        if i + 1 < k {
            e.cast::<T>().add(i).write(coefficient(i, i + 1));
        }
    }
    Ok(core)
}
fn build_data(
    scratch: &mut OwnedScratch,
    input: *const u8,
    m: usize,
    n: usize,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
    par: Par,
    threshold: usize,
) -> Result<Data, String> {
    // SAFETY: each scalar branch validates spans before dereferencing numeric pointers.
    unsafe {
        Ok(match tag {
            0 => Data::F32(build(scratch, input, m, n, d, e, bs, par, threshold)?),
            1 => Data::F64(build(scratch, input, m, n, d, e, bs, par, threshold)?),
            2 => Data::C32(build(scratch, input, m, n, d, e, bs, par, threshold)?),
            3 => Data::C64(build(scratch, input, m, n, d, e, bs, par, threshold)?),
            _ => return Err("unsupported compact bidiagonal scalar".into()),
        })
    }
}
fn write_req(req: StackReq, out: *mut usize) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    // SAFETY: caller preserves two writable UInt elements after address checks.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
fn construction_query(
    m: usize,
    n: usize,
    tag: i32,
    bs: usize,
    threshold: usize,
    par: ParShape,
    out: *mut usize,
) -> Result<(), String> {
    write_req(
        match tag {
            0 => construction_req::<f32>(m, n, bs, par, threshold)?,
            1 => construction_req::<f64>(m, n, bs, par, threshold)?,
            2 => construction_req::<c32>(m, n, bs, par, threshold)?,
            3 => construction_req::<c64>(m, n, bs, par, threshold)?,
            _ => return Err("unsupported compact bidiagonal scalar".into()),
        },
        out,
    )
}
#[julia]
pub fn compact_bidiagonal_requirements(
    m: usize,
    n: usize,
    tag: i32,
    bs: usize,
    threshold: usize,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| {
        with_team(team, |par| {
            construction_query(m, n, tag, bs, threshold, par.shape(), out)
        })
    })
}
#[julia]
impl OwnedCompactBidiagonal {
    #[julia]
    pub fn reduce(
        scratch: &mut OwnedScratch,
        input: *const u8,
        m: usize,
        n: usize,
        d: *mut u8,
        e: *mut u8,
        tag: i32,
        bs: usize,
        threshold: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<Self, String> {
        guarded(|| {
            with_team(team, |par| {
                Ok(Self {
                    data: build_data(scratch, input, m, n, d, e, tag, bs, par, threshold)?,
                })
            })
        })
    }

    #[julia]
    pub fn duplicate(source: &Self) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                data: source.data.clone(),
            })
        })
    }
}
impl OwnedCompactBidiagonal {
    fn metadata(&self) -> (usize, usize, usize, i32) {
        match &self.data {
            Data::F32(s) => (s.m, s.n, s.b(), 0),
            Data::F64(s) => (s.m, s.n, s.b(), 1),
            Data::C32(s) => (s.m, s.n, s.b(), 2),
            Data::C64(s) => (s.m, s.n, s.b(), 3),
        }
    }
    fn validate(&self, m: usize, n: usize, b: usize, tag: i32) -> Result<(), String> {
        if self.metadata() != (m, n, b, tag) {
            return Err("compact bidiagonal native scalar/shape metadata mismatch".into());
        }
        Ok(())
    }
}
#[julia]
impl OwnedCompactBidiagonal {
    #[julia]
    pub fn compact_bidiagonal_info(&self, out: &mut [usize]) -> Result<(), String> {
        let out = crate::workspace::fixed_out(out, 4)?;
        guarded(|| {
            buffer::<usize>(out.cast(), 4)?;
            let (m, n, b, tag) = self.metadata();
            unsafe {
                out.write(m);
                out.add(1).write(n);
                out.add(2).write(b);
                out.add(3).write(tag as usize);
            }
            Ok(())
        })
    }
    #[julia]
    pub fn compact_bidiagonal_validate(
        &self,
        m: usize,
        n: usize,
        b: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| self.validate(m, n, b, tag))
    }
    #[julia]
    pub fn compact_bidiagonal_apply_requirements(
        &self,
        m: usize,
        n: usize,
        b: usize,
        tag: i32,
        target: i32,
        u: bool,
        cols: usize,
        extent: usize,
        right: bool,
        t: bool,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            self.validate(m, n, b, tag)?;
            write_req(
                match target {
                    0 => apply_req::<f32>(m, n, b, u, cols, extent, right, t)?,
                    1 => apply_req::<f64>(m, n, b, u, cols, extent, right, t)?,
                    2 => apply_req::<c32>(m, n, b, u, cols, extent, right, t)?,
                    3 => apply_req::<c64>(m, n, b, u, cols, extent, right, t)?,
                    _ => return Err("unsupported compact bidiagonal application scalar".into()),
                },
                out,
            )
        })
    }
}
unsafe fn apply<T: ComplexField>(
    s: &Core<T>,
    scratch: &mut OwnedScratch,
    u: bool,
    cols: usize,
    extent: usize,
    right: bool,
    t: bool,
    c: bool,
    input: *const u8,
    out: *mut u8,
    par: Par,
) -> Result<(), String> {
    s.validate_cols(u, cols)?;
    let dim = s.dim(u);
    let idim = if right == t { cols } else { dim };
    let odim = if right == t { dim } else { cols };
    let ilen = idim
        .checked_mul(extent)
        .ok_or("compact bidiagonal input size overflow")?;
    let olen = odim
        .checked_mul(extent)
        .ok_or("compact bidiagonal output size overflow")?;
    no_overlap(buffer::<T>(input, ilen)?, buffer::<T>(out, olen)?)?;
    let stack = scratch.stack(apply_req::<T>(s.m, s.n, s.b(), u, cols, extent, right, t)?)?;
    for i in 0..ilen {
        if !is_finite(&*input.cast::<T>().add(i)) {
            return Err("compact bidiagonal action input must be finite".into());
        }
    }
    if cols == 0 {
        for i in 0..olen {
            out.cast::<T>().add(i).write(zero());
        }
        return Ok(());
    }
    let (rows, ncols) = if right { (extent, dim) } else { (dim, extent) };
    let (mut work, stack) = linalg::temp_mat_zeroed::<T, _, _>(rows, ncols, stack);
    let (ir, ic) = if right {
        (extent, idim)
    } else {
        (idim, extent)
    };
    for j in 0..ic {
        for i in 0..ir {
            work.as_mat_mut()[(i, j)] = copy(&*input.cast::<T>().add(i + j * ir));
        }
    }
    s.apply_in(u, work.as_mat_mut(), right, t, c, par, stack)?;
    crate::typed::finite(work.as_mat_ref())?;
    let (or, oc) = if right {
        (extent, odim)
    } else {
        (odim, extent)
    };
    for j in 0..oc {
        for i in 0..or {
            out.cast::<T>()
                .add(i + j * or)
                .write(copy(&work.as_mat_ref()[(i, j)]));
        }
    }
    Ok(())
}
fn apply_data(
    source: &OwnedCompactBidiagonal,
    scratch: &mut OwnedScratch,
    m: usize,
    n: usize,
    b: usize,
    tag: i32,
    u: bool,
    cols: usize,
    extent: usize,
    right: bool,
    t: bool,
    c: bool,
    input: *const u8,
    out: *mut u8,
    par: Par,
) -> Result<(), String> {
    source.validate(m, n, b, tag)?;
    // SAFETY: scalar branches validate all spans, dimensions and workspace first.
    unsafe {
        match &source.data {
            Data::F32(s) => apply(s, scratch, u, cols, extent, right, t, c, input, out, par),
            Data::F64(s) => apply(s, scratch, u, cols, extent, right, t, c, input, out, par),
            Data::C32(s) => apply(s, scratch, u, cols, extent, right, t, c, input, out, par),
            Data::C64(s) => apply(s, scratch, u, cols, extent, right, t, c, input, out, par),
        }
    }
}
#[julia]
impl OwnedCompactBidiagonal {
    #[julia]
    pub fn compact_bidiagonal_apply(
        &self,
        scratch: &mut OwnedScratch,
        m: usize,
        n: usize,
        b: usize,
        tag: i32,
        u: bool,
        cols: usize,
        extent: usize,
        right: bool,
        t: bool,
        c: bool,
        input: *const u8,
        out: *mut u8,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                apply_data(
                    self, scratch, m, n, b, tag, u, cols, extent, right, t, c, input, out, par,
                )
            })
        })
    }
}

/// `C = alpha*basis_action(B) + beta*C` read from and written to view
/// descriptors. The embedding/projection work matrix is the declared
/// scratch's `temp_mat_zeroed`, so no other buffer exists. Inputs, the
/// action and the combined result are checked finite before the first
/// destination write; `beta_zero` never reads C.
#[allow(clippy::too_many_arguments)]
unsafe fn mul_into<T: Arm>(
    s: &Core<T>,
    work: &mut OwnedScratch,
    shape: (bool, usize, usize, bool, bool, bool),
    dest: *mut u8,
    ds: Spec,
    src: *const u8,
    ss: Spec,
    coefficients: (*const u8, *const u8, bool, bool),
    par: Par,
) -> Result<(), String> {
    let (u, cols, extent, right, t, c) = shape;
    s.validate_cols(u, cols)?;
    let dim = s.dim(u);
    let idim = if right == t { cols } else { dim };
    let odim = if right == t { dim } else { cols };
    let (ir, ic) = if right {
        (extent, idim)
    } else {
        (idim, extent)
    };
    let (or, oc) = if right {
        (extent, odim)
    } else {
        (odim, extent)
    };
    if spec_shape(ss) != (ir, ic) || spec_shape(ds) != (or, oc) {
        return Err("compact bidiagonal action dimensions must match".into());
    }
    let stack = work.stack(apply_req::<T>(s.m, s.n, s.b(), u, cols, extent, right, t)?)?;
    let x = operand::<T>(src, ss, destination_span(dest, ds)?)?;
    let (a, conjugate) = x.get();
    if !a.is_all_finite() {
        return Err("compact bidiagonal action input must be finite".into());
    }
    let (alpha, beta, alpha_zero, beta_zero) = coefficients;
    let alpha = copy(&*alpha.cast::<T>());
    let beta = copy(&*beta.cast::<T>());
    let mut out =
        T::pick_mut(destination(dest, ds)?).ok_or("destination scalar types must match")?;
    let combine = |p: Option<&T>, y: &T| {
        let scaled = if beta_zero {
            zero::<T>()
        } else {
            mul(&beta, y)
        };
        match p {
            Some(p) if !alpha_zero => add(&mul(&alpha, p), &scaled),
            _ => scaled,
        }
    };
    if cols == 0 || alpha_zero {
        for j in 0..oc {
            for i in 0..or {
                if !is_finite(&combine(None, &out[(i, j)])) {
                    return Err("compact basis product overflowed the destination scalar".into());
                }
            }
        }
        for j in 0..oc {
            for i in 0..or {
                out[(i, j)] = combine(None, &out[(i, j)]);
            }
        }
        return Ok(());
    }
    let (rows, ncols) = if right { (extent, dim) } else { (dim, extent) };
    let (mut work, stack) = linalg::temp_mat_zeroed::<T, _, _>(rows, ncols, stack);
    let mut w = work.as_mat_mut();
    for j in 0..ic {
        for i in 0..ir {
            w[(i, j)] = if conjugate {
                conj(&a[(i, j)])
            } else {
                copy(&a[(i, j)])
            };
        }
    }
    s.apply_in(u, w.rb_mut(), right, t, c, par, stack)?;
    crate::typed::finite(w.as_ref())?;
    // Combine in the work block, check it, then copy: C is written only
    // after the complete result is known to be finite.
    let mut block = w.submatrix_mut(0, 0, or, oc);
    for j in 0..oc {
        for i in 0..or {
            let y = combine(Some(&block[(i, j)]), &out[(i, j)]);
            block[(i, j)] = y;
        }
    }
    if !block.as_ref().is_all_finite() {
        return Err("compact basis product overflowed the destination scalar".into());
    }
    out.copy_from(block.as_ref());
    Ok(())
}
#[julia]
impl OwnedCompactBidiagonal {
    /// Native destination form of [`Self::compact_bidiagonal_apply`]:
    /// `dest = alpha*Q*src + beta*dest` for view descriptors of one scalar.
    /// `metadata` is `(m, n, blocksize, cols, extent)`; the source is
    /// snapshot when it overlaps the destination.
    ///
    /// # Safety
    /// Julia roots the destination, source, descriptors and coefficients.
    #[allow(clippy::too_many_arguments)]
    #[julia]
    pub fn compact_bidiagonal_mul_into(
        &self,
        scratch: &mut OwnedScratch,
        metadata: &[usize],
        tag: i32,
        u: bool,
        right: bool,
        t: bool,
        c: bool,
        dest: *mut u8,
        dd: *const isize,
        src: *const u8,
        sd: *const isize,
        alpha: *const u8,
        beta: *const u8,
        alpha_zero: bool,
        beta_zero: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let [m, n, b, cols, extent] = metadata else {
                return Err(
                    "compact basis metadata must hold (m, n, blocksize, cols, extent)".into(),
                );
            };
            self.validate(*m, *n, *b, tag)?;
            let (ds, ss) = (Spec::read(dd)?, Spec::read(sd)?);
            let shape = (u, *cols, *extent, right, t, c);
            let k = (alpha, beta, alpha_zero, beta_zero);
            with_team(team, |par| match (&self.data, source_view(dest, ds)?) {
                (Data::F32(s), Borrowed::F32(_)) => {
                    mul_into(s, scratch, shape, dest, ds, src, ss, k, par)
                }
                (Data::F64(s), Borrowed::F64(_)) => {
                    mul_into(s, scratch, shape, dest, ds, src, ss, k, par)
                }
                (Data::C32(s), Borrowed::C32(_)) => {
                    mul_into(s, scratch, shape, dest, ds, src, ss, k, par)
                }
                (Data::C64(s), Borrowed::C64(_)) => {
                    mul_into(s, scratch, shape, dest, ds, src, ss, k, par)
                }
                _ => Err("compact basis and destination scalar types must match".into()),
            })
        })
    }
}

fn cast_core<T: ComplexField, U: ComplexField>(
    s: &Core<T>,
    convert: impl Fn(&T) -> Result<U, String>,
) -> Result<Core<U>, String> {
    let mut out = Core {
        m: s.m,
        n: s.n,
        packed: Mat::zeros(s.packed.nrows(), s.k()),
        left: Mat::zeros(s.b(), s.k()),
        right: Mat::zeros(s.b(), s.k().saturating_sub(1)),
    };
    for j in 0..s.k() {
        if real(&s.left[(j % s.b(), j)]) != infinity::<T::Real>() {
            for i in j + 1..s.packed.nrows() {
                out.packed[(i, j)] = convert(&s.packed[(i, j)])?;
            }
        }
    }
    for j in 0..s.k().saturating_sub(1) {
        if real(&s.right[(j % s.b(), j)]) != infinity::<T::Real>() {
            for i in j + 2..s.k() {
                out.packed[(j, i)] = convert(&s.packed[(j, i)])?;
            }
        }
    }
    for j in 0..s.left.ncols() {
        for i in 0..=j % s.b() {
            out.left[(i, j)] = convert(&s.left[(i, j)])?;
        }
    }
    for j in 0..s.right.ncols() {
        for i in 0..=j % s.b() {
            out.right[(i, j)] = convert(&s.right[(i, j)])?;
        }
    }
    Ok(out)
}
#[julia]
impl OwnedCompactBidiagonal {
    #[julia]
    pub fn compact_bidiagonal_component(
        &self,
        m: usize,
        n: usize,
        b: usize,
        tag: i32,
        u: bool,
        kind: i32,
        out: *mut u8,
        len: usize,
    ) -> Result<(), String> {
        guarded(|| {
            self.validate(m, n, b, tag)?;
            unsafe fn write<T: ComplexField>(
                s: &Core<T>,
                u: bool,
                kind: i32,
                out: *mut u8,
                len: usize,
            ) -> Result<(), String> {
                let left = s.left_side(u);
                let count = if left { s.k() } else { s.k().saturating_sub(1) };
                let rows = if kind == 1 {
                    s.b()
                } else if left {
                    s.dim(u)
                } else {
                    count
                };
                if !matches!(kind, 0 | 1) || rows.checked_mul(count) != Some(len) {
                    return Err("compact bidiagonal component shape mismatch".into());
                }
                buffer::<T>(out, len)?;
                let factor = if left { &s.left } else { &s.right };
                for j in 0..count {
                    for i in 0..rows {
                        let value = if kind == 1 {
                            if i <= j % s.b() {
                                copy(&factor[(i, j)])
                            } else {
                                zero()
                            }
                        } else if i < j {
                            zero()
                        } else if i == j {
                            one()
                        } else if real(&factor[(j % s.b(), j)]) == infinity::<T::Real>() {
                            zero()
                        } else if left {
                            copy(&s.packed[(i, j)])
                        } else {
                            copy(&s.packed[(j, i + 1)])
                        };
                        out.cast::<T>().add(i + j * rows).write(value);
                    }
                }
                Ok(())
            }
            // SAFETY: every scalar branch checks the exact component length and span.
            unsafe {
                match &self.data {
                    Data::F32(s) => write(s, u, kind, out, len),
                    Data::F64(s) => write(s, u, kind, out, len),
                    Data::C32(s) => write(s, u, kind, out, len),
                    Data::C64(s) => write(s, u, kind, out, len),
                }
            }
        })
    }
}

#[julia]
impl OwnedCompactBidiagonal {
    #[julia]
    pub fn converted(source: &Self, tag: i32) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                data: match (&source.data, tag) {
                    (Data::F32(s), 0) => Data::F32(cast_core(s, |v: &f32| {
                        let out = (*v) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 1) => Data::F64(cast_core(s, |v: &f32| {
                        let out = (*v) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 2) => Data::C32(cast_core(s, |v: &f32| {
                        let out = c32::new((*v) as f32, (0.0) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 3) => Data::C64(cast_core(s, |v: &f32| {
                        let out = c64::new((*v) as f64, (0.0) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 0) => Data::F32(cast_core(s, |v: &f64| {
                        let out = (*v) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 1) => Data::F64(cast_core(s, |v: &f64| {
                        let out = (*v) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 2) => Data::C32(cast_core(s, |v: &f64| {
                        let out = c32::new((*v) as f32, (0.0) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 3) => Data::C64(cast_core(s, |v: &f64| {
                        let out = c64::new((*v) as f64, (0.0) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 0) => Data::F32(cast_core(s, |v: &c32| {
                        if v.im != 0.0 {
                            return Err(
                                "cannot convert complex compact basis to real scalar".into()
                            );
                        }
                        let out = (v.re) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 1) => Data::F64(cast_core(s, |v: &c32| {
                        if v.im != 0.0 {
                            return Err(
                                "cannot convert complex compact basis to real scalar".into()
                            );
                        }
                        let out = (v.re) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 2) => Data::C32(cast_core(s, |v: &c32| {
                        let out = c32::new((v.re) as f32, (v.im) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 3) => Data::C64(cast_core(s, |v: &c32| {
                        let out = c64::new((v.re) as f64, (v.im) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 0) => Data::F32(cast_core(s, |v: &c64| {
                        if v.im != 0.0 {
                            return Err(
                                "cannot convert complex compact basis to real scalar".into()
                            );
                        }
                        let out = (v.re) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 1) => Data::F64(cast_core(s, |v: &c64| {
                        if v.im != 0.0 {
                            return Err(
                                "cannot convert complex compact basis to real scalar".into()
                            );
                        }
                        let out = (v.re) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 2) => Data::C32(cast_core(s, |v: &c64| {
                        let out = c32::new((v.re) as f32, (v.im) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 3) => Data::C64(cast_core(s, |v: &c64| {
                        let out = c64::new((v.re) as f64, (v.im) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact basis scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    _ => return Err("unsupported compact basis conversion scalar".into()),
                },
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn action<T: ComplexField>(
        s: &Core<T>,
        u: bool,
        cols: usize,
        right: bool,
        t: bool,
        c: bool,
        a: MatRef<'_, T>,
    ) -> Mat<T> {
        let extent = if right { a.nrows() } else { a.ncols() };
        let outdim = if right == t { s.dim(u) } else { cols };
        let (or, oc) = if right {
            (extent, outdim)
        } else {
            (outdim, extent)
        };
        let input: Vec<T> = (0..a.ncols())
            .flat_map(|j| (0..a.nrows()).map(move |i| copy(&a[(i, j)])))
            .collect();
        let mut output: Vec<T> = (0..or * oc).map(|_| zero()).collect();
        let mut work = OwnedScratch::new();
        let req = apply_req::<T>(s.m, s.n, s.b(), u, cols, extent, right, t).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // SAFETY: exact-size disjoint numeric input/output buffers live for the call.
        unsafe {
            apply(
                s,
                &mut work,
                u,
                cols,
                extent,
                right,
                t,
                c,
                input.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .unwrap();
        Mat::from_fn(or, oc, |i, j| copy(&output[i + j * or]))
    }
    #[test]
    fn mul_into_uses_only_the_declared_scratch() {
        use crate::allocation_probe::count;
        use crate::views::raw_descriptor;
        let (m, n, bs) = (6usize, 4usize, 2usize);
        let a = Mat::from_fn(m, n, |i, j| ((i * 5 + j * 3) % 7) as f64 - 2.5);
        let input: Vec<f64> = (0..n)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .map(|(i, j)| a[(i, j)])
            .collect();
        let mut d = vec![0.0f64; n];
        let mut e = vec![0.0f64; n - 1];
        let creq = construction_req::<f64>(m, n, bs, ParShape::Seq, 0).unwrap();
        let mut work = OwnedScratch::new();
        work.scratch_reserve(creq.size_bytes(), creq.align_bytes())
            .unwrap();
        let s = unsafe {
            build::<f64>(
                &mut work,
                input.as_ptr().cast(),
                m,
                n,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                bs,
                Par::Seq,
                0,
            )
        }
        .unwrap();
        let owner = OwnedCompactBidiagonal {
            data: Data::F64(s.clone()),
        };
        // Thin U (m x n) times B (n x 3) into a padded Julia destination.
        let (u, cols, extent) = (true, n, 3usize);
        let req = apply_req::<f64>(m, n, bs, u, cols, extent, false, false).unwrap();
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let b = Mat::from_fn(n, extent, |i, j| (1 + i + 2 * j) as f64);
        let bd = raw_descriptor(n, extent, b.col_stride(), 1);
        let dd = raw_descriptor(m, extent, 9, 1);
        let mut julia = vec![1.0f64; 9 * extent];
        let expected = action(&s, u, cols, false, false, false, b.as_ref());
        let (alpha, beta) = (2.0f64, 0.5f64);
        let meta = [m, n, bs, cols, extent];
        let mut call = |beta_zero: bool, julia: &mut Vec<f64>| {
            count(|| {
                owner
                    .compact_bidiagonal_mul_into(
                        &mut scratch,
                        &meta,
                        1,
                        u,
                        false,
                        false,
                        false,
                        julia.as_mut_ptr().cast(),
                        dd.as_ptr(),
                        b.as_ptr().cast(),
                        bd.as_ptr(),
                        (&alpha as *const f64).cast(),
                        (&beta as *const f64).cast(),
                        false,
                        beta_zero,
                        None,
                    )
                    .unwrap()
            })
        };
        assert_eq!(call(false, &mut julia), 0);
        for j in 0..extent {
            for i in 0..m {
                assert!((julia[i + 9 * j] - (alpha * expected[(i, j)] + beta)).abs() < 1e-12);
            }
        }
        assert_eq!(call(true, &mut julia), 0);
        assert_eq!(julia[m], 1.0);
    }
    fn check<T: ComplexField>(a: Mat<T>, bs: usize, tolerance: f64) {
        let (m, n) = (a.nrows(), a.ncols());
        let k = m.min(n);
        let input: Vec<T> = (0..n)
            .flat_map(|j| {
                (0..m).map({
                    let a = &a;
                    move |i| copy(&a[(i, j)])
                })
            })
            .collect();
        let mut d: Vec<T> = (0..k).map(|_| zero()).collect();
        let mut e: Vec<T> = (0..k.saturating_sub(1)).map(|_| zero()).collect();
        let req = construction_req::<T>(m, n, bs, ParShape::Seq, 0).unwrap();
        let mut work = OwnedScratch::new();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // SAFETY: disjoint numeric arrays have the required exact lengths/types.
        let s = unsafe {
            build::<T>(
                &mut work,
                input.as_ptr().cast(),
                m,
                n,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                bs,
                Par::Seq,
                0,
            )
        }
        .unwrap();
        let full_u = action(
            &s,
            true,
            m,
            false,
            false,
            false,
            Mat::from_fn(m, m, |i, j| if i == j { one() } else { zero() }).as_ref(),
        );
        let full_v = action(
            &s,
            false,
            n,
            false,
            false,
            false,
            Mat::from_fn(n, n, |i, j| if i == j { one() } else { zero() }).as_ref(),
        );
        let tol = from_f64::<T::Real>(tolerance);
        for j in 0..n {
            for i in 0..m {
                let mut value = zero::<T>();
                for r in 0..k {
                    value = add(
                        &value,
                        &mul(&mul(&full_u[(i, r)], &d[r]), &conj(&full_v[(j, r)])),
                    );
                    if r + 1 < k {
                        let (l, h) = if m >= n { (r, r + 1) } else { (r + 1, r) };
                        value = add(
                            &value,
                            &mul(&mul(&full_u[(i, l)], &e[r]), &conj(&full_v[(j, h)])),
                        );
                    }
                }
                assert!(
                    abs(&sub(&value, &a[(i, j)])) <= tol,
                    "reconstruction m={m} n={n} bs={bs}"
                );
            }
        }
        for (u, q) in [(true, full_u.as_ref()), (false, full_v.as_ref())] {
            let dim = s.dim(u);
            for j in 0..dim {
                for i in 0..dim {
                    let mut value = zero::<T>();
                    for r in 0..dim {
                        value = add(&value, &mul(&conj(&q[(r, i)]), &q[(r, j)]));
                    }
                    assert!(abs(&sub(&value, &if i == j { one() } else { zero() })) <= tol);
                }
            }
            for cols in [k, dim] {
                for right in [false, true] {
                    for t in [false, true] {
                        for c in [false, true] {
                            let idim = if right == t { cols } else { dim };
                            let odim = if right == t { dim } else { cols };
                            let (ir, ic) = if right { (3, idim) } else { (idim, 3) };
                            let rhs = Mat::from_fn(ir, ic, |i, j| {
                                from_f64::<T>((1 + i + 2 * j) as f64 * 0.17)
                            });
                            let output = action(&s, u, cols, right, t, c, rhs.as_ref());
                            for j in 0..output.ncols() {
                                for i in 0..output.nrows() {
                                    let mut expected = zero::<T>();
                                    for r in 0..idim {
                                        let (row, col) = if right {
                                            if t {
                                                (j, r)
                                            } else {
                                                (r, j)
                                            }
                                        } else if t {
                                            (r, i)
                                        } else {
                                            (i, r)
                                        };
                                        let value = if c {
                                            conj(&q[(row, col)])
                                        } else {
                                            copy(&q[(row, col)])
                                        };
                                        expected = add(
                                            &expected,
                                            &if right {
                                                mul(&rhs[(i, r)], &value)
                                            } else {
                                                mul(&value, &rhs[(r, j)])
                                            },
                                        );
                                    }
                                    assert!(abs(&sub(&output[(i,j)],&expected))<=tol,"action m={m} n={n} cols={cols} outdim={odim} right={right} t={t} c={c}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn compact_bidiagonal_all_scalars_shapes_and_orientations() {
        for (m, n) in [
            (0, 0),
            (0, 4),
            (4, 0),
            (1, 1),
            (4, 1),
            (1, 4),
            (4, 3),
            (3, 4),
            (3, 3),
            (7, 4),
            (4, 7),
        ] {
            for bs in [0, 1, 2.min(m.min(n).max(1))] {
                check(
                    Mat::from_fn(m, n, |i, j| (1 + i + 3 * j) as f32 * 0.13),
                    bs,
                    1e-3,
                );
                check(
                    Mat::from_fn(m, n, |i, j| (1 + i + 3 * j) as f64 * 0.13),
                    bs,
                    1e-11,
                );
                check(
                    Mat::from_fn(m, n, |i, j| {
                        c32::new((1 + i + 3 * j) as f32 * 0.13, (i as f32 - j as f32) * 0.21)
                    }),
                    bs,
                    1e-3,
                );
                check(
                    Mat::from_fn(m, n, |i, j| {
                        c64::new((1 + i + 3 * j) as f64 * 0.13, (i as f64 - j as f64) * 0.21)
                    }),
                    bs,
                    1e-11,
                );
            }
        }
    }
    #[test]
    fn compact_bidiagonal_failure_preserves_numeric_outputs() {
        let mut work = OwnedScratch::new();
        let req = construction_req::<f64>(3, 2, 1, ParShape::Seq, 0).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let input = [1.0, f64::NAN, 2.0, 4.0, 5.0, 6.0];
        let mut d = [71.0; 2];
        let mut e = [72.0];
        // SAFETY: exact disjoint arrays; a deliberate nonfinite numeric value.
        assert!(unsafe {
            build::<f64>(
                &mut work,
                input.as_ptr().cast(),
                3,
                2,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                1,
                Par::Seq,
                0,
            )
        }
        .is_err());
        assert_eq!(d, [71.0; 2]);
        assert_eq!(e, [72.0]);
        let input = [1.0, 3.0, 2.0, 4.0, 5.0, 6.0];
        let s = unsafe {
            build::<f64>(
                &mut work,
                input.as_ptr().cast(),
                3,
                2,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                1,
                Par::Seq,
                0,
            )
        }
        .unwrap();
        let mut short = OwnedScratch::new();
        short.scratch_reserve(0, 1).unwrap();
        let mut out = [91.0; 3];
        let rhs = [1.0, 2.0];
        assert!(unsafe {
            apply(
                &s,
                &mut short,
                true,
                2,
                1,
                false,
                false,
                false,
                rhs.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(out, [91.0; 3]);
    }
}
