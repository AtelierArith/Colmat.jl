//! Embedded compact Householder products for tridiagonal and Hessenberg Q.
//! The existing facade name is retained for compatibility. Storage owns its
//! matrices; no workspace or execution token escapes a call.
use super::construction::{
    block_size, normalize_phase, params, phase_magnitude, real_subdiagonal, validate_real_bands,
};
use super::workspace::buffer;
use crate::collections::{VectorData, VectorOwner};
use crate::teams::{with_team, OwnedTeam};
use crate::typed::{MatrixData, TypedMatrix};
use crate::OwnedMatrix;
use crate::{guarded, householder_sequence, workspace::OwnedScratch};
use faer::dyn_stack::StackReq;
use faer::linalg::evd;
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;

#[derive(Clone)]
struct Core<T> {
    packed: Mat<T>,
    factor: Mat<T>,
    phases: Vec<T>,
}
impl<T: ComplexField> Core<T> {
    fn n(&self) -> usize {
        self.packed.nrows()
    }
    fn k(&self) -> usize {
        self.n().saturating_sub(1)
    }
    fn b(&self) -> usize {
        self.factor.nrows()
    }
    fn requirement(&self, extent: usize, right: bool, transpose: bool) -> StackReq {
        householder_sequence::layout_requirement::<T>(
            self.k(),
            self.k(),
            self.b(),
            extent,
            right,
            transpose,
        )
    }
    fn apply(
        &self,
        mut a: MatMut<'_, T>,
        work: &mut OwnedScratch,
        right: bool,
        transpose: bool,
        conjugated: bool,
        mut par: Par,
    ) -> Result<(), String> {
        if (if right { a.ncols() } else { a.nrows() }) != self.n() {
            return Err("compact Q application dimensions must match".into());
        }
        let stack = work.stack(self.requirement(
            if right { a.nrows() } else { a.ncols() },
            right,
            transpose,
        ))?;
        let scale = |a: &mut MatMut<'_, T>| {
            for i in 0..self.n() {
                let p = if conjugated {
                    conj(&self.phases[i])
                } else {
                    copy(&self.phases[i])
                };
                if right {
                    for row in 0..a.nrows() {
                        a[(row, i)] = &a[(row, i)] * &p;
                    }
                } else {
                    for col in 0..a.ncols() {
                        a[(i, col)] = &p * &a[(i, col)];
                    }
                }
            }
        };
        if !self.phases.is_empty() && right == transpose {
            scale(&mut a);
        }
        if self.k() > 0 {
            let target = if right {
                a.rb_mut().subcols_mut(1, self.k())
            } else {
                a.rb_mut().subrows_mut(1, self.k())
            };
            householder_sequence::apply_views_in(
                self.packed.as_ref().submatrix(1, 0, self.k(), self.k()),
                self.factor.as_ref(),
                target,
                right,
                transpose,
                conjugated,
                par.reborrow(),
                stack,
            )?;
        }
        if !self.phases.is_empty() && right != transpose {
            scale(&mut a);
        }
        Ok(())
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
pub struct OwnedCompactTridiagonal {
    data: Data,
}
fn requirement<T: ComplexField>(
    n: usize,
    bs: usize,
    par: ParShape,
    threshold: usize,
) -> Result<StackReq, String> {
    block_size::<T>(n, bs)?;
    Ok(if n < 2 {
        StackReq::EMPTY
    } else {
        evd::tridiag::tridiag_in_place_scratch::<T>(
            n,
            if threshold == usize::MAX {
                ParShape::Seq
            } else {
                par
            },
            params::<T>(threshold),
        )
    })
}
fn query(
    n: usize,
    tag: i32,
    bs: usize,
    par: ParShape,
    threshold: usize,
    out: *mut usize,
) -> Result<(), String> {
    buffer::<usize>(out.cast(), 2)?;
    let req = match tag {
        0 => requirement::<f32>(n, bs, par, threshold)?,
        1 => requirement::<f64>(n, bs, par, threshold)?,
        2 => requirement::<c32>(n, bs, par, threshold)?,
        3 => requirement::<c64>(n, bs, par, threshold)?,
        _ => return Err("unsupported compact tridiagonal scalar".into()),
    };
    // SAFETY: the caller preserves two writable usize elements, checked above.
    unsafe {
        out.write(req.size_bytes());
        out.add(1).write(req.align_bytes());
    }
    Ok(())
}
#[julia]
pub fn compact_tridiagonal_requirements(
    n: usize,
    tag: i32,
    bs: usize,
    threshold: usize,
    out: &mut [usize],
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    let out = crate::workspace::requirement_out(out)?;
    guarded(|| with_team(team, |par| query(n, tag, bs, par.shape(), threshold, out)))
}
unsafe fn build<T: ComplexField>(
    work: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    d: *mut u8,
    e: *mut u8,
    bs: usize,
    mut par: Par,
    threshold: usize,
) -> Result<Core<T>, String> {
    let nn = n
        .checked_mul(n)
        .ok_or("compact tridiagonal size overflow")?;
    let ranges = [
        buffer::<T>(input, nn)?,
        buffer::<T::Real>(d, n)?,
        buffer::<T::Real>(e, n.saturating_sub(1))?,
    ];
    for i in 0..3 {
        for j in i + 1..3 {
            let (a, b) = ranges[i];
            let (c, d) = ranges[j];
            if a < b && c < d && a < d && c < b {
                return Err("compact tridiagonal buffers overlap".into());
            }
        }
    }
    if threshold == usize::MAX {
        par = Par::Seq;
    }
    let stack = work.stack(requirement::<T>(n, bs, par.shape(), threshold)?)?;
    let b = block_size::<T>(n, bs)?.max(1);
    let k = n.saturating_sub(1);
    // SAFETY: input is a preserved column-major n*n buffer; spans were checked.
    let ptr = if n == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr().cast_const()
    } else {
        input.cast::<T>()
    };
    let a = MatRef::from_raw_parts(ptr, n, n, 1, n as isize);
    crate::typed::finite(a)?;
    for j in 0..n {
        for i in j..n {
            if a[(i, j)] != conj(&a[(j, i)]) {
                return Err("compact tridiagonal input must be Hermitian".into());
            }
        }
    }
    let mut core = Core {
        packed: Mat::from_fn(n, n, |i, j| copy(&a[(i, j)])),
        factor: Mat::zeros(b, k),
        phases: (0..n).map(|_| one::<T>()).collect(),
    };
    if k > 0 {
        let scaling = crate::schur::scale_in(core.packed.as_mut());
        evd::tridiag::tridiag_in_place(
            core.packed.as_mut(),
            core.factor.as_mut(),
            par.reborrow(),
            stack,
            params::<T>(threshold),
        );
        if !T::IS_REAL {
            let mut phase = one::<T>();
            for i in 1..n {
                let value = &core.packed[(i, i - 1)];
                let norm = phase_magnitude(value);
                if norm != zero::<T::Real>() {
                    phase = &phase * normalize_phase::<T>(value, &norm);
                    phase = normalize_phase::<T>(&phase, &phase_magnitude(&phase));
                }
                core.phases[i] = copy(&phase);
            }
        }
        if let Some(s) = scaling {
            let inv = recip(&s);
            for i in 0..n {
                core.packed[(i, i)] = mul_real(&core.packed[(i, i)], &inv);
                if i + 1 < n {
                    core.packed[(i + 1, i)] = mul_real(&core.packed[(i + 1, i)], &inv);
                }
            }
        }
    }
    // Validate all returned coefficients before publishing either output band.
    for i in 0..n {
        if !is_finite(&core.packed[(i, i)]) || !is_finite(&core.phases[i]) {
            return Err("compact tridiagonalization overflowed the working type".into());
        }
        if i + 1 < n && !is_finite(&core.packed[(i + 1, i)]) {
            return Err("compact tridiagonalization overflowed the working type".into());
        }
    }
    for j in 0..k {
        for i in j + 2..n {
            if !is_finite(&core.packed[(i, j)]) {
                return Err("nonfinite compact tridiagonal reflector".into());
            }
        }
    }
    // Finite complex components can have an unrepresentable real magnitude.
    validate_real_bands(core.packed.as_ref(), None)?;
    // SAFETY: disjoint preserved d/e buffers were checked before computation.
    for i in 0..n {
        d.cast::<T::Real>().add(i).write(real(&core.packed[(i, i)]));
        if i + 1 < n {
            e.cast::<T::Real>()
                .add(i)
                .write(real_subdiagonal(&core.packed[(i + 1, i)]));
        }
    }
    Ok(core)
}
fn build_data(
    work: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    d: *mut u8,
    e: *mut u8,
    tag: i32,
    bs: usize,
    par: Par,
    threshold: usize,
) -> Result<Data, String> {
    // SAFETY: every branch validates the scalar-specific input/output spans.
    unsafe {
        Ok(match tag {
            0 => Data::F32(build(work, input, n, d, e, bs, par, threshold)?),
            1 => Data::F64(build(work, input, n, d, e, bs, par, threshold)?),
            2 => Data::C32(build(work, input, n, d, e, bs, par, threshold)?),
            3 => Data::C64(build(work, input, n, d, e, bs, par, threshold)?),
            _ => return Err("unsupported compact tridiagonal scalar".into()),
        })
    }
}
#[julia]
impl OwnedCompactTridiagonal {
    #[julia]
    pub fn reduce(
        work: &mut OwnedScratch,
        input: *const u8,
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
                    data: build_data(work, input, n, d, e, tag, bs, par, threshold)?,
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
impl OwnedCompactTridiagonal {
    fn validate(&self, n: usize, k: usize, b: usize, tag: i32) -> Result<(), String> {
        let (actual_n, actual_b, actual_tag) = match &self.data {
            Data::F32(s) => (s.n(), s.b(), 0),
            Data::F64(s) => (s.n(), s.b(), 1),
            Data::C32(s) => (s.n(), s.b(), 2),
            Data::C64(s) => (s.n(), s.b(), 3),
        };
        if n != actual_n || k != n.saturating_sub(1) || b != actual_b || tag != actual_tag {
            return Err("compact Q native metadata does not match requested scalar/shape".into());
        }
        Ok(())
    }
}
#[julia]
impl OwnedCompactTridiagonal {
    #[julia]
    pub fn compact_tridiagonal_validate(
        &self,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| self.validate(n, k, b, tag))
    }
    #[julia]
    pub fn compact_tridiagonal_apply_requirements(
        &self,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
        target_tag: i32,
        extent: usize,
        right: bool,
        transpose: bool,
        out: &mut [usize],
    ) -> Result<(), String> {
        let out = crate::workspace::requirement_out(out)?;
        guarded(|| {
            self.validate(n, k, b, tag)?;
            buffer::<usize>(out.cast(), 2)?;
            let req = match target_tag {
                0 => householder_sequence::layout_requirement::<f32>(
                    k, k, b, extent, right, transpose,
                ),
                1 => householder_sequence::layout_requirement::<f64>(
                    k, k, b, extent, right, transpose,
                ),
                2 => householder_sequence::layout_requirement::<c32>(
                    k, k, b, extent, right, transpose,
                ),
                3 => householder_sequence::layout_requirement::<c64>(
                    k, k, b, extent, right, transpose,
                ),
                _ => return Err("unsupported compact Q workspace scalar".into()),
            };
            // SAFETY: preserved two-entry output, checked above.
            unsafe {
                out.write(req.size_bytes());
                out.add(1).write(req.align_bytes());
            }
            Ok(())
        })
    }
    #[julia]
    pub fn compact_tridiagonal_component(
        &self,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
        kind: i32,
        out: *mut u8,
        len: usize,
    ) -> Result<(), String> {
        guarded(|| {
            self.validate(n, k, b, tag)?;
            unsafe fn write<T: ComplexField>(
                s: &Core<T>,
                kind: i32,
                out: *mut u8,
                len: usize,
            ) -> Result<(), String> {
                let k = s.k();
                let b = s.b();
                let expected = match kind {
                    0 => k.checked_mul(k),
                    1 => b.checked_mul(k),
                    2 => Some(s.n()),
                    _ => return Err("unknown compact tridiagonal component".into()),
                }
                .ok_or("compact component size overflow")?;
                if len != expected {
                    return Err("compact component length mismatch".into());
                }
                buffer::<T>(out, len)?;
                match kind {
                    0 => {
                        for j in 0..k {
                            for i in 0..k {
                                out.cast::<T>().add(i + j * k).write(if i < j {
                                    zero::<T>()
                                } else if i == j {
                                    one::<T>()
                                } else if real(&s.factor[(j % b, j)]) == infinity::<T::Real>() {
                                    zero::<T>()
                                } else {
                                    copy(&s.packed[(i + 1, j)])
                                });
                            }
                        }
                    }
                    1 => {
                        for j in 0..k {
                            for i in 0..b {
                                out.cast::<T>().add(i + j * b).write(if i <= j % b {
                                    copy(&s.factor[(i, j)])
                                } else {
                                    zero::<T>()
                                });
                            }
                        }
                    }
                    _ => {
                        for i in 0..s.n() {
                            out.cast::<T>().add(i).write(if s.phases.is_empty() {
                                one::<T>()
                            } else {
                                copy(&s.phases[i])
                            });
                        }
                    }
                }
                Ok(())
            }
            // SAFETY: each branch checks length, address span and alignment before writing.
            unsafe {
                match &self.data {
                    Data::F32(s) => write(s, kind, out, len),
                    Data::F64(s) => write(s, kind, out, len),
                    Data::C32(s) => write(s, kind, out, len),
                    Data::C64(s) => write(s, kind, out, len),
                }
            }
        })
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn compact_tridiagonal_workspace(
        &mut self,
        source: &OwnedCompactTridiagonal,
        work: &mut OwnedScratch,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugated: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                source.validate(n, k, b, tag)?;
                match &source.data {
                    Data::F64(s) => s.apply(
                        self.matrix.as_mut(),
                        work,
                        right,
                        transpose,
                        conjugated,
                        par,
                    ),
                    _ => Err("compact tridiagonal scalar types must match".into()),
                }
            })
        })
    }
}

#[julia]
impl OwnedMatrix {}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn compact_tridiagonal_workspace(
        &mut self,
        source: &OwnedCompactTridiagonal,
        work: &mut OwnedScratch,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugated: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                source.validate(n, k, b, tag)?;
                match (
                    &source.data,
                    self.data.as_mut().ok_or("matrix is not initialized")?,
                ) {
                    (Data::F32(s), MatrixData::F32(a)) => {
                        s.apply(a.as_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::F64(s), MatrixData::F64(a)) => {
                        s.apply(a.as_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::C32(s), MatrixData::C32(a)) => {
                        s.apply(a.as_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::C64(s), MatrixData::C64(a)) => {
                        s.apply(a.as_mut(), work, right, transpose, conjugated, par)
                    }
                    _ => Err("compact tridiagonal scalar types must match".into()),
                }
            })
        })
    }
}

#[julia]
impl TypedMatrix {}

// Promote only Q's meaningful data. Packed H coefficients can overflow in a
// narrower scalar even when the unitary Q is perfectly representable.
fn cast_core<T: ComplexField, U: ComplexField>(
    s: &Core<T>,
    convert: impl Fn(&T) -> Result<U, String>,
) -> Result<Core<U>, String> {
    let mut out = Core {
        packed: Mat::zeros(s.n(), s.n()),
        factor: Mat::zeros(s.b(), s.k()),
        phases: Vec::with_capacity(s.phases.len()),
    };
    for j in 0..s.k() {
        let active = real(&s.factor[(j % s.b(), j)]) != infinity::<T::Real>();
        if active {
            for i in j + 2..s.n() {
                out.packed[(i, j)] = convert(&s.packed[(i, j)])?;
            }
        }
        for i in 0..=j % s.b() {
            out.factor[(i, j)] = convert(&s.factor[(i, j)])?;
        }
    }
    for p in &s.phases {
        out.phases.push(convert(p)?);
    }
    Ok(out)
}

#[julia]
impl OwnedCompactTridiagonal {
    #[julia]
    pub fn converted(source: &Self, tag: i32) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                data: match (&source.data, tag) {
                    (Data::F32(s), 0) => Data::F32(cast_core(s, |v: &f32| {
                        let out = (*v) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 1) => Data::F64(cast_core(s, |v: &f32| {
                        let out = (*v) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 2) => Data::C32(cast_core(s, |v: &f32| {
                        let out = c32::new((*v) as f32, (0.0) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F32(s), 3) => Data::C64(cast_core(s, |v: &f32| {
                        let out = c64::new((*v) as f64, (0.0) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 0) => Data::F32(cast_core(s, |v: &f64| {
                        let out = (*v) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 1) => Data::F64(cast_core(s, |v: &f64| {
                        let out = (*v) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 2) => Data::C32(cast_core(s, |v: &f64| {
                        let out = c32::new((*v) as f32, (0.0) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::F64(s), 3) => Data::C64(cast_core(s, |v: &f64| {
                        let out = c64::new((*v) as f64, (0.0) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 0) => Data::F32(cast_core(s, |v: &c32| {
                        if v.im != 0.0 {
                            return Err("cannot convert complex compact Q to real scalar".into());
                        }
                        let out = (v.re) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 1) => Data::F64(cast_core(s, |v: &c32| {
                        if v.im != 0.0 {
                            return Err("cannot convert complex compact Q to real scalar".into());
                        }
                        let out = (v.re) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 2) => Data::C32(cast_core(s, |v: &c32| {
                        let out = c32::new((v.re) as f32, (v.im) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C32(s), 3) => Data::C64(cast_core(s, |v: &c32| {
                        let out = c64::new((v.re) as f64, (v.im) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 0) => Data::F32(cast_core(s, |v: &c64| {
                        if v.im != 0.0 {
                            return Err("cannot convert complex compact Q to real scalar".into());
                        }
                        let out = (v.re) as f32;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 1) => Data::F64(cast_core(s, |v: &c64| {
                        if v.im != 0.0 {
                            return Err("cannot convert complex compact Q to real scalar".into());
                        }
                        let out = (v.re) as f64;
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 2) => Data::C32(cast_core(s, |v: &c64| {
                        let out = c32::new((v.re) as f32, (v.im) as f32);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    (Data::C64(s), 3) => Data::C64(cast_core(s, |v: &c64| {
                        let out = c64::new((v.re) as f64, (v.im) as f64);
                        if is_finite(v) && !is_finite(&out) {
                            return Err("compact Q scalar conversion overflow".into());
                        }
                        Ok(out)
                    })?),
                    _ => return Err("unsupported compact Q conversion scalar".into()),
                },
            })
        })
    }
}

#[julia]
impl OwnedCompactTridiagonal {
    #[julia]
    pub fn compact_tridiagonal_info(&self, out: *mut usize) -> Result<(), String> {
        guarded(|| {
            buffer::<usize>(out.cast(), 4)?;
            let (n, b, tag) = match &self.data {
                Data::F32(s) => (s.n(), s.b(), 0),
                Data::F64(s) => (s.n(), s.b(), 1),
                Data::C32(s) => (s.n(), s.b(), 2),
                Data::C64(s) => (s.n(), s.b(), 3),
            };
            // SAFETY: the caller preserves a writable four-entry UInt buffer.
            unsafe {
                out.write(n);
                out.add(1).write(n.saturating_sub(1));
                out.add(2).write(b);
                out.add(3).write(tag);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reject_real_band_overflow<T: ComplexField>(value: T) {
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for n in [2, 3] {
            let early = mul_real(
                &one::<T>(),
                &mul(&real(&value), &from_f64::<T::Real>(0.125)),
            );
            let input: Vec<T> = (0..n * n)
                .map(|index| {
                    let (i, j) = (index % n, index / n);
                    if i == j + 1 {
                        copy(if j + 2 == n { &value } else { &early })
                    } else if j == i + 1 {
                        conj(if i + 2 == n { &value } else { &early })
                    } else {
                        zero::<T>()
                    }
                })
                .collect();
            for use_team in [false, true] {
                let run = |par: Par<'_>| {
                    let req = requirement::<T>(n, 1, par.shape(), 0).unwrap();
                    let mut work = OwnedScratch::new();
                    work.scratch_reserve(req.size_bytes(), req.align_bytes())
                        .unwrap();
                    let mut diagonal: Vec<T::Real> = (0..n).map(|_| from_f64(17.0)).collect();
                    let mut off: Vec<T::Real> = (0..n - 1).map(|_| from_f64(23.0)).collect();
                    let before_d = diagonal.clone();
                    let before_e = off.clone();
                    // SAFETY: initialized, disjoint numeric buffers remain alive for the call.
                    let error = unsafe {
                        build::<T>(
                            &mut work,
                            input.as_ptr().cast(),
                            n,
                            diagonal.as_mut_ptr().cast(),
                            off.as_mut_ptr().cast(),
                            1,
                            par,
                            0,
                        )
                    }
                    .err()
                    .expect("unrepresentable real band must fail");
                    assert!(error.contains("real bands"), "{error}");
                    assert_eq!(diagonal, before_d);
                    assert_eq!(off, before_e);
                };
                if use_team {
                    team.execute(run).unwrap();
                } else {
                    run(Par::Seq);
                }
            }
        }
    }
    #[test]
    fn compact_real_band_overflow_c32_preserves_outputs() {
        reject_real_band_overflow(c32::new(3.0e38, 3.0e38));
    }
    #[test]
    fn compact_real_band_overflow_c64_preserves_outputs() {
        reject_real_band_overflow(c64::new(1.3e308, 1.3e308));
    }
    fn tuned_hessenberg<T: ComplexField>(tolerance: f64, imaginary: T) {
        use crate::schur::hessenberg_tuning::Policy;
        let mut team = OwnedTeam::new();
        team.team_initialize(2).unwrap();
        for n in [0, 1, 2, 17, 65] {
            let input: Vec<T> = (0..n * n)
                .map(|k| {
                    add(
                        &from_f64::<T>(((k * 7 + 3) % 19) as f64 / 19.0),
                        &mul(&imaginary, &from_f64::<T>(((k * 3 + 2) % 13) as f64 / 13.0)),
                    )
                })
                .collect();
            let a = Mat::from_fn(n, n, |i, j| copy(&input[i + j * n]));
            for parallel in [0, 1, usize::MAX - 1, usize::MAX] {
                for blocking in [0, 256 * 256, usize::MAX] {
                    let policy = Policy { parallel, blocking };
                    for use_team in [false, true] {
                        let run = |par: Par<'_>| {
                            let req = hessenberg_req_tuned::<T>(n, par.shape(), policy).unwrap();
                            let mut work = OwnedScratch::new();
                            work.scratch_reserve(req.size_bytes(), req.align_bytes())
                                .unwrap();
                            let mut output: Vec<T> = (0..n * n).map(|_| zero()).collect();
                            // SAFETY: disjoint contiguous numeric buffers remain alive throughout.
                            let core = unsafe {
                                build_hessenberg_tuned::<T>(
                                    &mut work,
                                    input.as_ptr().cast(),
                                    n,
                                    output.as_mut_ptr().cast(),
                                    par,
                                    policy,
                                )
                            }
                            .unwrap();
                            let h = Mat::from_fn(n, n, |i, j| copy(&output[i + j * n]));
                            let req = core.requirement(n, false, false);
                            work.scratch_reserve(req.size_bytes(), req.align_bytes())
                                .unwrap();
                            let mut q = Mat::<T>::identity(n, n);
                            core.apply(q.as_mut(), &mut work, false, false, false, Par::Seq)
                                .unwrap();
                            assert!(
                                (&q * &h * q.adjoint() - &a).norm_max()
                                    <= from_f64::<T::Real>(tolerance)
                            );
                            assert!(
                                (q.adjoint() * &q - Mat::<T>::identity(n, n)).norm_max()
                                    <= from_f64::<T::Real>(tolerance)
                            );
                        };
                        if use_team {
                            team.execute(run).unwrap();
                        } else {
                            run(Par::Seq);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn compact_tuned_hessenberg_four_scalars() {
        tuned_hessenberg::<f32>(2e-4, 0.0);
        tuned_hessenberg::<f64>(2e-12, 0.0);
        tuned_hessenberg::<c32>(2e-4, c32::new(0.0, 1.0));
        tuned_hessenberg::<c64>(2e-12, c64::new(0.0, 1.0));
    }
    fn check<T: ComplexField>(a: Mat<T>, tag: i32, tolerance: f64) {
        let n = a.nrows();
        let mut d: Vec<T::Real> = (0..n).map(|_| zero()).collect();
        let mut e: Vec<T::Real> = (0..n.saturating_sub(1)).map(|_| zero()).collect();
        let mut work = OwnedScratch::new();
        let req = requirement::<T>(n, 1, ParShape::Seq, 0).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let input: Vec<T> = (0..n)
            .flat_map(|j| {
                (0..n).map({
                    let a = &a;
                    move |i| copy(&a[(i, j)])
                })
            })
            .collect();
        // SAFETY: all arrays have disjoint, correctly typed contiguous storage.
        let core = unsafe {
            build::<T>(
                &mut work,
                input.as_ptr().cast(),
                n,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                1,
                Par::Seq,
                0,
            )
        }
        .unwrap();
        assert_eq!(core.n(), n);
        assert_eq!(core.k(), n.saturating_sub(1));
        let req = core.requirement(n, false, false);
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let mut q = Mat::from_fn(n, n, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        core.apply(q.as_mut(), &mut work, false, false, false, Par::Seq)
            .unwrap();
        for j in 0..n {
            for i in 0..n {
                let mut reconstructed = zero::<T>();
                let mut orthogonal = zero::<T>();
                for r in 0..n {
                    reconstructed = add(
                        &reconstructed,
                        &mul(&mul_real(&q[(i, r)], &d[r]), &conj(&q[(j, r)])),
                    );
                    orthogonal = add(&orthogonal, &mul(&conj(&q[(r, i)]), &q[(r, j)]));
                    if r + 1 < n {
                        reconstructed = add(
                            &reconstructed,
                            &mul(&mul_real(&q[(i, r)], &e[r]), &conj(&q[(j, r + 1)])),
                        );
                        reconstructed = add(
                            &reconstructed,
                            &mul(&mul_real(&q[(i, r + 1)], &e[r]), &conj(&q[(j, r)])),
                        );
                    }
                }
                assert!(
                    abs(&sub(&reconstructed, &a[(i, j)])) <= from_f64::<T::Real>(tolerance),
                    "reconstruction tag={tag} n={n}"
                );
                assert!(
                    abs(&sub(
                        &orthogonal,
                        &if i == j { one::<T>() } else { zero::<T>() }
                    )) <= from_f64::<T::Real>(tolerance),
                    "unitarity tag={tag} n={n}"
                );
            }
        }
    }
    #[test]
    fn compact_native_four_scalars_and_empty_boundaries() {
        for n in [0, 1, 2, 5] {
            check(
                Mat::from_fn(n, n, |i, j| if i == j { (i + 1) as f32 } else { 0.25 }),
                0,
                2e-5,
            );
            check(
                Mat::from_fn(n, n, |i, j| if i == j { (i + 1) as f64 } else { 0.25 }),
                1,
                1e-12,
            );
            check(
                Mat::from_fn(n, n, |i, j| {
                    c32::new(
                        if i == j { (i + 1) as f32 } else { 0.25 },
                        if i > j {
                            0.4
                        } else if i < j {
                            -0.4
                        } else {
                            0.0
                        },
                    )
                }),
                2,
                2e-5,
            );
            check(
                Mat::from_fn(n, n, |i, j| {
                    c64::new(
                        if i == j { (i + 1) as f64 } else { 0.25 },
                        if i > j {
                            0.4
                        } else if i < j {
                            -0.4
                        } else {
                            0.0
                        },
                    )
                }),
                3,
                1e-12,
            );
        }
    }
    #[test]
    fn compact_native_failure_does_not_publish_band() {
        let input = [1.0f64, 1.0, 2.0, 3.0];
        let mut d = [71.0f64; 2];
        let mut e = [72.0f64];
        let mut work = OwnedScratch::new();
        let req = requirement::<f64>(2, 1, ParShape::Seq, 0).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // SAFETY: preserved disjoint buffers; deliberately non-Hermitian input.
        let err = unsafe {
            build::<f64>(
                &mut work,
                input.as_ptr().cast(),
                2,
                d.as_mut_ptr().cast(),
                e.as_mut_ptr().cast(),
                1,
                Par::Seq,
                0,
            )
        }
        .err()
        .unwrap();
        assert!(err.contains("Hermitian"));
        assert_eq!(d, [71.0; 2]);
        assert_eq!(e, [72.0]);
    }

    fn check_hessenberg<T: ComplexField>(a: Mat<T>, tolerance: f64) {
        let n = a.nrows();
        let input: Vec<T> = (0..n)
            .flat_map(|j| {
                (0..n).map({
                    let a = &a;
                    move |i| copy(&a[(i, j)])
                })
            })
            .collect();
        let mut band: Vec<T> = (0..n * n).map(|_| zero()).collect();
        let mut work = OwnedScratch::new();
        let req = hessenberg_req::<T>(n, ParShape::Seq).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        // SAFETY: disjoint, correctly typed, column-major numeric buffers.
        let core = unsafe {
            build_hessenberg::<T>(
                &mut work,
                input.as_ptr().cast(),
                n,
                band.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .unwrap();
        assert!(core.phases.is_empty());
        let h = Mat::from_fn(n, n, |i, j| copy(&band[i + j * n]));
        for j in 0..n {
            for i in j + 2..n {
                assert_eq!(abs(&h[(i, j)]), zero::<T::Real>());
            }
        }
        let req = core.requirement(n, false, false);
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let mut q = Mat::from_fn(n, n, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        core.apply(q.as_mut(), &mut work, false, false, false, Par::Seq)
            .unwrap();
        assert!((&q * &h * q.adjoint() - &a).norm_l2() <= from_f64::<T::Real>(tolerance));
        let identity = Mat::from_fn(n, n, |i, j| if i == j { one::<T>() } else { zero::<T>() });
        assert!((q.adjoint() * &q - identity).norm_l2() <= from_f64::<T::Real>(tolerance));
    }

    #[test]
    fn compact_hessenberg_four_scalars_and_boundaries() {
        for n in [0, 1, 2, 5, 33] {
            check_hessenberg(
                Mat::from_fn(n, n, |i, j| (1 + i + 3 * j) as f32 / 17.0),
                0.01,
            );
            check_hessenberg(
                Mat::from_fn(n, n, |i, j| (1 + i + 3 * j) as f64 / 17.0),
                1e-10,
            );
            check_hessenberg(
                Mat::from_fn(n, n, |i, j| {
                    c32::new((1 + i + 3 * j) as f32 / 17.0, (2 * i + j) as f32 / 19.0)
                }),
                0.01,
            );
            check_hessenberg(
                Mat::from_fn(n, n, |i, j| {
                    c64::new((1 + i + 3 * j) as f64 / 17.0, (2 * i + j) as f64 / 19.0)
                }),
                1e-10,
            );
        }
    }

    #[test]
    fn compact_hessenberg_failures_preserve_public_output() {
        let mut input = vec![1.0f64; 25];
        let mut output = vec![71.0f64; 25];
        let mut work = OwnedScratch::new();
        // SAFETY: input and output are preserved typed disjoint buffers; the
        // short scratch is rejected before computing or publishing any H.
        assert!(unsafe {
            build_hessenberg::<f64>(
                &mut work,
                input.as_ptr().cast(),
                5,
                output.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(output, vec![71.0; 25]);
        let req = hessenberg_req::<f64>(5, ParShape::Seq).unwrap();
        work.scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        input[3] = f64::NAN;
        // SAFETY: no invalid reference is formed from the finite-check failure.
        assert!(unsafe {
            build_hessenberg::<f64>(
                &mut work,
                input.as_ptr().cast(),
                5,
                output.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(output, vec![71.0; 25]);
        input[3] = 1.0;
        let old = input.clone();
        // SAFETY: alias is presented only as raw pointers and rejected before
        // constructing matrix references or touching either numeric buffer.
        assert!(unsafe {
            build_hessenberg::<f64>(
                &mut work,
                input.as_ptr().cast(),
                5,
                input.as_mut_ptr().cast(),
                Par::Seq,
            )
        }
        .is_err());
        assert_eq!(input, old);
        assert!(hessenberg_req::<f64>(usize::MAX, ParShape::Seq).is_err());
    }
}

#[julia]
impl VectorOwner {
    #[julia]
    pub fn compact_tridiagonal_workspace(
        &mut self,
        source: &OwnedCompactTridiagonal,
        work: &mut OwnedScratch,
        n: usize,
        k: usize,
        b: usize,
        tag: i32,
        right: bool,
        transpose: bool,
        conjugated: bool,
        team: Option<&OwnedTeam>,
    ) -> Result<(), String> {
        guarded(|| {
            with_team(team, |par| {
                source.validate(n, k, b, tag)?;
                match (
                    &source.data,
                    self.data.as_mut().ok_or("matrix is not initialized")?,
                ) {
                    (Data::F32(s), VectorData::F32(a)) => {
                        s.apply(a.as_mat_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::F64(s), VectorData::F64(a)) => {
                        s.apply(a.as_mat_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::C32(s), VectorData::C32(a)) => {
                        s.apply(a.as_mat_mut(), work, right, transpose, conjugated, par)
                    }
                    (Data::C64(s), VectorData::C64(a)) => {
                        s.apply(a.as_mat_mut(), work, right, transpose, conjugated, par)
                    }
                    _ => Err("compact tridiagonal scalar types must match".into()),
                }
            })
        })
    }
}

#[julia]
impl VectorOwner {}

// Only the unit tests build the untuned compact Hessenberg directly.
#[cfg(test)]
fn hessenberg_req<T: ComplexField>(n: usize, par: ParShape) -> Result<StackReq, String> {
    hessenberg_req_tuned::<T>(n, par, crate::schur::hessenberg_tuning::Policy::default())
}
fn hessenberg_req_tuned<T: ComplexField>(
    n: usize,
    par: ParShape,
    policy: crate::schur::hessenberg_tuning::Policy,
) -> Result<StackReq, String> {
    let par = policy.shape(par);
    n.checked_mul(n)
        .and_then(|nn| nn.checked_mul(core::mem::size_of::<T>()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("Hessenberg dimensions overflow")?;
    if n < 2 {
        return Ok(StackReq::EMPTY);
    }
    let b = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1);
    Ok(evd::hessenberg::hessenberg_in_place_scratch::<T>(
        n,
        b,
        par,
        policy.params::<T>(),
    ))
}
// Only the unit tests build the untuned compact Hessenberg directly.
#[cfg(test)]
unsafe fn build_hessenberg<T: ComplexField>(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    output: *mut u8,
    par: Par,
) -> Result<Core<T>, String> {
    build_hessenberg_tuned(
        scratch,
        input,
        n,
        output,
        par,
        crate::schur::hessenberg_tuning::Policy::default(),
    )
}
// Only the unit tests build the untuned compact Hessenberg directly.
#[cfg(test)]
unsafe fn build_hessenberg_tuned<T: ComplexField>(
    scratch: &mut OwnedScratch,
    input: *const u8,
    n: usize,
    output: *mut u8,
    par: Par,
    policy: crate::schur::hessenberg_tuning::Policy,
) -> Result<Core<T>, String> {
    let nn = n.checked_mul(n).ok_or("Hessenberg dimensions overflow")?;
    let (a, b) = buffer::<T>(input, nn)?;
    let (c, d) = buffer::<T>(output, nn)?;
    if a < b && c < d && a < d && c < b {
        return Err("Hessenberg input/output buffers overlap".into());
    }
    let ptr = if n == 0 {
        std::ptr::NonNull::<T>::dangling().as_ptr().cast_const()
    } else {
        input.cast::<T>()
    };
    // SAFETY: preserved column-major numeric input, checked for span/alignment.
    let a = MatRef::from_raw_parts(ptr, n, n, 1, n as isize);
    let (core, h) = reduce_hessenberg(a, scratch, par, policy)?;
    for j in 0..n {
        for i in 0..n {
            output.cast::<T>().add(i + j * n).write(copy(&h[(i, j)]));
        }
    }
    Ok(core)
}
fn reduce_hessenberg<T: ComplexField>(
    a: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    par: Par,
    policy: crate::schur::hessenberg_tuning::Policy,
) -> Result<(Core<T>, Mat<T>), String> {
    if a.nrows() != a.ncols() {
        return Err("Hessenberg input must be square".into());
    }
    let n = a.nrows();
    let mut par = policy.execution(par);
    let stack = scratch.stack(hessenberg_req_tuned::<T>(n, par.shape(), policy)?)?;
    crate::typed::finite(a)?;
    let b = if n < 2 {
        1
    } else {
        faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1)
    };
    let mut core = Core {
        packed: a.to_owned(),
        factor: Mat::zeros(b, n.saturating_sub(1)),
        phases: Vec::new(),
    };
    let scale = crate::schur::scale_in(core.packed.as_mut());
    if n > 1 {
        evd::hessenberg::hessenberg_in_place(
            core.packed.as_mut(),
            core.factor.as_mut(),
            par.reborrow(),
            stack,
            policy.params::<T>(),
        );
    }
    crate::typed::finite(core.packed.as_ref())?;
    for j in 0..core.k() {
        for i in 0..=j % b {
            let value = &core.factor[(i, j)];
            if i == j % b {
                if imag(value) != zero::<T::Real>() || !(real(value) > zero::<T::Real>()) {
                    return Err("invalid Hessenberg Householder denominator".into());
                }
            } else if !is_finite(value) {
                return Err("nonfinite Hessenberg Householder factor".into());
            }
        }
    }
    let inverse = scale.map(|s| recip(&s));
    let value = |i: usize, j: usize| {
        if i > j + 1 {
            zero::<T>()
        } else if let Some(ref s) = inverse {
            mul_real(&core.packed[(i, j)], s)
        } else {
            copy(&core.packed[(i, j)])
        }
    };
    // Validate the whole public H before publishing any output entry. Packed
    // reflector essentials remain in their working scale and are not restored.
    for j in 0..n {
        for i in 0..n {
            if !is_finite(&value(i, j)) {
                return Err("Hessenberg reduction overflowed the working type".into());
            }
        }
    }
    let h = Mat::from_fn(n, n, value);
    Ok((core, h))
}

#[julia]
pub fn compact_hessenberg_tuned_requirements(
    n: usize,
    tag: i32,
    parallel: usize,
    blocking: usize,
    team: Option<&OwnedTeam>,
    out: &mut [usize],
) -> Result<(), String> {
    use crate::schur::hessenberg_tuning::{with_team, Policy};
    guarded(|| {
        with_team(team, |par| {
            if out.len() != 2 {
                return Err("Hessenberg requirements output must have length two".into());
            }
            let policy = Policy { parallel, blocking };
            let req = match tag {
                0 => hessenberg_req_tuned::<f32>(n, par.shape(), policy)?,
                1 => hessenberg_req_tuned::<f64>(n, par.shape(), policy)?,
                2 => hessenberg_req_tuned::<c32>(n, par.shape(), policy)?,
                3 => hessenberg_req_tuned::<c64>(n, par.shape(), policy)?,
                _ => return Err("unsupported Hessenberg scalar".into()),
            };
            out[0] = req.size_bytes();
            out[1] = req.align_bytes();
            Ok(())
        })
    })
}
#[julia]
impl OwnedMatrix {
    #[julia]
    #[rustcall_owned_return(OwnedCompactTridiagonal)]
    pub fn compact_hessenberg_tuned(
        &self,
        out_h: &mut OwnedMatrix,
        scratch: &mut OwnedScratch,
        n: usize,
        tag: i32,
        parallel: usize,
        blocking: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<OwnedCompactTridiagonal, String> {
        use crate::schur::hessenberg_tuning::{with_team, Policy};
        guarded(|| {
            let policy = Policy { parallel, blocking };
            if tag != 1 || self.matrix.shape() != (n, n) {
                return Err("Hessenberg native metadata mismatch".into());
            }
            let (core, h) = with_team(team, |par| {
                reduce_hessenberg(self.matrix.as_ref(), scratch, par, policy)
            })?;
            out_h.matrix = h;
            Ok(OwnedCompactTridiagonal {
                data: Data::F64(core),
            })
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    #[rustcall_owned_return(OwnedCompactTridiagonal)]
    pub fn compact_hessenberg_tuned(
        &self,
        out_h: &mut TypedMatrix,
        scratch: &mut OwnedScratch,
        n: usize,
        tag: i32,
        parallel: usize,
        blocking: usize,
        team: Option<&OwnedTeam>,
    ) -> Result<OwnedCompactTridiagonal, String> {
        use crate::schur::hessenberg_tuning::{with_team, Policy};
        guarded(|| {
            let policy = Policy { parallel, blocking };
            let (data, h) = match self.data()? {
                MatrixData::F32(a) if tag == 0 && a.shape() == (n, n) => {
                    let (core, h) = with_team(team, |par| {
                        reduce_hessenberg(a.as_ref(), scratch, par, policy)
                    })?;
                    (Data::F32(core), MatrixData::F32(h))
                }
                MatrixData::F64(a) if tag == 1 && a.shape() == (n, n) => {
                    let (core, h) = with_team(team, |par| {
                        reduce_hessenberg(a.as_ref(), scratch, par, policy)
                    })?;
                    (Data::F64(core), MatrixData::F64(h))
                }
                MatrixData::C32(a) if tag == 2 && a.shape() == (n, n) => {
                    let (core, h) = with_team(team, |par| {
                        reduce_hessenberg(a.as_ref(), scratch, par, policy)
                    })?;
                    (Data::C32(core), MatrixData::C32(h))
                }
                MatrixData::C64(a) if tag == 3 && a.shape() == (n, n) => {
                    let (core, h) = with_team(team, |par| {
                        reduce_hessenberg(a.as_ref(), scratch, par, policy)
                    })?;
                    (Data::C64(core), MatrixData::C64(h))
                }
                _ => return Err("Hessenberg native metadata mismatch".into()),
            };
            out_h.data = Some(h);
            Ok(OwnedCompactTridiagonal { data })
        })
    }
}
