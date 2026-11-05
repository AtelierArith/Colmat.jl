//! Owned compact tridiagonal reflectors. No workspace or execution token escapes.
use super::construction::{block_size, normalize_phase, params, phase_magnitude};
use super::workspace::buffer;
use crate::collections::{VectorData, VectorOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::OwnedMatrix;
use crate::{guarded, householder_sequence, teams::OwnedTeam, workspace::OwnedScratch};
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
            return Err("compact tridiagonal application dimensions must match".into());
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
        if right == transpose {
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
        if right != transpose {
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
    out: *mut usize,
) -> Result<(), String> {
    guarded(|| query(n, tag, bs, ParShape::Seq, threshold, out))
}
#[julia]
impl OwnedTeam {
    #[julia]
    pub fn compact_tridiagonal_team_requirements(
        &self,
        n: usize,
        tag: i32,
        bs: usize,
        threshold: usize,
        out: *mut usize,
    ) -> Result<(), String> {
        guarded(|| self.execute(|par| query(n, tag, bs, par.shape(), threshold, out))?)
    }
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
    // SAFETY: disjoint preserved d/e buffers were checked before computation.
    for i in 0..n {
        d.cast::<T::Real>().add(i).write(real(&core.packed[(i, i)]));
        if i + 1 < n {
            e.cast::<T::Real>().add(i).write(if T::IS_REAL {
                real(&core.packed[(i + 1, i)])
            } else {
                phase_magnitude(&core.packed[(i + 1, i)])
            });
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
    ) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                data: build_data(work, input, n, d, e, tag, bs, Par::Seq, threshold)?,
            })
        })
    }
    #[julia]
    pub fn reduce_team(
        work: &mut OwnedScratch,
        input: *const u8,
        n: usize,
        d: *mut u8,
        e: *mut u8,
        tag: i32,
        bs: usize,
        threshold: usize,
        team: &OwnedTeam,
    ) -> Result<Self, String> {
        guarded(|| {
            team.execute(|par| {
                Ok(Self {
                    data: build_data(work, input, n, d, e, tag, bs, par, threshold)?,
                })
            })?
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
            return Err(
                "compact tridiagonal native metadata does not match requested scalar/shape".into(),
            );
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
        out: *mut usize,
    ) -> Result<(), String> {
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
                            out.cast::<T>().add(i).write(copy(&s.phases[i]));
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
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
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
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn compact_tridiagonal_workspace_team(
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
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
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
            })?
        })
    }
}

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
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
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
    }
}

#[julia]
impl TypedMatrix {
    #[julia]
    pub fn compact_tridiagonal_workspace_team(
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
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
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
            })?
        })
    }
}

// Promote only Q's meaningful data. Packed H coefficients can overflow in a
// narrower scalar even when the unitary Q is perfectly representable.
fn cast_core<T: ComplexField, U: ComplexField>(
    s: &Core<T>,
    convert: impl Fn(&T) -> Result<U, String>,
) -> Result<Core<U>, String> {
    let mut out = Core {
        packed: Mat::zeros(s.n(), s.n()),
        factor: Mat::zeros(s.b(), s.k()),
        phases: Vec::with_capacity(s.n()),
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
    ) -> Result<(), String> {
        guarded(|| {
            let par = Par::Seq;
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
    }
}

#[julia]
impl VectorOwner {
    #[julia]
    pub fn compact_tridiagonal_workspace_team(
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
        team: &OwnedTeam,
    ) -> Result<(), String> {
        guarded(|| {
            team.execute(|par| {
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
            })?
        })
    }
}
