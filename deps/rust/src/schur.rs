//! Dense Hessenberg/Schur operations. Outputs are always disjoint new owners.
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{evd, householder};
use faer::mat::{AsMatMut, AsMatRef};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Conj, Mat, MatMut, MatRef, Par, ParShape};
use rustcall_julia_macros::julia;
mod construction;
mod eigenvectors_workspace;
pub(crate) mod hessenberg_tuning;
mod reorder_workspace;

// Power-of-two scaling follows faer's general EVD driver. Keep the
// reduction/QR kernels in their safe range without changing the similarity Q.
pub(crate) fn scale<T: ComplexField>(h: &mut Mat<T>) -> Option<T::Real> {
    scale_in(h.as_mut())
}
pub(crate) fn scale_in<T: ComplexField>(mut h: MatMut<'_, T>) -> Option<T::Real> {
    let mut anrm = zero::<T::Real>();
    for j in 0..h.ncols() {
        for i in 0..h.nrows() {
            let re = abs(&real(&h[(i, j)]));
            let im = abs(&imag(&h[(i, j)]));
            if re > anrm {
                anrm = re;
            }
            if im > anrm {
                anrm = im;
            }
        }
    }
    let small = div(&sqrt(&min_positive::<T::Real>()), &eps::<T::Real>());
    let big = recip(&small);
    if anrm == zero::<T::Real>() || (anrm >= small && anrm <= big) {
        return None;
    }
    let mut f = one::<T::Real>();
    if anrm < small {
        while mul(&anrm, &f) < small {
            f = mul(&f, &from_f64::<T::Real>(2.0));
        }
    } else {
        while mul(&anrm, &f) > big {
            f = mul(&f, &from_f64::<T::Real>(0.5));
        }
    }
    for j in 0..h.ncols() {
        for i in 0..h.nrows() {
            h[(i, j)] = mul_real(&h[(i, j)], &f);
        }
    }
    Some(f)
}
pub(crate) fn restore<T: ComplexField>(h: &mut Mat<T>, factor: Option<T::Real>) {
    if let Some(f) = factor {
        let inv = recip(&f);
        for j in 0..h.ncols() {
            for i in 0..h.nrows() {
                h[(i, j)] = mul_real(&h[(i, j)], &inv);
            }
        }
    }
}

fn hessenberg<T: ComplexField>(a: MatRef<'_, T>) -> Result<(Mat<T>, Mat<T>), String> {
    let mut mem = MemBuffer::try_new(construction::hessenberg_requirement::<T>(
        a.nrows(),
        ParShape::Seq,
    ))
    .map_err(|e| format!("{e:?}"))?;
    hessenberg_in(a, Par::Seq, MemStack::new(&mut mem))
}
fn hessenberg_in<T: ComplexField>(
    a: MatRef<'_, T>,
    par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<T>, Mat<T>), String> {
    hessenberg_in_tuned(a, par, stack, hessenberg_tuning::Policy::default())
}
pub(crate) fn hessenberg_in_tuned<T: ComplexField>(
    a: MatRef<'_, T>,
    par: Par,
    stack: &mut MemStack,
    policy: hessenberg_tuning::Policy,
) -> Result<(Mat<T>, Mat<T>), String> {
    let mut par = policy.execution(par);
    if a.nrows() != a.ncols() {
        return Err("Hessenberg/Schur input must be square".into());
    }
    finite(a)?;
    let n = a.nrows();
    let mut h = a.to_owned();
    let mut z = Mat::identity(n, n);
    if n <= 1 {
        return Ok((h, z));
    }
    let scaling = scale(&mut h);
    let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T>(n - 1, n - 1);
    let (mut hh, stack) = faer::linalg::temp_mat_zeroed::<T, _, _>(bs, n - 1, stack);
    evd::hessenberg::hessenberg_in_place(
        h.as_mut(),
        hh.as_mat_mut(),
        par.reborrow(),
        stack,
        policy.params::<T>(),
    );
    householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
        h.as_ref().submatrix(1, 0, n - 1, n - 1),
        hh.as_mat_ref(),
        Conj::No,
        z.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
        par.reborrow(),
        stack,
    );
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = zero::<T>();
        }
    }
    restore(&mut h, scaling);
    finite(h.as_ref())?;
    Ok((h, z))
}
trait SchurScalar: ComplexField + Sized {
    type Value;
    fn step_scratch(n: usize, par: ParShape) -> StackReq;
    fn step_in(
        h: &mut Mat<Self>,
        z: &mut Mat<Self>,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<(), String>;
    fn step(h: &mut Mat<Self>, z: &mut Mat<Self>) -> Result<(), String> {
        let mut mem = MemBuffer::try_new(Self::step_scratch(h.nrows(), ParShape::Seq))
            .map_err(|e| format!("{e:?}"))?;
        Self::step_in(h, z, Par::Seq, MemStack::new(&mut mem))
    }
    fn move_block(
        h: &mut Mat<Self>,
        z: &mut Mat<Self>,
        from: usize,
        to: usize,
        par: Par,
        stack: &mut MemStack,
    ) -> Result<(), String>;
    fn block_size(h: MatRef<'_, Self>, i: usize) -> usize;
    fn validate(h: MatRef<'_, Self>) -> Result<(), String>;
    unsafe fn values(h: MatRef<'_, Self>, out: *mut u8);
}
macro_rules! real_scalar {
    ($r:ty,$c:ty,$variant:ident) => {
        impl SchurScalar for $r {
            type Value = $c;
            fn step_scratch(n: usize, par: ParShape) -> StackReq {
                if n == 0 {
                    return StackReq::EMPTY;
                }
                StackReq::new::<Self>(n)
                    .array(2)
                    .and(evd::hessenberg_schur_scratch::<Self>(
                        n,
                        true,
                        par,
                        Default::default(),
                    ))
            }
            fn step_in(
                h: &mut Mat<Self>,
                z: &mut Mat<Self>,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                let scaling = scale(h);
                let n = h.nrows();
                if n == 0 {
                    return Ok(());
                }
                let (mut wr, stack) = stack.make_with::<Self>(n, |_| Self::NAN);
                let (mut wi, stack) = stack.make_with::<Self>(n, |_| Self::NAN);
                evd::hessenberg_schur_real(
                    h.as_mut(),
                    Some(z.as_mut()),
                    faer::ColMut::from_slice_mut(&mut wr),
                    faer::ColMut::from_slice_mut(&mut wi),
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("{e:?}"))?;
                restore(h, scaling);
                finite(h.as_ref())?;
                finite(z.as_ref())?;
                Ok(())
            }
            fn move_block(
                h: &mut Mat<Self>,
                z: &mut Mat<Self>,
                from: usize,
                to: usize,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                evd::reorder_schur_real(h.as_mut(), Some(z.as_mut()), from, to, par, stack)
                    .map(|_| ())
                    .map_err(|e| format!("Schur reorder rejected: {e:?}"))
            }
            fn block_size(h: MatRef<'_, Self>, i: usize) -> usize {
                if i + 1 < h.nrows() && h[(i + 1, i)] != 0.0 {
                    2
                } else {
                    1
                }
            }
            fn validate(h: MatRef<'_, Self>) -> Result<(), String> {
                let n = h.nrows();
                let mut i = 0;
                for j in 0..n {
                    for k in j + 2..n {
                        if h[(k, j)] != 0.0 {
                            return Err("matrix is not in real Schur form".into());
                        }
                    }
                }
                while i < n {
                    let b = Self::block_size(h, i);
                    if b == 2
                        && (h[(i, i)] != h[(i + 1, i + 1)]
                            || h[(i, i + 1)] == 0.0
                            || (h[(i + 1, i)] > 0.0) == (h[(i, i + 1)] > 0.0)
                            || (i + 2 < n && h[(i + 2, i + 1)] != 0.0))
                    {
                        return Err(
                            "real Schur 2x2 blocks must be standardized conjugate pairs".into()
                        );
                    }
                    i += b;
                }
                Ok(())
            }
            unsafe fn values(h: MatRef<'_, Self>, out: *mut u8) {
                let mut i = 0;
                let out = out.cast::<$c>();
                while i < h.nrows() {
                    if Self::block_size(h, i) == 2 {
                        let re = h[(i, i)];
                        let im = h[(i, i + 1)].abs().sqrt() * h[(i + 1, i)].abs().sqrt();
                        out.add(i).write(<$c>::new(re, im));
                        out.add(i + 1).write(<$c>::new(re, -im));
                        i += 2;
                    } else {
                        out.add(i).write(<$c>::new(h[(i, i)], 0.0));
                        i += 1;
                    }
                }
            }
        }
    };
}
macro_rules! complex_scalar {
    ($c:ty,$variant:ident) => {
        impl SchurScalar for $c {
            type Value = $c;
            fn step_scratch(n: usize, par: ParShape) -> StackReq {
                if n == 0 {
                    return StackReq::EMPTY;
                }
                StackReq::new::<Self>(n).and(evd::hessenberg_schur_scratch::<Self>(
                    n,
                    true,
                    par,
                    Default::default(),
                ))
            }
            fn step_in(
                h: &mut Mat<Self>,
                z: &mut Mat<Self>,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                let scaling = scale(h);
                let n = h.nrows();
                if n == 0 {
                    return Ok(());
                }
                let (mut w, stack) =
                    stack.make_with::<Self>(n, |_| Self::new(f64::NAN as _, f64::NAN as _));
                evd::hessenberg_schur_cplx(
                    h.as_mut(),
                    Some(z.as_mut()),
                    faer::ColMut::from_slice_mut(&mut w),
                    par,
                    stack,
                    Default::default(),
                )
                .map_err(|e| format!("{e:?}"))?;
                restore(h, scaling);
                finite(h.as_ref())?;
                finite(z.as_ref())?;
                Ok(())
            }
            fn move_block(
                h: &mut Mat<Self>,
                z: &mut Mat<Self>,
                from: usize,
                to: usize,
                par: Par,
                stack: &mut MemStack,
            ) -> Result<(), String> {
                evd::reorder_schur_cplx(h.as_mut(), Some(z.as_mut()), from, to, par, stack);
                Ok(())
            }
            fn block_size(_h: MatRef<'_, Self>, _i: usize) -> usize {
                1
            }
            fn validate(h: MatRef<'_, Self>) -> Result<(), String> {
                for j in 0..h.ncols() {
                    for i in j + 1..h.nrows() {
                        if h[(i, j)] != Self::new(0.0, 0.0) {
                            return Err("complex Schur matrix must be upper triangular".into());
                        }
                    }
                }
                Ok(())
            }
            unsafe fn values(h: MatRef<'_, Self>, out: *mut u8) {
                for i in 0..h.nrows() {
                    out.cast::<Self>().add(i).write(h[(i, i)]);
                }
            }
        }
    };
}
real_scalar!(f32, c32, C32);
real_scalar!(f64, c64, C64);
complex_scalar!(c32, C32);
complex_scalar!(c64, C64);

fn validate<T: SchurScalar>(h: MatRef<'_, T>, z: MatRef<'_, T>) -> Result<(), String> {
    if h.nrows() != h.ncols() || h.shape() != z.shape() {
        return Err("Schur T and Z must have matching square shapes".into());
    }
    finite(h)?;
    finite(z)?;
    T::validate(h)
}
fn reordered<T: SchurScalar>(
    h: MatRef<'_, T>,
    z: MatRef<'_, T>,
    select: &[u8],
) -> Result<(Mat<T>, Mat<T>), String> {
    let mut mem = MemBuffer::try_new(reorder_workspace::requirement::<T>(h.nrows()))
        .map_err(|e| format!("{e:?}"))?;
    reordered_in(h, z, select, Par::Seq, MemStack::new(&mut mem))
}
fn reordered_in<T: SchurScalar>(
    h: MatRef<'_, T>,
    z: MatRef<'_, T>,
    select: &[u8],
    mut par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<T>, Mat<T>), String> {
    validate(h, z)?;
    if select.len() != h.nrows() {
        return Err("Schur selection length mismatch".into());
    }
    if select.iter().any(|&v| v > 1) {
        return Err("Schur selection flags must be zero or one".into());
    }
    let (mut flags, stack) = stack.make_with::<u8>(select.len(), |i| select[i]);
    let mut i = 0;
    while i < h.nrows() {
        let b = T::block_size(h, i);
        if b == 2 && (flags[i] != flags[i + 1]) {
            return Err(
                "both members of a real conjugate pair must have the same selection".into(),
            );
        }
        i += b;
    }
    let mut h = h.to_owned();
    let mut z = z.to_owned();
    let n = h.nrows();
    if n == 0 {
        return Ok((h, z));
    }
    let mut here = 0;
    let mut next = 0;
    while here < n {
        let b = T::block_size(h.as_ref(), here);
        if flags[here] != 0 {
            if here != next {
                T::move_block(&mut h, &mut z, here, next, par.reborrow(), stack)?;
                flags[next..here + b].rotate_right(b);
            }
            next += b;
        }
        here += b;
    }
    finite(h.as_ref())?;
    finite(z.as_ref())?;
    Ok((h, z))
}
unsafe fn selection<'a>(p: *const u8, n: usize) -> &'a [u8] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn schur_from(
        &mut self,
        source: &OwnedMatrix,
        z: &mut OwnedMatrix,
        values: *mut u8,
        hessenberg_only: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source;
            let (mut h, mut q) = hessenberg(a.matrix.as_ref())?;
            if !hessenberg_only {
                f64::step(&mut h, &mut q)?;
                unsafe { f64::values(h.as_ref(), values) }
            }
            self.matrix = h;
            z.matrix = q;
            Ok(())
        })
    }
    #[julia]
    pub fn schur_reorder_from(
        &mut self,
        t: &OwnedMatrix,
        z: &OwnedMatrix,
        out_z: &mut OwnedMatrix,
        select: *const u8,
        n: usize,
        values: *mut u8,
    ) -> Result<(), String> {
        guarded(|| {
            let (h, q) = reordered(t.matrix.as_ref(), z.matrix.as_ref(), unsafe {
                selection(select, n)
            })?;
            unsafe {
                f64::values(h.as_ref(), values);
                (out_z).matrix = q;
            }
            self.matrix = h;
            Ok(())
        })
    }
}
macro_rules! typed_schur {
    ($data:expr,$out_z:expr,$values:expr,$only:expr,$x:ident) => {
        match $data {
            MatrixData::F32($x) => {
                let (mut h, mut q) = hessenberg($x.as_ref())?;
                if !$only {
                    f32::step(&mut h, &mut q)?;
                    f32::values(h.as_ref(), $values);
                }
                $out_z.data = Some(MatrixData::F32(q));
                MatrixData::F32(h)
            }
            MatrixData::F64($x) => {
                let (mut h, mut q) = hessenberg($x.as_ref())?;
                if !$only {
                    f64::step(&mut h, &mut q)?;
                    f64::values(h.as_ref(), $values);
                }
                $out_z.data = Some(MatrixData::F64(q));
                MatrixData::F64(h)
            }
            MatrixData::C32($x) => {
                let (mut h, mut q) = hessenberg($x.as_ref())?;
                if !$only {
                    c32::step(&mut h, &mut q)?;
                    c32::values(h.as_ref(), $values);
                }
                $out_z.data = Some(MatrixData::C32(q));
                MatrixData::C32(h)
            }
            MatrixData::C64($x) => {
                let (mut h, mut q) = hessenberg($x.as_ref())?;
                if !$only {
                    c64::step(&mut h, &mut q)?;
                    c64::values(h.as_ref(), $values);
                }
                $out_z.data = Some(MatrixData::C64(q));
                MatrixData::C64(h)
            }
        }
    };
}
macro_rules! typed_reorder {
    ($t:expr,$z:expr,$out_z:expr,$select:expr,$values:expr) => {
        match ($t, $z) {
            (MatrixData::F32(t), MatrixData::F32(z)) => {
                let (h, q) = reordered(t.as_ref(), z.as_ref(), $select)?;
                f32::values(h.as_ref(), $values);
                $out_z.data = Some(MatrixData::F32(q));
                MatrixData::F32(h)
            }
            (MatrixData::F64(t), MatrixData::F64(z)) => {
                let (h, q) = reordered(t.as_ref(), z.as_ref(), $select)?;
                f64::values(h.as_ref(), $values);
                $out_z.data = Some(MatrixData::F64(q));
                MatrixData::F64(h)
            }
            (MatrixData::C32(t), MatrixData::C32(z)) => {
                let (h, q) = reordered(t.as_ref(), z.as_ref(), $select)?;
                c32::values(h.as_ref(), $values);
                $out_z.data = Some(MatrixData::C32(q));
                MatrixData::C32(h)
            }
            (MatrixData::C64(t), MatrixData::C64(z)) => {
                let (h, q) = reordered(t.as_ref(), z.as_ref(), $select)?;
                c64::values(h.as_ref(), $values);
                $out_z.data = Some(MatrixData::C64(q));
                MatrixData::C64(h)
            }
            _ => return Err("Schur scalar types must match".into()),
        }
    };
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_from(
        &mut self,
        source: &TypedMatrix,
        z: &mut TypedMatrix,
        values: *mut u8,
        hessenberg_only: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let a = (source).data()?;

            self.data = Some(typed_schur!(a, z, values, hessenberg_only, x));
            Ok(())
        })
    }
    #[julia]
    pub fn schur_reorder_from(
        &mut self,
        t: &TypedMatrix,
        z: &TypedMatrix,
        out_z: &mut TypedMatrix,
        select: *const u8,
        n: usize,
        values: *mut u8,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let t = (t).data()?;
            let z = (z).data()?;

            self.data = Some(typed_reorder!(t, z, out_z, selection(select, n), values));
            Ok(())
        })
    }
}

fn reduced<T: SchurScalar>(h: MatRef<'_, T>, z: MatRef<'_, T>) -> Result<(Mat<T>, Mat<T>), String> {
    let mut mem = MemBuffer::try_new(T::step_scratch(h.nrows(), ParShape::Seq))
        .map_err(|e| format!("{e:?}"))?;
    reduced_in(h, z, Par::Seq, MemStack::new(&mut mem))
}
fn reduced_with_initial<T: SchurScalar>(
    h: MatRef<'_, T>,
    z: Option<MatRef<'_, T>>,
    par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<T>, Mat<T>), String> {
    if h.nrows() != h.ncols() || z.is_some_and(|z| h.shape() != z.shape()) {
        return Err("H and Q must have matching square dimensions".into());
    }
    finite(h)?;
    if let Some(z) = z {
        finite(z)?;
    }
    for j in 0..h.ncols() {
        for i in j + 2..h.nrows() {
            if h[(i, j)] != zero::<T>() {
                return Err("input is not upper Hessenberg".into());
            }
        }
    }
    let mut h = h.to_owned();
    let mut z = z
        .map(|z| z.to_owned())
        .unwrap_or_else(|| Mat::identity(h.nrows(), h.nrows()));
    T::step_in(&mut h, &mut z, par, stack)?;
    Ok((h, z))
}
fn reduced_in<T: SchurScalar>(
    h: MatRef<'_, T>,
    z: MatRef<'_, T>,
    par: Par,
    stack: &mut MemStack,
) -> Result<(Mat<T>, Mat<T>), String> {
    reduced_with_initial(h, Some(z), par, stack)
}
mod identity_workspace;
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn schur_reduced_from(
        &mut self,
        h: &OwnedMatrix,
        q: &OwnedMatrix,
        out_q: &mut OwnedMatrix,
        values: *mut u8,
    ) -> Result<(), String> {
        guarded(|| {
            let (h, q) = reduced(h.matrix.as_ref(), q.matrix.as_ref())?;
            unsafe {
                f64::values(h.as_ref(), values);
                (out_q).matrix = q;
            }
            self.matrix = h;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn schur_reduced_from(
        &mut self,
        h: &TypedMatrix,
        q: &TypedMatrix,
        out_q: &mut TypedMatrix,
        values: *mut u8,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            let h = (h).data()?;
            let q = (q).data()?;
            let out = out_q;
            macro_rules! finish {
                ($h:expr,$q:expr,$ty:ty,$variant:ident) => {{
                    let (h, q) = reduced($h.as_ref(), $q.as_ref())?;
                    <$ty>::values(h.as_ref(), values);
                    out.data = Some(MatrixData::$variant(q));
                    MatrixData::$variant(h)
                }};
            }
            self.data = Some(match (h, q) {
                (MatrixData::F32(h), MatrixData::F32(q)) => finish!(h, q, f32, F32),
                (MatrixData::F64(h), MatrixData::F64(q)) => finish!(h, q, f64, F64),
                (MatrixData::C32(h), MatrixData::C32(q)) => finish!(h, q, c32, C32),
                (MatrixData::C64(h), MatrixData::C64(q)) => finish!(h, q, c64, C64),
                _ => return Err("Hessenberg scalar types must match".into()),
            });
            Ok(())
        })
    }
}
