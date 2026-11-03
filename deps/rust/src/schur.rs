//! Dense Hessenberg/Schur operations. Outputs are always disjoint new owners.
use crate::typed::{finite, MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{evd, householder};
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, Conj, Mat, MatRef, Par};
use rustcall_julia_macros::julia;

// Power-of-two scaling follows faer's general EVD driver. Keep the
// reduction/QR kernels in their safe range without changing the similarity Q.
pub(crate) fn scale<T: ComplexField>(h: &mut Mat<T>) -> Option<T::Real> {
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
    let mut hh = Mat::zeros(bs, n - 1);
    let req = StackReq::any_of(&[
        evd::hessenberg::hessenberg_in_place_scratch::<T>(
            n,
            bs,
            Par::Seq.shape(),
            Default::default(),
        ),
        householder::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(
            n - 1,
            bs,
            n - 1,
        ),
    ]);
    let mut mem = MemBuffer::new(req);
    evd::hessenberg::hessenberg_in_place(
        h.as_mut(),
        hh.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    );
    householder::apply_block_householder_sequence_on_the_right_in_place_with_conj(
        h.as_ref().submatrix(1, 0, n - 1, n - 1),
        hh.as_ref(),
        Conj::No,
        z.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
        Par::Seq,
        MemStack::new(&mut mem),
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
    fn step(h: &mut Mat<Self>, z: &mut Mat<Self>) -> Result<(), String>;
    fn move_block(
        h: &mut Mat<Self>,
        z: &mut Mat<Self>,
        from: usize,
        to: usize,
        mem: &mut MemBuffer,
    ) -> Result<(), String>;
    fn block_size(h: MatRef<'_, Self>, i: usize) -> usize;
    fn validate(h: MatRef<'_, Self>) -> Result<(), String>;
    unsafe fn values(h: MatRef<'_, Self>, out: *mut u8);
    fn eigenvectors(h: MatRef<'_, Self>, z: MatRef<'_, Self>) -> Result<MatrixData, String>;
}
macro_rules! real_scalar {
    ($r:ty,$c:ty,$variant:ident) => {
        impl SchurScalar for $r {
            fn step(h: &mut Mat<Self>, z: &mut Mat<Self>) -> Result<(), String> {
                let scaling = scale(h);
                let n = h.nrows();
                let mut wr = Col::zeros(n);
                let mut wi = Col::zeros(n);
                let mut mem = MemBuffer::new(evd::hessenberg_schur_scratch::<Self>(
                    n,
                    true,
                    Par::Seq.shape(),
                    Default::default(),
                ));
                evd::hessenberg_schur_real(
                    h.as_mut(),
                    Some(z.as_mut()),
                    wr.as_mut(),
                    wi.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut mem),
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
                mem: &mut MemBuffer,
            ) -> Result<(), String> {
                evd::reorder_schur_real(
                    h.as_mut(),
                    Some(z.as_mut()),
                    from,
                    to,
                    Par::Seq,
                    MemStack::new(mem),
                )
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
            fn eigenvectors(
                h: MatRef<'_, Self>,
                z: MatRef<'_, Self>,
            ) -> Result<MatrixData, String> {
                let n = h.nrows();
                let mut x = Mat::zeros(n, n);
                if n > 0 {
                    let mut mem = MemBuffer::new(evd::eigenvectors_from_schur_scratch::<Self>(n));
                    evd::eigenvectors_from_schur_real(
                        h,
                        Some(z),
                        x.as_mut(),
                        Par::Seq,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|e| format!("{e:?}"))?;
                }
                let mut out = Mat::<$c>::zeros(n, n);
                let mut j = 0;
                while j < n {
                    if Self::block_size(h, j) == 2 {
                        for i in 0..n {
                            out[(i, j)] = <$c>::new(x[(i, j)], x[(i, j + 1)]);
                            out[(i, j + 1)] = <$c>::new(x[(i, j)], -x[(i, j + 1)]);
                        }
                        j += 2;
                    } else {
                        for i in 0..n {
                            out[(i, j)] = <$c>::new(x[(i, j)], 0.0);
                        }
                        j += 1;
                    }
                }
                Ok(MatrixData::$variant(out))
            }
        }
    };
}
macro_rules! complex_scalar {
    ($c:ty,$variant:ident) => {
        impl SchurScalar for $c {
            fn step(h: &mut Mat<Self>, z: &mut Mat<Self>) -> Result<(), String> {
                let scaling = scale(h);
                let n = h.nrows();
                let mut w = Col::zeros(n);
                let mut mem = MemBuffer::new(evd::hessenberg_schur_scratch::<Self>(
                    n,
                    true,
                    Par::Seq.shape(),
                    Default::default(),
                ));
                evd::hessenberg_schur_cplx(
                    h.as_mut(),
                    Some(z.as_mut()),
                    w.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut mem),
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
                mem: &mut MemBuffer,
            ) -> Result<(), String> {
                evd::reorder_schur_cplx(
                    h.as_mut(),
                    Some(z.as_mut()),
                    from,
                    to,
                    Par::Seq,
                    MemStack::new(mem),
                );
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
            fn eigenvectors(
                h: MatRef<'_, Self>,
                z: MatRef<'_, Self>,
            ) -> Result<MatrixData, String> {
                let n = h.nrows();
                let mut x = Mat::zeros(n, n);
                if n > 0 {
                    let mut mem = MemBuffer::new(evd::eigenvectors_from_schur_scratch::<Self>(n));
                    evd::eigenvectors_from_schur_cplx(
                        h,
                        Some(z),
                        x.as_mut(),
                        Par::Seq,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|e| format!("{e:?}"))?;
                }
                Ok(MatrixData::$variant(x))
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
    validate(h, z)?;
    if select.len() != h.nrows() {
        return Err("Schur selection length mismatch".into());
    }
    let mut flags = select.to_vec();
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
    let mut mem = MemBuffer::new(evd::reorder_schur_scratch::<T>(n));
    let mut here = 0;
    let mut next = 0;
    while here < n {
        let b = T::block_size(h.as_ref(), here);
        if flags[here] != 0 {
            if here != next {
                T::move_block(&mut h, &mut z, here, next, &mut mem)?;
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
    #[julia]
    pub fn schur_eigenvectors_from(
        &mut self,
        t: *const u8,
        z: *const u8,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            self.data = Some(if tag == 1 {
                let t = &*t.cast::<OwnedMatrix>();
                let z = &*z.cast::<OwnedMatrix>();
                validate(t.matrix.as_ref(), z.matrix.as_ref())?;
                f64::eigenvectors(t.matrix.as_ref(), z.matrix.as_ref())?
            } else {
                let t = (&*t.cast::<TypedMatrix>()).data()?;
                let z = (&*z.cast::<TypedMatrix>()).data()?;
                match (t, z) {
                    (MatrixData::F32(t), MatrixData::F32(z)) if tag == 0 => {
                        validate(t.as_ref(), z.as_ref())?;
                        f32::eigenvectors(t.as_ref(), z.as_ref())?
                    }
                    (MatrixData::C32(t), MatrixData::C32(z)) if tag == 2 => {
                        validate(t.as_ref(), z.as_ref())?;
                        c32::eigenvectors(t.as_ref(), z.as_ref())?
                    }
                    (MatrixData::C64(t), MatrixData::C64(z)) if tag == 3 => {
                        validate(t.as_ref(), z.as_ref())?;
                        c64::eigenvectors(t.as_ref(), z.as_ref())?
                    }
                    _ => return Err("Schur scalar types must match".into()),
                }
            });
            Ok(())
        })
    }
}

fn reduced<T: SchurScalar>(h: MatRef<'_, T>, z: MatRef<'_, T>) -> Result<(Mat<T>, Mat<T>), String> {
    if h.nrows() != h.ncols() || h.shape() != z.shape() {
        return Err("H and Q must have matching square dimensions".into());
    }
    finite(h)?;
    finite(z)?;
    for j in 0..h.ncols() {
        for i in j + 2..h.nrows() {
            if h[(i, j)] != zero::<T>() {
                return Err("input is not upper Hessenberg".into());
            }
        }
    }
    let mut h = h.to_owned();
    let mut z = z.to_owned();
    T::step(&mut h, &mut z)?;
    Ok((h, z))
}
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
