//! Schur driver primitives over the native public Hessenberg-Schur APIs.
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Col, Mat};

pub(crate) trait SchurScalar: crate::householder::ReflectorScalar {
    fn reorder(t: &mut Mat<Self>, q: &mut Mat<Self>, from: usize, to: usize) -> bool;
    fn native(h: &mut Mat<Self>, q: &mut Mat<Self>) -> (Vec<Self>, Vec<f64>, bool);
}
macro_rules! real_schur {
    ($t:ty) => {
        impl SchurScalar for $t {
            fn reorder(t: &mut Mat<Self>, q: &mut Mat<Self>, from: usize, to: usize) -> bool {
                use colmat::linalg::evd::{reorder_schur_real, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    reorder_schur_real(
                        t.as_mut(),
                        Some(q.as_mut()),
                        from,
                        to,
                        par,
                        MemStack::new(&mut mem),
                    )
                    .is_ok()
                })
            }
            fn native(h: &mut Mat<Self>, q: &mut Mat<Self>) -> (Vec<Self>, Vec<f64>, bool) {
                use colmat::linalg::evd::{hessenberg_schur_real, hessenberg_schur_scratch};
                let n = h.nrows();
                let mut real = Col::<Self>::zeros(n);
                let mut imag = Col::<Self>::zeros(n);
                let success = crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<Self>(
                        n,
                        true,
                        par.shape(),
                        Default::default(),
                    ));
                    hessenberg_schur_real(
                        h.as_mut(),
                        Some(q.as_mut()),
                        real.as_mut(),
                        imag.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .is_ok()
                });
                (
                    (0..n).map(|i| real[i]).collect(),
                    (0..n).map(|i| imag[i] as f64).collect(),
                    success,
                )
            }
        }
    };
}
macro_rules! complex_schur {
    ($t:ty) => {
        impl SchurScalar for $t {
            fn reorder(t: &mut Mat<Self>, q: &mut Mat<Self>, from: usize, to: usize) -> bool {
                use colmat::linalg::evd::{reorder_schur_cplx, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    reorder_schur_cplx(
                        t.as_mut(),
                        Some(q.as_mut()),
                        from,
                        to,
                        par,
                        MemStack::new(&mut mem),
                    );
                });
                true
            }
            fn native(h: &mut Mat<Self>, q: &mut Mat<Self>) -> (Vec<Self>, Vec<f64>, bool) {
                use colmat::linalg::evd::{hessenberg_schur_cplx, hessenberg_schur_scratch};
                let n = h.nrows();
                let mut values = Col::<Self>::zeros(n);
                let success = crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<Self>(
                        n,
                        true,
                        par.shape(),
                        Default::default(),
                    ));
                    hessenberg_schur_cplx(
                        h.as_mut(),
                        Some(q.as_mut()),
                        values.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .is_ok()
                });
                ((0..n).map(|i| values[i]).collect(), vec![0.0; n], success)
            }
        }
    };
}
real_schur!(f32);
real_schur!(f64);
complex_schur!(c32);
complex_schur!(c64);

pub(crate) fn multiply<T: Scalar>(a: &Mat<T>, ta: u8, b: &Mat<T>, tb: u8) -> Mat<T> {
    let rows = if ta == b'N' { a.nrows() } else { a.ncols() };
    let inner = if ta == b'N' { a.ncols() } else { a.nrows() };
    let cols = if tb == b'N' { b.ncols() } else { b.nrows() };
    let mut output = Mat::<T>::zeros(rows, cols);
    let ld = output.as_ref().col_stride();
    unsafe {
        crate::kernels::gemm(
            ta,
            tb,
            rows as isize,
            cols as isize,
            inner as isize,
            T::o(),
            a.as_ref().as_ptr(),
            a.as_ref().col_stride(),
            b.as_ref().as_ptr(),
            b.as_ref().col_stride(),
            T::z(),
            output.as_mut().as_ptr_mut(),
            ld,
        );
    }
    output
}

pub(crate) fn active<T: SchurScalar>(
    h: &mut Mat<T>,
    z: Option<&mut Mat<T>>,
    lo: usize,
    hi: usize,
) -> (Vec<T>, Vec<f64>, usize) {
    let n = h.nrows();
    let size = hi - lo;
    let mut values: Vec<T> = (0..n).map(|i| h[(i, i)]).collect();
    let mut imaginary = vec![0.0; n];
    if size == 0 {
        return (values, imaginary, 0);
    }
    let mut block = Mat::from_fn(size, size, |i, j| {
        if i <= j + 1 {
            h[(lo + i, lo + j)]
        } else {
            T::z()
        }
    });
    let mut q = Mat::from_fn(size, size, |i, j| if i == j { T::o() } else { T::z() });
    let (w, wi, success) = T::native(&mut block, &mut q);
    let transform = Mat::from_fn(n, n, |i, j| {
        if i >= lo && i < hi && j >= lo && j < hi {
            q[(i - lo, j - lo)]
        } else if i == j {
            T::o()
        } else {
            T::z()
        }
    });
    let right = multiply(h, b'N', &transform, b'N');
    let result = multiply(&transform, b'C', &right, b'N');
    for j in 0..n {
        for i in 0..n {
            if i >= lo && i < hi && j >= lo && j < hi {
                h[(i, j)] = if i <= j + 1 {
                    block[(i - lo, j - lo)]
                } else {
                    T::z()
                };
            } else if (i >= lo && i < hi) || (j >= lo && j < hi) {
                h[(i, j)] = result[(i, j)];
            }
        }
    }
    if let Some(z) = z {
        let updated = multiply(z, b'N', &transform, b'N');
        *z = updated;
    }
    values[lo..hi].copy_from_slice(&w);
    imaginary[lo..hi].copy_from_slice(&wi);
    (values, imaginary, if success { 0 } else { hi })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reconstruct<T: SchurScalar>() {
        let n = 5;
        let mut h = Mat::from_fn(n, n, |i, j| {
            if i <= j + 1 {
                T::real(((i * 7 + j * 3 + 1) % 13) as f64)
            } else {
                T::z()
            }
        });
        h[(1, 0)] = T::z();
        h[(4, 3)] = T::z();
        let original = h.clone();
        let mut z = Mat::from_fn(n, n, |i, j| if i == j { T::o() } else { T::z() });
        let (_, _, info) = active(&mut h, Some(&mut z), 1, 4);
        assert_eq!(info, 0);
        let temp = multiply(&z, b'N', &h, b'N');
        let result = multiply(&temp, b'N', &z, b'C');
        for j in 0..n {
            for i in 0..n {
                assert!((result[(i, j)] - original[(i, j)]).magnitude() < 1e-10);
            }
        }
        assert_eq!(h[(0, 0)], original[(0, 0)]);
        assert_eq!(h[(4, 4)], original[(4, 4)]);
    }
    #[test]
    fn active_real_and_complex_similarity() {
        reconstruct::<f64>();
        reconstruct::<c64>();
    }
}

unsafe fn run_hessenberg<T: SchurScalar>(
    n: usize,
    lo: usize,
    hi: usize,
    h: *mut T,
    ldh: isize,
    z: *mut T,
    ldz: isize,
    initialize: bool,
    want_z: bool,
    zlo: usize,
    zhi: usize,
) -> (Vec<T>, Vec<f64>, usize) {
    let mut matrix = Mat::from_fn(n, n, |i, j| {
        if i <= j + 1 {
            *h.offset(i as isize + j as isize * ldh)
        } else {
            T::z()
        }
    });
    let mut vectors = if want_z {
        Some(Mat::from_fn(n, n, |i, j| {
            if initialize {
                if i == j {
                    T::o()
                } else {
                    T::z()
                }
            } else if i >= zlo && i < zhi {
                *z.offset(i as isize + j as isize * ldz)
            } else {
                T::z()
            }
        }))
    } else {
        None
    };
    let result = active(&mut matrix, vectors.as_mut(), lo, hi);
    for j in 0..n {
        for i in 0..n {
            *h.offset(i as isize + j as isize * ldh) = matrix[(i, j)];
        }
    }
    if let Some(vectors) = vectors {
        let rows = if initialize { 0..n } else { zlo..zhi };
        for j in 0..n {
            for i in rows.clone() {
                *z.offset(i as isize + j as isize * ldz) = vectors[(i, j)];
            }
        }
    }
    result
}
use crate::kernels::flag;
use paste::paste;
macro_rules! hseqr_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hseqr $s>](job:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,
            h:*mut $t,ldh:*const $i,wr:*mut $t,wi:*mut $t,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(job,comp)=(flag(job),flag(compz));
            let(n,lo,hi,lh,lz,lw)=(*n as isize,*ilo as isize,*ihi as isize,*ldh as isize,*ldz as isize,*lwork as isize);
            for(arg,ok)in [(1,matches!(job,b'E'|b'S')),(2,matches!(comp,b'N'|b'I'|b'V')),(3,n>=0),(4,lo>=1&&lo<=n.max(1)),(5,hi>=lo.min(n)&&hi<=n),
                (7,lh>=n.max(1)),(11,lz>=1&&(comp==b'N'||lz>=n)),(13,lw== -1||lw>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(n.max(1));if lw== -1 {return;}
            let(w,imag,err)=run_hessenberg(n as usize,(lo-1) as usize,hi as usize,h,lh,z,lz,comp==b'I',comp!=b'N',0,n as usize);
            for j in 0..n as usize {*wr.add(j)=w[j];*wi.add(j)=imag[j] as $t;}*info=err as $i;
        }
    }};
}
macro_rules! hseqr_complex_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hseqr $s>](job:*const u8,compz:*const u8,n:*const $i,ilo:*const $i,ihi:*const $i,
            h:*mut $t,ldh:*const $i,w:*mut $t,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;let(job,comp)=(flag(job),flag(compz));
            let(n,lo,hi,lh,lz,lw)=(*n as isize,*ilo as isize,*ihi as isize,*ldh as isize,*ldz as isize,*lwork as isize);
            for(arg,ok)in [(1,matches!(job,b'E'|b'S')),(2,matches!(comp,b'N'|b'I'|b'V')),(3,n>=0),(4,lo>=1&&lo<=n.max(1)),(5,hi>=lo.min(n)&&hi<=n),
                (7,lh>=n.max(1)),(10,lz>=1&&(comp==b'N'||lz>=n)),(12,lw== -1||lw>=n.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(n.max(1));if lw== -1 {return;}
            let(values,_,err)=run_hessenberg(n as usize,(lo-1) as usize,hi as usize,h,lh,z,lz,comp==b'I',comp!=b'N',0,n as usize);
            for j in 0..n as usize {*w.add(j)=values[j];}*info=err as $i;
        }
    }};
}
hseqr_real_exports!(s, f32, i32, _);
hseqr_real_exports!(d, f64, i32, _);
hseqr_real_exports!(s, f32, i64, _64_);
hseqr_real_exports!(d, f64, i64, _64_);
hseqr_complex_exports!(c, c32, i32, _);
hseqr_complex_exports!(z, c64, i32, _);
hseqr_complex_exports!(c, c32, i64, _64_);
hseqr_complex_exports!(z, c64, i64, _64_);

macro_rules! lahqr_real_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt) => {
        paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](
            wantt: *const $i,
            wantz: *const $i,
            n: *const $i,
            ilo: *const $i,
            ihi: *const $i,
            h: *mut $t,
            ldh: *const $i,
            wr: *mut $t,
            wi: *mut $t,
            iloz: *const $i,
            ihiz: *const $i,
            z: *mut $t,
            ldz: *const $i,
            info: *mut $i,
        ) {
            *info = 0;
            let _want_t = *wantt != 0;
            let want_z = *wantz != 0;
            let (n, lo, hi, lh, zlo, zhi, lz) = (
                *n as isize,
                *ilo as isize,
                *ihi as isize,
                *ldh as isize,
                *iloz as isize,
                *ihiz as isize,
                *ldz as isize,
            );
            for (arg, ok) in [
                (3, n >= 0),
                (4, lo >= 1 && lo <= n.max(1)),
                (5, hi >= lo.min(n) && hi <= n),
                (7, lh >= n.max(1)),
                (10, !want_z || (zlo >= 1 && zlo <= lo)),
                (11, !want_z || (zhi >= hi && zhi <= n)),
                (13, lz >= 1 && (!want_z || lz >= n)),
            ] {
                if !ok {
                    *info = -arg;
                    crate::invalid(arg as i32);
                    return;
                }
            }
            let (values, imag, err) = run_hessenberg(
                n as usize,
                (lo - 1) as usize,
                hi as usize,
                h,
                lh,
                z,
                lz,
                false,
                want_z,
                zlo.max(1) as usize - 1,
                zhi.max(0) as usize,
            );
            for j in 0..n as usize {
                *wr.add(j) = values[j];
                *wi.add(j) = imag[j] as $t;
            }
            *info = err as $i;
        }
        }
    };
}
macro_rules! laqr_real_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt) => {
        paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](
            wantt: *const $i,
            wantz: *const $i,
            n: *const $i,
            ilo: *const $i,
            ihi: *const $i,
            h: *mut $t,
            ldh: *const $i,
            wr: *mut $t,
            wi: *mut $t,
            iloz: *const $i,
            ihiz: *const $i,
            z: *mut $t,
            ldz: *const $i,
            work: *mut $t,
            lwork: *const $i,
            info: *mut $i,
        ) {
            *info = 0;
            let _want_t = *wantt != 0;
            let want_z = *wantz != 0;
            let (n, lo, hi, lh, zlo, zhi, lz) = (
                *n as isize,
                *ilo as isize,
                *ihi as isize,
                *ldh as isize,
                *iloz as isize,
                *ihiz as isize,
                *ldz as isize,
            );
            for (arg, ok) in [
                (3, n >= 0),
                (4, lo >= 1 && lo <= n.max(1)),
                (5, hi >= lo.min(n) && hi <= n),
                (7, lh >= n.max(1)),
                (10, !want_z || (zlo >= 1 && zlo <= lo)),
                (11, !want_z || (zhi >= hi && zhi <= n)),
                (13, lz >= 1 && (!want_z || lz >= n)),
                (15, *lwork == -1 || *lwork >= 1),
            ] {
                if !ok {
                    *info = -arg;
                    crate::invalid(arg as i32);
                    return;
                }
            }
            *work = crate::lapack::workspace::<$t>(n.max(1));
            if *lwork == -1 {
                return;
            }
            let (values, imag, err) = run_hessenberg(
                n as usize,
                (lo - 1) as usize,
                hi as usize,
                h,
                lh,
                z,
                lz,
                false,
                want_z,
                zlo.max(1) as usize - 1,
                zhi.max(0) as usize,
            );
            for j in 0..n as usize {
                *wr.add(j) = values[j];
                *wi.add(j) = imag[j] as $t;
            }
            *info = err as $i;
        }
        }
    };
}
macro_rules! lahqr_complex_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt) => {
        paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](
            wantt: *const $i,
            wantz: *const $i,
            n: *const $i,
            ilo: *const $i,
            ihi: *const $i,
            h: *mut $t,
            ldh: *const $i,
            w: *mut $t,
            iloz: *const $i,
            ihiz: *const $i,
            z: *mut $t,
            ldz: *const $i,
            info: *mut $i,
        ) {
            *info = 0;
            let _want_t = *wantt != 0;
            let want_z = *wantz != 0;
            let (n, lo, hi, lh, zlo, zhi, lz) = (
                *n as isize,
                *ilo as isize,
                *ihi as isize,
                *ldh as isize,
                *iloz as isize,
                *ihiz as isize,
                *ldz as isize,
            );
            for (arg, ok) in [
                (3, n >= 0),
                (4, lo >= 1 && lo <= n.max(1)),
                (5, hi >= lo.min(n) && hi <= n),
                (7, lh >= n.max(1)),
                (9, !want_z || (zlo >= 1 && zlo <= lo)),
                (10, !want_z || (zhi >= hi && zhi <= n)),
                (12, lz >= 1 && (!want_z || lz >= n)),
            ] {
                if !ok {
                    *info = -arg;
                    crate::invalid(arg as i32);
                    return;
                }
            }
            let (values, imag, err) = run_hessenberg(
                n as usize,
                (lo - 1) as usize,
                hi as usize,
                h,
                lh,
                z,
                lz,
                false,
                want_z,
                zlo.max(1) as usize - 1,
                zhi.max(0) as usize,
            );
            let _ = imag;
            for j in 0..n as usize {
                *w.add(j) = values[j];
            }
            *info = err as $i;
        }
        }
    };
}
macro_rules! laqr_complex_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt) => {
        paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](
            wantt: *const $i,
            wantz: *const $i,
            n: *const $i,
            ilo: *const $i,
            ihi: *const $i,
            h: *mut $t,
            ldh: *const $i,
            w: *mut $t,
            iloz: *const $i,
            ihiz: *const $i,
            z: *mut $t,
            ldz: *const $i,
            work: *mut $t,
            lwork: *const $i,
            info: *mut $i,
        ) {
            *info = 0;
            let _want_t = *wantt != 0;
            let want_z = *wantz != 0;
            let (n, lo, hi, lh, zlo, zhi, lz) = (
                *n as isize,
                *ilo as isize,
                *ihi as isize,
                *ldh as isize,
                *iloz as isize,
                *ihiz as isize,
                *ldz as isize,
            );
            for (arg, ok) in [
                (3, n >= 0),
                (4, lo >= 1 && lo <= n.max(1)),
                (5, hi >= lo.min(n) && hi <= n),
                (7, lh >= n.max(1)),
                (9, !want_z || (zlo >= 1 && zlo <= lo)),
                (10, !want_z || (zhi >= hi && zhi <= n)),
                (12, lz >= 1 && (!want_z || lz >= n)),
                (14, *lwork == -1 || *lwork >= 1),
            ] {
                if !ok {
                    *info = -arg;
                    crate::invalid(arg as i32);
                    return;
                }
            }
            *work = crate::lapack::workspace::<$t>(n.max(1));
            if *lwork == -1 {
                return;
            }
            let (values, imag, err) = run_hessenberg(
                n as usize,
                (lo - 1) as usize,
                hi as usize,
                h,
                lh,
                z,
                lz,
                false,
                want_z,
                zlo.max(1) as usize - 1,
                zhi.max(0) as usize,
            );
            let _ = imag;
            for j in 0..n as usize {
                *w.add(j) = values[j];
            }
            *info = err as $i;
        }
        }
    };
}
lahqr_real_exports!(s, lahqr, f32, i32, _);
laqr_real_exports!(s, laqr0, f32, i32, _);
laqr_real_exports!(s, laqr4, f32, i32, _);
lahqr_real_exports!(s, lahqr, f32, i64, _64_);
laqr_real_exports!(s, laqr0, f32, i64, _64_);
laqr_real_exports!(s, laqr4, f32, i64, _64_);
lahqr_real_exports!(d, lahqr, f64, i32, _);
laqr_real_exports!(d, laqr0, f64, i32, _);
laqr_real_exports!(d, laqr4, f64, i32, _);
lahqr_real_exports!(d, lahqr, f64, i64, _64_);
laqr_real_exports!(d, laqr0, f64, i64, _64_);
laqr_real_exports!(d, laqr4, f64, i64, _64_);
lahqr_complex_exports!(c, lahqr, c32, i32, _);
laqr_complex_exports!(c, laqr0, c32, i32, _);
laqr_complex_exports!(c, laqr4, c32, i32, _);
lahqr_complex_exports!(c, lahqr, c32, i64, _64_);
laqr_complex_exports!(c, laqr0, c32, i64, _64_);
laqr_complex_exports!(c, laqr4, c32, i64, _64_);
lahqr_complex_exports!(z, lahqr, c64, i32, _);
laqr_complex_exports!(z, laqr0, c64, i32, _);
laqr_complex_exports!(z, laqr4, c64, i32, _);
lahqr_complex_exports!(z, lahqr, c64, i64, _64_);
laqr_complex_exports!(z, laqr0, c64, i64, _64_);
laqr_complex_exports!(z, laqr4, c64, i64, _64_);

// Eigenvalues in LAPACK's real conjugate-pair order after native block swaps.
fn spectrum<T: SchurScalar>(t: &Mat<T>) -> (Vec<T>, Vec<f64>) {
    let n = t.nrows();
    let mut w: Vec<T> = (0..n).map(|i| t[(i, i)]).collect();
    let mut wi = vec![0.0; n];
    let mut i = 0;
    while i < n {
        if !T::COMPLEX && i + 1 < n && t[(i + 1, i)] != T::z() {
            let a = t[(i, i)].re();
            let d = t[(i + 1, i + 1)].re();
            let b = t[(i, i + 1)].re();
            let c = t[(i + 1, i)].re();
            let scale = a.abs().max(b.abs()).max(c.abs()).max(d.abs());
            let re = a * 0.5 + d * 0.5;
            let delta = ((a / scale - d / scale) * 0.5).powi(2) + (b / scale) * (c / scale);
            let im = (-delta).max(0.0).sqrt() * scale;
            w[i] = T::real(re);
            w[i + 1] = T::real(re);
            wi[i] = im;
            wi[i + 1] = -im;
            i += 2;
        } else {
            i += 1;
        }
    }
    (w, wi)
}

unsafe fn general_schur<T: SchurScalar>(
    n: usize,
    a: *mut T,
    lda: isize,
    vs: *mut T,
    ldvs: isize,
    want_vs: bool,
    sort: bool,
    mut select: impl FnMut(T, f64) -> bool,
) -> (Vec<T>, Vec<f64>, Vec<bool>, usize, usize) {
    // Normalize reduction, iteration and reordering; SELECT always sees original units.
    let mut largest = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            largest = largest.max((*a.offset(i as isize + j as isize * lda)).magnitude());
        }
    }
    let scale = if largest > 0.0 && largest.is_finite() {
        largest
    } else {
        1.0
    };
    let mut h = Mat::from_fn(n, n, |i, j| {
        *a.offset(i as isize + j as isize * lda) / T::real(scale)
    });
    let ld = h.as_ref().col_stride();
    let mut tau = vec![T::z(); n.saturating_sub(1)];
    crate::hessenberg::reduce(n, 0, n, h.as_mut().as_ptr_mut(), ld, tau.as_mut_ptr());
    let mut q = h.clone();
    let lq = q.as_ref().col_stride();
    crate::hessenberg::explicit(n, 0, n, q.as_mut().as_ptr_mut(), lq, tau.as_ptr());
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = T::z();
        }
    }
    let (mut w, mut wi, mut info) = active(&mut h, Some(&mut q), 0, n);
    for i in 0..n {
        w[i] = w[i] * T::real(scale);
        wi[i] *= scale;
    }
    let mut chosen = vec![false; n];
    let mut sdim = 0;
    if sort && info == 0 {
        for i in 0..n {
            chosen[i] = select(w[i], wi[i]);
        }
        let mut i = 0;
        while i < n {
            let size = if !T::COMPLEX && i + 1 < n && h[(i + 1, i)] != T::z() {
                2
            } else {
                1
            };
            let take = chosen[i..i + size].iter().any(|&x| x);
            for flag in &mut chosen[i..i + size] {
                *flag = take;
            }
            if take {
                if i != sdim {
                    if !T::reorder(&mut h, &mut q, i, sdim) {
                        info = n + 1;
                        break;
                    }
                    chosen[sdim..i + size].rotate_right(size);
                }
                sdim += size;
            }
            i += size;
        }
        (w, wi) = spectrum(&h);
        for i in 0..n {
            w[i] = w[i] * T::real(scale);
            wi[i] *= scale;
        }
        if info == 0 {
            let mut actual = vec![false; n];
            for i in 0..n {
                actual[i] = select(w[i], wi[i]);
            }
            let mut i = 0;
            while i < n {
                let size = if !T::COMPLEX && i + 1 < n && h[(i + 1, i)] != T::z() {
                    2
                } else {
                    1
                };
                let take = actual[i..i + size].iter().any(|&x| x);
                for flag in &mut actual[i..i + size] {
                    *flag = take;
                }
                i += size;
            }
            if actual
                .iter()
                .enumerate()
                .any(|(i, &take)| take != (i < sdim))
            {
                info = n + 2;
            }
            sdim = actual.iter().filter(|&&x| x).count();
            chosen = actual;
        }
    }
    for j in 0..n {
        for i in 0..n {
            h[(i, j)] = h[(i, j)] * T::real(scale);
        }
    }
    crate::lapack::output(&h, a, lda);
    if want_vs {
        crate::lapack::output(&q, vs, ldvs);
    }
    (w, wi, chosen, sdim, info)
}

macro_rules! gees_real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gees $s>](jobvs:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,n:*const $i,a:*mut $t,lda:*const $i,sdim:*mut $i,wr:*mut $t,wi:*mut $t,vs:*mut $t,ldvs:*const $i,work:*mut $t,lwork:*const $i,bwork:*mut $i,info:*mut $i) {
            *info=0;let(job,sorting)=(flag(jobvs),flag(sort));let(n,la,lv,lw)=(*n as isize,*lda as isize,*ldvs as isize,*lwork as isize);
            for(arg,ok)in [(1,matches!(job,b'N'|b'V')),(2,matches!(sorting,b'N'|b'S')),(4,n>=0),(6,la>=n.max(1)),(11,lv>=1&&(job==b'N'||lv>=n)),(13,lw== -1||lw>=(3*n).max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>((3*n).max(1));if lw== -1{return;}
            *sdim=0;
            let(w,imag,chosen,count,err)=general_schur(n as usize,a,la,vs,lv,job==b'V',sorting==b'S',|re,im| {
                let im=im as $t;select.expect("GEES SORT=S requires a SELECT callback")(&re,&im)!=0
            });
            for j in 0..n as usize {*wr.add(j)=w[j];*wi.add(j)=imag[j] as $t;if sorting==b'S'{*bwork.add(j)=chosen[j] as $i;}}
            *sdim=count as $i;*info=err as $i;
        }
    }};
}
macro_rules! gees_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gees $s>](jobvs:*const u8,sort:*const u8,select:Option<unsafe extern "C" fn(*const $t)->$i>,n:*const $i,a:*mut $t,lda:*const $i,sdim:*mut $i,w:*mut $t,vs:*mut $t,ldvs:*const $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,bwork:*mut $i,info:*mut $i) {
            *info=0;let(job,sorting)=(flag(jobvs),flag(sort));let(n,la,lv,lw)=(*n as isize,*lda as isize,*ldvs as isize,*lwork as isize);
            for(arg,ok)in [(1,matches!(job,b'N'|b'V')),(2,matches!(sorting,b'N'|b'S')),(4,n>=0),(6,la>=n.max(1)),(10,lv>=1&&(job==b'N'||lv>=n)),(12,lw== -1||lw>=(2*n).max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>((2*n).max(1));if lw== -1{return;}
            *sdim=0;
            let(values,_,chosen,count,err)=general_schur(n as usize,a,la,vs,lv,job==b'V',sorting==b'S',|value,_|select.expect("GEES SORT=S requires a SELECT callback")(&value)!=0);
            for j in 0..n as usize {*w.add(j)=values[j];if sorting==b'S'{*bwork.add(j)=chosen[j] as $i;}}
            *sdim=count as $i;*info=err as $i;
        }
    }};
}
gees_real_exports!(s, f32, i32, _);
gees_real_exports!(d, f64, i32, _);
gees_real_exports!(s, f32, i64, _64_);
gees_real_exports!(d, f64, i64, _64_);
gees_complex_exports!(c, c32, f32, i32, _);
gees_complex_exports!(z, c64, f64, i32, _);
gees_complex_exports!(c, c32, f32, i64, _64_);
gees_complex_exports!(z, c64, f64, i64, _64_);
