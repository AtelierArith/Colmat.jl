//! Schur driver primitives over the native public Hessenberg-Schur APIs.
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Col, Mat};

pub(crate) trait SchurScalar: Scalar {
    fn native(h: &mut Mat<Self>, q: &mut Mat<Self>) -> (Vec<Self>, Vec<f64>, bool);
}
macro_rules! real_schur {
    ($t:ty) => {
        impl SchurScalar for $t {
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

fn multiply<T: Scalar>(a: &Mat<T>, ta: u8, b: &Mat<T>, tb: u8) -> Mat<T> {
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

#[no_mangle]
pub unsafe extern "C" fn slahqr_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f32,
    ldh: *const i32,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f32,
    ldz: *const i32,
    info: *mut i32,
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn slaqr0_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f32,
    ldh: *const i32,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f32,
    ldz: *const i32,
    work: *mut f32,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<f32>(n.max(1));
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn slaqr4_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f32,
    ldh: *const i32,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f32,
    ldz: *const i32,
    work: *mut f32,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<f32>(n.max(1));
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn slahqr_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f32,
    ldh: *const i64,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f32,
    ldz: *const i64,
    info: *mut i64,
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn slaqr0_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f32,
    ldh: *const i64,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f32,
    ldz: *const i64,
    work: *mut f32,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<f32>(n.max(1));
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn slaqr4_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f32,
    ldh: *const i64,
    wr: *mut f32,
    wi: *mut f32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f32,
    ldz: *const i64,
    work: *mut f32,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<f32>(n.max(1));
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
        *wi.add(j) = imag[j] as f32;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn dlahqr_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f64,
    ldh: *const i32,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f64,
    ldz: *const i32,
    info: *mut i32,
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn dlaqr0_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f64,
    ldh: *const i32,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f64,
    ldz: *const i32,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<f64>(n.max(1));
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn dlaqr4_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut f64,
    ldh: *const i32,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut f64,
    ldz: *const i32,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<f64>(n.max(1));
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn dlahqr_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f64,
    ldh: *const i64,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f64,
    ldz: *const i64,
    info: *mut i64,
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn dlaqr0_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f64,
    ldh: *const i64,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f64,
    ldz: *const i64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<f64>(n.max(1));
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn dlaqr4_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut f64,
    ldh: *const i64,
    wr: *mut f64,
    wi: *mut f64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut f64,
    ldz: *const i64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<f64>(n.max(1));
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
        *wi.add(j) = imag[j] as f64;
    }
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn clahqr_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c32,
    ldh: *const i32,
    w: *mut c32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c32,
    ldz: *const i32,
    info: *mut i32,
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn claqr0_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c32,
    ldh: *const i32,
    w: *mut c32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c32,
    ldz: *const i32,
    work: *mut c32,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<c32>(n.max(1));
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn claqr4_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c32,
    ldh: *const i32,
    w: *mut c32,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c32,
    ldz: *const i32,
    work: *mut c32,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<c32>(n.max(1));
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn clahqr_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c32,
    ldh: *const i64,
    w: *mut c32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c32,
    ldz: *const i64,
    info: *mut i64,
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
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn claqr0_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c32,
    ldh: *const i64,
    w: *mut c32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c32,
    ldz: *const i64,
    work: *mut c32,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<c32>(n.max(1));
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
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn claqr4_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c32,
    ldh: *const i64,
    w: *mut c32,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c32,
    ldz: *const i64,
    work: *mut c32,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<c32>(n.max(1));
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
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn zlahqr_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c64,
    ldh: *const i32,
    w: *mut c64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c64,
    ldz: *const i32,
    info: *mut i32,
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn zlaqr0_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c64,
    ldh: *const i32,
    w: *mut c64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c64,
    ldz: *const i32,
    work: *mut c64,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<c64>(n.max(1));
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn zlaqr4_(
    wantt: *const i32,
    wantz: *const i32,
    n: *const i32,
    ilo: *const i32,
    ihi: *const i32,
    h: *mut c64,
    ldh: *const i32,
    w: *mut c64,
    iloz: *const i32,
    ihiz: *const i32,
    z: *mut c64,
    ldz: *const i32,
    work: *mut c64,
    lwork: *const i32,
    info: *mut i32,
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
    *work = crate::lapack::workspace::<c64>(n.max(1));
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
    *info = err as i32;
}

#[no_mangle]
pub unsafe extern "C" fn zlahqr_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c64,
    ldh: *const i64,
    w: *mut c64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c64,
    ldz: *const i64,
    info: *mut i64,
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
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn zlaqr0_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c64,
    ldh: *const i64,
    w: *mut c64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c64,
    ldz: *const i64,
    work: *mut c64,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<c64>(n.max(1));
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
    *info = err as i64;
}

#[no_mangle]
pub unsafe extern "C" fn zlaqr4_64_(
    wantt: *const i64,
    wantz: *const i64,
    n: *const i64,
    ilo: *const i64,
    ihi: *const i64,
    h: *mut c64,
    ldh: *const i64,
    w: *mut c64,
    iloz: *const i64,
    ihiz: *const i64,
    z: *mut c64,
    ldz: *const i64,
    work: *mut c64,
    lwork: *const i64,
    info: *mut i64,
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
    *work = crate::lapack::workspace::<c64>(n.max(1));
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
    *info = err as i64;
}
