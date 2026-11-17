//! Bidiagonal SVD LAPACK ABI adapters.
//! BDSQR/BDSDC(I)/BDSVDX/LASQ1 use the public native bidiagonal SVD. BDSDC
//! COMPQ='P' has no public compact representation in the native API, so that
//! one expert mode delegates to the saved original ILPACK implementation.
use crate::kernels::{flag, Scalar};
use crate::lapack::LapackInt;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::traits::math_utils;
use colmat::{Mat, MatMut, MatRef};
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

static BDSDC_FALLBACKS: [AtomicPtr<c_void>; 2] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; 2];
#[no_mangle]
pub extern "C" fn colmat_bdsdc_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = BDSDC_FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_bdsdc_has_fallback(id: i32) -> i32 {
    BDSDC_FALLBACKS
        .get(id as usize)
        .is_some_and(|p| !p.load(Ordering::Acquire).is_null()) as i32
}
fn bdsdc_fallback(id: usize) -> *mut c_void {
    let pointer = BDSDC_FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat BDSDC compact mode requires a configured ILP64 LAPACK backend"
    );
    pointer
}

trait RealF64: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}
impl RealF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl RealF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

pub(crate) fn bidiagonal_svd<T: Scalar>(
    d: &[T::Real],
    e: &[T::Real],
    upper: bool,
    vectors: bool,
) -> Option<(Vec<T::Real>, Mat<T>, Mat<T>)>
where
    T::Real: RealF64,
{
    let n = d.len();
    if n == 0 {
        return Some((Vec::new(), Mat::<T>::zeros(0, 0), Mat::<T>::zeros(0, 0)));
    }
    let mut diag: Vec<T> = d.iter().map(|x| T::real(x.to_f64())).collect();
    let offdiag: Vec<T> = e.iter().map(|x| T::real(x.to_f64())).collect();
    let mut u = Mat::<T>::zeros(n, n);
    let mut v = Mat::<T>::zeros(n, n);
    use colmat::linalg::svd::*;
    let mut params: colmat::Spec<SvdParams, T> = Default::default();
    params.config.recursion_threshold = usize::MAX;
    let result = crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(bidiagonal_svd_scratch::<T>(n, par.shape(), params));
        bidiagonal_svd(
            &mut diag,
            &offdiag,
            upper,
            u.as_mut(),
            v.as_mut(),
            par,
            MemStack::new(&mut memory),
            params,
        )
    });
    result.ok()?;
    let values: Vec<T::Real> = diag.iter().map(|x| T::Real::from_f64(T::re(*x))).collect();
    Some((values, u, v))
}

unsafe fn out<T: Scalar, I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}

unsafe fn bdsqr<T: Scalar, I: LapackInt>(
    uplo: u8,
    n: isize,
    ncvt: isize,
    nru: isize,
    ncc: isize,
    d: *mut T::Real,
    e: *mut T::Real,
    vt: *mut T,
    ldvt: isize,
    u: *mut T,
    ldu: isize,
    c: *mut T,
    ldc: isize,
    info: *mut I,
) where
    T::Real: RealF64,
{
    *info = I::from_i64(0);
    if !matches!(uplo, b'U' | b'L') {
        out::<T, I>(info, 1);
        return;
    }
    if n < 0 {
        out::<T, I>(info, 2);
        return;
    }
    if ncvt < 0 {
        out::<T, I>(info, 3);
        return;
    }
    if nru < 0 {
        out::<T, I>(info, 4);
        return;
    }
    if ncc < 0 {
        out::<T, I>(info, 5);
        return;
    }
    let nn = n as usize;
    if ncvt > 0 && ldvt < 1.max(n) {
        out::<T, I>(info, 8);
        return;
    }
    if nru > 0 && ldu < 1.max(nru) {
        out::<T, I>(info, 10);
        return;
    }
    if ncc > 0 && ldc < 1.max(n) {
        out::<T, I>(info, 12);
        return;
    }
    if n == 0 {
        return;
    }
    let d_in: Vec<T::Real> = (0..nn).map(|i| *d.add(i)).collect();
    let e_in: Vec<T::Real> = (0..nn.saturating_sub(1)).map(|i| *e.add(i)).collect();
    let Some((values, left, right)) = bidiagonal_svd::<T>(&d_in, &e_in, uplo == b'U', true) else {
        *info = I::from_i64(1);
        return;
    };
    for i in 0..nn {
        *d.add(i) = values[i];
        if i + 1 < nn {
            *e.add(i) = math_utils::zero::<T::Real>();
        }
    }
    let (ncvt, nru, ncc) = (ncvt as usize, nru as usize, ncc as usize);
    if ncvt > 0 {
        let input = MatRef::from_raw_parts(vt, nn, ncvt, 1, ldvt);
        let mut output = Mat::<T>::zeros(nn, ncvt);
        for j in 0..ncvt {
            for i in 0..nn {
                let mut sum = T::z();
                for k in 0..nn {
                    sum += right[(k, i)].cj() * input[(k, j)];
                }
                output[(i, j)] = sum;
            }
        }
        crate::lapack::output(&output, vt, ldvt);
    }
    if nru > 0 {
        let input = MatRef::from_raw_parts(u as *const T, nru, nn, 1, ldu);
        let mut output = Mat::<T>::zeros(nru, nn);
        for j in 0..nn {
            for i in 0..nru {
                let mut sum = T::z();
                for k in 0..nn {
                    sum += input[(i, k)] * left[(k, j)];
                }
                output[(i, j)] = sum;
            }
        }
        crate::lapack::output(&output, u, ldu);
    }
    if ncc > 0 {
        let input = MatRef::from_raw_parts(c, nn, ncc, 1, ldc);
        let mut output = Mat::<T>::zeros(nn, ncc);
        for j in 0..ncc {
            for i in 0..nn {
                let mut sum = T::z();
                for k in 0..nn {
                    sum += left[(k, i)].cj() * input[(k, j)];
                }
                output[(i, j)] = sum;
            }
        }
        crate::lapack::output(&output, c, ldc);
    }
}

trait Zero {
    fn default_zero() -> Self;
}
impl Zero for f32 {
    fn default_zero() -> Self {
        0.0
    }
}
impl Zero for f64 {
    fn default_zero() -> Self {
        0.0
    }
}

macro_rules! real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p bdsqr $s>](
                uplo:*const u8,n:*const $i,ncvt:*const $i,nru:*const $i,ncc:*const $i,
                d:*mut $t,e:*mut $t,vt:*mut $t,ldvt:*const $i,u:*mut $t,ldu:*const $i,
                c:*mut $t,ldc:*const $i,work:*mut $t,info:*mut $i,
            ) {
                let _=work;
                bdsqr::<$t,$i>(flag(uplo),(*n).val(),(*ncvt).val(),(*nru).val(),(*ncc).val(),
                    d,e,vt,(*ldvt).val(),u,(*ldu).val(),c,(*ldc).val(),info)
            }
        }
    };
}
macro_rules! complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p bdsqr $s>](
                uplo:*const u8,n:*const $i,ncvt:*const $i,nru:*const $i,ncc:*const $i,
                d:*mut $r,e:*mut $r,vt:*mut $t,ldvt:*const $i,u:*mut $t,ldu:*const $i,
                c:*mut $t,ldc:*const $i,rwork:*mut $r,info:*mut $i,
            ) {
                let _=rwork;
                bdsqr::<$t,$i>(flag(uplo),(*n).val(),(*ncvt).val(),(*nru).val(),(*ncc).val(),
                    d,e,vt,(*ldvt).val(),u,(*ldu).val(),c,(*ldc).val(),info)
            }
        }
    };
}
macro_rules! bdsdc_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$id:expr,$wide:expr) => { paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p bdsdc $s>](
            uplo:*const u8,compq:*const u8,n:*const $i,d:*mut $t,e:*mut $t,
            u:*mut $t,ldu:*const $i,vt:*mut $t,ldvt:*const $i,q:*mut $t,iq:*mut $i,
            work:*mut $t,iwork:*mut $i,info:*mut $i,
        ) where $t: Scalar<Real= $t>
        {
            let comp=flag(compq);
            if comp==b'P' {
                // Convert to the saved ILP64 compact driver.
                let nn=*n as i64;
                let (ldu,ldvt)=(*ldu as i64,*ldvt as i64);
                let mut iq64:Vec<i64>=(0..nn.max(0) as usize).map(|i| *iq.add(i) as i64).collect();
                let mut iwork64:Vec<i64>=(0..8*nn.max(0) as usize).map(|i| *iwork.add(i) as i64).collect();
                let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut i64,*mut $t,*mut i64,*mut i64)=core::mem::transmute(bdsdc_fallback($id));
                let mut status=0i64;
                f(uplo,compq,&nn,d,e,u,&ldu,vt,&ldvt,q,iq64.as_mut_ptr(),work,iwork64.as_mut_ptr(),&mut status);
                for i in 0..nn.max(0) as usize {*iq.add(i)=iq64[i] as $i;}
                for i in 0..8*nn.max(0) as usize {*iwork.add(i)=iwork64[i] as $i;}
                *info=status as $i;
                return;
            }
            *info=0;
            let nn=*n as usize;
            if !matches!(flag(uplo),b'U'|b'L') {out::<$t,$i>(info,1);return;}
            if !matches!(comp,b'N'|b'I') {out::<$t,$i>(info,2);return;}
            if *n<0 {out::<$t,$i>(info,3);return;}
            if comp==b'I' && *ldu<1.max(*n) {out::<$t,$i>(info,6);return;}
            if comp==b'I' && *ldvt<1.max(*n) {out::<$t,$i>(info,8);return;}
            if nn==0 {return;}
            let d_in:Vec<$t>=(0..nn).map(|i| *d.add(i)).collect();
            let e_in:Vec<$t>=(0..nn.saturating_sub(1)).map(|i| *e.add(i)).collect();
            let Some((values,left,right))=bidiagonal_svd::<$t>(&d_in,&e_in,flag(uplo)==b'U',comp==b'I') else {*info=1;return;};
            for i in 0..nn {*d.add(i)=values[i]; if i+1<nn {*e.add(i)=0.0;}}
            if comp==b'I' {
                let mut vtmat=Mat::<$t>::zeros(nn,nn);
                for j in 0..nn {for i in 0..nn {vtmat[(i,j)]=right[(j,i)];}}
                crate::lapack::output(&left,u,*ldu as isize);
                crate::lapack::output(&vtmat,vt,*ldvt as isize);
            }
        }
    }};
}

real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
complex_exports!(c, colmat::c32, f32, i32, _);
complex_exports!(z, colmat::c64, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, colmat::c32, f32, i64, _64_);
complex_exports!(z, colmat::c64, f64, i64, _64_);
bdsdc_exports!(s, f32, i32, _, 0, false);
bdsdc_exports!(d, f64, i32, _, 1, false);
bdsdc_exports!(s, f32, i64, _64_, 0, true);
bdsdc_exports!(d, f64, i64, _64_, 1, true);

unsafe fn bdsvdx<T: Scalar<Real = T> + RealF64, I: LapackInt>(
    uplo: u8,
    jobz: u8,
    range: u8,
    n: isize,
    d: *mut T,
    e: *mut T,
    vl: T,
    vu: T,
    il: isize,
    iu: isize,
    ns: *mut I,
    s: *mut T,
    z: *mut T,
    ldz: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let (jobz, range) = (jobz.to_ascii_uppercase(), range.to_ascii_uppercase());
    if !matches!(uplo, b'U' | b'L') {
        out::<T, I>(info, 1);
        return;
    }
    if !matches!(jobz, b'N' | b'V') {
        out::<T, I>(info, 2);
        return;
    }
    if !matches!(range, b'A' | b'V' | b'I') {
        out::<T, I>(info, 3);
        return;
    }
    if n < 0 {
        out::<T, I>(info, 4);
        return;
    }
    if range == b'V' && !(vu.to_f64() > vl.to_f64()) {
        out::<T, I>(info, 8);
        return;
    }
    if range == b'I' && (il < 1 || il > iu.max(0) || iu > n) {
        out::<T, I>(info, if il < 1 { 9 } else { 10 });
        return;
    }
    if jobz == b'V' && ldz < 1.max(2 * n) {
        out::<T, I>(info, 13);
        return;
    }
    let nn = n as usize;
    if nn == 0 {
        *ns = I::from_i64(0);
        return;
    }
    let d_in: Vec<T> = (0..nn).map(|i| *d.add(i)).collect();
    let e_in: Vec<T> = (0..nn.saturating_sub(1)).map(|i| *e.add(i)).collect();
    let Some((values, left, right)) = bidiagonal_svd::<T>(&d_in, &e_in, uplo == b'U', jobz == b'V')
    else {
        *info = I::from_i64(1);
        return;
    };
    // `values` is descending. RANGE selects a contiguous block for 'I' and a
    // value interval for 'V'; BDSVDX returns the selected values ascending.
    let selected: Vec<usize> = match range {
        b'A' => (0..nn).rev().collect(),
        b'V' => {
            let mut out = Vec::new();
            for k in (0..nn).rev() {
                if values[k].to_f64() >= vl.to_f64() && values[k].to_f64() < vu.to_f64() {
                    out.push(k);
                }
            }
            out
        }
        _ => {
            let lo = nn.saturating_sub(iu.max(0) as usize);
            let hi = nn.saturating_sub(il.max(1) as usize - 1);
            (lo..hi.min(nn)).rev().collect()
        }
    };
    *ns = I::from_i64(selected.len() as i64);
    for (j, &k) in selected.iter().enumerate() {
        *s.add(j) = values[k];
        if jobz == b'V' {
            for i in 0..nn {
                *z.add(i + j * ldz as usize) = left[(i, k)];
                *z.add(nn + i + j * ldz as usize) = right[(i, k)];
            }
        }
    }
    let _ = e;
}

unsafe fn lasq1<T: Scalar<Real = T> + RealF64, I: LapackInt>(
    n: isize,
    d: *mut T,
    e: *mut T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n < 0 {
        out::<T, I>(info, 1);
        return;
    }
    let nn = n as usize;
    if nn == 0 {
        return;
    }
    let d_in: Vec<T> = (0..nn).map(|i| *d.add(i)).collect();
    let e_in: Vec<T> = (0..nn.saturating_sub(1)).map(|i| *e.add(i)).collect();
    let Some((values, _, _)) = bidiagonal_svd::<T>(&d_in, &e_in, false, false) else {
        *info = I::from_i64(2);
        return;
    };
    for i in 0..nn {
        *d.add(i) = values[i];
        if i + 1 < nn {
            *e.add(i) = math_utils::zero::<T>();
        }
    }
}

macro_rules! diagnostic_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p bdsvdx $s>](
                uplo:*const u8,jobz:*const u8,range:*const u8,n:*const $i,
                d:*mut $t,e:*mut $t,vl:*const $t,vu:*const $t,il:*const $i,iu:*const $i,
                ns:*mut $i,s:*mut $t,z:*mut $t,ldz:*const $i,work:*mut $t,iwork:*mut $i,
                info:*mut $i,
            ) {
                let _=(work,iwork);
                bdsvdx::<$t,$i>(flag(uplo),flag(jobz),flag(range),(*n).val(),d,e,*vl,*vu,
                    (*il).val(),(*iu).val(),ns,s,z,(*ldz).val(),info)
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p lasq1 $s>](
                n:*const $i,d:*mut $t,e:*mut $t,work:*mut $t,info:*mut $i,
            ) {
                let _=work;
                lasq1::<$t,$i>((*n).val(),d,e,info)
            }
        }
    };
}
diagnostic_exports!(s, f32, i32, _);
diagnostic_exports!(d, f64, i32, _);
diagnostic_exports!(s, f32, i64, _64_);
diagnostic_exports!(d, f64, i64, _64_);
