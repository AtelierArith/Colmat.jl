//! LAPACK homogeneous generalized eigenproblem representation over native GEVD.
use crate::householder::ReflectorScalar;
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Mat, MatMut, MatRef};

pub(crate) struct EigenResult<T: ReflectorScalar> {
    pub alpha: Vec<T>,
    pub imaginary: Vec<f64>,
    pub beta: Vec<T>,
    pub left: Mat<T>,
    pub right: Mat<T>,
}
pub(crate) trait EigenScalar: ReflectorScalar {
    fn eigen(
        a: MatMut<'_, Self>,
        b: MatMut<'_, Self>,
        left: bool,
        right: bool,
    ) -> Result<EigenResult<Self>, ()>;
}
macro_rules! real_eigen {
    ($t:ty) => {
        impl EigenScalar for $t {
            fn eigen(
                mut a: MatMut<'_, Self>,
                mut b: MatMut<'_, Self>,
                left: bool,
                right: bool,
            ) -> Result<EigenResult<Self>, ()> {
                use colmat::linalg::evd::ComputeEigenvectors;
                use colmat::linalg::gevd::{gevd_real, gevd_scratch};
                let n = a.nrows();
                let mut ar = Diag::<Self>::zeros(n);
                let mut ai = Diag::<Self>::zeros(n);
                let mut beta = Diag::<Self>::zeros(n);
                let mut vl = Mat::<Self>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<Self>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let flag = |yes| {
                        if yes {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(gevd_scratch::<Self>(
                        n,
                        flag(left),
                        flag(right),
                        par.shape(),
                        Default::default(),
                    ));
                    gevd_real(
                        a.as_mut(),
                        b.as_mut(),
                        ar.as_mut(),
                        ai.as_mut(),
                        beta.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(EigenResult {
                    alpha: (0..n).map(|i| ar[i]).collect(),
                    imaginary: (0..n).map(|i| ai[i] as f64).collect(),
                    beta: (0..n).map(|i| beta[i]).collect(),
                    left: vl,
                    right: vr,
                })
            }
        }
    };
}
macro_rules! complex_eigen {
    ($t:ty) => {
        impl EigenScalar for $t {
            fn eigen(
                mut a: MatMut<'_, Self>,
                mut b: MatMut<'_, Self>,
                left: bool,
                right: bool,
            ) -> Result<EigenResult<Self>, ()> {
                use colmat::linalg::evd::ComputeEigenvectors;
                use colmat::linalg::gevd::{gevd_cplx, gevd_scratch};
                let n = a.nrows();
                let mut alpha = Diag::<Self>::zeros(n);
                let mut beta = Diag::<Self>::zeros(n);
                let mut vl = Mat::<Self>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<Self>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let flag = |yes| {
                        if yes {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(gevd_scratch::<Self>(
                        n,
                        flag(left),
                        flag(right),
                        par.shape(),
                        Default::default(),
                    ));
                    gevd_cplx(
                        a.as_mut(),
                        b.as_mut(),
                        alpha.as_mut(),
                        beta.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(EigenResult {
                    alpha: (0..n).map(|i| alpha[i]).collect(),
                    imaginary: vec![0.0; n],
                    beta: (0..n).map(|i| beta[i]).collect(),
                    left: vl,
                    right: vr,
                })
            }
        }
    };
}
real_eigen!(f32);
real_eigen!(f64);
complex_eigen!(c32);
complex_eigen!(c64);

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: EigenScalar>() {
        let n = 3;
        let mut a = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real((i + 2) as f64)
            } else if i < j {
                T::real((i + j + 1) as f64)
            } else {
                T::z()
            }
        });
        let mut b = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real((i + 1) as f64)
            } else {
                T::z()
            }
        });
        let result = T::eigen(a.as_mut(), b.as_mut(), true, true).unwrap();
        for j in 0..n {
            assert_eq!(result.imaginary[j], 0.0);
            let alpha = result.alpha[j];
            let beta = result.beta[j];
            for i in 0..n {
                let mut right = T::z();
                let mut left = T::z();
                for k in 0..n {
                    right = right + (beta * a[(i, k)] - alpha * b[(i, k)]) * result.right[(k, j)];
                    left = left + result.left[(k, j)].cj() * (beta * a[(k, i)] - alpha * b[(k, i)]);
                }
                assert!(right.magnitude() < 1e-10);
                assert!(left.magnitude() < 1e-10);
            }
        }
    }
    #[test]
    fn homogeneous_left_and_right_residuals() {
        check::<f64>();
        check::<c64>();
    }
}

fn normalize<T: EigenScalar>(v: &mut Mat<T>, imaginary: &[f64]) {
    let n = v.nrows();
    let mut j = 0;
    while j < v.ncols() {
        let pair = !T::COMPLEX && j + 1 < v.ncols() && imaginary[j] > 0.0;
        let mut largest = 0.0_f64;
        for i in 0..n {
            let magnitude = if pair {
                v[(i, j)].magnitude() + v[(i, j + 1)].magnitude()
            } else {
                v[(i, j)].abs1()
            };
            largest = largest.max(magnitude);
        }
        if largest > 0.0 {
            for i in 0..n {
                v[(i, j)] = v[(i, j)] / T::real(largest);
                if pair {
                    v[(i, j + 1)] = v[(i, j + 1)] / T::real(largest);
                }
            }
        }
        j += if pair { 2 } else { 1 };
    }
}
// Track the pinned GEVD implementation's power-of-two prescale so its Schur
// matrices can be returned in original units (GGEVX's stronger output contract).
fn native_unscale<T: EigenScalar>(a: MatRef<'_, T>) -> f64 {
    let mut norm = 0.0_f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            norm = norm.max(a[(i, j)].re().abs()).max(a[(i, j)].im().abs());
        }
    }
    if !norm.is_finite() {
        return 1.0;
    }
    let minimum = T::SAFE * (T::EPS / 2.0);
    let small = minimum.sqrt() / T::EPS;
    let large = 1.0 / small;
    let mut factor = 1.0;
    if norm > 0.0 && norm < small {
        while norm * factor < small {
            factor *= 2.0;
        }
    } else if norm > large {
        while norm * factor > large {
            factor *= 0.5;
        }
    }
    1.0 / factor
}
unsafe fn run<T: EigenScalar>(
    n: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    vl: *mut T,
    ldvl: isize,
    vr: *mut T,
    ldvr: isize,
    left: bool,
    right: bool,
) -> Result<(Vec<T>, Vec<f64>, Vec<T>), ()> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let (sa, sb) = {
        let aa = crate::lapack::input_ref(a, n, n, lda);
        let bb = crate::lapack::input_ref(b, n, n, ldb);
        (native_unscale(aa), native_unscale(bb))
    };
    let mut result = T::eigen(
        crate::lapack::input_mut(a, n, n, lda),
        crate::lapack::input_mut(b, n, n, ldb),
        left,
        right,
    )?;
    {
        let mut aa = crate::lapack::input_mut(a, n, n, lda);
        let mut bb = crate::lapack::input_mut(b, n, n, ldb);
        for j in 0..n {
            for i in 0..n {
                aa[(i, j)] = aa[(i, j)] * T::real(sa);
                bb[(i, j)] = bb[(i, j)] * T::real(sb);
            }
        }
    }
    if left {
        // Native real GEVD packs transpose-left vectors; LAPACK packs adjoint-left
        // vectors for the positive-imaginary member, so conjugate each real pair.
        if !T::COMPLEX {
            for j in 0..n.saturating_sub(1) {
                if result.imaginary[j] > 0.0 {
                    for i in 0..n {
                        result.left[(i, j + 1)] = -result.left[(i, j + 1)];
                    }
                }
            }
        }
        normalize(&mut result.left, &result.imaginary);
        crate::lapack::output(&result.left, vl, ldvl);
    }
    if right {
        normalize(&mut result.right, &result.imaginary);
        crate::lapack::output(&result.right, vr, ldvr);
    }
    Ok((result.alpha, result.imaginary, result.beta))
}
use crate::kernels::flag;
use paste::paste;
macro_rules! real_exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](jobvl:*const u8,jobvr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alphar:*mut $t,alphai:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(jl,jr)=(flag(jobvl),flag(jobvr));let(n,la,lb,ll,lr,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvl as isize,*ldvr as isize,*lwork as isize);
            let minimum=(8*n).max(1);
            for(arg,ok)in[(1,matches!(jl,b'N'|b'V')),(2,matches!(jr,b'N'|b'V')),(3,n>=0),(5,la>=n.max(1)),(7,lb>=n.max(1)),(12,ll>=1&&(jl==b'N'||ll>=n)),(14,lr>=1&&(jr==b'N'||lr>=n)),(16,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
            match run(n as usize,a,la,b,lb,vl,ll,vr,lr,jl==b'V',jr==b'V'){
                Ok((ar,ai,be))=>for i in 0..n as usize{*alphar.add(i)=ar[i];*alphai.add(i)=ai[i] as $t;*beta.add(i)=be[i];},
                Err(())=>*info=n as $i,
            }
        }
    }};
}
macro_rules! complex_exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](jobvl:*const u8,jobvr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,work:*mut $t,lwork:*const $i,_rwork:*mut $r,info:*mut $i){
            *info=0;let(jl,jr)=(flag(jobvl),flag(jobvr));let(n,la,lb,ll,lr,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvl as isize,*ldvr as isize,*lwork as isize);
            let minimum=(2*n).max(1);
            for(arg,ok)in[(1,matches!(jl,b'N'|b'V')),(2,matches!(jr,b'N'|b'V')),(3,n>=0),(5,la>=n.max(1)),(7,lb>=n.max(1)),(11,ll>=1&&(jl==b'N'||ll>=n)),(13,lr>=1&&(jr==b'N'||lr>=n)),(15,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
            match run(n as usize,a,la,b,lb,vl,ll,vr,lr,jl==b'V',jr==b'V'){
                Ok((ar,_,be))=>for i in 0..n as usize{*alpha.add(i)=ar[i];*beta.add(i)=be[i];},
                Err(())=>*info=n as $i,
            }
        }
    }};
}
real_exports!(s, ggev, f32, i32, _);
real_exports!(s, ggev3, f32, i32, _);
real_exports!(s, gegv, f32, i32, _);
real_exports!(s, ggev, f32, i64, _64_);
real_exports!(s, ggev3, f32, i64, _64_);
real_exports!(s, gegv, f32, i64, _64_);
real_exports!(d, ggev, f64, i32, _);
real_exports!(d, ggev3, f64, i32, _);
real_exports!(d, gegv, f64, i32, _);
real_exports!(d, ggev, f64, i64, _64_);
real_exports!(d, ggev3, f64, i64, _64_);
real_exports!(d, gegv, f64, i64, _64_);
complex_exports!(c, ggev, c32, f32, i32, _);
complex_exports!(c, ggev3, c32, f32, i32, _);
complex_exports!(c, gegv, c32, f32, i32, _);
complex_exports!(c, ggev, c32, f32, i64, _64_);
complex_exports!(c, ggev3, c32, f32, i64, _64_);
complex_exports!(c, gegv, c32, f32, i64, _64_);
complex_exports!(z, ggev, c64, f64, i32, _);
complex_exports!(z, ggev3, c64, f64, i32, _);
complex_exports!(z, gegv, c64, f64, i32, _);
complex_exports!(z, ggev, c64, f64, i64, _64_);
complex_exports!(z, ggev3, c64, f64, i64, _64_);
complex_exports!(z, gegv, c64, f64, i64, _64_);

use core::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static EXPERT: [AtomicPtr<c_void>; 4] = [const { AtomicPtr::new(core::ptr::null_mut()) }; 4];
#[no_mangle]
pub extern "C" fn colmat_ggevx_set_expert_fallback(id: i32, p: *mut c_void) {
    if (0..4).contains(&id) {
        EXPERT[id as usize].store(p, Ordering::Relaxed);
    }
}
#[no_mangle]
pub extern "C" fn colmat_ggevx_has_expert_fallback(id: i32) -> i32 {
    ((0..4).contains(&id) && !EXPERT[id as usize].load(Ordering::Relaxed).is_null()) as i32
}
unsafe fn norm_one<T: crate::kernels::Scalar>(n: usize, a: *const T, lda: isize) -> f64 {
    let mut result = 0.0_f64;
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += (*a.offset(i as isize + j as isize * lda)).magnitude();
        }
        result = result.max(sum);
    }
    result
}

macro_rules! expert_real_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p ggevx $s>](balanc:*const u8,jobvl:*const u8,jobvr:*const u8,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alpha:*mut $t,alphai:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,ilo:*mut $i,ihi:*mut $i,lscale:*mut $r,rscale:*mut $r,abnrm:*mut $r,bbnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,work:*mut $t,lwork:*const $i,_iwork:*mut $i,_bwork:*mut $i,info:*mut $i){
*info=0;let(bal,jl,jr,sens)=(flag(balanc),flag(jobvl),flag(jobvr),flag(sense));
let(nn,la,lb,ll,lr,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvl as isize,*ldvr as isize,*lwork as isize);
let mut minimum=if jl==b'V'||jr==b'V'{6*nn}else{2*nn};if matches!(sens,b'E'|b'B'){minimum=minimum.max(10*nn);}if matches!(sens,b'V'|b'B'){minimum=minimum.max(2*nn*(nn+4)+16);}if nn==0{minimum=1;}minimum=minimum.max(1);
for(arg,ok)in[(1,matches!(bal,b'N'|b'P'|b'S'|b'B')),(2,matches!(jl,b'N'|b'V')),(3,matches!(jr,b'N'|b'V')),(4,matches!(sens,b'N'|b'E'|b'V'|b'B')),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1)),(14,ll>=1&&(jl==b'N'||ll>=nn)),(16,lr>=1&&(jr==b'N'||lr>=nn)),(26,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
if bal!=b'N'||sens!=b'N'{
    // Colmat's Julia loader guarantees a configured ILP64 expert backend before
    // publishing this entry point. Integer scratch belongs to this call.
    let pointer=EXPERT[$id].load(Ordering::Relaxed);
    assert!(!pointer.is_null(),"initialize GGEVX expert fallback before using expert modes");
    let(n64,la64,lb64,ll64,lr64,lw64)=(nn as i64,la as i64,lb as i64,ll as i64,lr as i64,lw as i64);
    let(mut lo64,mut hi64,mut err64)=(0_i64,0_i64,0_i64);
    let count=if lw== -1{0}else{nn as usize};
    let mut iw=vec![0_i64;count+6];let mut bw=vec![0_i64;count];
let fallback:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut i64,*mut i64,*mut $r,*mut $r,*mut $r,*mut $r,*mut $r,*mut $r,*mut $t,*const i64,*mut i64,*mut i64,*mut i64,usize,usize,usize,usize)=core::mem::transmute(pointer);
fallback(balanc,jobvl,jobvr,sense,&n64,a,&la64,b,&lb64,alpha,alphai,beta,vl,&ll64,vr,&lr64,&mut lo64,&mut hi64,lscale,rscale,abnrm,bbnrm,rconde,rcondv,work,&lw64,iw.as_mut_ptr(),bw.as_mut_ptr(),&mut err64,1,1,1,1);
if lw!= -1{*ilo=lo64 as $i;*ihi=hi64 as $i;}*info=err64 as $i;return;
}
*work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
*ilo=1;*ihi=nn as $i;*abnrm=norm_one(nn as usize,a,la) as $r;*bbnrm=norm_one(nn as usize,b,lb) as $r;
for i in 0..nn as usize{*lscale.add(i)=1.0;*rscale.add(i)=1.0;}
match run(nn as usize,a,la,b,lb,vl,ll,vr,lr,jl==b'V',jr==b'V'){
Ok((ar,im,be))=>for i in 0..nn as usize{*alpha.add(i)=ar[i];*alphai.add(i)=im[i] as $t;*beta.add(i)=be[i];},Err(())=>*info=nn as $i,}
}
}};}

macro_rules! expert_complex_exports {($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p ggevx $s>](balanc:*const u8,jobvl:*const u8,jobvr:*const u8,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,ilo:*mut $i,ihi:*mut $i,lscale:*mut $r,rscale:*mut $r,abnrm:*mut $r,bbnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,work:*mut $t,lwork:*const $i,rwork:*mut $r,_iwork:*mut $i,_bwork:*mut $i,info:*mut $i){
*info=0;let(bal,jl,jr,sens)=(flag(balanc),flag(jobvl),flag(jobvr),flag(sense));
let(nn,la,lb,ll,lr,lw)=(*n as isize,*lda as isize,*ldb as isize,*ldvl as isize,*ldvr as isize,*lwork as isize);
let minimum=if nn==0{1}else if matches!(sens,b'V'|b'B'){2*nn*(nn+1)}else if sens==b'E'{4*nn}else{2*nn}.max(1);
for(arg,ok)in[(1,matches!(bal,b'N'|b'P'|b'S'|b'B')),(2,matches!(jl,b'N'|b'V')),(3,matches!(jr,b'N'|b'V')),(4,matches!(sens,b'N'|b'E'|b'V'|b'B')),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1)),(13,ll>=1&&(jl==b'N'||ll>=nn)),(15,lr>=1&&(jr==b'N'||lr>=nn)),(25,lw== -1||lw>=minimum)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
if bal!=b'N'||sens!=b'N'{
    // Colmat's Julia loader guarantees a configured ILP64 expert backend before
    // publishing this entry point. Integer scratch belongs to this call.
    let pointer=EXPERT[$id].load(Ordering::Relaxed);
    assert!(!pointer.is_null(),"initialize GGEVX expert fallback before using expert modes");
    let(n64,la64,lb64,ll64,lr64,lw64)=(nn as i64,la as i64,lb as i64,ll as i64,lr as i64,lw as i64);
    let(mut lo64,mut hi64,mut err64)=(0_i64,0_i64,0_i64);
    let count=if lw== -1{0}else{nn as usize};
    let mut iw=vec![0_i64;count+2];let mut bw=vec![0_i64;count];
let fallback:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*mut $t,*mut $t,*const i64,*mut $t,*const i64,*mut i64,*mut i64,*mut $r,*mut $r,*mut $r,*mut $r,*mut $r,*mut $r,*mut $t,*const i64,*mut $r,*mut i64,*mut i64,*mut i64,usize,usize,usize,usize)=core::mem::transmute(pointer);
fallback(balanc,jobvl,jobvr,sense,&n64,a,&la64,b,&lb64,alpha,beta,vl,&ll64,vr,&lr64,&mut lo64,&mut hi64,lscale,rscale,abnrm,bbnrm,rconde,rcondv,work,&lw64,rwork,iw.as_mut_ptr(),bw.as_mut_ptr(),&mut err64,1,1,1,1);
if lw!= -1{*ilo=lo64 as $i;*ihi=hi64 as $i;}*info=err64 as $i;return;
}
*work=crate::lapack::workspace::<$t>(minimum);if lw== -1{return;}
*ilo=1;*ihi=nn as $i;*abnrm=norm_one(nn as usize,a,la) as $r;*bbnrm=norm_one(nn as usize,b,lb) as $r;
for i in 0..nn as usize{*lscale.add(i)=1.0;*rscale.add(i)=1.0;}
match run(nn as usize,a,la,b,lb,vl,ll,vr,lr,jl==b'V',jr==b'V'){
Ok((ar,im,be))=>for i in 0..nn as usize{*alpha.add(i)=ar[i];let _=&im;*beta.add(i)=be[i];},Err(())=>*info=nn as $i,}
}
}};}
expert_real_exports!(s, f32, f32, i32, _, 0);
expert_real_exports!(s, f32, f32, i64, _64_, 0);
expert_real_exports!(d, f64, f64, i32, _, 1);
expert_real_exports!(d, f64, f64, i64, _64_, 1);
expert_complex_exports!(c, c32, f32, i32, _, 2);
expert_complex_exports!(c, c32, f32, i64, _64_, 2);
expert_complex_exports!(z, c64, f64, i32, _, 3);
expert_complex_exports!(z, c64, f64, i64, _64_, 3);
