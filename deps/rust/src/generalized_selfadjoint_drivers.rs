//! Generalized definite eigendrivers using native congruence, EVD and back-transform.
use crate::generalized_selfadjoint::{self as reduction, GeneralizedScalar};
use crate::kernels::{flag, Scalar};
use colmat::{Conj, Mat};
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

static FALLBACKS: [AtomicPtr<c_void>; 12] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 12];
#[no_mangle]
pub extern "C" fn colmat_selfadjoint_gv_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_selfadjoint_gv_has_fallback(id: i32) -> i32 {
    FALLBACKS
        .get(id as usize)
        .is_some_and(|p| !p.load(Ordering::Acquire).is_null()) as i32
}
fn fallback(id: usize) -> *mut c_void {
    let pointer = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat generalized self-adjoint fallback must be configured before use"
    );
    pointer
}
struct NativeResult<T: GeneralizedScalar> {
    values: Vec<f64>,
    vectors: Mat<T>,
    reduced: Mat<T>,
}
enum Failure {
    NonPositive(usize),
    Native,
}
unsafe fn snapshot<T: GeneralizedScalar>(u: u8, n: usize, b: *const T, ld: isize) -> Vec<T> {
    let mut saved = Vec::new();
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                let pointer = b.offset(i as isize + j as isize * ld);
                saved.push(if i == j {
                    T::diagonal(pointer)
                } else {
                    *pointer
                });
            }
        }
    }
    saved
}
unsafe fn restore<T: GeneralizedScalar>(saved: &[T], u: u8, n: usize, b: *mut T, ld: isize) {
    let mut k = 0;
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                *b.offset(i as isize + j as isize * ld) = saved[k];
                k += 1;
            }
        }
    }
}
unsafe fn native<T: GeneralizedScalar>(
    kind: u8,
    job: u8,
    u: u8,
    n: usize,
    a: *const T,
    la: isize,
    b: *mut T,
    lb: isize,
) -> Result<NativeResult<T>, Failure> {
    let saved = snapshot(u, n, b, lb);
    // Complex native LLT uses the opposite triangle as scratch. Stage it and
    // copy only the selected output, without reading ignored diagonal parts.
    let status = if T::COMPLEX {
        let mut factor = reduction::hermitian(u, n, b, lb);
        let status = crate::lapack::cholesky(factor.as_mut());
        for j in 0..n {
            for i in 0..n {
                if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                    let value = if u == b'L' {
                        factor[(i, j)]
                    } else {
                        factor[(j, i)].cj()
                    };
                    *b.offset(i as isize + j as isize * lb) =
                        if i == j { T::real(value.re()) } else { value };
                }
            }
        }
        status
    } else {
        let factor = if u == b'L' {
            crate::lapack::input_mut(b, n, n, lb)
        } else {
            crate::lapack::input_mut_transposed(b, n, lb)
        };
        crate::lapack::cholesky(factor)
    };
    if status != 0 {
        return Err(Failure::NonPositive(status));
    }
    let matrix = reduction::hermitian(u, n, a, la);
    let reduced = reduction::with_lower(u, n, b, lb, |lower| {
        reduction::standardize(kind, matrix, lower)
    });
    let result = if reduced.as_ref().is_all_finite() {
        crate::lapack::selfadj(reduced.as_ref(), job == b'V')
    } else {
        Err(())
    };
    let (values, mut vectors) = match result {
        Ok(result) => result,
        Err(()) => {
            restore(&saved, u, n, b, lb);
            return Err(Failure::Native);
        }
    };
    if job == b'V' {
        reduction::with_lower(u, n, b, lb, |lower| {
            crate::threading::with_par(|par| {
                if kind == 3 {
                    use colmat::linalg::matmul::triangular::{matmul, BlockStructure::*};
                    let mut output = Mat::<T>::zeros(n, n);
                    matmul(
                        output.as_mut(),
                        Rectangular,
                        colmat::Accum::Replace,
                        lower,
                        TriangularLower,
                        vectors.as_ref(),
                        Rectangular,
                        T::o(),
                        par,
                    );
                    vectors = output;
                } else {
                    colmat::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(
                        lower.transpose(),
                        Conj::Yes,
                        vectors.as_mut(),
                        par,
                    );
                }
            })
        });
    }
    if !values.iter().all(|x| x.is_finite()) || !vectors.as_ref().is_all_finite() {
        restore(&saved, u, n, b, lb);
        return Err(Failure::Native);
    }
    Ok(NativeResult {
        values,
        vectors,
        reduced,
    })
}
unsafe fn full_output<T: GeneralizedScalar, R: Scalar>(
    result: NativeResult<T>,
    job: u8,
    u: u8,
    a: *mut T,
    la: isize,
    w: *mut R,
) {
    for (i, &value) in result.values.iter().enumerate() {
        *w.add(i) = R::real(value);
    }
    if job == b'V' {
        crate::lapack::output(&result.vectors, a, la);
    } else {
        reduction::output(&result.reduced, u, a, la);
    }
}
use paste::paste;
macro_rules! gv_real {
    ($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p sygv $s>](itype:*const $i,jobz:*const u8,uplo:*const u8,n:*const $i,a:*mut $r,lda:*const $i,b:*mut $r,ldb:*const $i,w:*mut $r,work:*mut $r,lwork:*const $i,info:*mut $i) {
            *info=0;let(kind,job,u,nn,la,lb,lw)=(*itype as isize,flag(jobz),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(u,b'U'|b'L')),(4,nn>=0),(6,la>=nn.max(1)),(8,lb>=nn.max(1))] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let minimum=nn.saturating_mul(3).saturating_sub(1).max(1);*work=crate::lapack::workspace::<$r>(minimum);
            if lw!= -1&&lw<minimum{*info=-11;crate::invalid(11);return;}
            if lw== -1||nn==0{return;}
            match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                Ok(result)=>full_output(result,job,u,a,la,w),
                Err(Failure::NonPositive(index))=>*info=(nn as usize+index) as $i,
                Err(Failure::Native)=>{
                    let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $r,*const i64,*mut $r,*const i64,*mut $r,*mut $r,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut status=0i64;f(&(kind as i64),jobz,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),w,work,&(lw as i64),&mut status);*info=status as $i;
                }
            }
        }
    }};
}
macro_rules! gv_complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hegv $s>](itype:*const $i,jobz:*const u8,uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,w:*mut $r,work:*mut $t,lwork:*const $i,rwork:*mut $r,info:*mut $i) {
            *info=0;let(kind,job,u,nn,la,lb,lw)=(*itype as isize,flag(jobz),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(u,b'U'|b'L')),(4,nn>=0),(6,la>=nn.max(1)),(8,lb>=nn.max(1))] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let minimum=nn.saturating_mul(2).saturating_sub(1).max(1);*work=crate::lapack::workspace::<$t>(minimum);
            if lw!= -1&&lw<minimum{*info=-11;crate::invalid(11);return;}
            if lw== -1||nn==0{return;}
            match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                Ok(result)=>full_output(result,job,u,a,la,w),
                Err(Failure::NonPositive(index))=>*info=(nn as usize+index) as $i,
                Err(Failure::Native)=>{
                    let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*mut $t,*const i64,*mut $r,*mut i64)=std::mem::transmute(fallback($id));
                    let mut status=0i64;f(&(kind as i64),jobz,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),w,work,&(lw as i64),rwork,&mut status);*info=status as $i;
                }
            }
        }
    }};
}
gv_real!(s, f32, i32, _, 0);
gv_real!(d, f64, i32, _, 1);
gv_complex!(c, colmat::c32, f32, i32, _, 2);
gv_complex!(z, colmat::c64, f64, i32, _, 3);
gv_real!(s, f32, i64, _64_, 0);
gv_real!(d, f64, i64, _64_, 1);
gv_complex!(c, colmat::c32, f32, i64, _64_, 2);
gv_complex!(z, colmat::c64, f64, i64, _64_, 3);

fn dc_workspace(n: isize, vectors: bool, complex: bool) -> (isize, isize, isize) {
    if n <= 1 {
        return (1, 1, 1);
    }
    let square = n.saturating_mul(n);
    if complex {
        if vectors {
            (
                n.saturating_mul(2).saturating_add(square),
                1isize
                    .saturating_add(n.saturating_mul(5))
                    .saturating_add(square.saturating_mul(2)),
                3isize.saturating_add(n.saturating_mul(5)),
            )
        } else {
            (n.saturating_add(1), n, 1)
        }
    } else if vectors {
        (
            1isize
                .saturating_add(n.saturating_mul(6))
                .saturating_add(square.saturating_mul(2)),
            1,
            3isize.saturating_add(n.saturating_mul(5)),
        )
    } else {
        (n.saturating_mul(2).saturating_add(1), 1, 1)
    }
}
macro_rules! gvd_real {
    ($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p sygvd $s>](itype:*const $i,jobz:*const u8,uplo:*const u8,n:*const $i,a:*mut $r,lda:*const $i,b:*mut $r,ldb:*const $i,w:*mut $r,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;let(kind,job,u,nn,la,lb,lw,li)=(*itype as isize,flag(jobz),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*lwork as isize,*liwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(u,b'U'|b'L')),(4,nn>=0),(6,la>=nn.max(1)),(8,lb>=nn.max(1))] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let(minimum,_,integers)=dc_workspace(nn,job==b'V',false);let query=lw== -1||li== -1;
            *work=crate::lapack::workspace::<$r>(minimum);*iwork=integers as $i;
            for(arg,valid)in[(11,query||lw>=minimum),(13,query||li>=integers)] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            if query||nn==0{return;}
            match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                Ok(result)=>full_output(result,job,u,a,la,w),
                Err(Failure::NonPositive(index))=>*info=(nn as usize+index) as $i,
                Err(Failure::Native)=>{
                    let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $r,*const i64,*mut $r,*const i64,*mut $r,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut scratch=vec![0i64;integers as usize];let mut status=0i64;
                    f(&(kind as i64),jobz,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),w,work,&(lw as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);*iwork=scratch[0] as $i;*info=status as $i;
                }
            }
        }
    }};
}
macro_rules! gvd_complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hegvd $s>](itype:*const $i,jobz:*const u8,uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,w:*mut $r,work:*mut $t,lwork:*const $i,rwork:*mut $r,lrwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;let(kind,job,u,nn,la,lb,lw,lr,li)=(*itype as isize,flag(jobz),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*lwork as isize,*lrwork as isize,*liwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(u,b'U'|b'L')),(4,nn>=0),(6,la>=nn.max(1)),(8,lb>=nn.max(1))] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let(minimum,reals,integers)=dc_workspace(nn,job==b'V',true);let query=lw== -1||lr== -1||li== -1;
            *work=crate::lapack::workspace::<$t>(minimum);*rwork=crate::lapack::workspace::<$r>(reals);*iwork=integers as $i;
            for(arg,valid)in[(11,query||lw>=minimum),(13,query||lr>=reals),(15,query||li>=integers)] {if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            if query||nn==0{return;}
            match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                Ok(result)=>full_output(result,job,u,a,la,w),
                Err(Failure::NonPositive(index))=>*info=(nn as usize+index) as $i,
                Err(Failure::Native)=>{
                    let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*mut $t,*const i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut scratch=vec![0i64;integers as usize];let mut status=0i64;
                    f(&(kind as i64),jobz,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),w,work,&(lw as i64),rwork,&(lr as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);*iwork=scratch[0] as $i;*info=status as $i;
                }
            }
        }
    }};
}
gvd_real!(s, f32, i32, _, 4);
gvd_real!(d, f64, i32, _, 5);
gvd_complex!(c, colmat::c32, f32, i32, _, 6);
gvd_complex!(z, colmat::c64, f64, i32, _, 7);
gvd_real!(s, f32, i64, _64_, 4);
gvd_real!(d, f64, i64, _64_, 5);
gvd_complex!(c, colmat::c32, f32, i64, _64_, 6);
gvd_complex!(z, colmat::c64, f64, i64, _64_, 7);

enum Selection {
    All,
    Value(f64, f64),
    Index(usize, usize),
}
unsafe fn selection<R: Scalar, I: crate::lapack::LapackInt>(
    mode: u8,
    n: isize,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
) -> Result<Selection, i32> {
    match mode {
        b'A' => Ok(Selection::All),
        b'V' => {
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if n > 0 && hi <= lo {
                Err(11)
            } else {
                Ok(Selection::Value(lo, hi))
            }
        }
        b'I' => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > n.max(1) {
                Err(12)
            } else if hi < lo.min(n) || hi > n {
                Err(13)
            } else {
                Ok(Selection::Index((lo - 1) as usize, hi as usize))
            }
        }
        _ => Err(3),
    }
}
unsafe fn range_output<T: GeneralizedScalar, R: Scalar, I: crate::lapack::LapackInt>(
    result: NativeResult<T>,
    selected: Selection,
    job: u8,
    u: u8,
    a: *mut T,
    la: isize,
    m: *mut I,
    w: *mut R,
    z: *mut T,
    lz: isize,
    ifail: *mut I,
) {
    let indices: Vec<usize> = (0..result.values.len())
        .filter(|&i| match selected {
            Selection::All => true,
            Selection::Value(lo, hi) => lo < result.values[i] && result.values[i] <= hi,
            Selection::Index(lo, hi) => lo <= i && i < hi,
        })
        .collect();
    *m = I::from_i64(indices.len() as i64);
    for (column, &source) in indices.iter().enumerate() {
        *w.add(column) = R::real(result.values[source]);
        if job == b'V' {
            for row in 0..result.vectors.nrows() {
                *z.offset(row as isize + column as isize * lz) = result.vectors[(row, source)];
            }
            *ifail.add(column) = I::from_i64(0);
        }
    }
    reduction::output(&result.reduced, u, a, la);
}
macro_rules! gvx_real {
    ($p:ident,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p sygvx $s>](itype:*const $i,jobz:*const u8,range:*const u8,uplo:*const u8,n:*const $i,a:*mut $r,lda:*const $i,b:*mut $r,ldb:*const $i,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $r,ldz:*const $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            *info=0;let(kind,job,mode,u,nn,la,lb,lz,lw)=(*itype as isize,flag(jobz),flag(range),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*ldz as isize,*lwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(mode,b'A'|b'V'|b'I')),(4,matches!(u,b'U'|b'L')),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1))]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let selected=match selection(mode,nn,vl,vu,il,iu){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if lz<1||(job==b'V'&&lz<nn){*info=-18;crate::invalid(18);return;}
            let minimum=nn.saturating_mul(8).max(1);*work=crate::lapack::workspace::<$r>(minimum);
            if lw!= -1&&lw<minimum{*info=-20;crate::invalid(20);return;}
            if lw== -1{return;}
            *m=0;if nn==0{return;}
            if *abstol<=0.0 {
                match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                    Ok(result)=>{range_output(result,selected,job,u,a,la,m,w,z,lz,ifail);return;},
                    Err(Failure::NonPositive(index))=>{*info=(nn as usize+index)as $i;return;},
                    Err(Failure::Native)=>{},
                }
            }
            let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const u8,*const i64,*mut $r,*const i64,*mut $r,*const i64,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $r,*const i64,*mut $r,*const i64,*mut i64,*mut i64,*mut i64)=std::mem::transmute(fallback($id));
            let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;5*(nn as usize)];let mut failed=vec![0i64;nn as usize];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(&(kind as i64),jobz,range,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),vl,vu,&lo,&hi,abstol,&mut count,w,z,&(lz as i64),work,&(lw as i64),scratch.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            if job==b'V'{for k in 0..count.max(0).min(nn as i64)as usize{*ifail.add(k)=failed[k]as $i;}}
            let _=iwork;
        }
    }};
}
macro_rules! gvx_complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hegvx $s>](itype:*const $i,jobz:*const u8,range:*const u8,uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            *info=0;let(kind,job,mode,u,nn,la,lb,lz,lw)=(*itype as isize,flag(jobz),flag(range),flag(uplo),*n as isize,*lda as isize,*ldb as isize,*ldz as isize,*lwork as isize);
            for(arg,valid)in[(1,(1..=3).contains(&kind)),(2,matches!(job,b'N'|b'V')),(3,matches!(mode,b'A'|b'V'|b'I')),(4,matches!(u,b'U'|b'L')),(5,nn>=0),(7,la>=nn.max(1)),(9,lb>=nn.max(1))]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            let selected=match selection(mode,nn,vl,vu,il,iu){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if lz<1||(job==b'V'&&lz<nn){*info=-18;crate::invalid(18);return;}
            let minimum=nn.saturating_mul(2).max(1);*work=crate::lapack::workspace::<$t>(minimum);
            if lw!= -1&&lw<minimum{*info=-20;crate::invalid(20);return;}
            if lw== -1{return;}
            *m=0;if nn==0{return;}
            if *abstol<=0.0 {
                match native(kind as u8,job,u,nn as usize,a,la,b,lb) {
                    Ok(result)=>{range_output(result,selected,job,u,a,la,m,w,z,lz,ifail);return;},
                    Err(Failure::NonPositive(index))=>{*info=(nn as usize+index)as $i;return;},
                    Err(Failure::Native)=>{},
                }
            }
            let f:unsafe extern "C" fn(*const i64,*const u8,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,*const i64,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*mut i64,*mut i64,*mut i64)=std::mem::transmute(fallback($id));
            let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;5*(nn as usize)];let mut failed=vec![0i64;nn as usize];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(&(kind as i64),jobz,range,uplo,&(nn as i64),a,&(la as i64),b,&(lb as i64),vl,vu,&lo,&hi,abstol,&mut count,w,z,&(lz as i64),work,&(lw as i64),rwork,scratch.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            if job==b'V'{for k in 0..count.max(0).min(nn as i64)as usize{*ifail.add(k)=failed[k]as $i;}}
            let _=iwork;
        }
    }};
}
gvx_real!(s, f32, i32, _, 8);
gvx_real!(d, f64, i32, _, 9);
gvx_complex!(c, colmat::c32, f32, i32, _, 10);
gvx_complex!(z, colmat::c64, f64, i32, _, 11);
gvx_real!(s, f32, i64, _64_, 8);
gvx_real!(d, f64, i64, _64_, 9);
gvx_complex!(c, colmat::c32, f32, i64, _64_, 10);
gvx_complex!(z, colmat::c64, f64, i64, _64_, 11);
