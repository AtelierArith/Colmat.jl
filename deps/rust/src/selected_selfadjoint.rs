//! SYEVX/HEEVX selection over native full-spectrum self-adjoint EVD.
//! Positive ABSTOL retains the saved expert driver rather than claiming
//! Sturm/bisection accuracy contracts for the native QR/DC algorithms.
use crate::generalized_selfadjoint::{hermitian, GeneralizedScalar};
use crate::kernels::{flag, Scalar};
use paste::paste;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

pub(crate) static FALLBACKS: [AtomicPtr<c_void>; 4] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; 4];
fn fallback(id: usize) -> *mut c_void {
    let pointer = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat selected self-adjoint EVD requires a saved expert driver"
    );
    pointer
}
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
            if n == 0 {
                return Ok(Selection::All);
            }
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if hi <= lo {
                Err(8)
            } else {
                Ok(Selection::Value(lo, hi))
            }
        }
        b'I' => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > n.max(1) {
                Err(9)
            } else if hi < lo.min(n) || hi > n {
                Err(10)
            } else {
                Ok(Selection::Index((lo - 1) as usize, hi as usize))
            }
        }
        _ => Err(2),
    }
}
impl Selection {
    fn contains(&self, index: usize, value: f64) -> bool {
        match *self {
            Self::All => true,
            Self::Value(lo, hi) => lo < value && value <= hi,
            Self::Index(lo, hi) => lo <= index && index < hi,
        }
    }
}
unsafe fn native<T: GeneralizedScalar, R: Scalar, I: crate::lapack::LapackInt>(
    uplo: u8,
    n: usize,
    a: *const T,
    lda: isize,
    job: u8,
    selected: &Selection,
    m: *mut I,
    w: *mut R,
    z: *mut T,
    ldz: isize,
    ifail: *mut I,
) -> Result<(), ()> {
    let matrix = hermitian(uplo, n, a, lda);
    if !matrix.as_ref().is_all_finite() {
        return Err(());
    }
    let (values, vectors) = crate::lapack::selfadj(matrix.as_ref(), job == b'V')?;
    if !values.iter().all(|x| x.is_finite()) || !vectors.as_ref().is_all_finite() {
        return Err(());
    }
    let mut count = 0;
    for (i, &value) in values.iter().enumerate() {
        if !selected.contains(i, value) {
            continue;
        }
        *w.add(count) = R::real(value);
        if job == b'V' {
            *ifail.add(count) = I::from_i64(0);
            for row in 0..n {
                *z.offset(row as isize + count as isize * ldz) = vectors[(row, i)];
            }
        }
        count += 1;
    }
    *m = I::from_i64(count as i64);
    Ok(())
}
macro_rules! evx {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr,$factor:expr,($($extra:ident:$extra_ty:ty,)*))=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $suffix>](jobz:*const u8,range:*const u8,uplo:*const u8,
            n:*const $i,a:*mut $t,lda:*const $i,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,
            abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,
            work:*mut $t,lwork:*const $i,$($extra:$extra_ty,)*iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            let(job,mode,triangle,nn,la,lz,lw)=(flag(jobz),flag(range),flag(uplo),
                *n as isize,*lda as isize,*ldz as isize,*lwork as isize);
            *info=0;
            for(arg,valid)in [(1,matches!(job,b'N'|b'V')),(2,matches!(mode,b'A'|b'I'|b'V')),
                (3,matches!(triangle,b'U'|b'L')),(4,nn>=0),(6,la>=nn.max(1))] {
                if !valid {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let selected=match selection(mode,nn,vl,vu,il,iu) {Ok(s)=>s,
                Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if lz<1||(job==b'V'&&lz<nn) {*info=-15;crate::invalid(15);return;}
            let minimum=if nn<=1 {1} else {nn.saturating_mul($factor)};
            *work=crate::lapack::workspace::<$t>(minimum);
            if lw != -1 && lw<minimum {*info=-17;crate::invalid(17);return;}
            if lw== -1 {return;}
            *m=0;
            if nn==0 {return;}
            if nn==1 {
                let value=<$t as GeneralizedScalar>::diagonal(a).re();
                if selected.contains(0,value) {*m=1;*w=value as $r;if job==b'V'{*ifail=0;}}
                // LAPACK's scalar quick return writes Z even when the interval
                // excludes its single eigenvalue.
                if job==b'V' {*z=<$t as Scalar>::real(1.0);}
                return;
            }
            if *abstol<=0.0 && native(triangle,nn as usize,a,la,job,&selected,m,w,z,lz,ifail).is_ok() {return;}
            let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const i64,*mut $t,*const i64,
                *const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $t,*const i64,
                *mut $t,*const i64,$($extra_ty,)* *mut i64,*mut i64,*mut i64)=std::mem::transmute(fallback($id));
            let(mut count,mut status)=(0i64,0i64);
            let mut scratch=vec![0i64;5*nn as usize];let mut failed=vec![0i64;nn as usize];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(jobz,range,uplo,&(nn as i64),a,&(la as i64),vl,vu,&lo,&hi,abstol,&mut count,w,z,
                &(lz as i64),work,&(lw as i64),$($extra,)*scratch.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            for j in 0..scratch.len() {*iwork.add(j)=scratch[j] as $i;}
            if job==b'V' {for j in 0..count.max(0).min(nn as i64) as usize {*ifail.add(j)=failed[j] as $i;}}
        }
    }};
}
evx!(s, syevx, f32, f32, i32, _, 0, 8, ());
evx!(d, syevx, f64, f64, i32, _, 1, 8, ());
evx!(c,heevx,colmat::c32,f32,i32,_,2,2,(rwork:*mut f32,));
evx!(z,heevx,colmat::c64,f64,i32,_,3,2,(rwork:*mut f64,));
evx!(s, syevx, f32, f32, i64, _64_, 0, 8, ());
evx!(d, syevx, f64, f64, i64, _64_, 1, 8, ());
evx!(c,heevx,colmat::c32,f32,i64,_64_,2,2,(rwork:*mut f32,));
evx!(z,heevx,colmat::c64,f64,i64,_64_,3,2,(rwork:*mut f64,));
