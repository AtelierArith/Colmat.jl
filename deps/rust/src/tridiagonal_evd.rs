//! Native kernels shared by the tridiagonal LAPACK drivers.
use crate::kernels::Scalar;
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::linalg::evd::*;
use colmat::Mat;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

// Saved ILP64 implementations provide LAPACK's partial-result contract when
// the native API reports only NoConvergence. LP64 entry points widen scalars.
pub(crate) static FALLBACKS: [AtomicPtr<c_void>; 24] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; 24];
fn fallback(id: usize) -> *mut c_void {
    let pointer = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !pointer.is_null(),
        "Colmat tridiagonal fallback must be configured before use"
    );
    pointer
}

// Keep the native error distinct: its public API does not expose the partially
// converged D/E needed to synthesize a LAPACK nonconvergence result.
pub(crate) fn eigen<T: Scalar>(
    diagonal: &[T],
    offdiagonal: &[T],
    vectors: bool,
    divide: bool,
) -> Result<(Vec<T>, Mat<T>), ()> {
    let n = diagonal.len();
    assert_eq!(offdiagonal.len(), n.saturating_sub(1));
    if !diagonal
        .iter()
        .chain(offdiagonal)
        .all(|x| x.magnitude().is_finite())
    {
        return Err(());
    }
    let mut d = Diag::<T>::zeros(n);
    let mut e = Diag::<T>::zeros(n.saturating_sub(1));
    for i in 0..n {
        d[i] = diagonal[i];
    }
    for i in 0..n.saturating_sub(1) {
        e[i] = offdiagonal[i];
    }
    let mut s = Diag::<T>::zeros(n);
    let mut u = Mat::<T>::zeros(n, if vectors { n } else { 0 });
    let mut params: colmat::Spec<SelfAdjointEvdParams, T> = Default::default();
    if !divide {
        params.config.recursion_threshold = usize::MAX;
    }
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(tridiagonal_self_adjoint_evd_scratch::<T>(
            n,
            if vectors {
                ComputeEigenvectors::Yes
            } else {
                ComputeEigenvectors::No
            },
            par.shape(),
            params,
        ));
        tridiagonal_self_adjoint_evd(
            d.as_ref(),
            e.as_ref(),
            s.as_mut(),
            if vectors { Some(u.as_mut()) } else { None },
            par,
            MemStack::new(&mut memory),
            params,
        )
        .map_err(|_| ())?;
        if !(0..n).all(|i| s[i].magnitude().is_finite())
            || !(0..u.ncols()).all(|j| (0..n).all(|i| u[(i, j)].magnitude().is_finite()))
        {
            return Err(());
        }
        Ok(((0..n).map(|i| s[i]).collect(), u))
    })
}

use crate::kernels::flag;
use paste::paste;
unsafe fn accumulate<T: Scalar>(z: *const T, n: usize, ld: isize, vectors: &Mat<T>) -> Mat<T> {
    let mut result = Mat::<T>::zeros(n, n);
    crate::threading::with_par(|par| {
        colmat::linalg::matmul::matmul(
            result.as_mut(),
            colmat::Accum::Replace,
            crate::lapack::input_ref(z, n, n, ld),
            vectors.as_ref(),
            T::o(),
            par,
        );
    });
    result
}
macro_rules! qr_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p steqr $suffix>](compz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $r,info:*mut $i) {
            *info=0;
            let (mode,nn,ld)=(flag(compz),*n as isize,*ldz as isize);
            for (argument,valid) in [(1,matches!(mode,b'N'|b'I'|b'V')),(2,nn>=0),(6,ld>=1&&(mode==b'N'||ld>=nn.max(1)))] {
                if !valid { *info=-argument;crate::invalid(argument as i32);return; }
            }
            if nn==0 { return; }
            if nn==1 { if mode==b'I' { *z=<$t as Scalar>::o(); } return; }
            let size=nn as usize;
            let diagonal=std::slice::from_raw_parts(d,size);
            let offdiagonal=std::slice::from_raw_parts(e,size.saturating_sub(1));
            match eigen::<$r>(diagonal,offdiagonal,mode!=b'N',false) {
                Ok((values,vectors))=>{
                    if mode!=b'N' {
                        let vectors=Mat::<$t>::from_fn(size,size,|i,j| <$t as Scalar>::real(vectors[(i,j)] as f64));
                        let result=if mode==b'V' { accumulate(z,size,ld,&vectors) } else { vectors };
                        crate::lapack::output(&result,z,ld);
                    }
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size.saturating_sub(1) { *e.add(i)=0.0; }
                },
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const i64,*mut $r,*mut $r,*mut $t,*const i64,*mut $r,*mut i64)=std::mem::transmute(fallback($id));
                    let mut status=0i64;
                    f(compz,&(nn as i64),d,e,z,&(ld as i64),work,&mut status);
                    *info=status as $i;
                }
            }
        }
    }};
}
macro_rules! sterf_exports {
    ($p:ident,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p sterf $suffix>](n:*const $i,d:*mut $r,e:*mut $r,info:*mut $i) {
            *info=0;let nn=*n as isize;
            if nn<0 { *info=-1;crate::invalid(1);return; }
            if nn<=1 { return; }
            let size=nn as usize;
            match eigen::<$r>(std::slice::from_raw_parts(d,size),std::slice::from_raw_parts(e,size-1),false,false) {
                Ok((values,_))=>{
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size-1 { *e.add(i)=0.0; }
                },
                Err(())=>{
                    let f:unsafe extern "C" fn(*const i64,*mut $r,*mut $r,*mut i64)=std::mem::transmute(fallback($id));
                    let mut status=0i64;f(&(nn as i64),d,e,&mut status);*info=status as $i;
                }
            }
        }
    }};
}
qr_exports!(s, f32, f32, i32, _, 0);
qr_exports!(d, f64, f64, i32, _, 1);
qr_exports!(c, colmat::c32, f32, i32, _, 2);
qr_exports!(z, colmat::c64, f64, i32, _, 3);
qr_exports!(s, f32, f32, i64, _64_, 0);
qr_exports!(d, f64, f64, i64, _64_, 1);
qr_exports!(c, colmat::c32, f32, i64, _64_, 2);
qr_exports!(z, colmat::c64, f64, i64, _64_, 3);
sterf_exports!(s, f32, i32, _, 4);
sterf_exports!(d, f64, i32, _, 5);
sterf_exports!(s, f32, i64, _64_, 4);
sterf_exports!(d, f64, i64, _64_, 5);

macro_rules! driver_exports {
    ($p:ident,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stev $suffix>](jobz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:*const $i,work:*mut $r,info:*mut $i) {
            let mode=flag(jobz);
            if !matches!(mode,b'N'|b'V') { *info=-1;crate::invalid(1);return; }
            [<$p steqr $suffix>](&(if mode==b'V' { b'I' } else { b'N' }),n,d,e,z,ldz,work,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stevd $suffix>](jobz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:*const $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;
            let(mode,nn,ld,lw,li)=(flag(jobz),*n as isize,*ldz as isize,*lwork as isize,*liwork as isize);
            let query=lw == -1 || li == -1;
            for(argument,valid)in[(1,matches!(mode,b'N'|b'V')),(2,nn>=0),(6,ld>=1&&(mode==b'N'||ld>=nn.max(1)))] {
                if !valid { *info=-argument;crate::invalid(argument as i32);return; }
            }
            let (minimum,integers)=if nn>1&&mode==b'V' {(1isize.saturating_add(4isize.saturating_mul(nn)).saturating_add(nn.saturating_mul(nn)),3isize.saturating_add(5isize.saturating_mul(nn)))}else{(1,1)};
            *work=crate::lapack::workspace::<$r>(minimum);*iwork=integers as $i;
            for(argument,valid)in[(8,query||lw>=minimum),(10,query||li>=integers)] {
                if !valid { *info=-argument;crate::invalid(argument as i32);return; }
            }
            if query||nn==0 { return; }
            if nn==1 { if mode==b'V' { *z=1.0; } return; }
            let size=nn as usize;
            match eigen::<$r>(std::slice::from_raw_parts(d,size),std::slice::from_raw_parts(e,size-1),mode==b'V',true) {
                Ok((values,vectors))=>{
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size-1 { *e.add(i)=0.0; }
                    if mode==b'V' { crate::lapack::output(&vectors,z,ld); }
                },
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const i64,*mut $r,*mut $r,*mut $r,*const i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut scratch=vec![0i64;integers as usize];let mut status=0i64;
                    f(jobz,&(nn as i64),d,e,z,&(ld as i64),work,&(lw as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);
                    *iwork=scratch[0] as $i;*info=status as $i;
                }
            }
        }
    }};
}
driver_exports!(s, f32, i32, _, 6);
driver_exports!(d, f64, i32, _, 7);
driver_exports!(s, f32, i64, _64_, 6);
driver_exports!(d, f64, i64, _64_, 7);

macro_rules! positive_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p pteqr $suffix>](compz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $r,info:*mut $i) {
            *info=0;
            let(mode,nn,ld)=(flag(compz),*n as isize,*ldz as isize);
            for(argument,valid)in[(1,matches!(mode,b'N'|b'I'|b'V')),(2,nn>=0),(6,ld>=1&&(mode==b'N'||ld>=nn.max(1)))] {
                if !valid { *info=-argument;crate::invalid(argument as i32);return; }
            }
            if nn==0 { return; }
            if nn==1 { if mode!=b'N' { *z=<$t as Scalar>::o(); } return; }
            let size=nn as usize;
            let mut diagonal=std::slice::from_raw_parts(d,size).to_vec();
            let mut offdiagonal=std::slice::from_raw_parts(e,size-1).to_vec();
            match positive(&mut diagonal,&mut offdiagonal) {
                Ok((values,vectors))=>{
                    if mode!=b'N' {
                        let vectors=Mat::<$t>::from_fn(size,size,|i,j| <$t as Scalar>::real(vectors[(i,j)] as f64));
                        let result=if mode==b'V' { accumulate(z,size,ld,&vectors) } else { vectors };
                        crate::lapack::output(&result,z,ld);
                    }
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size-1 { *e.add(i)=0.0; }
                },
                Err(PositiveFailure::NonPositive(index))=>{
                    *info=index as $i;
                    for i in 0..size { *d.add(i)=diagonal[i]; }
                    for i in 0..size-1 { *e.add(i)=offdiagonal[i]; }
                    if mode==b'I' { for j in 0..size { for i in 0..size { *z.offset(i as isize+j as isize*ld)=if i==j { <$t as Scalar>::o() } else { <$t as Scalar>::z() }; } } }
                },
                Err(PositiveFailure::NoConvergence)=>{
                    let f:unsafe extern "C" fn(*const u8,*const i64,*mut $r,*mut $r,*mut $t,*const i64,*mut $r,*mut i64)=std::mem::transmute(fallback($id));
                    let mut status=0i64;f(compz,&(nn as i64),d,e,z,&(ld as i64),work,&mut status);*info=status as $i;
                }
            }
        }
    }};
}
positive_exports!(s, f32, f32, i32, _, 8);
positive_exports!(d, f64, f64, i32, _, 9);
positive_exports!(c, colmat::c32, f32, i32, _, 10);
positive_exports!(z, colmat::c64, f64, i32, _, 11);
positive_exports!(s, f32, f32, i64, _64_, 8);
positive_exports!(d, f64, f64, i64, _64_, 9);
positive_exports!(c, colmat::c32, f32, i64, _64_, 10);
positive_exports!(z, colmat::c64, f64, i64, _64_, 11);

// Match LAPACK's small-problem QR workspace (SMLSIZ=25). Larger problems use
// the documented divide-and-conquer minima; the actual scratch is owned here.
fn dc_workspace(n: isize, mode: u8, complex: bool) -> (isize, isize, isize) {
    if n <= 1 || mode == b'N' {
        return (1, 1, 1);
    }
    if n <= 25 {
        return if complex {
            (1, 2 * (n - 1), 1)
        } else {
            (2 * (n - 1), 1, 1)
        };
    }
    let lg = (isize::BITS - (n - 1).leading_zeros()) as isize;
    let square = n.saturating_mul(n);
    let (real, integer) = if mode == b'V' {
        (
            1isize
                .saturating_add(n.saturating_mul(3))
                .saturating_add(n.saturating_mul(lg).saturating_mul(2))
                .saturating_add(4isize.saturating_mul(square)),
            6isize
                .saturating_add(n.saturating_mul(6))
                .saturating_add(n.saturating_mul(lg).saturating_mul(5)),
        )
    } else {
        (
            1isize
                .saturating_add(n.saturating_mul(4))
                .saturating_add(if complex {
                    2isize.saturating_mul(square)
                } else {
                    square
                }),
            3isize.saturating_add(n.saturating_mul(5)),
        )
    };
    if complex {
        (if mode == b'V' { square } else { 1 }, real, integer)
    } else {
        (real, 1, integer)
    }
}

macro_rules! dc_real_exports {
    ($p:ident,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stedc $suffix>](compz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $r,ldz:*const $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;let(mode,nn,ld,lw,li)=(flag(compz),*n as isize,*ldz as isize,*lwork as isize,*liwork as isize);
            let query=lw == -1 || li == -1;
            for(argument,valid)in[(1,matches!(mode,b'N'|b'I'|b'V')),(2,nn>=0),(6,ld>=1&&(mode==b'N'||ld>=nn.max(1)))] { if !valid { *info=-argument;crate::invalid(argument as i32);return; } }
            let(minimum,_,integers)=dc_workspace(nn,mode,false);
            *work=crate::lapack::workspace::<$r>(minimum);*iwork=integers as $i;
            for(argument,valid)in[(8,query||lw>=minimum),(10,query||li>=integers)] { if !valid { *info=-argument;crate::invalid(argument as i32);return; } }
            if query||nn==0 { return; }
            if nn==1 { if mode!=b'N' { *z=1.0; } return; }
            let size=nn as usize;
            match eigen::<$r>(std::slice::from_raw_parts(d,size),std::slice::from_raw_parts(e,size-1),mode!=b'N',true) {
                Ok((values,vectors))=>{
                    if mode!=b'N' {
                        let result=if mode==b'V' { accumulate(z,size,ld,&vectors) }else{vectors};
                        crate::lapack::output(&result,z,ld);
                    }
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size-1 { *e.add(i)=0.0; }
                },
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const i64,*mut $r,*mut $r,*mut $r,*const i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut scratch=vec![0i64;integers as usize];let mut status=0i64;
                    f(compz,&(nn as i64),d,e,z,&(ld as i64),work,&(lw as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);
                    *iwork=scratch[0] as $i;*info=status as $i;
                }
            }
        }
    }};
}
macro_rules! dc_complex_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stedc $suffix>](compz:*const u8,n:*const $i,d:*mut $r,e:*mut $r,z:*mut $t,ldz:*const $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,lrwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;let(mode,nn,ld,lw,lr,li)=(flag(compz),*n as isize,*ldz as isize,*lwork as isize,*lrwork as isize,*liwork as isize);
            let query=lw == -1 || lr == -1 || li == -1;
            for(argument,valid)in[(1,matches!(mode,b'N'|b'I'|b'V')),(2,nn>=0),(6,ld>=1&&(mode==b'N'||ld>=nn.max(1)))] { if !valid { *info=-argument;crate::invalid(argument as i32);return; } }
            let(minimum,reals,integers)=dc_workspace(nn,mode,true);
            *work=crate::lapack::workspace::<$t>(minimum);*rwork=crate::lapack::workspace::<$r>(reals);*iwork=integers as $i;
            for(argument,valid)in[(8,query||lw>=minimum),(10,query||lr>=reals),(12,query||li>=integers)] { if !valid { *info=-argument;crate::invalid(argument as i32);return; } }
            if query||nn==0 { return; }
            if nn==1 { if mode!=b'N' { *z=<$t as Scalar>::o(); } return; }
            let size=nn as usize;
            match eigen::<$r>(std::slice::from_raw_parts(d,size),std::slice::from_raw_parts(e,size-1),mode!=b'N',true) {
                Ok((values,vectors))=>{
                    if mode!=b'N' {
                        let vectors=Mat::<$t>::from_fn(size,size,|i,j| <$t as Scalar>::real(vectors[(i,j)] as f64));
                        let result=if mode==b'V' { accumulate(z,size,ld,&vectors) }else{vectors};
                        crate::lapack::output(&result,z,ld);
                    }
                    for i in 0..size { *d.add(i)=values[i]; }
                    for i in 0..size-1 { *e.add(i)=0.0; }
                },
                Err(())=>{
                    let f:unsafe extern "C" fn(*const u8,*const i64,*mut $r,*mut $r,*mut $t,*const i64,*mut $t,*const i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                    let mut scratch=vec![0i64;integers as usize];let mut status=0i64;
                    f(compz,&(nn as i64),d,e,z,&(ld as i64),work,&(lw as i64),rwork,&(lr as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);
                    *iwork=scratch[0] as $i;*info=status as $i;
                }
            }
        }
    }};
}
dc_real_exports!(s, f32, i32, _, 12);
dc_real_exports!(d, f64, i32, _, 13);
dc_complex_exports!(c, colmat::c32, f32, i32, _, 14);
dc_complex_exports!(z, colmat::c64, f64, i32, _, 15);
dc_real_exports!(s, f32, i64, _64_, 12);
dc_real_exports!(d, f64, i64, _64_, 13);
dc_complex_exports!(c, colmat::c32, f32, i64, _64_, 14);
dc_complex_exports!(z, colmat::c64, f64, i64, _64_, 15);

enum Selection {
    All,
    Value(f64, f64),
    Index(usize, usize),
}
impl Selection {
    fn full(&self, n: usize) -> bool {
        matches!(self, Self::All) || matches!(self,Self::Index(0,hi) if *hi==n)
    }
    fn indices<T: Scalar>(&self, values: &[T]) -> Vec<usize> {
        (0..values.len())
            .filter(|&i| match *self {
                Self::All => true,
                Self::Value(lo, hi) => values[i].re() > lo && values[i].re() <= hi,
                Self::Index(lo, hi) => i >= lo && i < hi,
            })
            .collect()
    }
}
unsafe fn selection<R: Scalar, I: crate::lapack::LapackInt>(
    mode: u8,
    n: isize,
    vl: *const R,
    vu: *const R,
    il: *const I,
    iu: *const I,
    stemr: bool,
) -> Result<Selection, i32> {
    match mode {
        b'A' => Ok(Selection::All),
        b'V' => {
            let (lo, hi) = ((*vl).re(), (*vu).re());
            if n > 0 && hi <= lo {
                Err(7)
            } else {
                Ok(Selection::Value(lo, hi))
            }
        }
        b'I' => {
            let (lo, hi) = ((*il).val(), (*iu).val());
            if lo < 1 || lo > if stemr { n } else { n.max(1) } {
                return Err(8);
            }
            if hi < if stemr { lo } else { lo.min(n) } || hi > n {
                return Err(9);
            }
            Ok(Selection::Index((lo - 1) as usize, hi as usize))
        }
        _ => Err(2),
    }
}
unsafe fn spectrum<R: Scalar>(
    d: *const R,
    e: *const R,
    n: usize,
    vectors: bool,
) -> Result<(Vec<R>, Mat<R>), ()> {
    let diagonal = if n == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(d, n).to_vec()
    };
    let offdiagonal = if n <= 1 {
        vec![]
    } else {
        std::slice::from_raw_parts(e, n - 1).to_vec()
    };
    eigen(&diagonal, &offdiagonal, vectors, true)
}
unsafe fn selected_output<R: Scalar, T: Scalar, I: crate::lapack::LapackInt>(
    values: &[R],
    vectors: &Mat<R>,
    indices: &[usize],
    w: *mut R,
    z: *mut T,
    ld: isize,
    wantz: bool,
    support: *mut I,
    failed: *mut I,
) {
    for (column, &source) in indices.iter().enumerate() {
        *w.add(column) = values[source];
        if wantz {
            for row in 0..vectors.nrows() {
                *z.offset(row as isize + column as isize * ld) =
                    T::real(vectors[(row, source)].re());
            }
            if !support.is_null() {
                let first = (0..vectors.nrows())
                    .find(|&i| vectors[(i, source)] != R::z())
                    .unwrap_or(0);
                let last = (0..vectors.nrows())
                    .rfind(|&i| vectors[(i, source)] != R::z())
                    .unwrap_or(first);
                *support.add(2 * column) = I::from_i64((first + 1) as i64);
                *support.add(2 * column + 1) = I::from_i64((last + 1) as i64);
            }
            if !failed.is_null() {
                *failed.add(column) = I::from_i64(0);
            }
        }
    }
}

macro_rules! range_real_exports {
    ($p:ident,$r:ty,$i:ty,$suffix:tt,$rid:expr,$xid:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stevr $suffix>](jobz:*const u8,range:*const u8,n:*const $i,d:*mut $r,e:*mut $r,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $r,ldz:*const $i,isuppz:*mut $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;
            let(job,mode,nn,ld,lw,li)=(flag(jobz),flag(range),*n as isize,*ldz as isize,*lwork as isize,*liwork as isize);
            for(argument,valid)in[(1,matches!(job,b'N'|b'V')),(2,matches!(mode,b'A'|b'V'|b'I')),(3,nn>=0)]{ if !valid{*info=-argument;crate::invalid(argument as i32);return;} }
            let selected=match selection(mode,nn,vl,vu,il,iu,false){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if ld<1||(job==b'V'&&ld<nn) { *info=-14;crate::invalid(14);return; }
            let(minimum,integers)=(20isize.saturating_mul(nn).max(1),10isize.saturating_mul(nn).max(1));
            let query=lw == -1||li == -1;
            *work=crate::lapack::workspace::<$r>(minimum);*iwork=integers as $i;
            for(argument,valid)in[(17,query||lw>=minimum),(19,query||li>=integers)] { if !valid{*info=-argument;crate::invalid(argument as i32);return;} }
            if query { return; }
            *m=0;if nn==0 {return;}
            if *abstol<=0.0 {
                if let Ok((values,vectors))=spectrum(d,e,nn as usize,job==b'V') {
                    let indices=selected.indices(&values);*m=indices.len() as $i;
                    selected_output(&values,&vectors,&indices,w,z,ld,job==b'V',if selected.full(nn as usize){isuppz}else{std::ptr::null_mut()},std::ptr::null_mut::<$i>());return;
                }
            }
            let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*mut $r,*mut $r,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $r,*const i64,*mut i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($rid));
            let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;integers as usize];let mut support=vec![0i64;2*(nn as usize).max(1)];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(jobz,range,&(nn as i64),d,e,vl,vu,&lo,&hi,abstol,&mut count,w,z,&(ld as i64),support.as_mut_ptr(),work,&(lw as i64),scratch.as_mut_ptr(),&(integers as i64),&mut status);
            *m=count as $i;*info=status as $i;*iwork=scratch[0] as $i;
            if job==b'V'&&selected.full(nn as usize) { for k in 0..2*(count.max(0).min(nn as i64) as usize){*isuppz.add(k)=support[k] as $i;} }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stevx $suffix>](jobz:*const u8,range:*const u8,n:*const $i,d:*mut $r,e:*mut $r,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $r,ldz:*const $i,work:*mut $r,iwork:*mut $i,ifail:*mut $i,info:*mut $i) {
            *info=0;let(job,mode,nn,ld)=(flag(jobz),flag(range),*n as isize,*ldz as isize);
            for(argument,valid)in[(1,matches!(job,b'N'|b'V')),(2,matches!(mode,b'A'|b'V'|b'I')),(3,nn>=0)]{ if !valid{*info=-argument;crate::invalid(argument as i32);return;} }
            let selected=match selection(mode,nn,vl,vu,il,iu,false){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if ld<1||(job==b'V'&&ld<nn) { *info=-14;crate::invalid(14);return; }
            *m=0;if nn==0 {return;}
            if *abstol<=0.0 {
                if let Ok((values,vectors))=spectrum(d,e,nn as usize,job==b'V') {
                    let indices=selected.indices(&values);*m=indices.len() as $i;
                    selected_output(&values,&vectors,&indices,w,z,ld,job==b'V',std::ptr::null_mut::<$i>(),ifail);return;
                }
            }
            let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*mut $r,*mut $r,*const $r,*const $r,*const i64,*const i64,*const $r,*mut i64,*mut $r,*mut $r,*const i64,*mut $r,*mut i64,*mut i64,*mut i64)=std::mem::transmute(fallback($xid));
            let(mut count,mut status)=(0i64,0i64);let mut scratch=vec![0i64;5*(nn as usize).max(1)];let mut failed=vec![0i64;(nn as usize).max(1)];
            let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
            f(jobz,range,&(nn as i64),d,e,vl,vu,&lo,&hi,abstol,&mut count,w,z,&(ld as i64),work,scratch.as_mut_ptr(),failed.as_mut_ptr(),&mut status);
            *m=count as $i;*info=status as $i;
            if job==b'V' { for k in 0..count.max(0).min(nn as i64) as usize {*ifail.add(k)=failed[k] as $i;} }
            let _=iwork; // Native and bridged fallback own integer scratch.
        }
    }};
}
range_real_exports!(s, f32, i32, _, 16, 18);
range_real_exports!(d, f64, i32, _, 17, 19);
range_real_exports!(s, f32, i64, _64_, 16, 18);
range_real_exports!(d, f64, i64, _64_, 17, 19);

macro_rules! stemr_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt,$id:expr) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stemr $suffix>](jobz:*const u8,range:*const u8,n:*const $i,d:*mut $r,e:*mut $r,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,nzc:*const $i,isuppz:*mut $i,tryrac:*mut $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            *info=0;
            let(job,mode,nn,ld,nz,lw,li)=(flag(jobz),flag(range),*n as isize,*ldz as isize,*nzc as isize,*lwork as isize,*liwork as isize);
            for(argument,valid)in[(1,matches!(job,b'N'|b'V')),(2,matches!(mode,b'A'|b'V'|b'I')),(3,nn>=0)]{ if !valid{*info=-argument;crate::invalid(argument as i32);return;} }
            let selected=match selection(mode,nn,vl,vu,il,iu,true){Ok(s)=>s,Err(arg)=>{*info=-arg as $i;crate::invalid(arg);return;}};
            if ld<1||(job==b'V'&&ld<nn) { *info=-13;crate::invalid(13);return; }
            let(minimum,integers)=(nn.saturating_mul(if job==b'V'{18}else{12}),nn.saturating_mul(if job==b'V'{10}else{8}));
            let query=lw == -1||li == -1;let zquery=nz == -1;
            for(argument,valid)in[(17,query||lw>=minimum),(19,query||li>=integers)] { if !valid{*info=-argument;crate::invalid(argument as i32);return;} }
            *work=crate::lapack::workspace::<$r>(minimum);*iwork=integers as $i;
            let forward=move || {
                let f:unsafe extern "C" fn(*const u8,*const u8,*const i64,*mut $r,*mut $r,*const $r,*const $r,*const i64,*const i64,*mut i64,*mut $r,*mut $t,*const i64,*const i64,*mut i64,*mut i64,*mut $r,*const i64,*mut i64,*const i64,*mut i64)=std::mem::transmute(fallback($id));
                let(mut count,mut status)=(0i64,0i64);
                let mut logical=if query||zquery{0}else{*tryrac as i64};
                let mut scratch=vec![0i64;if query{1}else{integers.max(1) as usize}];let mut support=vec![0i64;2*(nn as usize).max(1)];
                let(lo,hi)=if mode==b'I'{(*il as i64,*iu as i64)}else{(1,nn as i64)};
                f(jobz,range,&(nn as i64),d,e,vl,vu,&lo,&hi,&mut count,w,z,&(ld as i64),&(nz as i64),support.as_mut_ptr(),&mut logical,work,&(lw as i64),scratch.as_mut_ptr(),&(if query{li as i64}else{integers as i64}),&mut status);
                *info=status as $i;*iwork=scratch[0] as $i;
                if !query&&!zquery {
                    *m=count as $i;*tryrac=logical as $i;
                    if job==b'V'{for k in 0..2*(count.max(0).min(nn as i64) as usize){*isuppz.add(k)=support[k] as $i;}}
                }
            };
            // Relative-accuracy preserving MRRR remains an explicit fallback.
            if !query&&!zquery&&nn>0&&*tryrac!=0 { forward();return; }
            let count=if job==b'N'{0}else{match selected {
                Selection::All=>nn as usize,
                Selection::Index(lo,hi)=>hi-lo,
                Selection::Value(_,_)=>match spectrum(d,e,nn as usize,false){Ok((values,_))=>selected.indices(&values).len(),Err(())=>{forward();return;}},
            }};
            if zquery { *z=<$t as Scalar>::real(count as f64); }
            else if nz<(count as isize) { *info=-14;crate::invalid(14);return; }
            if query||zquery { return; }
            *m=0;if nn==0 {return;}
            match spectrum(d,e,nn as usize,job==b'V') {
                Ok((values,vectors))=>{
                    let indices=selected.indices(&values);*m=indices.len() as $i;
                    selected_output(&values,&vectors,&indices,w,z,ld,job==b'V',isuppz,std::ptr::null_mut::<$i>());
                },
                Err(())=>forward(),
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p stegr $suffix>](jobz:*const u8,range:*const u8,n:*const $i,d:*mut $r,e:*mut $r,vl:*const $r,vu:*const $r,il:*const $i,iu:*const $i,abstol:*const $r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:*const $i,isuppz:*mut $i,work:*mut $r,lwork:*const $i,iwork:*mut $i,liwork:*const $i,info:*mut $i) {
            let mut relative=0 as $i;
            [<$p stemr $suffix>](jobz,range,n,d,e,vl,vu,il,iu,m,w,z,ldz,n,isuppz,&mut relative,work,lwork,iwork,liwork,info);
            let _=abstol; // Like Reference LAPACK, STEGR does not use ABSTOL.
        }
    }};
}
stemr_exports!(s, f32, f32, i32, _, 20);
stemr_exports!(d, f64, f64, i32, _, 21);
stemr_exports!(c, colmat::c32, f32, i32, _, 22);
stemr_exports!(z, colmat::c64, f64, i32, _, 23);
stemr_exports!(s, f32, f32, i64, _64_, 20);
stemr_exports!(d, f64, f64, i64, _64_, 21);
stemr_exports!(c, colmat::c32, f32, i64, _64_, 22);
stemr_exports!(z, colmat::c64, f64, i64, _64_, 23);

#[derive(Debug, PartialEq)]
pub(crate) enum PositiveFailure {
    NonPositive(usize),
    NoConvergence,
}

// PTEQR uses singular values of the bidiagonal Cholesky factor rather than
// the ordinary tridiagonal EVD: tiny positive eigenvalues need relative accuracy.
pub(crate) fn positive<T: Scalar>(
    diagonal: &mut [T],
    offdiagonal: &mut [T],
) -> Result<(Vec<T>, Mat<T>), PositiveFailure> {
    use colmat::linalg::svd::*;
    assert!(!T::COMPLEX);
    let n = diagonal.len();
    assert_eq!(offdiagonal.len(), n.saturating_sub(1));
    for i in 0..n {
        if diagonal[i].re() <= 0.0 {
            return Err(PositiveFailure::NonPositive(i + 1));
        }
        if i + 1 < n {
            let entry = offdiagonal[i];
            offdiagonal[i] = entry / diagonal[i];
            diagonal[i + 1] = diagonal[i + 1] - offdiagonal[i] * entry;
        }
    }
    for i in 0..n {
        diagonal[i] = T::real(diagonal[i].re().sqrt());
        if i + 1 < n {
            offdiagonal[i] = offdiagonal[i] * diagonal[i];
        }
    }
    let mut values = diagonal.to_vec();
    let mut u = Mat::<T>::zeros(n, n);
    let mut v = Mat::<T>::zeros(n, n);
    let mut params: colmat::Spec<SvdParams, T> = Default::default();
    params.config.recursion_threshold = usize::MAX;
    crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(bidiagonal_svd_scratch::<T>(n, par.shape(), params));
        bidiagonal_svd(
            &mut values,
            offdiagonal,
            false,
            u.as_mut(),
            v.as_mut(),
            par,
            MemStack::new(&mut memory),
            params,
        )
        .map_err(|_| PositiveFailure::NoConvergence)
    })?;
    for x in &mut values {
        *x = *x * *x;
    }
    Ok((values, u))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tridiagonal_reconstruction() {
        for n in [0usize, 1, 2, 7, 40] {
            for divide in [false, true] {
                let d: Vec<f64> = (0..n).map(|i| 2.0 + i as f64 / 9.0).collect();
                let e: Vec<f64> = (0..n.saturating_sub(1))
                    .map(|i| if i == 3 { 0.0 } else { -0.25 })
                    .collect();
                let (s, u) = eigen(&d, &e, true, divide).unwrap();
                let (values, _) = eigen(&d, &e, false, divide).unwrap();
                for j in 0..n {
                    assert!((s[j] - values[j]).abs() < 1e-11);
                    if j > 0 {
                        assert!(s[j - 1] <= s[j]);
                    }
                    for i in 0..n {
                        let mut product = d[i] * u[(i, j)];
                        if i > 0 {
                            product += e[i - 1] * u[(i - 1, j)];
                        }
                        if i + 1 < n {
                            product += e[i] * u[(i + 1, j)];
                        }
                        assert!((product - s[j] * u[(i, j)]).abs() < 1e-11);
                        let dot: f64 = (0..n).map(|k| u[(k, i)] * u[(k, j)]).sum();
                        assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-11);
                    }
                }
            }
        }
    }

    #[test]
    fn positive_tridiagonal_relative_accuracy_and_failure() {
        let mut d = [1.0f64, 1e-100, 1e-200];
        let mut e = [0.0, 0.0];
        let (s, _) = positive(&mut d, &mut e).unwrap();
        for (got, expected) in s.iter().zip([1.0, 1e-100, 1e-200]) {
            assert!((got / expected - 1.0).abs() < 1e-14);
        }
        let mut d = [4.0f64, 3.0, 2.0];
        let mut e = [0.5, -0.25];
        let (s, u) = positive(&mut d, &mut e).unwrap();
        for j in 0..3 {
            for i in 0..3 {
                let mut product = [4.0, 3.0, 2.0][i] * u[(i, j)];
                if i > 0 {
                    product += [0.5, -0.25][i - 1] * u[(i - 1, j)];
                }
                if i < 2 {
                    product += [0.5, -0.25][i] * u[(i + 1, j)];
                }
                assert!((product - s[j] * u[(i, j)]).abs() < 1e-12);
            }
        }
        let mut d = [1.0f64, 1.0, 3.0];
        let mut e = [2.0, 0.5];
        assert!(matches!(
            positive(&mut d, &mut e),
            Err(PositiveFailure::NonPositive(2))
        ));
        assert_eq!(d, [1.0, -3.0, 3.0]);
        assert_eq!(e, [2.0, 0.5]);
    }
}
