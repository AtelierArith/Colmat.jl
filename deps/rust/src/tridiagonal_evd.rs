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
static FALLBACKS: [AtomicPtr<c_void>; 12] = [const { AtomicPtr::new(std::ptr::null_mut()) }; 12];
#[no_mangle]
pub extern "C" fn colmat_tridiagonal_set_fallback(id: i32, pointer: *mut c_void) {
    if let Some(slot) = FALLBACKS.get(id as usize) {
        slot.store(pointer, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_tridiagonal_has_fallback(id: i32) -> i32 {
    FALLBACKS
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
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
        if !(0..n).all(|i| s[i].magnitude().is_finite()) {
            return Err(());
        }
        Ok(((0..n).map(|i| s[i]).collect(), u))
    })
}

use crate::kernels::flag;
use paste::paste;
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
                        let result=if mode==b'V' { &crate::lapack::input(z,size,size,ld) * &vectors } else { vectors };
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
                        let result=if mode==b'V' { &crate::lapack::input(z,size,size,ld) * &vectors } else { vectors };
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
