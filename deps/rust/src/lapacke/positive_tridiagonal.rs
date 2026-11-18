//! LAPACKE PTTRF/PTTRS high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{pttrf,pttrs}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrf $($suffix)*>](n:$i,d:*mut $t,e:*mut $t)->$i {
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -2;}
                if nan_vec((n-1).max(0) as i64,e) {return -3;}
            }
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrf $fortran>](&n,d,e,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrf_work $($suffix)*>](n:$i,d:*mut $t,e:*mut $t)->$i {
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrf $fortran>](&n,d,e,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrs $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,d:*const $t,e:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -6;}
                if nan_vec(n as i64,d) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
            }
            [<LAPACKE_ $p pttrs_work $($suffix)*>](layout,n,nrhs,d,e,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrs_work $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,d:*const $t,e:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -7;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::positive_tridiagonal::[<$p pttrs $fortran>](&n,&nrhs,d,e,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrs $fortran>](&n,&nrhs,d,e,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrf $($suffix)*>](n:$i,d:*mut $r,e:*mut $t)->$i {
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -2;}
                if nan_vec((n-1).max(0) as i64,e) {return -3;}
            }
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrf $fortran>](&n,d,e,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrf_work $($suffix)*>](n:$i,d:*mut $r,e:*mut $t)->$i {
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrf $fortran>](&n,d,e,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrs $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,d:*const $r,e:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -7;}
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec((n-1).max(0) as i64,e) {return -6;}
            }
            [<LAPACKE_ $p pttrs_work $($suffix)*>](layout,uplo,n,nrhs,d,e,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pttrs_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,d:*const $r,e:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -8;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::positive_tridiagonal::[<$p pttrs $fortran>](&uplo,&n,&nrhs,d,e,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p pttrs $fortran>](&uplo,&n,&nrhs,d,e,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
real!(s, f32, i32, [], _);
real!(d, f64, i32, [], _);
complex!(c, colmat::c32, f32, i32, [], _);
complex!(z, colmat::c64, f64, i32, [], _);
real!(s, f32, i64, [64_], _64_);
real!(d, f64, i64, [64_], _64_);
complex!(c, colmat::c32, f32, i64, [64_], _64_);
complex!(z, colmat::c64, f64, i64, [64_], _64_);
