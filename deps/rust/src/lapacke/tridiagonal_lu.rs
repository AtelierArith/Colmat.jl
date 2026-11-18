//! LAPACKE GTTRF/GTTRS/GTSV high/work adapters for general tridiagonal LU.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{gttrf,gttrs,gtsv}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gttrf $($suffix)*>](
            n:$i,dl:*mut $t,d:*mut $t,du:*mut $t,du2:*mut $t,ipiv:*mut $i,
        )->$i {
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -3;}
                if nan_vec((n-1).max(0) as i64,dl) {return -2;}
                if nan_vec((n-1).max(0) as i64,du) {return -4;}
            }
            let mut info:$i=0;
            crate::tridiagonal_lu::[<$p gttrf $fortran>](&n,dl,d,du,du2,ipiv,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gttrf_work $($suffix)*>](
            n:$i,dl:*mut $t,d:*mut $t,du:*mut $t,du2:*mut $t,ipiv:*mut $i,
        )->$i {
            let mut info:$i=0;
            crate::tridiagonal_lu::[<$p gttrf $fortran>](&n,dl,d,du,du2,ipiv,&mut info);
            info
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gttrs $($suffix)*>](
            layout:i32,trans:u8,n:$i,nrhs:$i,dl:*const $t,d:*const $t,
            du:*const $t,du2:*const $t,ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -10;}
                if nan_vec(n as i64,d) {return -6;}
                if nan_vec((n-1).max(0) as i64,dl) {return -5;}
                if nan_vec((n-1).max(0) as i64,du) {return -7;}
                if nan_vec((n-2).max(0) as i64,du2) {return -8;}
            }
            [<LAPACKE_ $p gttrs_work $($suffix)*>](layout,trans,n,nrhs,dl,d,du,du2,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gttrs_work $($suffix)*>](
            layout:i32,trans:u8,n:$i,nrhs:$i,dl:*const $t,d:*const $t,
            du:*const $t,du2:*const $t,ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -11;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::tridiagonal_lu::[<$p gttrs $fortran>](&trans,&n,&nrhs,dl,d,du,du2,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_lu::[<$p gttrs $fortran>](&trans,&n,&nrhs,dl,d,du,du2,ipiv,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gtsv $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,dl:*mut $t,d:*mut $t,du:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -7;}
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec((n-1).max(0) as i64,dl) {return -4;}
                if nan_vec((n-1).max(0) as i64,du) {return -6;}
            }
            [<LAPACKE_ $p gtsv_work $($suffix)*>](layout,n,nrhs,dl,d,du,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gtsv_work $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,dl:*mut $t,d:*mut $t,du:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -8;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::tridiagonal_lu::[<$p gtsv $fortran>](&n,&nrhs,dl,d,du,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::tridiagonal_lu::[<$p gtsv $fortran>](&n,&nrhs,dl,d,du,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
