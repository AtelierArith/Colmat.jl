//! LAPACKE LACGV/LACP2 high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{c,z}
//! {lacgv,lacp2}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! lacgv {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacgv $($suffix)*>](
            n:$i,x:*mut $t,incx:$i,
        )->$i {
            if LAPACKE_get_nancheck()!=0 && n>0 && incx!=0 {
                let first=crate::kernels::start(n as isize,incx as isize);
                for i in 0..n as isize {
                    let v=*x.offset(first+i*incx as isize);
                    if v!=v {return -2;}
                }
            }
            [<LAPACKE_ $p lacgv_work $($suffix)*>](n,x,incx)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacgv_work $($suffix)*>](
            n:$i,x:*mut $t,incx:$i,
        )->$i {
            crate::complex_auxiliary::[<$p lacgv $fortran>](&n,x,&incx);
            0
        }
    }};
}
macro_rules! lacp2 {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacp2 $($suffix)*>](
            layout:i32,uplo:u8,m:$i,n:$i,a:*const $r,lda:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
            }
            [<LAPACKE_ $p lacp2_work $($suffix)*>](layout,uplo,m,n,a,lda,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacp2_work $($suffix)*>](
            layout:i32,uplo:u8,m:$i,n:$i,a:*const $r,lda:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR {
                if lda<n {return -6;}
                if ldb<n {return -8;}
            }
            if layout==COL_MAJOR || m==0 || n==0 {
                crate::complex_auxiliary::[<$p lacp2 $fortran>](&uplo,&m,&n,a,&lda,b,&ldb);
                return 0;
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(mm,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lb)=(mm.max(1) as $i,mm.max(1) as $i);
            crate::complex_auxiliary::[<$p lacp2 $fortran>](&uplo,&m,&n,at.as_mut_ptr(),&la,bt.as_mut_ptr(),&lb);
            for i in 0..mm {for j in 0..nn {*b.add(i*ldb as usize+j)=bt[i+j*mm];}}
            0
        }
    }};
}
lacgv!(c, colmat::c32, f32, i32, [], _);
lacgv!(z, colmat::c64, f64, i32, [], _);
lacgv!(c, colmat::c32, f32, i64, [64_], _64_);
lacgv!(z, colmat::c64, f64, i64, [64_], _64_);
lacp2!(c, colmat::c32, f32, i32, [], _);
lacp2!(z, colmat::c64, f64, i32, [], _);
lacp2!(c, colmat::c32, f32, i64, [64_], _64_);
lacp2!(z, colmat::c64, f64, i64, [64_], _64_);
