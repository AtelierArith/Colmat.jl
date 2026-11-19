//! LAPACKE SPTRD/HPTRD high/work adapters for packed symmetric/Hermitian
//! tridiagonal reduction.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{sptrd,hptrd}{,_work}.c.
use super::packed_lu::{packed_from_column, packed_nan, packed_to_column};
use super::{LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,d:*mut $r,e:*mut $r,tau:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && packed_nan(layout,uplo,n.max(0) as usize,ap,<$t>::COMPLEX) {
                return -4;
            }
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,uplo,n,ap,d,e,tau)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,d:*mut $r,e:*mut $r,tau:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n<=0 {
                let mut info:$i=0;
                crate::structured_tridiagonal::[<$p $f $fort>](&uplo,&n,ap,d,e,tau,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tmp=packed_to_column(layout,uplo,nn,ap);
            let mut info:$i=0;
            crate::structured_tridiagonal::[<$p $f $fort>](&uplo,&n,tmp.as_mut_ptr(),d,e,tau,&mut info);
            if info>=0 {packed_from_column(layout,uplo,nn,ap,&tmp);}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, sptrd, f32, f32, i32, [], _);
exports!(d, sptrd, f64, f64, i32, [], _);
exports!(c, hptrd, colmat::c32, f32, i32, [], _);
exports!(z, hptrd, colmat::c64, f64, i32, [], _);
exports!(s, sptrd, f32, f32, i64, [64_], _64_);
exports!(d, sptrd, f64, f64, i64, [64_], _64_);
exports!(c, hptrd, colmat::c32, f32, i64, [64_], _64_);
exports!(z, hptrd, colmat::c64, f64, i64, [64_], _64_);
