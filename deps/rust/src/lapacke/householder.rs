//! LAPACKE LARFG high/work adapters (no matrix-layout argument).
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}larfg{,_work}.c.
use super::LAPACKE_get_nancheck;
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfg $($suffix)*>](
            n:$i,alpha:*mut $t,x:*mut $t,incx:$i,tau:*mut $t,
        )->$i {
            if LAPACKE_get_nancheck()!=0 && n>0 {
                if *alpha != *alpha {return -2;}
                if incx!=0 {
                    let first=crate::kernels::start(n as isize,incx as isize);
                    for i in 0..n as isize {
                        let v=*x.offset(first+i*incx as isize);
                        if v!=v {return -3;}
                    }
                }
            }
            [<LAPACKE_ $p larfg_work $($suffix)*>](n,alpha,x,incx,tau)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfg_work $($suffix)*>](
            n:$i,alpha:*mut $t,x:*mut $t,incx:$i,tau:*mut $t,
        )->$i {
            crate::householder::[<$p larfg $fortran>](&n,alpha,x,&incx,tau);
            0
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
