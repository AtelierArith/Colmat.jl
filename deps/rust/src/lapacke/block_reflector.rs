//! LAPACKE LARFT high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}larft{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larft $($suffix)*>](
            layout:i32,direct:u8,storev:u8,n:$i,k:$i,v:*const $t,ldv:$i,
            tau:*const $t,t:*mut $t,ldt:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (nr,nc)=if storev.to_ascii_uppercase()==b'C' {(n,k)} else {(k,n)};
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,nr as i64,nc as i64,v,ldv as i64).unwrap_or(false) {return -6;}
                if (0..k.max(0) as usize).any(|i| {let x=*tau.add(i);x!=x}) {return -8;}
            }
            [<LAPACKE_ $p larft_work $($suffix)*>](layout,direct,storev,n,k,v,ldv,tau,t,ldt)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larft_work $($suffix)*>](
            layout:i32,direct:u8,storev:u8,n:$i,k:$i,v:*const $t,ldv:$i,
            tau:*const $t,t:*mut $t,ldt:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (nr,nc)=if storev.to_ascii_uppercase()==b'C' {(n,k.max(0))} else {(k.max(0),n)};
            if layout==ROW_MAJOR && ldt<k {return -10;}
            if layout==ROW_MAJOR && ldv<nc {return -7;}
            if layout==COL_MAJOR || n==0 || k==0 {
                crate::householder::[<$p larft $fortran>](&direct,&storev,&n,&k,v,&ldv,tau,t,&ldt);
                return 0;
            }
            let (nn,kk)=(nr as usize,if storev.to_ascii_uppercase()==b'C' {k.max(0) as usize} else {n as usize});
            let mut vt=match row_to_column(nn,kk,v,ldv as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut tt=vec![<$t>::z();(k.max(0) as usize) * (k.max(0) as usize)];
            let ldv_t=nr.max(1) as $i; let ldt_t=k.max(1) as $i;
            crate::householder::[<$p larft $fortran>](&direct,&storev,&n,&k,vt.as_mut_ptr(),&ldv_t,tau,tt.as_mut_ptr(),&ldt_t);
            for i in 0..k as usize {for j in 0..k as usize {*t.add(i*ldt as usize+j)=tt[i+j*k as usize];}}
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
