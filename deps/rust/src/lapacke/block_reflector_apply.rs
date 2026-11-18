//! LAPACKE LARFB high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}larfb{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfb $($suffix)*>](
            layout:i32,side:u8,trans:u8,direct:u8,storev:u8,m:$i,n:$i,k:$i,
            v:*const $t,ldv:$i,t:*const $t,ldt:$i,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let left=side.to_ascii_uppercase()==b'L';
            let col=storev.to_ascii_uppercase()==b'C';
            let (nrv,ncv)=match (col,left) {
                (true,true)=>(m,k.max(0)),(true,false)=>(n,k.max(0)),
                (false,true)=>(k.max(0),m),(false,false)=>(k.max(0),n),
            };
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,nrv as i64,ncv as i64,v,ldv as i64).unwrap_or(false) {return -8;}
                if ge_nancheck(layout,k as i64,k as i64,t,ldt as i64).unwrap_or(false) {return -11;}
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -13;}
            }
            let mut work=vec![<$t>::z();(k.max(0) as usize).max(1)];
            [<LAPACKE_ $p larfb_work $($suffix)*>](layout,side,trans,direct,storev,m,n,k,v,ldv,t,ldt,c,ldc,work.as_mut_ptr(),k.max(0))
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larfb_work $($suffix)*>](
            layout:i32,side:u8,trans:u8,direct:u8,storev:u8,m:$i,n:$i,k:$i,
            v:*const $t,ldv:$i,t:*const $t,ldt:$i,c:*mut $t,ldc:$i,
            work:*mut $t,ldwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let left=side.to_ascii_uppercase()==b'L';
            let col=storev.to_ascii_uppercase()==b'C';
            let (nrv,ncv)=match (col,left) {
                (true,true)=>(m,k.max(0)),(true,false)=>(n,k.max(0)),
                (false,true)=>(k.max(0),m),(false,false)=>(k.max(0),n),
            };
            if layout==ROW_MAJOR && ldc<n {return -14;}
            if layout==ROW_MAJOR && ldt<k {return -12;}
            if layout==ROW_MAJOR && ldv<ncv {return -10;}
            if layout==COL_MAJOR || m==0 || n==0 || k==0 {
                crate::householder::[<$p larfb $fortran>](&side,&trans,&direct,&storev,&m,&n,&k,v,&ldv,t,&ldt,c,&ldc,work,&ldwork);
                return 0;
            }
            let (mm,nn,kk)=(m as usize,n as usize,k as usize);
            let (r,cg)=(nrv as usize,ncv as usize);
            let mut vt=match row_to_column(r,cg,v,ldv as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut tt=match row_to_column(kk,kk,t,ldt as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (lv,lt,lc)=(r.max(1) as $i,kk.max(1) as $i,mm.max(1) as $i);
            crate::householder::[<$p larfb $fortran>](&side,&trans,&direct,&storev,&m,&n,&k,vt.as_mut_ptr(),&lv,tt.as_mut_ptr(),&lt,ct.as_mut_ptr(),&lc,work,&ldwork);
            for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}
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
