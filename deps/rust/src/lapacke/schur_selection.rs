//! LAPACKE TRSEN high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}trsen{,_work}.c.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

fn geometry(side: i32, ldt: i64, ldq: i64, n: i64) -> Option<i64> {
    if side != ROW_MAJOR && side != COL_MAJOR {
        return Some(-1);
    }
    if side == ROW_MAJOR && ldt < n {
        return Some(-7);
    }
    if side == ROW_MAJOR && ldq < n {
        return Some(-9);
    }
    if side == COL_MAJOR && ldt < n.max(1) {
        return Some(-7);
    }
    if side == COL_MAJOR && ldq < n.max(1) {
        return Some(-9);
    }
    None
}

unsafe fn high_nan<T: Scalar>(
    layout: i32,
    n: i64,
    t: *const T,
    ldt: i64,
    q: *const T,
    ldq: i64,
) -> Option<i64> {
    if LAPACKE_get_nancheck() != 0 {
        if ge_nancheck(layout, n, n, t, ldt).unwrap_or(false) {
            return Some(-6);
        }
        if ge_nancheck(layout, n, n, q, ldq).unwrap_or(false) {
            return Some(-8);
        }
    }
    None
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trsen $($suffix)*>](
            layout:i32,job:u8,compq:u8,select:*const $i,n:$i,t:*mut $t,ldt:$i,
            q:*mut $t,ldq:$i,wr:*mut $t,wi:*mut $t,m:*mut $i,s:*mut $t,sep:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if let Some(e)=high_nan(layout,n as i64,t,ldt as i64,q,ldq as i64){return e as $i;}
            let mut wq=<$t>::z();let mut iq:$i=-1;
            let mut info=$i::from_i64(0);
            crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,t,&ldt,q,&ldq,wr,wi,m,s,sep,&mut wq,&mut (-1 as $i),&mut iq,&mut (-1 as $i),&mut info);
            let _=info;
            let lw=wq.re().max(1.0) as usize;
            let li=iq.max(1) as usize;
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![$i::from_i64(0);li];
            [<LAPACKE_ $p trsen_work $($suffix)*>](layout,job,compq,select,n,t,ldt,q,ldq,wr,wi,m,s,sep,work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trsen_work $($suffix)*>](
            layout:i32,job:u8,compq:u8,select:*const $i,n:$i,t:*mut $t,ldt:$i,
            q:*mut $t,ldq:$i,wr:*mut $t,wi:*mut $t,m:*mut $i,s:*mut $t,sep:*mut $t,
            work:*mut $t,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            if let Some(e)=geometry(layout,ldt as i64,ldq as i64,n as i64){return e as $i;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,t,&ldt,q,&ldq,wr,wi,m,s,sep,work,&lwork,iwork,&liwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tt=match row_to_column(nn,nn,t,ldt as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut qt=match row_to_column(nn,nn,q,ldq as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,tt.as_mut_ptr(),&ld,qt.as_mut_ptr(),&ld,wr,wi,m,s,sep,work,&lwork,iwork,&liwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*t.add(i*ldt as usize+j)=tt[i+j*nn];*q.add(i*ldq as usize+j)=qt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trsen $($suffix)*>](
            layout:i32,job:u8,compq:u8,select:*const $i,n:$i,t:*mut $t,ldt:$i,
            q:*mut $t,ldq:$i,w:*mut $t,m:*mut $i,s:*mut $r,sep:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if let Some(e)=high_nan(layout,n as i64,t,ldt as i64,q,ldq as i64){return e as $i;}
            let mut wq=<$t>::z();
            let mut info=$i::from_i64(0);
            crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,t,&ldt,q,&ldq,w,m,s,sep,&mut wq,&mut (-1 as $i),&mut info);
            let _=info;
            let lw=wq.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p trsen_work $($suffix)*>](layout,job,compq,select,n,t,ldt,q,ldq,w,m,s,sep,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trsen_work $($suffix)*>](
            layout:i32,job:u8,compq:u8,select:*const $i,n:$i,t:*mut $t,ldt:$i,
            q:*mut $t,ldq:$i,w:*mut $t,m:*mut $i,s:*mut $r,sep:*mut $r,
            work:*mut $t,lwork:$i,
        )->$i {
            if let Some(e)=geometry(layout,ldt as i64,ldq as i64,n as i64){return e as $i;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,t,&ldt,q,&ldq,w,m,s,sep,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tt=match row_to_column(nn,nn,t,ldt as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut qt=match row_to_column(nn,nn,q,ldq as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur_reorder::[<$p trsen $fortran>](&job,&compq,select,&n,tt.as_mut_ptr(),&ld,qt.as_mut_ptr(),&ld,w,m,s,sep,work,&lwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*t.add(i*ldt as usize+j)=tt[i+j*nn];*q.add(i*ldq as usize+j)=qt[i+j*nn];}}}
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
