//! LAPACKE GEES high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}gees{,_work}.c.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gees $($suffix)*>](
            layout:i32,jobvs:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,sdim:*mut $i,wr:*mut $t,wi:*mut $t,
            vs:*mut $t,ldvs:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -6;}
            let mut wq=<$t>::z();
            let mut info=$i::from_i64(0);
            let mut bwork=vec![$i::from_i64(0);n.max(1) as usize];
            crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,a,&lda,sdim,wr,wi,vs,&ldvs,&mut wq,&mut (-1 as $i),bwork.as_mut_ptr(),&mut info);
            if info!=0 && info.val()!=0 {return info;}
            let lw=wq.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p gees_work $($suffix)*>](layout,jobvs,sort,select,n,a,lda,sdim,wr,wi,vs,ldvs,work.as_mut_ptr(),lw as $i,bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gees_work $($suffix)*>](
            layout:i32,jobvs:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,sdim:*mut $i,wr:*mut $t,wi:*mut $t,
            vs:*mut $t,ldvs:$i,work:*mut $t,lwork:$i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            if layout==ROW_MAJOR && ldvs<n {return -12;}
            if layout==COL_MAJOR || n==0 {
                crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,a,&lda,sdim,wr,wi,vs,&ldvs,work,&lwork,bwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut vt=match row_to_column(nn,nn,vs,ldvs as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,at.as_mut_ptr(),&ld,sdim,wr,wi,vt.as_mut_ptr(),&ld,work,&lwork,bwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*vs.add(i*ldvs as usize+j)=vt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gees $($suffix)*>](
            layout:i32,jobvs:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,sdim:*mut $i,w:*mut $t,vs:*mut $t,ldvs:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -6;}
            let mut wq=<$t>::z();
            let mut info=$i::from_i64(0);
            let mut bwork=vec![$i::from_i64(0);n.max(1) as usize];
            let mut rwork=vec![<$r>::z();n.max(1) as usize];
            crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,a,&lda,sdim,w,vs,&ldvs,&mut wq,&mut (-1 as $i),rwork.as_mut_ptr(),bwork.as_mut_ptr(),&mut info);
            if info.val()!=0 {return info;}
            let lw=wq.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p gees_work $($suffix)*>](layout,jobvs,sort,select,n,a,lda,sdim,w,vs,ldvs,work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr(),bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gees_work $($suffix)*>](
            layout:i32,jobvs:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,sdim:*mut $i,w:*mut $t,vs:*mut $t,ldvs:$i,
            work:*mut $t,lwork:$i,rwork:*mut $r,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            if layout==ROW_MAJOR && ldvs<n {return -11;}
            if layout==COL_MAJOR || n==0 {
                crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,a,&lda,sdim,w,vs,&ldvs,work,&lwork,rwork,bwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut vt=match row_to_column(nn,nn,vs,ldvs as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::schur::[<$p gees $fortran>](&jobvs,&sort,select,&n,at.as_mut_ptr(),&ld,sdim,w,vt.as_mut_ptr(),&ld,work,&lwork,rwork,bwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*vs.add(i*ldvs as usize+j)=vt[i+j*nn];}}}
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
