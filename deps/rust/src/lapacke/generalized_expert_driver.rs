//! LAPACKE GGEVX high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}ggevx{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggevx $($suffix)*>](
            layout:i32,balanc:u8,jl:u8,jr:u8,sense:u8,n:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,ar:*mut $t,ai:*mut $t,be:*mut $t,vl:*mut $t,
            ldvl:$i,vr:*mut $t,ldvr:$i,ilo:*mut $i,ihi:*mut $i,ls:*mut $t,
            rs:*mut $t,anrm:*mut $t,bnrm:*mut $t,rc:*mut $t,rv:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
            }
            let mut q=<$t>::z(); let mut info:$i=0;
            let mut iw=vec![$i::from_i64(0);(6*n.max(0) as usize).max(1)];
            let mut bw=vec![$i::from_i64(0);n.max(1) as usize];
            crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,a,&lda,b,&ldb,ar,ai,be,vl,&ldvl,vr,&ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,&mut q,&mut (-1 as $i),iw.as_mut_ptr(),bw.as_mut_ptr(),&mut info);
            let lw=q.re().max(1.0) as usize; let mut w=vec![<$t>::z();lw];
            [<LAPACKE_ $p ggevx_work $($suffix)*>](layout,balanc,jl,jr,sense,n,a,lda,b,ldb,ar,ai,be,vl,ldvl,vr,ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,w.as_mut_ptr(),lw as $i,iw.as_mut_ptr(),bw.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggevx_work $($suffix)*>](
            layout:i32,balanc:u8,jl:u8,jr:u8,sense:u8,n:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,ar:*mut $t,ai:*mut $t,be:*mut $t,vl:*mut $t,
            ldvl:$i,vr:*mut $t,ldvr:$i,ilo:*mut $i,ihi:*mut $i,ls:*mut $t,
            rs:*mut $t,anrm:*mut $t,bnrm:*mut $t,rc:*mut $t,rv:*mut $t,
            work:*mut $t,lwork:$i,iwork:*mut $i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && ldvl<n {return -15;}
            if layout==ROW_MAJOR && ldvr<n {return -17;}
            if layout==COL_MAJOR || n==0 {
                crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,a,&lda,b,&ldb,ar,ai,be,vl,&ldvl,vr,&ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,work,&lwork,iwork,bwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=match row_to_column(nn,nn,vl,ldvl as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut rt=match row_to_column(nn,nn,vr,ldvr as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i; let mut info:$i=0;
            crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,ar,ai,be,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,work,&lwork,iwork,bwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*b.add(i*ldb as usize+j)=bt[i+j*nn];*vl.add(i*ldvl as usize+j)=lt[i+j*nn];*vr.add(i*ldvr as usize+j)=rt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggevx $($suffix)*>](
            layout:i32,balanc:u8,jl:u8,jr:u8,sense:u8,n:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,
            vr:*mut $t,ldvr:$i,ilo:*mut $i,ihi:*mut $i,ls:*mut $r,rs:*mut $r,
            anrm:*mut $r,bnrm:*mut $r,rc:*mut $r,rv:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
            }
            let mut q=<$t>::z(); let mut info:$i=0;
            let mut rw=vec![<$r>::z();(8*n.max(0) as usize).max(1)];
            let mut iw=vec![$i::from_i64(0);(6*n.max(0) as usize).max(1)];
            let mut bw=vec![$i::from_i64(0);n.max(1) as usize];
            crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,a,&lda,b,&ldb,alpha,beta,vl,&ldvl,vr,&ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,&mut q,&mut (-1 as $i),rw.as_mut_ptr(),iw.as_mut_ptr(),bw.as_mut_ptr(),&mut info);
            let lw=q.re().max(1.0) as usize; let mut w=vec![<$t>::z();lw];
            [<LAPACKE_ $p ggevx_work $($suffix)*>](layout,balanc,jl,jr,sense,n,a,lda,b,ldb,alpha,beta,vl,ldvl,vr,ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,w.as_mut_ptr(),lw as $i,rw.as_mut_ptr(),iw.as_mut_ptr(),bw.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggevx_work $($suffix)*>](
            layout:i32,balanc:u8,jl:u8,jr:u8,sense:u8,n:$i,a:*mut $t,lda:$i,
            b:*mut $t,ldb:$i,alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,
            vr:*mut $t,ldvr:$i,ilo:*mut $i,ihi:*mut $i,ls:*mut $r,rs:*mut $r,
            anrm:*mut $r,bnrm:*mut $r,rc:*mut $r,rv:*mut $r,
            work:*mut $t,lwork:$i,rwork:*mut $r,iwork:*mut $i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && ldvl<n {return -14;}
            if layout==ROW_MAJOR && ldvr<n {return -16;}
            if layout==COL_MAJOR || n==0 {
                crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,a,&lda,b,&ldb,alpha,beta,vl,&ldvl,vr,&ldvr,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,work,&lwork,rwork,iwork,bwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=match row_to_column(nn,nn,vl,ldvl as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut rt=match row_to_column(nn,nn,vr,ldvr as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i; let mut info:$i=0;
            crate::generalized_eigen::[<$p ggevx $fortran>](&balanc,&jl,&jr,&sense,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,alpha,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,ilo,ihi,ls,rs,anrm,bnrm,rc,rv,work,&lwork,rwork,iwork,bwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*b.add(i*ldb as usize+j)=bt[i+j*nn];*vl.add(i*ldvl as usize+j)=lt[i+j*nn];*vr.add(i*ldvr as usize+j)=rt[i+j*nn];}}}
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
