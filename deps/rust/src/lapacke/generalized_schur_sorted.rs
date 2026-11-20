//! LAPACKE GGESX high/work adapters for the generalized Schur form with
//! ordering and condition estimates (SENSE).
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}ggesx{,_work}.c.
//! SELECTCTG is called by the native sorter; row-major A, B, VSL and VSR are
//! transposed and written back whole, and WORK/LWORK/LIWORK are queried.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggesx $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t,*const $t)->$i>,
            sense:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,rconde:*mut $t,rcondv:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -8;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -10;}
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            let r=[<LAPACKE_ $p ggesx_work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,sense,n,a,lda,b,ldb,sdim,alphar,alphai,beta,vsl,ldvsl,vsr,ldvsr,rconde,rcondv,
                &mut qw,-1,&mut qi,-1,core::ptr::null_mut());
            if r!=0 {return r;}
            // The condition-estimate TGSYL step needs LWORK one past the reported minimum.
            let lw=qw.re().max(1.0) as usize+2*(n.max(0) as usize)*(n.max(0) as usize)+1;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![0 as $i;li];
            let mut bwork=vec![0 as $i;(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p ggesx_work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,sense,n,a,lda,b,ldb,sdim,alphar,alphai,beta,vsl,ldvsl,vsr,ldvsr,rconde,rcondv,
                work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i,bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggesx_work $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t,*const $t)->$i>,
            sense:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,rconde:*mut $t,rcondv:*mut $t,
            work:*mut $t,lwork:$i,iwork:*mut $i,liwork:$i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -9;}
            if layout==ROW_MAJOR && ldb<n {return -11;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -17;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -19;}
            if layout==COL_MAJOR || n==0 || lwork== -1 || liwork== -1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {0};
                let (la,lb,ll,lr)=if layout==ROW_MAJOR {(ld,ld,ld,ld)} else {(lda,ldb,ldvsl,ldvsr)};
                crate::qz_reorder_driver::[<$p ggesx $fortran>](&jobvsl,&jobvsr,&sort,select,&sense,&n,a,&la,b,&lb,sdim,alphar,alphai,beta,vsl,&ll,vsr,&lr,rconde,rcondv,work,&lwork,iwork,&liwork,bwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let wantl=jobvsl.to_ascii_uppercase()==b'V';
            let wantr=jobvsr.to_ascii_uppercase()==b'V';
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=if wantl {match row_to_column(nn,nn,vsl,ldvsl as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let mut rt=if wantr {match row_to_column(nn,nn,vsr,ldvsr as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::qz_reorder_driver::[<$p ggesx $fortran>](&jobvsl,&jobvsr,&sort,select,&sense,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,sdim,alphar,alphai,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,rconde,rcondv,work,&lwork,iwork,&liwork,bwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    *a.add(i*lda as usize+j)=at[i+j*nn];
                    *b.add(i*ldb as usize+j)=bt[i+j*nn];
                    if wantl {*vsl.add(i*ldvsl as usize+j)=lt[i+j*nn];}
                    if wantr {*vsr.add(i*ldvsr as usize+j)=rt[i+j*nn];}
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggesx $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            sense:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alpha:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,rconde:*mut $r,rcondv:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -8;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -10;}
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            let mut rwork=vec![<$r>::z();(8*n.max(0) as usize).max(1)];
            let r=[<LAPACKE_ $p ggesx_work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,sense,n,a,lda,b,ldb,sdim,alpha,beta,vsl,ldvsl,vsr,ldvsr,rconde,rcondv,
                &mut qw,-1,rwork.as_mut_ptr(),&mut qi,-1,core::ptr::null_mut());
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize+2*(n.max(0) as usize)*(n.max(0) as usize)+1;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![0 as $i;li];
            let mut bwork=vec![0 as $i;(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p ggesx_work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,sense,n,a,lda,b,ldb,sdim,alpha,beta,vsl,ldvsl,vsr,ldvsr,rconde,rcondv,
                work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr(),iwork.as_mut_ptr(),li as $i,bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ggesx_work $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            sense:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alpha:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,rconde:*mut $r,rcondv:*mut $r,
            work:*mut $t,lwork:$i,rwork:*mut $r,iwork:*mut $i,liwork:$i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -9;}
            if layout==ROW_MAJOR && ldb<n {return -11;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -16;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -18;}
            if layout==COL_MAJOR || n==0 || lwork== -1 || liwork== -1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {0};
                let (la,lb,ll,lr)=if layout==ROW_MAJOR {(ld,ld,ld,ld)} else {(lda,ldb,ldvsl,ldvsr)};
                crate::qz_reorder_driver::[<$p ggesx $fortran>](&jobvsl,&jobvsr,&sort,select,&sense,&n,a,&la,b,&lb,sdim,alpha,beta,vsl,&ll,vsr,&lr,rconde,rcondv,work,&lwork,rwork,iwork,&liwork,bwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let wantl=jobvsl.to_ascii_uppercase()==b'V';
            let wantr=jobvsr.to_ascii_uppercase()==b'V';
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=if wantl {match row_to_column(nn,nn,vsl,ldvsl as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let mut rt=if wantr {match row_to_column(nn,nn,vsr,ldvsr as usize){Ok(v)=>v,Err(e)=>return e as $i}} else {Vec::new()};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::qz_reorder_driver::[<$p ggesx $fortran>](&jobvsl,&jobvsr,&sort,select,&sense,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,sdim,alpha,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,rconde,rcondv,work,&lwork,rwork,iwork,&liwork,bwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    *a.add(i*lda as usize+j)=at[i+j*nn];
                    *b.add(i*ldb as usize+j)=bt[i+j*nn];
                    if wantl {*vsl.add(i*ldvsl as usize+j)=lt[i+j*nn];}
                    if wantr {*vsr.add(i*ldvsr as usize+j)=rt[i+j*nn];}
                }}
            }
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
