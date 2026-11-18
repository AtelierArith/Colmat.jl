//! LAPACKE GGES/GGES3 high/work adapters for the generalized Schur form.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{gges,gges3}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! real {
    ($p:ident,$family:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
            }
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -16;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -18;}
            let mut query=<$t>::z();
            let mut info=<$i>::from_i64(0);
            crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,a,&lda,b,&ldb,sdim,alphar,alphai,beta,vsl,&ldvsl,vsr,&ldvsr,&mut query,&mut (-1 as $i),core::ptr::null_mut(),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            let mut bwork=vec![<$i>::from_i64(0);(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,n,a,lda,b,ldb,sdim,alphar,alphai,beta,vsl,ldvsl,vsr,ldvsr,work.as_mut_ptr(),lw as $i,bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,
            work:*mut $t,lwork:$i,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -16;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -18;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,a,&lda,b,&ldb,sdim,alphar,alphai,beta,vsl,&ldvsl,vsr,&ldvsr,work,&lwork,bwork,&mut info);
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
            crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,sdim,alphar,alphai,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,work,&lwork,bwork,&mut info);
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
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alpha:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -9;}
            }
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -16;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -18;}
            let mut query=<$t>::z();
            let mut rwork=vec![<$r>::z();(8*n.max(0) as usize).max(1)];
            let mut info=<$i>::from_i64(0);
            crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,a,&lda,b,&ldb,sdim,alpha,beta,vsl,&ldvsl,vsr,&ldvsr,&mut query,&mut (-1 as $i),rwork.as_mut_ptr(),core::ptr::null_mut(),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            let mut bwork=vec![<$i>::from_i64(0);(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,jobvsl,jobvsr,sort,select,n,a,lda,b,ldb,sdim,alpha,beta,vsl,ldvsl,vsr,ldvsr,work.as_mut_ptr(),lw as $i,rwork.as_mut_ptr(),bwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,jobvsl:u8,jobvsr:u8,sort:u8,
            select:Option<unsafe extern "C" fn(*const $t,*const $t)->$i>,
            n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,sdim:*mut $i,
            alpha:*mut $t,beta:*mut $t,
            vsl:*mut $t,ldvsl:$i,vsr:*mut $t,ldvsr:$i,
            work:*mut $t,lwork:$i,rwork:*mut $r,bwork:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldb<n {return -10;}
            if layout==ROW_MAJOR && (ldvsl<1 || (jobvsl.to_ascii_uppercase()==b'V' && ldvsl<n)) {return -16;}
            if layout==ROW_MAJOR && (ldvsr<1 || (jobvsr.to_ascii_uppercase()==b'V' && ldvsr<n)) {return -18;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,a,&lda,b,&ldb,sdim,alpha,beta,vsl,&ldvsl,vsr,&ldvsr,work,&lwork,rwork,bwork,&mut info);
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
            crate::qz::[<$p $family $fortran>](&jobvsl,&jobvsr,&sort,select,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,sdim,alpha,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,work,&lwork,rwork,bwork,&mut info);
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
real!(s, gges, f32, i32, [], _);
real!(d, gges, f64, i32, [], _);
complex!(c, gges, colmat::c32, f32, i32, [], _);
complex!(z, gges, colmat::c64, f64, i32, [], _);
real!(s, gges, f32, i64, [64_], _64_);
real!(d, gges, f64, i64, [64_], _64_);
complex!(c, gges, colmat::c32, f32, i64, [64_], _64_);
complex!(z, gges, colmat::c64, f64, i64, [64_], _64_);
real!(s, gges3, f32, i32, [], _);
real!(d, gges3, f64, i32, [], _);
complex!(c, gges3, colmat::c32, f32, i32, [], _);
complex!(z, gges3, colmat::c64, f64, i32, [], _);
real!(s, gges3, f32, i64, [64_], _64_);
real!(d, gges3, f64, i64, [64_], _64_);
complex!(c, gges3, colmat::c32, f32, i64, [64_], _64_);
complex!(z, gges3, colmat::c64, f64, i64, [64_], _64_);
