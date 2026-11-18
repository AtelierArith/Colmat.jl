//! LAPACKE GGEV/GGEV3 high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{ggev,ggev3}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! real {
    ($p:ident,$family:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,
            vr:*mut $t,ldvr:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -7;}
            }
            let mut q=<$t>::z(); let mut info:$i=0;
            crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,a,&lda,b,&ldb,alphar,alphai,beta,vl,&ldvl,vr,&ldvr,&mut q,&mut (-1 as $i),&mut info);
            let lw=q.re().max(1.0) as usize; let mut w=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,jl,jr,n,a,lda,b,ldb,alphar,alphai,beta,vl,ldvl,vr,ldvr,w.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,
            vr:*mut $t,ldvr:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<n {return -8;}
            if layout==ROW_MAJOR && ldvl<n {return -13;}
            if layout==ROW_MAJOR && ldvr<n {return -15;}
            if layout==COL_MAJOR || n==0 {
                crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,a,&lda,b,&ldb,alphar,alphai,beta,vl,&ldvl,vr,&ldvr,work,&lwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=match row_to_column(nn,nn,vl,ldvl as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut rt=match row_to_column(nn,nn,vr,ldvr as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i; let mut info:$i=0;
            crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,alphar,alphai,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,work,&lwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*b.add(i*ldb as usize+j)=bt[i+j*nn];*vl.add(i*ldvl as usize+j)=lt[i+j*nn];*vr.add(i*ldvr as usize+j)=rt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! complex {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -7;}
            }
            let mut q=<$t>::z(); let mut info:$i=0; let mut rw=vec![<$r>::z();(8*n.max(0) as usize).max(1)];
            crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,a,&lda,b,&ldb,alpha,beta,vl,&ldvl,vr,&ldvr,&mut q,&mut (-1 as $i),rw.as_mut_ptr(),&mut info);
            let lw=q.re().max(1.0) as usize; let mut w=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,jl,jr,n,a,lda,b,ldb,alpha,beta,vl,ldvl,vr,ldvr,w.as_mut_ptr(),lw as $i,rw.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,jl:u8,jr:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alpha:*mut $t,beta:*mut $t,vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,
            work:*mut $t,lwork:$i,rwork:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<n {return -8;}
            if layout==ROW_MAJOR && ldvl<n {return -13;}
            if layout==ROW_MAJOR && ldvr<n {return -15;}
            if layout==COL_MAJOR || n==0 {
                crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,a,&lda,b,&ldb,alpha,beta,vl,&ldvl,vr,&ldvr,work,&lwork,rwork,&mut 0);
                return 0;
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut lt=match row_to_column(nn,nn,vl,ldvl as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut rt=match row_to_column(nn,nn,vr,ldvr as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i; let mut info:$i=0;
            crate::generalized_eigen::[<$p $family $fortran>](&jl,&jr,&n,at.as_mut_ptr(),&ld,bt.as_mut_ptr(),&ld,alpha,beta,lt.as_mut_ptr(),&ld,rt.as_mut_ptr(),&ld,work,&lwork,rwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];*b.add(i*ldb as usize+j)=bt[i+j*nn];*vl.add(i*ldvl as usize+j)=lt[i+j*nn];*vr.add(i*ldvr as usize+j)=rt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
real!(s, ggev, f32, i32, [], _);
real!(d, ggev, f64, i32, [], _);
complex!(c, ggev, colmat::c32, f32, i32, [], _);
complex!(z, ggev, colmat::c64, f64, i32, [], _);
real!(s, ggev, f32, i64, [64_], _64_);
real!(d, ggev, f64, i64, [64_], _64_);
complex!(c, ggev, colmat::c32, f32, i64, [64_], _64_);
complex!(z, ggev, colmat::c64, f64, i64, [64_], _64_);
real!(s, ggev3, f32, i32, [], _);
real!(d, ggev3, f64, i32, [], _);
complex!(c, ggev3, colmat::c32, f32, i32, [], _);
complex!(z, ggev3, colmat::c64, f64, i32, [], _);
real!(s, ggev3, f32, i64, [64_], _64_);
real!(d, ggev3, f64, i64, [64_], _64_);
complex!(c, ggev3, colmat::c32, f32, i64, [64_], _64_);
complex!(z, ggev3, colmat::c64, f64, i64, [64_], _64_);
