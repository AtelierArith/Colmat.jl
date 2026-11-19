//! LAPACKE SYTRF/SYTRS/SYSV and HETRF/HETRS/HESV high/work adapters for
//! dense symmetric/Hermitian indefinite LU (Bunch-Kaufman).
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d}{sytrf,sytrs,sysv}{,_work}.c and
//! LAPACKE/src/lapacke_{c,z}{hetrf,hetrs,hesv}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$family:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trf $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -4;
            }
            if layout==ROW_MAJOR && lda<n {return -5;}
            let mut query=<$t>::z();
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family trf $fortran>](&uplo,&n,a,&lda,ipiv,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family trf_work $($suffix)*>](layout,uplo,n,a,lda,ipiv,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trf_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*mut $i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family trf $fortran>](&uplo,&n,a,&lda,ipiv,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family trf $fortran>](&uplo,&n,at.as_mut_ptr(),&ld,ipiv,work,&lwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trs $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*const $t,lda:$i,
            ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            [<LAPACKE_ $p $family trs_work $($suffix)*>](layout,uplo,n,nrhs,a,lda,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trs_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*const $t,lda:$i,
            ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family trs $fortran>](&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lb)=(nn.max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family trs $fortran>](&uplo,&n,&nrhs,at.as_ptr(),&la,ipiv,bt.as_mut_ptr(),&lb,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trs2 $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*const $t,lda:$i,
            ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            let mut work=vec![<$t>::z();n.max(1) as usize];
            [<LAPACKE_ $p $family trs2_work $($suffix)*>](layout,uplo,n,nrhs,a,lda,ipiv,b,ldb,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family trs2_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*const $t,lda:$i,
            ipiv:*const $i,b:*mut $t,ldb:$i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family trs2 $fortran>](&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lb)=(nn.max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family trs2 $fortran>](&uplo,&n,&nrhs,at.as_ptr(),&la,ipiv,bt.as_mut_ptr(),&lb,work,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family sv $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*mut $t,lda:$i,
            ipiv:*mut $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            let mut query=<$t>::z();
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family sv $fortran>](&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,&mut query,&mut (-1 as $i),&mut info);
            if info!=0 {return info;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family sv_work $($suffix)*>](layout,uplo,n,nrhs,a,lda,ipiv,b,ldb,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family sv_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,a:*mut $t,lda:$i,
            ipiv:*mut $i,b:*mut $t,ldb:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family sv $fortran>](&uplo,&n,&nrhs,a,&lda,ipiv,b,&ldb,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lb)=(nn.max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family sv $fortran>](&uplo,&n,&nrhs,at.as_mut_ptr(),&la,ipiv,bt.as_mut_ptr(),&lb,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}
                for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, sy, f32, i32, [], _);
exports!(d, sy, f64, i32, [], _);
exports!(c, he, colmat::c32, i32, [], _);
exports!(z, he, colmat::c64, i32, [], _);
exports!(s, sy, f32, i64, [64_], _64_);
exports!(d, sy, f64, i64, [64_], _64_);
exports!(c, he, colmat::c32, i64, [64_], _64_);
exports!(z, he, colmat::c64, i64, [64_], _64_);
