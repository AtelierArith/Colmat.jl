//! LAPACKE GETSLS, GETSQRHRT, ORGTSQR_ROW/UNGTSQR_ROW and ORHR_COL/UNHR_COL
//! high/work adapters for tall-skinny least squares and the TSQR Householder
//! reconstruction family.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{getsls,getsqrhrt,orgtsqr_row,ungtsqr_row,orhr_col,unhr_col}{,_work}.c.
//! T is an NB-by-(N*slots) array; its row-major leading dimension must cover
//! the full column count (checked against N here and by the native checker).
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

fn slots(m: i64, n: i64, mb: i64) -> i64 {
    if mb > n && mb < m {
        (m - n) / (mb - n) + i64::from((m - n) % (mb - n) != 0)
    } else {
        1
    }
}

unsafe fn store<T: Scalar>(src: &[T], rows: usize, cols: usize, dst: *mut T, ld: usize) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add(i * ld + j) = src[i + j * rows];
        }
    }
}

macro_rules! getsls_exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getsls $($suffix)*>](
            layout:i32,trans:u8,m:$i,n:$i,nrhs:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -6;}
                if ge_nancheck(layout,(m as i64).max(n as i64),nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p getsls_work $($suffix)*>](layout,trans,m,n,nrhs,a,lda,b,ldb,&mut qw,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p getsls_work $($suffix)*>](layout,trans,m,n,nrhs,a,lda,b,ldb,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getsls_work $($suffix)*>](
            layout:i32,trans:u8,m:$i,n:$i,nrhs:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || m<=0 || n<=0 || nrhs<=0 || lwork== -1 || lwork== -2 {
                let mut info:$i=0;
                let (la,lb)=if layout==ROW_MAJOR {(m.max(1),m.max(n).max(1))} else {(lda,ldb)};
                crate::least_squares::[<$p getsls $fort>](&trans,&m,&n,&nrhs,a,&la,b,&lb,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,rr)=(m as usize,n as usize,nrhs as usize);
            let rows_b=mm.max(nn);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=match row_to_column(rows_b,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lb)=(mm.max(1) as $i,rows_b.max(1) as $i);
            let mut info:$i=0;
            crate::least_squares::[<$p getsls $fort>](&trans,&m,&n,&nrhs,at.as_mut_ptr(),&la,bt.as_mut_ptr(),&lb,work,&lwork,&mut info);
            if info>=0 {
                store(&at,mm,nn,a,lda as usize);
                store(&bt,rows_b,rr,b,ldb as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! getsqrhrt_exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getsqrhrt $($suffix)*>](
            layout:i32,m:$i,n:$i,mb1:$i,nb1:$i,nb2:$i,a:*mut $t,lda:$i,t:*mut $t,ldt:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {
                return -7;
            }
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p getsqrhrt_work $($suffix)*>](layout,m,n,mb1,nb1,nb2,a,lda,t,ldt,&mut qw,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p getsqrhrt_work $($suffix)*>](layout,m,n,mb1,nb1,nb2,a,lda,t,ldt,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p getsqrhrt_work $($suffix)*>](
            layout:i32,m:$i,n:$i,mb1:$i,nb1:$i,nb2:$i,a:*mut $t,lda:$i,t:*mut $t,ldt:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            if layout==ROW_MAJOR && ldt<n {return -10;}
            if layout==COL_MAJOR || m<=0 || n<=0 || lwork== -1 {
                let mut info:$i=0;
                let (la,lt)=if layout==ROW_MAJOR {(m.max(1),nb2.max(1))} else {(lda,ldt)};
                crate::tsqr_driver::[<$p getsqrhrt $fort>](&m,&n,&mb1,&nb1,&nb2,a,&la,t,&lt,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,kk)=(m as usize,n as usize,nb2.max(1) as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut tt=vec![<$t>::z();kk*nn];
            let (la,lt)=(mm.max(1) as $i,kk as $i);
            let mut info:$i=0;
            crate::tsqr_driver::[<$p getsqrhrt $fort>](&m,&n,&mb1,&nb1,&nb2,at.as_mut_ptr(),&la,tt.as_mut_ptr(),&lt,work,&lwork,&mut info);
            if info>=0 {
                store(&at,mm,nn,a,lda as usize);
                store(&tt,kk,nn,t,ldt as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! orgtsqr_exports {
    ($p:ident,$org:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $org _row $($suffix)*>](
            layout:i32,m:$i,n:$i,mb:$i,nb:$i,a:*mut $t,lda:$i,t:*const $t,ldt:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -6;}
                let rows=(nb as i64).min(n as i64);
                let cols=(n as i64)*slots(m as i64,n as i64,mb as i64);
                if ge_nancheck(layout,rows,cols,t,ldt as i64).unwrap_or(false) {return -8;}
            }
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p $org _row_work $($suffix)*>](layout,m,n,mb,nb,a,lda,t,ldt,&mut qw,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $org _row_work $($suffix)*>](layout,m,n,mb,nb,a,lda,t,ldt,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $org _row_work $($suffix)*>](
            layout:i32,m:$i,n:$i,mb:$i,nb:$i,a:*mut $t,lda:$i,t:*const $t,ldt:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            let cols=(n as i64)*slots(m as i64,n as i64,mb as i64);
            if layout==ROW_MAJOR && (ldt as i64)<cols.max(1) {return -9;}
            if layout==COL_MAJOR || m<=0 || n<=0 || lwork== -1 {
                let mut info:$i=0;
                let (la,lt)=if layout==ROW_MAJOR {(m.max(1),(nb.min(n)).max(1))} else {(lda,ldt)};
                crate::tsqr_driver::[<$p $org _row $fort>](&m,&n,&mb,&nb,a,&la,t,&lt,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let rows=(nb.min(n)).max(1) as usize;
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let tt=match row_to_column(rows,cols as usize,t,ldt as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (la,lt)=(mm.max(1) as $i,rows as $i);
            let mut info:$i=0;
            crate::tsqr_driver::[<$p $org _row $fort>](&m,&n,&mb,&nb,at.as_mut_ptr(),&la,tt.as_ptr(),&lt,work,&lwork,&mut info);
            if info>=0 {store(&at,mm,nn,a,lda as usize);}
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! orhr_exports {
    ($p:ident,$f:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,m:$i,n:$i,nb:$i,a:*mut $t,lda:$i,t:*mut $t,ldt:$i,d:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {
                return -5;
            }
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,m,n,nb,a,lda,t,ldt,d)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,m:$i,n:$i,nb:$i,a:*mut $t,lda:$i,t:*mut $t,ldt:$i,d:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==ROW_MAJOR && ldt<n {return -8;}
            if layout==COL_MAJOR || m<=0 || n<=0 {
                let mut info:$i=0;
                let (la,lt)=if layout==ROW_MAJOR {(m.max(1),nb.max(1))} else {(lda,ldt)};
                crate::householder_reconstruction::[<$p $f $fort>](&m,&n,&nb,a,&la,t,&lt,d,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,kk)=(m as usize,n as usize,nb.max(1) as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut tt=vec![<$t>::z();kk*nn];
            let (la,lt)=(mm.max(1) as $i,kk as $i);
            let mut info:$i=0;
            crate::householder_reconstruction::[<$p $f $fort>](&m,&n,&nb,at.as_mut_ptr(),&la,tt.as_mut_ptr(),&lt,d,&mut info);
            if info>=0 {
                store(&at,mm,nn,a,lda as usize);
                store(&tt,kk,nn,t,ldt as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        getsls_exports!(s,f32,$i,[$($suffix)*],$fort);
        getsls_exports!(d,f64,$i,[$($suffix)*],$fort);
        getsls_exports!(c,colmat::c32,$i,[$($suffix)*],$fort);
        getsls_exports!(z,colmat::c64,$i,[$($suffix)*],$fort);
        getsqrhrt_exports!(s,f32,$i,[$($suffix)*],$fort);
        getsqrhrt_exports!(d,f64,$i,[$($suffix)*],$fort);
        getsqrhrt_exports!(c,colmat::c32,$i,[$($suffix)*],$fort);
        getsqrhrt_exports!(z,colmat::c64,$i,[$($suffix)*],$fort);
        orgtsqr_exports!(s,orgtsqr,f32,$i,[$($suffix)*],$fort);
        orgtsqr_exports!(d,orgtsqr,f64,$i,[$($suffix)*],$fort);
        orgtsqr_exports!(c,ungtsqr,colmat::c32,$i,[$($suffix)*],$fort);
        orgtsqr_exports!(z,ungtsqr,colmat::c64,$i,[$($suffix)*],$fort);
        orhr_exports!(s,orhr_col,f32,$i,[$($suffix)*],$fort);
        orhr_exports!(d,orhr_col,f64,$i,[$($suffix)*],$fort);
        orhr_exports!(c,unhr_col,colmat::c32,$i,[$($suffix)*],$fort);
        orhr_exports!(z,unhr_col,colmat::c64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
