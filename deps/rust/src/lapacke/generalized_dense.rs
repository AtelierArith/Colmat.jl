//! LAPACKE SYGST/HEGST, SYGV/HEGV and SYGVD/HEGVD high/work adapters for the
//! dense generalized self-adjoint definite problem.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sygst,hegst,sygv,hegv,sygvd,hegvd}{,_work}.c.
//! Only the UPLO triangle of A and B is scanned for NaN; row-major A is
//! written back whole for JOBZ='V' (eigenvectors) and as a triangle otherwise.
use super::cholesky::po_nancheck;
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn in_triangle(uplo: u8, i: usize, j: usize) -> bool {
    if uplo.to_ascii_uppercase() == b'L' {
        i >= j
    } else {
        i <= j
    }
}

/// Shared row-major plumbing; `call(a, lda, b, ldb)` returns a layout-shifted
/// status. `write_b` is false for GST, whose B argument is read only.
#[allow(clippy::too_many_arguments)]
unsafe fn dense_job<T: Scalar>(
    layout: i32,
    uplo: u8,
    full_a: bool,
    write_b: bool,
    n: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    query: bool,
    errs: (i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && lda < n {
        return errs.0;
    }
    if layout == ROW_MAJOR && ldb < n {
        return errs.1;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(a, lda, b, ldb);
    }
    let nn = n as usize;
    let ld = n.max(1);
    if query {
        return call(a, ld, b, ld);
    }
    let mut at = match row_to_column(nn, nn, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match row_to_column(nn, nn, b, ldb as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let info = call(at.as_mut_ptr(), ld, bt.as_mut_ptr(), ld);
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                if full_a || in_triangle(uplo, i, j) {
                    *a.add(i * lda as usize + j) = at[i + j * nn];
                }
                if write_b && in_triangle(uplo, i, j) {
                    *b.add(i * ldb as usize + j) = bt[i + j * nn];
                }
            }
        }
    }
    info
}

unsafe fn nan_ab<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i64,
    a: *const T,
    lda: i64,
    b: *const T,
    ldb: i64,
    codes: (i64, i64),
) -> i64 {
    if LAPACKE_get_nancheck() == 0 {
        return 0;
    }
    match po_nancheck(layout, uplo, n, a, lda) {
        Ok(true) => return codes.0,
        Err(e) => return e,
        Ok(false) => {}
    }
    match po_nancheck(layout, uplo, n, b, ldb) {
        Ok(true) => codes.1,
        Err(e) => e,
        Ok(false) => 0,
    }
}

macro_rules! gst_exports {
    ($p:ident,$f:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,itype:$i,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*const $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_ab(layout,uplo,n as i64,a,lda as i64,b,ldb as i64,(-5,-7));
            if e!=0 {return e as $i;}
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,itype,uplo,n,a,lda,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,itype:$i,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*const $t,ldb:$i,
        )->$i {
            dense_job(layout,uplo,true,false,n as i64,a,lda as i64,b as *mut $t,ldb as i64,false,(-6,-8),|a,la,bb,lb|{
                let mut info:$i=0;
                crate::generalized_selfadjoint::[<$p $f $fort>](&itype,&uplo,&n,a,&(la as $i),bb as *const $t,&(lb as $i),&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! gv_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,w:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_ab(layout,uplo,n as i64,a,lda as i64,b,ldb as i64,(-6,-8));
            if e!=0 {return e as $i;}
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p $f _work $($suffix)*>](layout,itype,jobz,uplo,n,a,lda,b,ldb,w,&mut qw,-1$(,{let _=stringify!($rw);std::ptr::null_mut::<$r>()})*);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();(3*n.max(0) as usize).max(1)];)*
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,itype,jobz,uplo,n,a,lda,b,ldb,w,work.as_mut_ptr(),lw as $i$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,w:*mut $r,
            work:*mut $t,lwork:$i,$($rw:*mut $r,)*
        )->$i {
            let want=jobz.to_ascii_uppercase()==b'V';
            dense_job(layout,uplo,want,true,n as i64,a,lda as i64,b,ldb as i64,lwork== -1,(-7,-9),|a,la,bb,lb|{
                let mut info:$i=0;
                crate::generalized_selfadjoint_drivers::[<$p $f $fort>](&itype,&jobz,&uplo,&n,a,&(la as $i),bb,&(lb as $i),w,work,&lwork,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! gvd_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,w:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_ab(layout,uplo,n as i64,a,lda as i64,b,ldb as i64,(-6,-8));
            if e!=0 {return e as $i;}
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            $(let mut [<q $rw>]=<$r>::z();)*
            let r=[<LAPACKE_ $p $f _work $($suffix)*>](layout,itype,jobz,uplo,n,a,lda,b,ldb,w,&mut qw,-1,$(&mut [<q $rw>],-1,)*&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let [<l $rw>]=[<q $rw>].re().max(1.0) as usize;let mut $rw=vec![<$r>::z();[<l $rw>]];)*
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,itype,jobz,uplo,n,a,lda,b,ldb,w,work.as_mut_ptr(),lw as $i,$($rw.as_mut_ptr(),[<l $rw>] as $i,)*iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,w:*mut $r,
            work:*mut $t,lwork:$i,$($rw:*mut $r,[<l $rw>]:$i,)*iwork:*mut $i,liwork:$i,
        )->$i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $rw>]== -1)*;
            let want=jobz.to_ascii_uppercase()==b'V';
            dense_job(layout,uplo,want,true,n as i64,a,lda as i64,b,ldb as i64,query,(-7,-9),|a,la,bb,lb|{
                let mut info:$i=0;
                crate::generalized_selfadjoint_drivers::[<$p $f $fort>](&itype,&jobz,&uplo,&n,a,&(la as $i),bb,&(lb as $i),w,work,&lwork,$($rw,&[<l $rw>],)*iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        gst_exports!(s,sygst,f32,$i,[$($suffix)*],$fort);
        gst_exports!(d,sygst,f64,$i,[$($suffix)*],$fort);
        gst_exports!(c,hegst,colmat::c32,$i,[$($suffix)*],$fort);
        gst_exports!(z,hegst,colmat::c64,$i,[$($suffix)*],$fort);
        gv_exports!(s,sygv,f32,f32,$i,[$($suffix)*],$fort,[]);
        gv_exports!(d,sygv,f64,f64,$i,[$($suffix)*],$fort,[]);
        gv_exports!(c,hegv,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gv_exports!(z,hegv,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
        gvd_exports!(s,sygvd,f32,f32,$i,[$($suffix)*],$fort,[]);
        gvd_exports!(d,sygvd,f64,f64,$i,[$($suffix)*],$fort,[]);
        gvd_exports!(c,hegvd,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gvd_exports!(z,hegvd,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
