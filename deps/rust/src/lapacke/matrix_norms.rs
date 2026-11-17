//! LAPACKE dense/symmetric/triangular matrix norms.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}
//! {lange,lansy,lantr}{,_work}.c. Row-major storage is viewed as its transpose;
//! 1/infinity norm names and UPLO are swapped, and no padding is read.
use super::{ge_nancheck, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn swap_norm(norm: u8) -> u8 {
    match norm.to_ascii_uppercase() {
        b'1' | b'O' => b'I',
        b'I' => b'1',
        other => other,
    }
}
fn swap_uplo(u: u8) -> u8 {
    match u.to_ascii_uppercase() {
        b'U' => b'L',
        b'L' => b'U',
        other => other,
    }
}
unsafe fn workspace<T: Scalar, R: Scalar>(norm: u8, count: usize) -> Option<Vec<R>> {
    if matches!(norm.to_ascii_uppercase(), b'I') {
        let mut out = Vec::new();
        out.try_reserve_exact(count.max(1)).ok()?;
        out.resize(count.max(1), R::z());
        Some(out)
    } else {
        None
    }
}

macro_rules! dense_high {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lange $($suffix)*>](
            layout:i32,norm:u8,m:$i,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,m as i64,n as i64,a,lda as i64) {
                    Ok(true)=>return <$r>::real(-5.0),Err(info)=>return <$r>::real(info as f64),Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p lange_work $($suffix)*>](layout,norm,m,n,a,lda)
        }
    }};
}
macro_rules! dense_work {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lange_work $($suffix)*>](
            layout:i32,norm:u8,m:$i,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if layout==ROW_MAJOR && lda<n {return <$r>::real(-6.0);}
            if layout==COL_MAJOR && lda<m.max(1) {return <$r>::real(-6.0);}
            if m<0 {return <$r>::real(-3.0);}
            if n<0 {return <$r>::real(-4.0);}
            let mut wr=workspace::<$t,$r>(norm,if layout==ROW_MAJOR {n as usize} else {m as usize});
            let wp=match wr.as_mut() {Some(v)=>v.as_mut_ptr(),None=>core::ptr::null_mut()};
            if layout==COL_MAJOR {
                crate::matrix_norms::[<$p lange $fortran>](&norm,&m,&n,a,&lda,wp)
            } else {
                crate::matrix_norms::[<$p lange $fortran>](&swap_norm(norm),&n,&m,a,&lda,wp)
            }
        }
    }};
}

macro_rules! symmetric_high {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lansy $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if LAPACKE_get_nancheck()!=0 {
                match super::cholesky::po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true)=>return <$r>::real(-5.0),Err(info)=>return <$r>::real(info as f64),Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p lansy_work $($suffix)*>](layout,norm,uplo,n,a,lda)
        }
    }};
}
macro_rules! symmetric_work {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lansy_work $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if layout==ROW_MAJOR && lda<n {return <$r>::real(-6.0);}
            if layout==COL_MAJOR && lda<n.max(1) {return <$r>::real(-6.0);}
            let mut wr=workspace::<$t,$r>(norm,n as usize);
            let wp=match wr.as_mut() {Some(v)=>v.as_mut_ptr(),None=>core::ptr::null_mut()};
            if layout==COL_MAJOR {
                crate::matrix_norms::[<$p lansy $fortran>](&norm,&uplo,&n,a,&lda,wp)
            } else {
                crate::matrix_norms::[<$p lansy $fortran>](&norm,&swap_uplo(uplo),&n,a,&lda,wp)
            }
        }
    }};
}

macro_rules! triangular_high {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lantr $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,diag:u8,m:$i,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if LAPACKE_get_nancheck()!=0 {
                match super::triangular::nancheck(layout,uplo,diag,m as i64,a,lda as i64) {
                    Ok(true)=>return <$r>::real(-7.0),Err(info)=>return <$r>::real(info as f64),Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p lantr_work $($suffix)*>](layout,norm,uplo,diag,m,n,a,lda)
        }
    }};
}
macro_rules! triangular_work {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lantr_work $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,diag:u8,m:$i,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return <$r>::real(-1.0);}
            if layout==ROW_MAJOR && lda<n {return <$r>::real(-8.0);}
            if layout==COL_MAJOR && lda<m.max(1) {return <$r>::real(-8.0);}
            let mut wr=workspace::<$t,$r>(norm,m.max(n) as usize);
            let wp=match wr.as_mut() {Some(v)=>v.as_mut_ptr(),None=>core::ptr::null_mut()};
            if layout==COL_MAJOR {
                crate::matrix_norms::[<$p lantr $fortran>](&norm,&uplo,&diag,&m,&n,a,&lda,wp)
            } else {
                crate::matrix_norms::[<$p lantr $fortran>](
                    &swap_norm(norm),&swap_uplo(uplo),&diag,&n,&m,a,&lda,wp,
                )
            }
        }
    }};
}

macro_rules! all {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{
        dense_high!($p,$t,$r,$i,[$($suffix)*]);
        dense_work!($p,$t,$r,$i,[$($suffix)*],$fortran);
        symmetric_high!($p,$t,$r,$i,[$($suffix)*]);
        symmetric_work!($p,$t,$r,$i,[$($suffix)*],$fortran);
        triangular_high!($p,$t,$r,$i,[$($suffix)*]);
        triangular_work!($p,$t,$r,$i,[$($suffix)*],$fortran);
    };
}

all!(s, f32, f32, i32, [], _);
all!(d, f64, f64, i32, [], _);
all!(c, colmat::c32, f32, i32, [], _);
all!(z, colmat::c64, f64, i32, [], _);
all!(s, f32, f32, i64, [64_], _64_);
all!(d, f64, f64, i64, [64_], _64_);
all!(c, colmat::c32, f32, i64, [64_], _64_);
all!(z, colmat::c64, f64, i64, [64_], _64_);
