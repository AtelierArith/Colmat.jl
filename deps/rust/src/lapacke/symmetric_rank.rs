//! Complex symmetric rank-one LAPACKE auxiliary entries.
//! Reference: LAPACK v3.12.1 lapacke_csyr{,_work}.c. The Fortran BLAS call
//! has no INFO output: work returns zero for numerical argument errors,
//! with diagnostics supplied by the existing BLAS error mechanism.
use super::{
    cholesky::po_nancheck, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR, TRANSPOSE_MEMORY_ERROR,
};
use crate::kernels::{start, Scalar};

unsafe fn vector_nancheck<T: Scalar>(n: i64, x: *const T, incx: i64) -> Result<bool, i64> {
    if n <= 0 || x.is_null() || incx == 0 {
        return Ok(false);
    }
    let span = (n as usize - 1)
        .checked_mul(incx.unsigned_abs() as usize)
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(core::mem::size_of::<T>()));
    if span.is_none_or(|v| v > isize::MAX as usize) {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    let stride = incx as isize;
    let first = start(n as isize, stride);
    for i in 0..n as isize {
        let value = *x.offset(first + i * stride);
        if value != value {
            return Ok(true);
        }
    }
    Ok(false)
}

macro_rules! syr {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p syr $($suffix)*>](
            layout:i32,uplo:u8,n:$i,alpha:$t,x:*const $t,incx:$i,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true)=>return -7,Err(info)=>return info as $i,Ok(false)=>{},
                }
                if alpha!=alpha {return -4;}
                // Honor the caller's actual stride. The v3.12.1 high wrapper
                // passes literal 1 to its vector NaN checker; that would read
                // unused padding for nonunit strides and miss active entries.
                match vector_nancheck(n as i64,x,incx as i64) {
                    Ok(true)=>return -5,Err(info)=>return info as $i,Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p syr_work $($suffix)*>](layout,uplo,n,alpha,x,incx,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p syr_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,alpha:$t,x:*const $t,incx:$i,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -8;}
            // A symmetric (not Hermitian) update is invariant under transpose:
            // view row storage as columns and swap UPLO, without conjugation.
            let flag=if layout==ROW_MAJOR {match uplo.to_ascii_uppercase() {
                b'U'=>b'L',b'L'=>b'U',other=>other,
            }} else {uplo};
            // Empty row storage permits LDA=0; the reference uses a temporary
            // with a legal Fortran LDA even for that empty call.
            let ld=if layout==ROW_MAJOR && n==0 {1} else {lda};
            crate::abi::[<$p syr $fortran>](&flag,&n,&alpha,x,&incx,a,&ld);
            0
        }
    }};
}
syr!(c, colmat::c32, i32, [], _);
syr!(z, colmat::c64, i32, [], _);
syr!(c, colmat::c32, i64, [64_], _64_);
syr!(z, colmat::c64, i64, [64_], _64_);
