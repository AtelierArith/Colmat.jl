//! Dense positive-definite LAPACKE adapters. Only the selected triangle is read.
use super::{
    ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR, TRANSPOSE_MEMORY_ERROR,
};
use crate::kernels::Scalar;

fn selected(uplo: u8, i: usize, j: usize) -> bool {
    if uplo.to_ascii_uppercase() == b'L' {
        i >= j
    } else {
        i <= j
    }
}

pub(super) unsafe fn po_nancheck<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i64,
    a: *const T,
    lda: i64,
) -> Result<bool, i64> {
    if a.is_null() || n <= 0 || lda <= 0 || !crate::kernels::uplo(uplo.to_ascii_uppercase()) {
        return Ok(false);
    }
    let span = (n as usize - 1)
        .checked_mul(lda as usize)
        .and_then(|x| x.checked_add(n.min(lda) as usize))
        .and_then(|x| x.checked_mul(core::mem::size_of::<T>()));
    if span.is_none_or(|v| v > isize::MAX as usize) {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    for outer in 0..n as usize {
        for inner in 0..n.min(lda) as usize {
            let (i, j) = if layout == ROW_MAJOR {
                (outer, inner)
            } else {
                (inner, outer)
            };
            if selected(uplo, i, j) {
                let x = *a.add(outer * lda as usize + inner);
                if x != x {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

unsafe fn triangle_to_column<T: Scalar>(
    uplo: u8,
    n: usize,
    a: *const T,
    lda: usize,
) -> Result<Vec<T>, i64> {
    let count = n.checked_mul(n).ok_or(TRANSPOSE_MEMORY_ERROR)?;
    let span = (n - 1)
        .checked_mul(lda)
        .and_then(|x| x.checked_add(n))
        .and_then(|x| x.checked_mul(core::mem::size_of::<T>()))
        .ok_or(TRANSPOSE_MEMORY_ERROR)?;
    if span > isize::MAX as usize {
        return Err(TRANSPOSE_MEMORY_ERROR);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| TRANSPOSE_MEMORY_ERROR)?;
    out.resize(count, T::z());
    for j in 0..n {
        for i in 0..n {
            if selected(uplo, i, j) {
                out[i + j * n] = *a.add(i * lda + j);
            }
        }
    }
    Ok(out)
}

unsafe fn triangle_from_column<T: Scalar>(uplo: u8, n: usize, source: &[T], a: *mut T, lda: usize) {
    for j in 0..n {
        for i in 0..n {
            if selected(uplo, i, j) {
                *a.add(i * lda + j) = source[i + j * n];
            }
        }
    }
}

macro_rules! single {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$routine:ident)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine $($suffix)*>](layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i)->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true)=>return -4,Err(info)=>return info as $i,Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p $routine _work $($suffix)*>](layout,uplo,n,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine _work $($suffix)*>](layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i)->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if !crate::kernels::uplo(uplo.to_ascii_uppercase()) {return -2;}
            if n<0 {return -3;}
            if layout==COL_MAJOR && lda<1.max(n) {return -5;}
            if n==0 {return 0;}
            let mut info:$i=0;
            if layout==COL_MAJOR {
                crate::lapack::[<$p $routine $fortran>](&uplo,&n,a,&lda,&mut info);
            } else {
                let mut aa=match triangle_to_column(uplo,n as usize,a,lda as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                crate::lapack::[<$p $routine $fortran>](&uplo,&n,aa.as_mut_ptr(),&n,&mut info);
                triangle_from_column(uplo,n as usize,&aa,a,lda as usize);
            }
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! rhs {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$routine:ident,$mode:ident,$write:expr)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine $($suffix)*>](layout:i32,uplo:u8,n:$i,nrhs:$i,a:*$mode $t,lda:$i,b:*mut $t,ldb:$i)->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {
                    Ok(true)=>return -5,Err(info)=>return info as $i,Ok(false)=>{},
                }
                match ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64) {
                    Ok(true)=>return -7,Err(info)=>return info as $i,Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p $routine _work $($suffix)*>](layout,uplo,n,nrhs,a,lda,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $routine _work $($suffix)*>](layout:i32,uplo:u8,n:$i,nrhs:$i,a:*$mode $t,lda:$i,b:*mut $t,ldb:$i)->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR {
                if lda<n {return -6;}
                if ldb<nrhs {return -8;}
            }
            if !crate::kernels::uplo(uplo.to_ascii_uppercase()) {return -2;}
            if n<0 {return -3;}
            if nrhs<0 {return -4;}
            if layout==COL_MAJOR {
                if lda<1.max(n) {return -6;}
                if ldb<1.max(n) {return -8;}
            }
            if n==0 || (!$write && nrhs==0) {return 0;}
            let mut info:$i=0;
            if layout==COL_MAJOR {
                crate::lapack::[<$p $routine $fortran>](&uplo,&n,&nrhs,a,&lda,b,&ldb,&mut info);
            } else {
                let mut aa=match triangle_to_column(uplo,n as usize,a,lda as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                let mut bb=match row_to_column(n as usize,nrhs as usize,b,ldb as usize) {
                    Ok(v)=>v,Err(info)=>return info as $i,
                };
                crate::lapack::[<$p $routine $fortran>](&uplo,&n,&nrhs,aa.as_mut_ptr(),&n,bb.as_mut_ptr(),&n,&mut info);
                if $write {triangle_from_column(uplo,n as usize,&aa,a as *mut $t,lda as usize);}
                for j in 0..nrhs as usize {for i in 0..n as usize {
                    *b.add(i*ldb as usize+j)=bb[i+j*n as usize];
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! all {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{
        single!($p,$t,$i,[$($suffix)*],$fortran,potrf);
        single!($p,$t,$i,[$($suffix)*],$fortran,potrf2);
        single!($p,$t,$i,[$($suffix)*],$fortran,potri);
        rhs!($p,$t,$i,[$($suffix)*],$fortran,potrs,const,false);
        rhs!($p,$t,$i,[$($suffix)*],$fortran,posv,mut,true);
    };
}
all!(s, f32, i32, [], _);
all!(d, f64, i32, [], _);
all!(c, colmat::c32, i32, [], _);
all!(z, colmat::c64, i32, [], _);
all!(s, f32, i64, [64_], _64_);
all!(d, f64, i64, [64_], _64_);
all!(c, colmat::c32, i64, [64_], _64_);
all!(z, colmat::c64, i64, [64_], _64_);
