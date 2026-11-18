//! LAPACKE SPTRF/SPTRS/SPSV high/work adapters for packed symmetric
//! indefinite LU (Bunch-Kaufman).
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d}{sptrf,sptrs,spsv}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}
fn row_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        i * n - i * (i + 1) / 2 + j
    } else {
        i * (i + 1) / 2 + j
    }
}
fn col_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        j * (j + 1) / 2 + i
    } else {
        j * n - j * (j + 1) / 2 + i
    }
}
unsafe fn packed_to_column<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> Vec<T> {
    let upper = uplo.to_ascii_uppercase() == b'U';
    let mut out = vec![T::z(); packed_len(n)];
    if layout == COL_MAJOR {
        for k in 0..packed_len(n) {
            out[k] = *ap.add(k);
        }
        return out;
    }
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            out[col_index(upper, n, i, j)] = *ap.add(row_index(upper, n, i, j));
        }
    }
    out
}
unsafe fn packed_from_column<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *mut T, source: &[T]) {
    if layout == COL_MAJOR {
        for k in 0..packed_len(n) {
            *ap.add(k) = source[k];
        }
        return;
    }
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            *ap.add(row_index(upper, n, i, j)) = source[col_index(upper, n, i, j)];
        }
    }
}
unsafe fn packed_nan<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> bool {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            let k = if layout == ROW_MAJOR {
                row_index(upper, n, i, j)
            } else {
                col_index(upper, n, i, j)
            };
            let v = *ap.add(k);
            if v != v {
                return true;
            }
        }
    }
    false
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sptrf $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,ipiv:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -4;}
            [<LAPACKE_ $p sptrf_work $($suffix)*>](layout,uplo,n,ap,ipiv)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sptrf_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,ipiv:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p sptrf $fortran>](&uplo,&n,ap,ipiv,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tmp=packed_to_column(layout,uplo,nn,ap);
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p sptrf $fortran>](&uplo,&n,tmp.as_mut_ptr(),ipiv,&mut info);
            packed_from_column(layout,uplo,nn,ap,&tmp);
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sptrs $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*const $t,ipiv:*const $i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -5;}
            if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -7;}
            [<LAPACKE_ $p sptrs_work $($suffix)*>](layout,uplo,n,nrhs,ap,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p sptrs_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*const $t,ipiv:*const $i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -8;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p sptrs $fortran>](&uplo,&n,&nrhs,ap,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let tmp=packed_to_column(layout,uplo,nn,ap);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p sptrs $fortran>](&uplo,&n,&nrhs,tmp.as_ptr(),ipiv,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p spsv $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*mut $t,ipiv:*mut $i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -5;}
            if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -7;}
            [<LAPACKE_ $p spsv_work $($suffix)*>](layout,uplo,n,nrhs,ap,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p spsv_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*mut $t,ipiv:*mut $i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -8;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p spsv $fortran>](&uplo,&n,&nrhs,ap,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut tmp=packed_to_column(layout,uplo,nn,ap);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p spsv $fortran>](&uplo,&n,&nrhs,tmp.as_mut_ptr(),ipiv,bt.as_mut_ptr(),&ld,&mut info);
            packed_from_column(layout,uplo,nn,ap,&tmp);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
