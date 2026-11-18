//! LAPACKE TBTRS, TPTRS and TPTRI high/work adapters for triangular band and
//! packed triangular systems.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{tbtrs,tptrs,tptri}{,_work}.c.
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

fn band_index(row_major: bool, upper: bool, kd: usize, ld: usize, i: usize, j: usize) -> usize {
    let r = if upper { kd + i - j } else { i - j };
    if row_major {
        r * ld + j
    } else {
        r + j * ld
    }
}
unsafe fn band_to_column<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    kd: usize,
    ab: *const T,
    ldab: usize,
) -> Vec<T> {
    let ld = kd + 1;
    let upper = uplo.to_ascii_uppercase() == b'U';
    let mut out = vec![T::z(); ld * n];
    for j in 0..n {
        let (lo, hi) = if upper {
            (j.saturating_sub(kd), j)
        } else {
            (j, (j + kd).min(n - 1))
        };
        for i in lo..=hi {
            out[band_index(false, upper, kd, ld, i, j)] =
                *ab.add(band_index(layout == ROW_MAJOR, upper, kd, ldab, i, j));
        }
    }
    out
}
unsafe fn band_nan<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    kd: usize,
    ab: *const T,
    ldab: usize,
) -> bool {
    let row_major = layout == ROW_MAJOR;
    let upper = uplo.to_ascii_uppercase() == b'U';
    let ld = if row_major { ldab } else { kd + 1 };
    for j in 0..n {
        let (lo, hi) = if upper {
            (j.saturating_sub(kd), j)
        } else {
            (j, (j + kd).min(n - 1))
        };
        for i in lo..=hi {
            let v = *ab.add(band_index(row_major, upper, kd, ld, i, j));
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
        pub unsafe extern "C" fn [<LAPACKE_ $p tbtrs $($suffix)*>](
            layout:i32,uplo:u8,trans:u8,diag:u8,n:$i,kd:$i,nrhs:$i,
            ab:*const $t,ldab:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize) {return -8;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -10;}
            }
            [<LAPACKE_ $p tbtrs_work $($suffix)*>](layout,uplo,trans,diag,n,kd,nrhs,ab,ldab,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tbtrs_work $($suffix)*>](
            layout:i32,uplo:u8,trans:u8,diag:u8,n:$i,kd:$i,nrhs:$i,
            ab:*const $t,ldab:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -9;}
            if layout==ROW_MAJOR && ldb<nrhs {return -11;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::structured_triangular_lapack::[<$p tbtrs $fortran>](&uplo,&trans,&diag,&n,&kd,&nrhs,ab,&ldab,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk,rr)=(n as usize,kd.max(0) as usize,nrhs as usize);
            let tmp=band_to_column(layout,uplo,nn,kk,ab,ldab as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldab_t,ldb_t)=((kk+1).max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::structured_triangular_lapack::[<$p tbtrs $fortran>](&uplo,&trans,&diag,&n,&kd,&nrhs,tmp.as_ptr(),&ldab_t,bt.as_mut_ptr(),&ldb_t,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tptrs $($suffix)*>](
            layout:i32,uplo:u8,trans:u8,diag:u8,n:$i,nrhs:$i,
            ap:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -7;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            [<LAPACKE_ $p tptrs_work $($suffix)*>](layout,uplo,trans,diag,n,nrhs,ap,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tptrs_work $($suffix)*>](
            layout:i32,uplo:u8,trans:u8,diag:u8,n:$i,nrhs:$i,
            ap:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::structured_triangular_lapack::[<$p tptrs $fortran>](&uplo,&trans,&diag,&n,&nrhs,ap,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let tmp=packed_to_column(layout,uplo,nn,ap);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::structured_triangular_lapack::[<$p tptrs $fortran>](&uplo,&trans,&diag,&n,&nrhs,tmp.as_ptr(),bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tptri $($suffix)*>](
            layout:i32,uplo:u8,diag:u8,n:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -5;}
            [<LAPACKE_ $p tptri_work $($suffix)*>](layout,uplo,diag,n,ap)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tptri_work $($suffix)*>](
            layout:i32,uplo:u8,diag:u8,n:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::structured_triangular_lapack::[<$p tptri $fortran>](&uplo,&diag,&n,ap,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tmp=packed_to_column(layout,uplo,nn,ap);
            let mut info:$i=0;
            crate::structured_triangular_lapack::[<$p tptri $fortran>](&uplo,&diag,&n,tmp.as_mut_ptr(),&mut info);
            if info>=0 {packed_from_column(layout,uplo,nn,ap,&tmp);}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
