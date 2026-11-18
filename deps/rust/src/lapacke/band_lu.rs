//! LAPACKE GBTRF/GBTRS/GBSV high/work adapters for general band LU.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{gbtrf,gbtrs,gbsv}{,_work}.c.
use super::{ge_nancheck, row_to_column, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn band_rows(kl: usize, ku: usize) -> usize {
    2 * kl + ku + 1
}
fn source_index(row_major: bool, kl: usize, ku: usize, ld: usize, i: usize, j: usize) -> usize {
    let r = kl + ku + i - j;
    if row_major {
        r * ld + j
    } else {
        r + j * ld
    }
}
unsafe fn band_to_column<T: Scalar>(
    layout: i32,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> Vec<T> {
    let ld = band_rows(kl, ku);
    let mut out = vec![T::z(); ld * n];
    let row_major = layout == ROW_MAJOR;
    for j in 0..n {
        let lo = j.saturating_sub(ku);
        let hi = (j + kl).min(m.saturating_sub(1));
        for i in lo..=hi {
            out[source_index(false, kl, ku, ld, i, j)] =
                *ab.add(source_index(row_major, kl, ku, ldab, i, j));
        }
    }
    out
}
unsafe fn band_from_column<T: Scalar>(
    layout: i32,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *mut T,
    ldab: usize,
    source: &[T],
) {
    if layout == COL_MAJOR {
        let ld = band_rows(kl, ku);
        for k in 0..ld * n {
            *ab.add(k) = source[k];
        }
        return;
    }
    let ld = band_rows(kl, ku);
    for j in 0..n {
        let lo = j.saturating_sub(ku);
        let hi = (j + kl).min(m.saturating_sub(1));
        for i in lo..=hi {
            *ab.add(source_index(true, kl, ku, ldab, i, j)) =
                source[source_index(false, kl, ku, ld, i, j)];
        }
    }
}
unsafe fn band_nan<T: Scalar>(
    layout: i32,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> bool {
    let row_major = layout == ROW_MAJOR;
    let ld = if row_major { ldab } else { band_rows(kl, ku) };
    for j in 0..n {
        let lo = j.saturating_sub(ku);
        let hi = (j + kl).min(m.saturating_sub(1));
        for i in lo..=hi {
            let v = *ab.add(source_index(row_major, kl, ku, ld, i, j));
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
        pub unsafe extern "C" fn [<LAPACKE_ $p gbtrf $($suffix)*>](
            layout:i32,m:$i,n:$i,kl:$i,ku:$i,ab:*mut $t,ldab:$i,ipiv:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if band_nan(layout,m.max(0) as usize,n.max(0) as usize,kl.max(0) as usize,ku.max(0) as usize,ab,ldab.max(0) as usize) {return -6;}
            [<LAPACKE_ $p gbtrf_work $($suffix)*>](layout,m,n,kl,ku,ab,ldab,ipiv)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gbtrf_work $($suffix)*>](
            layout:i32,m:$i,n:$i,kl:$i,ku:$i,ab:*mut $t,ldab:$i,ipiv:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -7;}
            if layout==COL_MAJOR || m==0 || n==0 {
                let mut info:$i=0;
                crate::band_lu::[<$p gbtrf $fortran>](&m,&n,&kl,&ku,ab,&ldab,ipiv,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,kk,uu)=(m as usize,n as usize,kl.max(0) as usize,ku.max(0) as usize);
            let mut tmp=band_to_column(layout,mm,nn,kk,uu,ab,ldab as usize);
            let ld=band_rows(kk,uu).max(1) as $i;
            let mut info:$i=0;
            crate::band_lu::[<$p gbtrf $fortran>](&m,&n,&kl,&ku,tmp.as_mut_ptr(),&ld,ipiv,&mut info);
            band_from_column(layout,mm,nn,kk,uu,ab,ldab as usize,&tmp);
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gbtrs $($suffix)*>](
            layout:i32,trans:u8,n:$i,kl:$i,ku:$i,nrhs:$i,
            ab:*const $t,ldab:$i,ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if band_nan(layout,n.max(0) as usize,n.max(0) as usize,kl.max(0) as usize,ku.max(0) as usize,ab,ldab.max(0) as usize) {return -7;}
            if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -10;}
            [<LAPACKE_ $p gbtrs_work $($suffix)*>](layout,trans,n,kl,ku,nrhs,ab,ldab,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gbtrs_work $($suffix)*>](
            layout:i32,trans:u8,n:$i,kl:$i,ku:$i,nrhs:$i,
            ab:*const $t,ldab:$i,ipiv:*const $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -8;}
            if layout==ROW_MAJOR && ldb<nrhs {return -11;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::band_lu::[<$p gbtrs $fortran>](&trans,&n,&kl,&ku,&nrhs,ab,&ldab,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk,uu,rr)=(n as usize,kl.max(0) as usize,ku.max(0) as usize,nrhs as usize);
            let tmp=band_to_column(layout,nn,nn,kk,uu,ab,ldab as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldab_t,ldb_t)=(band_rows(kk,uu).max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::band_lu::[<$p gbtrs $fortran>](&trans,&n,&kl,&ku,&nrhs,tmp.as_ptr(),&ldab_t,ipiv,bt.as_mut_ptr(),&ldb_t,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gbsv $($suffix)*>](
            layout:i32,n:$i,kl:$i,ku:$i,nrhs:$i,ab:*mut $t,ldab:$i,
            ipiv:*mut $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if band_nan(layout,n.max(0) as usize,n.max(0) as usize,kl.max(0) as usize,ku.max(0) as usize,ab,ldab.max(0) as usize) {return -6;}
            if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -9;}
            [<LAPACKE_ $p gbsv_work $($suffix)*>](layout,n,kl,ku,nrhs,ab,ldab,ipiv,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p gbsv_work $($suffix)*>](
            layout:i32,n:$i,kl:$i,ku:$i,nrhs:$i,ab:*mut $t,ldab:$i,
            ipiv:*mut $i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -7;}
            if layout==ROW_MAJOR && ldb<nrhs {return -10;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::band_lu::[<$p gbsv $fortran>](&n,&kl,&ku,&nrhs,ab,&ldab,ipiv,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk,uu,rr)=(n as usize,kl.max(0) as usize,ku.max(0) as usize,nrhs as usize);
            let mut tmp=band_to_column(layout,nn,nn,kk,uu,ab,ldab as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldab_t,ldb_t)=(band_rows(kk,uu).max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::band_lu::[<$p gbsv $fortran>](&n,&kl,&ku,&nrhs,tmp.as_mut_ptr(),&ldab_t,ipiv,bt.as_mut_ptr(),&ldb_t,&mut info);
            band_from_column(layout,nn,nn,kk,uu,ab,ldab as usize,&tmp);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
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
