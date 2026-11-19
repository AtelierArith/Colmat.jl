//! LAPACKE PBEQU, PBSTF and PBSV high/work adapters for the positive-definite
//! band extras.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{pbequ,pbstf,pbsv}{,_work}.c.
//! The band uses the KD+1-row convention AB(KD+1+i-j, j).
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn index(row_major: bool, upper: bool, kd: usize, ld: usize, i: usize, j: usize) -> usize {
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
            out[index(false, upper, kd, ld, i, j)] =
                *ab.add(index(layout == ROW_MAJOR, upper, kd, ldab, i, j));
        }
    }
    out
}
unsafe fn band_from_column<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    kd: usize,
    ab: *mut T,
    ldab: usize,
    source: &[T],
) {
    if layout == COL_MAJOR {
        let ld = kd + 1;
        for k in 0..ld * n {
            *ab.add(k) = source[k];
        }
        return;
    }
    let upper = uplo.to_ascii_uppercase() == b'U';
    let ld = kd + 1;
    for j in 0..n {
        let (lo, hi) = if upper {
            (j.saturating_sub(kd), j)
        } else {
            (j, (j + kd).min(n - 1))
        };
        for i in lo..=hi {
            *ab.add(index(true, upper, kd, ldab, i, j)) = source[index(false, upper, kd, ld, i, j)];
        }
    }
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
            let v = *ab.add(index(row_major, upper, kd, ld, i, j));
            if v != v {
                return true;
            }
        }
    }
    false
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbequ $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kd:$i,ab:*const $t,ldab:$i,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize) {return -5;}
            [<LAPACKE_ $p pbequ_work $($suffix)*>](layout,uplo,n,kd,ab,ldab,s,scond,amax)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbequ_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kd:$i,ab:*const $t,ldab:$i,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -6;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::positive_equilibration::[<$p pbequ $fortran>](&uplo,&n,&kd,ab,&ldab,s,scond,amax,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk)=(n as usize,kd.max(0) as usize);
            let tmp=band_to_column(layout,uplo,nn,kk,ab,ldab as usize);
            let ld=(kk+1).max(1) as $i;
            let mut info:$i=0;
            crate::positive_equilibration::[<$p pbequ $fortran>](&uplo,&n,&kd,tmp.as_ptr(),&ld,s,scond,amax,&mut info);
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbstf $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kb:$i,bb:*mut $t,ldbb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if band_nan(layout,uplo,n.max(0) as usize,kb.max(0) as usize,bb,ldbb.max(0) as usize) {return -5;}
            [<LAPACKE_ $p pbstf_work $($suffix)*>](layout,uplo,n,kb,bb,ldbb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbstf_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kb:$i,bb:*mut $t,ldbb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldbb<n {return -6;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::band_cholesky::[<$p pbstf $fortran>](&uplo,&n,&kb,bb,&ldbb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk)=(n as usize,kb.max(0) as usize);
            let mut tmp=band_to_column(layout,uplo,nn,kk,bb,ldbb as usize);
            let ld=(kk+1).max(1) as $i;
            let mut info:$i=0;
            crate::band_cholesky::[<$p pbstf $fortran>](&uplo,&n,&kb,tmp.as_mut_ptr(),&ld,&mut info);
            if info>=0 {band_from_column(layout,uplo,nn,kk,bb,ldbb as usize,&tmp);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbsv $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kd:$i,nrhs:$i,ab:*mut $t,ldab:$i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize) {return -6;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -8;}
            }
            [<LAPACKE_ $p pbsv_work $($suffix)*>](layout,uplo,n,kd,nrhs,ab,ldab,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pbsv_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,kd:$i,nrhs:$i,ab:*mut $t,ldab:$i,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldab<n {return -7;}
            if layout==ROW_MAJOR && ldb<nrhs {return -9;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::band_cholesky::[<$p pbsv $fortran>](&uplo,&n,&kd,&nrhs,ab,&ldab,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk,rr)=(n as usize,kd.max(0) as usize,nrhs as usize);
            let mut tmp=band_to_column(layout,uplo,nn,kk,ab,ldab as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldab_t,ldb_t)=((kk+1).max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::band_cholesky::[<$p pbsv $fortran>](&uplo,&n,&kd,&nrhs,tmp.as_mut_ptr(),&ldab_t,bt.as_mut_ptr(),&ldb_t,&mut info);
            band_from_column(layout,uplo,nn,kk,ab,ldab as usize,&tmp);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, f32, i32, [], _);
exports!(d, f64, f64, i32, [], _);
exports!(c, colmat::c32, f32, i32, [], _);
exports!(z, colmat::c64, f64, i32, [], _);
exports!(s, f32, f32, i64, [64_], _64_);
exports!(d, f64, f64, i64, [64_], _64_);
exports!(c, colmat::c32, f32, i64, [64_], _64_);
exports!(z, colmat::c64, f64, i64, [64_], _64_);
