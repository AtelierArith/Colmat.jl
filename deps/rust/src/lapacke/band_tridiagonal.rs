//! LAPACKE SBTRD/HBTRD high/work adapters for banded symmetric/Hermitian
//! tridiagonal reduction.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}{sbtrd,hbtrd}{,_work}.c.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn index(row_major: bool, upper: bool, kd: usize, ld: usize, i: usize, j: usize) -> usize {
    if row_major {
        (if upper { kd + i - j } else { i - j }) * ld + j
    } else if upper {
        (kd + i - j) + j * ld
    } else {
        (i - j) + j * ld
    }
}

pub(super) unsafe fn band_to_column<T: Scalar>(
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

pub(super) unsafe fn band_from_column<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    kd: usize,
    ab: *mut T,
    ldab: usize,
    source: &[T],
) {
    if layout == COL_MAJOR {
        for k in 0..(kd + 1) * n {
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

pub(super) unsafe fn band_nan<T: Scalar>(
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
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,vect:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,
            d:*mut $r,e:*mut $r,q:*mut $t,ldq:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize) {return -6;}
                if vect.to_ascii_uppercase()==b'V'
                    && super::ge_nancheck(layout,n as i64,n as i64,q,ldq as i64).unwrap_or(false)
                {
                    return -10;
                }
            }
            let mut work=vec![<$t>::z();(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,vect,uplo,n,kd,ab,ldab,d,e,q,ldq,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,vect:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,
            d:*mut $r,e:*mut $r,q:*mut $t,ldq:$i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let want=vect.to_ascii_uppercase()==b'V';
            if layout==ROW_MAJOR && ldab<n {return -7;}
            if layout==ROW_MAJOR && want && ldq<n {return -11;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::structured_tridiagonal::[<$p $family $fortran>](&vect,&uplo,&n,&kd,ab,&ldab,d,e,q,&ldq,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,kk)=(n as usize,kd.max(0) as usize);
            let mut tmp=band_to_column(layout,uplo,nn,kk,ab,ldab as usize);
            let mut qt=if want {
                match row_to_column(nn,nn,q,ldq as usize){Ok(v)=>v,Err(err)=>return err as $i}
            } else {Vec::new()};
            let (ldab_t,ldq_t)=((kk+1).max(1) as $i,nn.max(1) as $i);
            let mut info:$i=0;
            crate::structured_tridiagonal::[<$p $family $fortran>](&vect,&uplo,&n,&kd,tmp.as_mut_ptr(),&ldab_t,d,e,qt.as_mut_ptr(),&ldq_t,work,&mut info);
            band_from_column(layout,uplo,nn,kk,ab,ldab as usize,&tmp);
            if info>=0 && want {
                for i in 0..nn {for j in 0..nn {*q.add(i*ldq as usize+j)=qt[i+j*nn];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, sbtrd, f32, f32, i32, [], _);
exports!(d, sbtrd, f64, f64, i32, [], _);
exports!(c, hbtrd, colmat::c32, f32, i32, [], _);
exports!(z, hbtrd, colmat::c64, f64, i32, [], _);
exports!(s, sbtrd, f32, f32, i64, [64_], _64_);
exports!(d, sbtrd, f64, f64, i64, [64_], _64_);
exports!(c, hbtrd, colmat::c32, f32, i64, [64_], _64_);
exports!(z, hbtrd, colmat::c64, f64, i64, [64_], _64_);
