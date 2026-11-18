//! LAPACKE LANGB and LANHE high/work adapters.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}langb{,_work}.c and
//! LAPACKE/src/lapacke_{c,z}lanhe{,_work}.c.
//! The band input uses the analysis convention AB(KU+1+i-j, j) with
//! LDAB >= KL+KU+1. Row-major results equal the column-major results for the
//! same logical matrix, unlike the upstream swap trick which is wrong for
//! non-symmetric norms.
use super::{LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn band_nan<T: Scalar>(
    layout: i32,
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> bool {
    let row_major = layout == ROW_MAJOR;
    for j in 0..n {
        let lo = j.saturating_sub(ku);
        let hi = (j + kl).min(n.saturating_sub(1));
        for i in lo..=hi {
            let v = if row_major {
                *ab.add((ku + i - j) * ldab + j)
            } else {
                *ab.add((ku + i - j) + j * ldab)
            };
            if v != v {
                return true;
            }
        }
    }
    false
}

unsafe fn band_to_column<T: Scalar>(
    n: usize,
    kl: usize,
    ku: usize,
    ab: *const T,
    ldab: usize,
) -> Vec<T> {
    let ld = kl + ku + 1;
    let mut out = vec![T::z(); ld * n];
    for j in 0..n {
        let lo = j.saturating_sub(ku);
        let hi = (j + kl).min(n.saturating_sub(1));
        for i in lo..=hi {
            out[(ku + i - j) + j * ld] = *ab.add((ku + i - j) * ldab + j);
        }
    }
    out
}

unsafe fn triangle_nan<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    a: *const T,
    lda: usize,
) -> bool {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper {
            (0, j)
        } else {
            (j, n.saturating_sub(1))
        };
        for i in lo..=hi {
            let v = if layout == ROW_MAJOR {
                *a.add(i * lda + j)
            } else {
                *a.add(i + j * lda)
            };
            if v != v {
                return true;
            }
        }
    }
    false
}

macro_rules! langb {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p langb $($suffix)*>](
            layout:i32,norm:u8,n:$i,kl:$i,ku:$i,ab:*const $t,ldab:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1.0 as $r;}
            if LAPACKE_get_nancheck()!=0
                && band_nan(layout,n.max(0) as usize,kl.max(0) as usize,ku.max(0) as usize,ab,ldab.max(0) as usize)
            {
                return -6.0 as $r;
            }
            let mut work=vec![<$r>::z();(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p langb_work $($suffix)*>](layout,norm,n,kl,ku,ab,ldab,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p langb_work $($suffix)*>](
            layout:i32,norm:u8,n:$i,kl:$i,ku:$i,ab:*const $t,ldab:$i,work:*mut $r,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1.0 as $r;}
            if layout==ROW_MAJOR && (ldab as i64)<(kl as i64)+(ku as i64)+1 {return -7.0 as $r;}
            if layout==COL_MAJOR || n==0 {
                return crate::matrix_norms::[<$p langb $fortran>](&norm,&n,&kl,&ku,ab,&ldab,work);
            }
            let (nn,kk,uu)=(n as usize,kl.max(0) as usize,ku.max(0) as usize);
            let tmp=band_to_column(nn,kk,uu,ab,ldab as usize);
            let ld=(kk+uu+1).max(1) as $i;
            crate::matrix_norms::[<$p langb $fortran>](&norm,&n,&kl,&ku,tmp.as_ptr(),&ld,work)
        }
    }};
}

macro_rules! lanhe {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lanhe $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,n:$i,a:*const $t,lda:$i,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1.0 as $r;}
            if LAPACKE_get_nancheck()!=0
                && triangle_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize)
            {
                return -5.0 as $r;
            }
            let mut work=vec![<$r>::z();(n.max(0) as usize).max(1)];
            [<LAPACKE_ $p lanhe_work $($suffix)*>](layout,norm,uplo,n,a,lda,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lanhe_work $($suffix)*>](
            layout:i32,norm:u8,uplo:u8,n:$i,a:*const $t,lda:$i,work:*mut $r,
        )->$r {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1.0 as $r;}
            if layout==ROW_MAJOR && lda<n {return -6.0 as $r;}
            if layout==COL_MAJOR || n==0 {
                return crate::matrix_norms::[<$p lanhe $fortran>](&norm,&uplo,&n,a,&lda,work);
            }
            // The native norm reads only the selected triangle, so converting
            // that triangle over the logical rectangle is exact.
            let nn=n as usize;
            let upper=uplo.to_ascii_uppercase()==b'U';
            let ld=nn.max(1) as $i;
            let mut tmp=vec![<$t>::z();nn*nn];
            for j in 0..nn {
                let (lo,hi)=if upper {(0,j)} else {(j,nn-1)};
                for i in lo..=hi {
                    tmp[i+j*nn]=*a.add(i*lda as usize+j);
                }
            }
            crate::matrix_norms::[<$p lanhe $fortran>](&norm,&uplo,&n,tmp.as_ptr(),&ld,work)
        }
    }};
}
langb!(s, f32, f32, i32, [], _);
langb!(d, f64, f64, i32, [], _);
langb!(c, colmat::c32, f32, i32, [], _);
langb!(z, colmat::c64, f64, i32, [], _);
langb!(s, f32, f32, i64, [64_], _64_);
langb!(d, f64, f64, i64, [64_], _64_);
langb!(c, colmat::c32, f32, i64, [64_], _64_);
langb!(z, colmat::c64, f64, i64, [64_], _64_);
lanhe!(c, colmat::c32, f32, i32, [], _);
lanhe!(z, colmat::c64, f64, i32, [], _);
lanhe!(c, colmat::c32, f32, i64, [64_], _64_);
lanhe!(z, colmat::c64, f64, i64, [64_], _64_);
