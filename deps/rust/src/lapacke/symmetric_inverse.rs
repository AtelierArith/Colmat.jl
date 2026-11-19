//! LAPACKE SYTRI/SYTRI2/SYTRI2X (S/D) and HETRI/HETRI2/HETRI2X (C/Z)
//! high/work adapters over the native Bunch-Kaufman inverse.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d}{sytri,sytri2,sytri2x}{,_work}.c and
//! LAPACKE/src/lapacke_{c,z}{hetri,hetri2,hetri2x}{,_work}.c.
//! Only the UPLO triangle is scanned for NaN (Hermitian diagonal imaginary
//! parts are ignored) and only that triangle is written back in row-major.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn tri_nan<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    a: *const T,
    lda: usize,
    hermitian: bool,
) -> bool {
    if a.is_null() {
        return false;
    }
    let upper = uplo.to_ascii_uppercase() == b'U';
    if !upper && uplo.to_ascii_uppercase() != b'L' {
        return false;
    }
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            // Reference scans min(extent, lda) entries per stored line.
            let (r, c) = (i, j);
            let k = if layout == ROW_MAJOR {
                if c >= lda {
                    continue;
                }
                r * lda + c
            } else {
                if r >= lda {
                    continue;
                }
                c * lda + r
            };
            let v = *a.add(k);
            let bad = if hermitian && i == j {
                v.re() != v.re()
            } else {
                v != v
            };
            if bad {
                return true;
            }
        }
    }
    false
}

unsafe fn write_back<T: Scalar>(uplo: u8, n: usize, a: *mut T, lda: usize, at: &[T]) {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for i in 0..n {
        for j in 0..n {
            if (upper && i <= j) || (!upper && i >= j) {
                *a.add(i * lda + j) = at[i + j * n];
            }
        }
    }
}

macro_rules! exports {
    ($p:ident,$family:ident,$herm:expr,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        // ---- TRI ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && tri_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize,$herm)
            {
                return -4;
            }
            let mut work=vec![<$t>::z();(n.max(1)) as usize];
            [<LAPACKE_ $p $family tri_work $($suffix)*>](layout,uplo,n,a,lda,ipiv,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family tri $fortran>](&uplo,&n,a,&lda,ipiv,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family tri $fortran>](&uplo,&n,at.as_mut_ptr(),&ld,ipiv,work,&mut info);
            if info>=0 {write_back(uplo,nn,a,lda as usize,&at);}
            if info<0 {info-1} else {info}
        }
        // ---- TRI2 ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri2 $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && tri_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize,$herm)
            {
                return -4;
            }
            let mut query=<$t>::z();
            let r=[<LAPACKE_ $p $family tri2_work $($suffix)*>](layout,uplo,n,a,lda,ipiv,&mut query,-1);
            if r!=0 {return r;}
            let lw=query.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family tri2_work $($suffix)*>](layout,uplo,n,a,lda,ipiv,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri2_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 || lwork==-1 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family tri2 $fortran>](&uplo,&n,a,&lda,ipiv,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family tri2 $fortran>](&uplo,&n,at.as_mut_ptr(),&ld,ipiv,work,&lwork,&mut info);
            if info>=0 {write_back(uplo,nn,a,lda as usize,&at);}
            if info<0 {info-1} else {info}
        }
        // ---- TRI2X ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri2x $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,nb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && tri_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize,$herm)
            {
                return -4;
            }
            let lw=((n.max(0) as i128+nb.max(0) as i128+1)*(nb.max(0) as i128+3)).max(1) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $family tri2x_work $($suffix)*>](layout,uplo,n,a,lda,ipiv,work.as_mut_ptr(),nb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family tri2x_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,ipiv:*const $i,work:*mut $t,nb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::bunch_kaufman::[<$p $family tri2x $fortran>](&uplo,&n,a,&lda,ipiv,work,&nb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::bunch_kaufman::[<$p $family tri2x $fortran>](&uplo,&n,at.as_mut_ptr(),&ld,ipiv,work,&nb,&mut info);
            if info>=0 {write_back(uplo,nn,a,lda as usize,&at);}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, sy, false, f32, i32, [], _);
exports!(d, sy, false, f64, i32, [], _);
exports!(c, he, true, colmat::c32, i32, [], _);
exports!(z, he, true, colmat::c64, i32, [], _);
exports!(s, sy, false, f32, i64, [64_], _64_);
exports!(d, sy, false, f64, i64, [64_], _64_);
exports!(c, he, true, colmat::c32, i64, [64_], _64_);
exports!(z, he, true, colmat::c64, i64, [64_], _64_);
