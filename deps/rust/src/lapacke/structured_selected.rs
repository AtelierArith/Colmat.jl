//! LAPACKE SBEVX/HBEVX and SPEVX/HPEVX high/work adapters for selected
//! eigenpairs of banded and packed symmetric/Hermitian matrices.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sbevx,hbevx,spevx,hpevx}{,_work}.c.
//! Only the stored band/packed triangle is scanned for NaN and only the
//! returned eigenvector columns are written back in row-major.
use super::band_tridiagonal::{band_from_column, band_nan, band_to_column};
use super::packed_lu::{packed_from_column, packed_nan, packed_to_column};
use super::{LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

fn z_columns(jobz: u8, range: u8, n: i64, il: i64, iu: i64) -> i128 {
    if jobz.to_ascii_uppercase() != b'V' {
        1
    } else if matches!(range.to_ascii_uppercase(), b'A' | b'V') {
        n as i128
    } else if range.to_ascii_uppercase() == b'I' {
        iu as i128 - il as i128 + 1
    } else {
        1
    }
}

unsafe fn write_z<T: Scalar, I: LapackInt>(
    n: usize,
    n1: bool,
    m: *const I,
    z: *mut T,
    ldz: usize,
    zt: &[T],
) {
    let cols = if n1 {
        1
    } else {
        (*m).val().clamp(0, n as isize) as usize
    };
    for i in 0..n {
        for j in 0..cols {
            *z.add(i * ldz + j) = zt[i + j * n];
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn band_evx<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i64,
    kd: i64,
    ab: *mut T,
    ldab: i64,
    q: *mut T,
    ldq: i64,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    errs: (i64, i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR {
        if ldab < n {
            return errs.0;
        }
        if ldq < n {
            return errs.1;
        }
        if (ldz as i128) < z_columns(jobz, range, n, il, iu) {
            return errs.2;
        }
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ab, ldab, q, ldq, z, ldz);
    }
    let nn = n as usize;
    let kk = kd.max(0) as usize;
    let ldab_t = (kk + 1) as i64;
    let ld = n.max(1);
    let want = jobz.to_ascii_uppercase() == b'V';
    let mut at = band_to_column(layout, uplo, nn, kk, ab, ldab as usize);
    let mut qt = vec![T::z(); nn * nn];
    let mut zt = vec![T::z(); nn * nn];
    let info = call(
        at.as_mut_ptr(),
        ldab_t,
        qt.as_mut_ptr(),
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info >= 0 {
        band_from_column(layout, uplo, nn, kk, ab, ldab as usize, &at);
        if want {
            for i in 0..nn {
                for j in 0..nn {
                    *q.add(i * ldq as usize + j) = qt[i + j * nn];
                }
            }
            write_z(nn, nn == 1, m, z, ldz as usize, &zt);
        }
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn packed_evx<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i64,
    ap: *mut T,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    err: i64,
    mut call: impl FnMut(*mut T, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && (ldz as i128) < z_columns(jobz, range, n, il, iu) {
        return err;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ap, z, ldz);
    }
    let nn = n as usize;
    let ld = n.max(1);
    let want = jobz.to_ascii_uppercase() == b'V';
    let mut at = packed_to_column(layout, uplo, nn, ap);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(at.as_mut_ptr(), zt.as_mut_ptr(), ld);
    if info >= 0 {
        packed_from_column(layout, uplo, nn, ap, &at);
        if want {
            write_z(nn, nn == 1, m, z, ldz as usize, &zt);
        }
    }
    info
}

macro_rules! evx_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,[$($rw:ident)*])=>{paste::paste!{
        // ---- band ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,q:*mut $t,ldq:$i,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize) {return -7;}
                if abstol.is_nan() {return -15;}
                if range.to_ascii_uppercase()==b'V' {
                    if vl.is_nan() {return -11;}
                    if vu.is_nan() {return -12;}
                }
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$t>::z();(7*nn).max(1)];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,range,uplo,n,kd,ab,ldab,q,ldq,vl,vu,il,iu,abstol,m,w,z,ldz,
                work.as_mut_ptr()$(,$rw.as_mut_ptr())*,iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,q:*mut $t,ldq:$i,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            band_evx::<$t,$i>(layout,jobz,range,uplo,n as i64,kd as i64,ab,ldab as i64,q,ldq as i64,il as i64,iu as i64,m,z,ldz as i64,(-8,-10,-19),|a,la,qq,lq,zz,lz|{
                let mut info:$i=0;
                crate::structured_selected_evd::[<$p $bf $f>](&jobz,&range,&uplo,&n,&kd,a,&(la as $i),qq,&(lq as $i),
                    &vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),work,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- packed ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,ap:*mut $t,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if packed_nan(layout,uplo,n.max(0) as usize,ap,<$t>::COMPLEX) {return -6;}
                if abstol.is_nan() {return -11;}
                if range.to_ascii_uppercase()==b'V' {
                    if vl.is_nan() {return -7;}
                    if vu.is_nan() {return -8;}
                }
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$t>::z();(8*nn).max(1)];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,jobz,range,uplo,n,ap,vl,vu,il,iu,abstol,m,w,z,ldz,
                work.as_mut_ptr()$(,$rw.as_mut_ptr())*,iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,ap:*mut $t,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            packed_evx::<$t,$i>(layout,jobz,range,uplo,n as i64,ap,il as i64,iu as i64,m,z,ldz as i64,-15,|a,zz,lz|{
                let mut info:$i=0;
                crate::structured_selected_evd::[<$p $pf $f>](&jobz,&range,&uplo,&n,a,&vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),
                    work,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$f:tt)=>{
        evx_exports!(s,sbevx,spevx,f32,f32,$i,[$($suffix)*],$f,[]);
        evx_exports!(d,sbevx,spevx,f64,f64,$i,[$($suffix)*],$f,[]);
        evx_exports!(c,hbevx,hpevx,colmat::c32,f32,$i,[$($suffix)*],$f,[rwork]);
        evx_exports!(z,hbevx,hpevx,colmat::c64,f64,$i,[$($suffix)*],$f,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
