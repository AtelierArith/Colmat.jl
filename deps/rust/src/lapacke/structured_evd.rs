//! LAPACKE SBEV/HBEV, SPEV/HPEV, SBEVD/HBEVD and SPEVD/HPEVD high/work
//! adapters for banded and packed symmetric/Hermitian eigenproblems.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sbev,hbev,spev,hpev,sbevd,hbevd,spevd,hpevd}{,_work}.c.
use super::band_tridiagonal::{band_from_column, band_nan, band_to_column};
use super::packed_lu::{packed_from_column, packed_nan, packed_to_column};
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn z_out<T: Scalar>(n: usize, z: *mut T, ldz: usize, zt: &[T]) {
    for i in 0..n {
        for j in 0..n {
            *z.add(i * ldz + j) = zt[i + j * n];
        }
    }
}

/// Shared row-major plumbing. `call(ab, ldab, z, ldz)` runs the native solver
/// and returns an already layout-shifted status.
#[allow(clippy::too_many_arguments)]
unsafe fn band_job<T: Scalar>(
    layout: i32,
    jobz: u8,
    uplo: u8,
    n: i64,
    kd: i64,
    ab: *mut T,
    ldab: i64,
    z: *mut T,
    ldz: i64,
    query: bool,
    errs: (i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && ldab < n {
        return errs.0;
    }
    if layout == ROW_MAJOR && ldz < n {
        return errs.1;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ab, ldab, z, ldz);
    }
    let (nn, kk) = (n as usize, kd.max(0) as usize);
    let ldab_t = (kk + 1) as i64;
    let ldz_t = nn.max(1) as i64;
    if query {
        return call(ab, ldab_t, z, ldz_t);
    }
    let want = jobz.to_ascii_uppercase() == b'V';
    let mut tmp = band_to_column(layout, uplo, nn, kk, ab, ldab as usize);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(tmp.as_mut_ptr(), ldab_t, zt.as_mut_ptr(), ldz_t);
    band_from_column(layout, uplo, nn, kk, ab, ldab as usize, &tmp);
    if info >= 0 && want {
        z_out(nn, z, ldz as usize, &zt);
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn packed_job<T: Scalar>(
    layout: i32,
    jobz: u8,
    uplo: u8,
    n: i64,
    ap: *mut T,
    z: *mut T,
    ldz: i64,
    query: bool,
    err: i64,
    mut call: impl FnMut(*mut T, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && ldz < n {
        return err;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ap, z, ldz);
    }
    let nn = n as usize;
    let ldz_t = nn.max(1) as i64;
    if query {
        return call(ap, z, ldz_t);
    }
    let want = jobz.to_ascii_uppercase() == b'V';
    let mut tmp = packed_to_column(layout, uplo, nn, ap);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(tmp.as_mut_ptr(), zt.as_mut_ptr(), ldz_t);
    packed_from_column(layout, uplo, nn, ap, &tmp);
    if info >= 0 && want {
        z_out(nn, z, ldz as usize, &zt);
    }
    info
}

macro_rules! ev_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize)
            {
                return -6;
            }
            let lw=(3*n.max(0) as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();lw];)*
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,kd,ab,ldab,w,z,ldz,work.as_mut_ptr()$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*
        )->$i {
            band_job(layout,jobz,uplo,n as i64,kd as i64,ab,ldab as i64,z,ldz as i64,false,(-7,-10),|a,la,zz,lz|{
                let mut info:$i=0;
                crate::structured_evd::[<$p $bf $f>](&jobz,&uplo,&n,&kd,a,&(la as $i),w,zz,&(lz as $i),work,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ap:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && packed_nan(layout,uplo,n.max(0) as usize,ap,<$t>::COMPLEX)
            {
                return -5;
            }
            let lw=(3*n.max(0) as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();lw];)*
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,jobz,uplo,n,ap,w,z,ldz,work.as_mut_ptr()$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ap:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*
        )->$i {
            packed_job(layout,jobz,uplo,n as i64,ap,z,ldz as i64,false,-8,|a,zz,lz|{
                let mut info:$i=0;
                crate::structured_evd::[<$p $pf $f>](&jobz,&uplo,&n,a,w,zz,&(lz as $i),work,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! evd_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$f:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && band_nan(layout,uplo,n.max(0) as usize,kd.max(0) as usize,ab,ldab.max(0) as usize)
            {
                return -6;
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            $(let mut [<q $rw>]=<$r>::z();)*
            let r=[<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,kd,ab,ldab,w,z,ldz,&mut qw,-1,$(&mut [<q $rw>],-1,)*&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let [<l $rw>]=[<q $rw>].re().max(1.0) as usize;let mut $rw=vec![<$r>::z();[<l $rw>]];)*
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,kd,ab,ldab,w,z,ldz,work.as_mut_ptr(),lw as $i,$($rw.as_mut_ptr(),[<l $rw>] as $i,)*iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,kd:$i,ab:*mut $t,ldab:$i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,$($rw:*mut $r,[<l $rw>]:$i,)*iwork:*mut $i,liwork:$i,
        )->$i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $rw>]== -1)*;
            band_job(layout,jobz,uplo,n as i64,kd as i64,ab,ldab as i64,z,ldz as i64,query,(-7,-10),|a,la,zz,lz|{
                let mut info:$i=0;
                crate::structured_evd::[<$p $bf $f>](&jobz,&uplo,&n,&kd,a,&(la as $i),w,zz,&(lz as $i),work,&lwork,$($rw,&[<l $rw>],)*iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ap:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && packed_nan(layout,uplo,n.max(0) as usize,ap,<$t>::COMPLEX)
            {
                return -5;
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            $(let mut [<q $rw>]=<$r>::z();)*
            let r=[<LAPACKE_ $p $pf _work $($suffix)*>](layout,jobz,uplo,n,ap,w,z,ldz,&mut qw,-1,$(&mut [<q $rw>],-1,)*&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let [<l $rw>]=[<q $rw>].re().max(1.0) as usize;let mut $rw=vec![<$r>::z();[<l $rw>]];)*
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,jobz,uplo,n,ap,w,z,ldz,work.as_mut_ptr(),lw as $i,$($rw.as_mut_ptr(),[<l $rw>] as $i,)*iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ap:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,$($rw:*mut $r,[<l $rw>]:$i,)*iwork:*mut $i,liwork:$i,
        )->$i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $rw>]== -1)*;
            packed_job(layout,jobz,uplo,n as i64,ap,z,ldz as i64,query,-8,|a,zz,lz|{
                let mut info:$i=0;
                crate::structured_evd::[<$p $pf $f>](&jobz,&uplo,&n,a,w,zz,&(lz as $i),work,&lwork,$($rw,&[<l $rw>],)*iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$f:tt)=>{
        ev_exports!(s,sbev,spev,f32,f32,$i,[$($suffix)*],$f,[]);
        ev_exports!(d,sbev,spev,f64,f64,$i,[$($suffix)*],$f,[]);
        ev_exports!(c,hbev,hpev,colmat::c32,f32,$i,[$($suffix)*],$f,[rwork]);
        ev_exports!(z,hbev,hpev,colmat::c64,f64,$i,[$($suffix)*],$f,[rwork]);
        evd_exports!(s,sbevd,spevd,f32,f32,$i,[$($suffix)*],$f,[]);
        evd_exports!(d,sbevd,spevd,f64,f64,$i,[$($suffix)*],$f,[]);
        evd_exports!(c,hbevd,hpevd,colmat::c32,f32,$i,[$($suffix)*],$f,[rwork]);
        evd_exports!(z,hbevd,hpevd,colmat::c64,f64,$i,[$($suffix)*],$f,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
