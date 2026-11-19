//! LAPACKE SBGST/HBGST, SPGST/HPGST, SBGV/HBGV, SPGV/HPGV, SBGVD/HBGVD and
//! SPGVD/HPGVD high/work adapters for banded and packed generalized
//! self-adjoint definite problems.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sbgst,hbgst,spgst,hpgst,sbgv,hbgv,spgv,hpgv,sbgvd,hbgvd,spgvd,hpgvd}{,_work}.c.
//! Only the stored band/packed triangles are scanned for NaN.
use super::band_tridiagonal::{band_from_column, band_nan, band_to_column};
use super::packed_lu::{packed_from_column, packed_nan, packed_to_column};
use super::{LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

unsafe fn z_out<T: Scalar>(n: usize, z: *mut T, ldz: usize, zt: &[T]) {
    for i in 0..n {
        for j in 0..n {
            *z.add(i * ldz + j) = zt[i + j * n];
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn band_gen_job<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i64,
    ka: i64,
    kb: i64,
    ab: *mut T,
    ldab: i64,
    bb: *mut T,
    ldbb: i64,
    z: *mut T,
    ldz: i64,
    check_z: bool,
    want_z: bool,
    write_bb: bool,
    query: bool,
    errs: (i64, i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && ldab < n {
        return errs.0;
    }
    if layout == ROW_MAJOR && ldbb < n {
        return errs.1;
    }
    if layout == ROW_MAJOR && check_z && ldz < n {
        return errs.2;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ab, ldab, bb, ldbb, z, ldz);
    }
    let nn = n as usize;
    let (kka, kkb) = (ka.max(0) as usize, kb.max(0) as usize);
    let (ldab_t, ldbb_t, ldz_t) = ((kka + 1) as i64, (kkb + 1) as i64, n.max(1));
    if query {
        return call(ab, ldab_t, bb, ldbb_t, z, ldz_t);
    }
    let mut at = band_to_column(layout, uplo, nn, kka, ab, ldab as usize);
    let mut bt = band_to_column(layout, uplo, nn, kkb, bb, ldbb as usize);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(
        at.as_mut_ptr(),
        ldab_t,
        bt.as_mut_ptr(),
        ldbb_t,
        zt.as_mut_ptr(),
        ldz_t,
    );
    if info >= 0 {
        band_from_column(layout, uplo, nn, kka, ab, ldab as usize, &at);
        if write_bb {
            band_from_column(layout, uplo, nn, kkb, bb, ldbb as usize, &bt);
        }
        if want_z {
            z_out(nn, z, ldz as usize, &zt);
        }
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn packed_gen_job<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i64,
    ap: *mut T,
    bp: *mut T,
    z: *mut T,
    ldz: i64,
    check_z: bool,
    want_z: bool,
    write_bp: bool,
    query: bool,
    err: i64,
    mut call: impl FnMut(*mut T, *mut T, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && check_z && ldz < n {
        return err;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ap, bp, z, ldz);
    }
    let nn = n as usize;
    let ldz_t = n.max(1);
    if query {
        return call(ap, bp, z, ldz_t);
    }
    let mut at = packed_to_column(layout, uplo, nn, ap);
    let mut bt = packed_to_column(layout, uplo, nn, bp);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(at.as_mut_ptr(), bt.as_mut_ptr(), zt.as_mut_ptr(), ldz_t);
    if info >= 0 {
        packed_from_column(layout, uplo, nn, ap, &at);
        if write_bp {
            packed_from_column(layout, uplo, nn, bp, &bt);
        }
        if want_z {
            z_out(nn, z, ldz as usize, &zt);
        }
    }
    info
}

unsafe fn nan_band<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i32,
    ka: i64,
    kb: i64,
    ab: *const T,
    ldab: i64,
    bb: *const T,
    ldbb: i64,
    codes: (i64, i64),
) -> i64 {
    if LAPACKE_get_nancheck() == 0 {
        return 0;
    }
    let nn = n.max(0) as usize;
    if band_nan(
        layout,
        uplo,
        nn,
        ka.max(0) as usize,
        ab,
        ldab.max(0) as usize,
    ) {
        return codes.0;
    }
    if band_nan(
        layout,
        uplo,
        nn,
        kb.max(0) as usize,
        bb,
        ldbb.max(0) as usize,
    ) {
        return codes.1;
    }
    0
}
unsafe fn nan_packed<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: i32,
    ap: *const T,
    bp: *const T,
    codes: (i64, i64),
) -> i64 {
    if LAPACKE_get_nancheck() == 0 {
        return 0;
    }
    let nn = n.max(0) as usize;
    if packed_nan(layout, uplo, nn, ap, T::COMPLEX) {
        return codes.0;
    }
    if packed_nan(layout, uplo, nn, bp, T::COMPLEX) {
        return codes.1;
    }
    0
}

macro_rules! gv_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        // ---- band GV ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_band(layout,uplo,n as i32,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,(-7,-9));
            if e!=0 {return e as $i;}
            let lw=(3*n.max(0) as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();lw];)*
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,ka,kb,ab,ldab,bb,ldbb,w,z,ldz,work.as_mut_ptr()$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*
        )->$i {
            let want=jobz.to_ascii_uppercase()==b'V';
            band_gen_job(layout,uplo,n as i64,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,z,ldz as i64,true,want,true,false,(-8,-10,-13),|a,la,b,lb,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized_band::[<$p $bf $fort>](&jobz,&uplo,&n,&ka,&kb,a,&(la as $i),b,&(lb as $i),w,zz,&(lz as $i),work,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- packed GV ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_packed(layout,uplo,n as i32,ap,bp,(-6,-7));
            if e!=0 {return e as $i;}
            let lw=(3*n.max(0) as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();lw];)*
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,itype,jobz,uplo,n,ap,bp,w,z,ldz,work.as_mut_ptr()$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*
        )->$i {
            let want=jobz.to_ascii_uppercase()==b'V';
            packed_gen_job(layout,uplo,n as i64,ap,bp,z,ldz as i64,true,want,true,false,-10,|a,b,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized::[<$p $pf $fort>](&itype,&jobz,&uplo,&n,a,b,w,zz,&(lz as $i),work,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! gvd_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        // ---- band GVD ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_band(layout,uplo,n as i32,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,(-7,-9));
            if e!=0 {return e as $i;}
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            $(let mut [<q $rw>]=<$r>::z();)*
            let r=[<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,ka,kb,ab,ldab,bb,ldbb,w,z,ldz,&mut qw,-1,$(&mut [<q $rw>],-1,)*&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let [<l $rw>]=[<q $rw>].re().max(1.0) as usize;let mut $rw=vec![<$r>::z();[<l $rw>]];)*
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,uplo,n,ka,kb,ab,ldab,bb,ldbb,w,z,ldz,work.as_mut_ptr(),lw as $i,$($rw.as_mut_ptr(),[<l $rw>] as $i,)*iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,$($rw:*mut $r,[<l $rw>]:$i,)*iwork:*mut $i,liwork:$i,
        )->$i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $rw>]== -1)*;
            let want=jobz.to_ascii_uppercase()==b'V';
            band_gen_job(layout,uplo,n as i64,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,z,ldz as i64,true,want,true,query,(-8,-10,-13),|a,la,b,lb,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized_band::[<$p $bf $fort>](&jobz,&uplo,&n,&ka,&kb,a,&(la as $i),b,&(lb as $i),w,zz,&(lz as $i),work,&lwork,$($rw,&[<l $rw>],)*iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- packed GVD ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_packed(layout,uplo,n as i32,ap,bp,(-6,-7));
            if e!=0 {return e as $i;}
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            $(let mut [<q $rw>]=<$r>::z();)*
            let r=[<LAPACKE_ $p $pf _work $($suffix)*>](layout,itype,jobz,uplo,n,ap,bp,w,z,ldz,&mut qw,-1,$(&mut [<q $rw>],-1,)*&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let [<l $rw>]=[<q $rw>].re().max(1.0) as usize;let mut $rw=vec![<$r>::z();[<l $rw>]];)*
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,itype,jobz,uplo,n,ap,bp,w,z,ldz,work.as_mut_ptr(),lw as $i,$($rw.as_mut_ptr(),[<l $rw>] as $i,)*iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,lwork:$i,$($rw:*mut $r,[<l $rw>]:$i,)*iwork:*mut $i,liwork:$i,
        )->$i {
            let query=lwork== -1 || liwork== -1 $(|| [<l $rw>]== -1)*;
            let want=jobz.to_ascii_uppercase()==b'V';
            packed_gen_job(layout,uplo,n as i64,ap,bp,z,ldz as i64,true,want,true,query,-10,|a,b,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized::[<$p $pf $fort>](&itype,&jobz,&uplo,&n,a,b,w,zz,&(lz as $i),work,&lwork,$($rw,&[<l $rw>],)*iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! gst_exports {
    ($p:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        // ---- band GST ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,vect:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*const $t,ldbb:$i,x:*mut $t,ldx:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_band(layout,uplo,n as i32,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,(-7,-9));
            if e!=0 {return e as $i;}
            let lw=(3*n.max(0) as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();lw];)*
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,vect,uplo,n,ka,kb,ab,ldab,bb,ldbb,x,ldx,work.as_mut_ptr()$(,$rw.as_mut_ptr())*)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,vect:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*const $t,ldbb:$i,x:*mut $t,ldx:$i,
            work:*mut $t,$($rw:*mut $r,)*
        )->$i {
            let want=vect.to_ascii_uppercase()==b'V';
            band_gen_job(layout,uplo,n as i64,ka as i64,kb as i64,ab,ldab as i64,bb as *mut $t,ldbb as i64,x,ldx as i64,want,want,false,false,(-8,-10,-12),|a,la,b,lb,xx,lx|{
                let mut info:$i=0;
                crate::structured_generalized_band::[<$p $bf $fort>](&vect,&uplo,&n,&ka,&kb,a,&(la as $i),b as *const $t,&(lb as $i),xx,&(lx as $i),work,$($rw,)*&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- packed GST ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,itype:$i,uplo:u8,n:$i,ap:*mut $t,bp:*const $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_packed(layout,uplo,n as i32,ap,bp,(-5,-6));
            if e!=0 {return e as $i;}
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,itype,uplo,n,ap,bp)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,itype:$i,uplo:u8,n:$i,ap:*mut $t,bp:*const $t,
        )->$i {
            packed_gen_job(layout,uplo,n as i64,ap,bp as *mut $t,std::ptr::null_mut(),1,false,false,false,false,0,|a,b,_zz,_lz|{
                let mut info:$i=0;
                crate::structured_generalized::[<$p $pf $fort>](&itype,&uplo,&n,a,b as *const $t,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        gv_exports!(s,sbgv,spgv,f32,f32,$i,[$($suffix)*],$fort,[]);
        gv_exports!(d,sbgv,spgv,f64,f64,$i,[$($suffix)*],$fort,[]);
        gv_exports!(c,hbgv,hpgv,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gv_exports!(z,hbgv,hpgv,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
        gvd_exports!(s,sbgvd,spgvd,f32,f32,$i,[$($suffix)*],$fort,[]);
        gvd_exports!(d,sbgvd,spgvd,f64,f64,$i,[$($suffix)*],$fort,[]);
        gvd_exports!(c,hbgvd,hpgvd,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gvd_exports!(z,hbgvd,hpgvd,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
        gst_exports!(s,sbgst,spgst,f32,f32,$i,[$($suffix)*],$fort,[]);
        gst_exports!(d,sbgst,spgst,f64,f64,$i,[$($suffix)*],$fort,[]);
        gst_exports!(c,hbgst,hpgst,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gst_exports!(z,hbgst,hpgst,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
