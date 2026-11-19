//! LAPACKE SYEVX/HEEVX, SYGVX/HEGVX, SBGVX/HBGVX and SPGVX/HPGVX high/work
//! adapters for selected eigenpairs of dense, banded and packed (generalized)
//! self-adjoint problems.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{syevx,heevx,sygvx,hegvx,sbgvx,hbgvx,spgvx,hpgvx}{,_work}.c.
//! Only the stored triangles are scanned for NaN; row-major Z is written back
//! only for the returned columns.
use super::band_tridiagonal::{band_from_column, band_nan, band_to_column};
use super::cholesky::po_nancheck;
use super::packed_lu::{packed_from_column, packed_nan, packed_to_column};
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
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
fn in_triangle(uplo: u8, i: usize, j: usize) -> bool {
    if uplo.to_ascii_uppercase() == b'L' {
        i >= j
    } else {
        i <= j
    }
}
unsafe fn write_z<T: Scalar, I: LapackInt>(n: usize, m: *const I, z: *mut T, ldz: usize, zt: &[T]) {
    let cols = if n == 1 {
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
unsafe fn dense_evx<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    query: bool,
    errs: (i64, i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    let has_b = !b.is_null();
    if layout == ROW_MAJOR {
        if lda < n {
            return errs.0;
        }
        if has_b && ldb < n {
            return errs.1;
        }
        if (ldz as i128) < z_columns(jobz, range, n, il, iu) {
            return errs.2;
        }
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(a, lda, b, ldb, z, ldz);
    }
    let ld = n.max(1);
    if query {
        return call(a, ld, b, ld, z, ld);
    }
    let nn = n as usize;
    let mut at = match row_to_column(nn, nn, a, lda as usize) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = if has_b {
        match row_to_column(nn, nn, b, ldb as usize) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let mut zt = vec![T::z(); nn * nn];
    let info = call(
        at.as_mut_ptr(),
        ld,
        if has_b { bt.as_mut_ptr() } else { b },
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info >= 0 {
        for i in 0..nn {
            for j in 0..nn {
                if in_triangle(uplo, i, j) {
                    *a.add(i * lda as usize + j) = at[i + j * nn];
                    if has_b {
                        *b.add(i * ldb as usize + j) = bt[i + j * nn];
                    }
                }
            }
        }
        if jobz.to_ascii_uppercase() == b'V' {
            write_z(nn, m, z, ldz as usize, &zt);
        }
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn band_gvx<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i64,
    ka: i64,
    kb: i64,
    ab: *mut T,
    ldab: i64,
    bb: *mut T,
    ldbb: i64,
    q: *mut T,
    ldq: i64,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    errs: (i64, i64, i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR {
        if ldab < n {
            return errs.0;
        }
        if ldbb < n {
            return errs.1;
        }
        if ldq < n {
            return errs.2;
        }
        if (ldz as i128) < z_columns(jobz, range, n, il, iu) {
            return errs.3;
        }
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ab, ldab, bb, ldbb, q, ldq, z, ldz);
    }
    let nn = n as usize;
    let (kka, kkb) = (ka.max(0) as usize, kb.max(0) as usize);
    let ld = n.max(1);
    let mut at = band_to_column(layout, uplo, nn, kka, ab, ldab as usize);
    let mut bt = band_to_column(layout, uplo, nn, kkb, bb, ldbb as usize);
    let mut qt = vec![T::z(); nn * nn];
    let mut zt = vec![T::z(); nn * nn];
    let info = call(
        at.as_mut_ptr(),
        (kka + 1) as i64,
        bt.as_mut_ptr(),
        (kkb + 1) as i64,
        qt.as_mut_ptr(),
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info >= 0 {
        band_from_column(layout, uplo, nn, kka, ab, ldab as usize, &at);
        band_from_column(layout, uplo, nn, kkb, bb, ldbb as usize, &bt);
        if jobz.to_ascii_uppercase() == b'V' {
            for i in 0..nn {
                for j in 0..nn {
                    *q.add(i * ldq as usize + j) = qt[i + j * nn];
                }
            }
            write_z(nn, m, z, ldz as usize, &zt);
        }
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn packed_gvx<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i64,
    ap: *mut T,
    bp: *mut T,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    err: i64,
    mut call: impl FnMut(*mut T, *mut T, *mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && (ldz as i128) < z_columns(jobz, range, n, il, iu) {
        return err;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(ap, bp, z, ldz);
    }
    let nn = n as usize;
    let mut at = packed_to_column(layout, uplo, nn, ap);
    let mut bt = packed_to_column(layout, uplo, nn, bp);
    let mut zt = vec![T::z(); nn * nn];
    let info = call(at.as_mut_ptr(), bt.as_mut_ptr(), zt.as_mut_ptr(), n.max(1));
    if info >= 0 {
        packed_from_column(layout, uplo, nn, ap, &at);
        packed_from_column(layout, uplo, nn, bp, &bt);
        if jobz.to_ascii_uppercase() == b'V' {
            write_z(nn, m, z, ldz as usize, &zt);
        }
    }
    info
}

fn range_nan<R: PartialOrd + Copy>(range: u8, vl: R, vu: R) -> (bool, bool) {
    if range.to_ascii_uppercase() == b'V' {
        (vl != vl, vu != vu)
    } else {
        (false, false)
    }
}

macro_rules! evx_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -6,Err(e)=>return e as $i,Ok(false)=>{}}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -8;}
                if bu {return -9;}
                if abstol.is_nan() {return -12;}
            }
            let nn=n.max(0) as usize;
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p $f _work $($suffix)*>](layout,jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z,ldz,&mut qw,-1,
                $({let _=stringify!($rw);std::ptr::null_mut::<$r>()},)*std::ptr::null_mut(),ifail);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,jobz,range,uplo,n,a,lda,vl,vu,il,iu,abstol,m,w,z,ldz,work.as_mut_ptr(),lw as $i,
                $($rw.as_mut_ptr(),)*iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,work:*mut $t,lwork:$i,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            dense_evx::<$t,$i>(layout,jobz,range,uplo,n as i64,a,lda as i64,std::ptr::null_mut(),1,il as i64,iu as i64,m,z,ldz as i64,
                lwork== -1,(-7,0,-16),|a,la,_b,_lb,zz,lz|{
                let mut info:$i=0;
                crate::selected_selfadjoint::[<$p $f $fort>](&jobz,&range,&uplo,&n,a,&(la as $i),&vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),
                    work,&lwork,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! gvx_exports {
    ($p:ident,$df:ident,$bf:ident,$pf:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,[$($rw:ident)*])=>{paste::paste!{
        // ---- dense GVX ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $df $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,range:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -7,Err(e)=>return e as $i,Ok(false)=>{}}
                match po_nancheck(layout,uplo,n as i64,b,ldb as i64) {Ok(true)=>return -9,Err(e)=>return e as $i,Ok(false)=>{}}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -11;}
                if bu {return -12;}
                if abstol.is_nan() {return -15;}
            }
            let nn=n.max(0) as usize;
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p $df _work $($suffix)*>](layout,itype,jobz,range,uplo,n,a,lda,b,ldb,vl,vu,il,iu,abstol,m,w,z,ldz,&mut qw,-1,
                $({let _=stringify!($rw);std::ptr::null_mut::<$r>()},)*std::ptr::null_mut(),ifail);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $df _work $($suffix)*>](layout,itype,jobz,range,uplo,n,a,lda,b,ldb,vl,vu,il,iu,abstol,m,w,z,ldz,work.as_mut_ptr(),lw as $i,
                $($rw.as_mut_ptr(),)*iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $df _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,range:u8,uplo:u8,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,vl:$r,vu:$r,il:$i,iu:$i,
            abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,work:*mut $t,lwork:$i,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            dense_evx::<$t,$i>(layout,jobz,range,uplo,n as i64,a,lda as i64,b,ldb as i64,il as i64,iu as i64,m,z,ldz as i64,
                lwork== -1,(-8,-10,-19),|a,la,bb,lb,zz,lz|{
                let mut info:$i=0;
                crate::generalized_selfadjoint_drivers::[<$p $df $fort>](&itype,&jobz,&range,&uplo,&n,a,&(la as $i),bb,&(lb as $i),&vl,&vu,&il,&iu,
                    &abstol,m,w,zz,&(lz as $i),work,&lwork,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- band GVX ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,q:*mut $t,ldq:$i,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nn=n.max(0) as usize;
                if band_nan(layout,uplo,nn,ka.max(0) as usize,ab,ldab.max(0) as usize) {return -8;}
                if band_nan(layout,uplo,nn,kb.max(0) as usize,bb,ldbb.max(0) as usize) {return -10;}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -14;}
                if bu {return -15;}
                if abstol.is_nan() {return -18;}
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$t>::z();(7*nn).max(1)];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $bf _work $($suffix)*>](layout,jobz,range,uplo,n,ka,kb,ab,ldab,bb,ldbb,q,ldq,vl,vu,il,iu,abstol,m,w,z,ldz,
                work.as_mut_ptr(),$($rw.as_mut_ptr(),)*iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $bf _work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,uplo:u8,n:$i,ka:$i,kb:$i,ab:*mut $t,ldab:$i,bb:*mut $t,ldbb:$i,q:*mut $t,ldq:$i,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            band_gvx::<$t,$i>(layout,jobz,range,uplo,n as i64,ka as i64,kb as i64,ab,ldab as i64,bb,ldbb as i64,q,ldq as i64,
                il as i64,iu as i64,m,z,ldz as i64,(-9,-11,-13,-22),|a,la,b,lb,qq,lq,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized_selected::[<$p $bf $fort>](&jobz,&range,&uplo,&n,&ka,&kb,a,&(la as $i),b,&(lb as $i),qq,&(lq as $i),
                    &vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),work,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
        // ---- packed GVX ----
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,range:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nn=n.max(0) as usize;
                if packed_nan(layout,uplo,nn,ap,<$t>::COMPLEX) {return -7;}
                if packed_nan(layout,uplo,nn,bp,<$t>::COMPLEX) {return -8;}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -9;}
                if bu {return -10;}
                if abstol.is_nan() {return -13;}
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$t>::z();(8*nn).max(1)];
            $(let mut $rw=vec![<$r>::z();(7*nn).max(1)];)*
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p $pf _work $($suffix)*>](layout,itype,jobz,range,uplo,n,ap,bp,vl,vu,il,iu,abstol,m,w,z,ldz,
                work.as_mut_ptr(),$($rw.as_mut_ptr(),)*iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $pf _work $($suffix)*>](
            layout:i32,itype:$i,jobz:u8,range:u8,uplo:u8,n:$i,ap:*mut $t,bp:*mut $t,
            vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,
            work:*mut $t,$($rw:*mut $r,)*iwork:*mut $i,ifail:*mut $i,
        )->$i {
            packed_gvx::<$t,$i>(layout,jobz,range,uplo,n as i64,ap,bp,il as i64,iu as i64,m,z,ldz as i64,-17,|a,b,zz,lz|{
                let mut info:$i=0;
                crate::structured_generalized_selected::[<$p $pf $fort>](&itype,&jobz,&range,&uplo,&n,a,b,&vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),
                    work,$($rw,)*iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        evx_exports!(s,syevx,f32,f32,$i,[$($suffix)*],$fort,[]);
        evx_exports!(d,syevx,f64,f64,$i,[$($suffix)*],$fort,[]);
        evx_exports!(c,heevx,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        evx_exports!(z,heevx,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
        gvx_exports!(s,sygvx,sbgvx,spgvx,f32,f32,$i,[$($suffix)*],$fort,[]);
        gvx_exports!(d,sygvx,sbgvx,spgvx,f64,f64,$i,[$($suffix)*],$fort,[]);
        gvx_exports!(c,hegvx,hbgvx,hpgvx,colmat::c32,f32,$i,[$($suffix)*],$fort,[rwork]);
        gvx_exports!(z,hegvx,hbgvx,hpgvx,colmat::c64,f64,$i,[$($suffix)*],$fort,[rwork]);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
