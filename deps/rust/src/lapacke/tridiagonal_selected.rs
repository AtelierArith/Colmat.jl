//! LAPACKE STEVX, STEVR, STEGR and STEMR high/work adapters for selected
//! eigenpairs of a symmetric tridiagonal matrix.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d}{stevx,stevr}{,_work}.c and
//! lapacke_{s,d,c,z}{stegr,stemr}{,_work}.c.
//! D/E are scanned for NaN (E has N-1 entries, N for STEMR) and the row-major
//! Z is written back only for the returned columns.
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
unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}
fn range_nan<R: PartialOrd + Copy>(range: u8, vl: R, vu: R) -> (bool, bool) {
    if range.to_ascii_uppercase() == b'V' {
        (vl != vl, vu != vu)
    } else {
        (false, false)
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn tri_job<T: Scalar, I: LapackInt>(
    layout: i32,
    jobz: u8,
    range: u8,
    n: i64,
    il: i64,
    iu: i64,
    m: *const I,
    z: *mut T,
    ldz: i64,
    query: bool,
    err: i64,
    mut call: impl FnMut(*mut T, i64) -> i64,
) -> i64 {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR && (ldz as i128) < z_columns(jobz, range, n, il, iu) {
        return err;
    }
    if layout == COL_MAJOR || n <= 0 {
        return call(z, ldz);
    }
    let ld = n.max(1);
    if query {
        return call(z, ld);
    }
    let nn = n as usize;
    let mut zt = vec![T::z(); nn * nn];
    let info = call(zt.as_mut_ptr(), ld);
    if info >= 0 && jobz.to_ascii_uppercase() == b'V' {
        let cols = if nn == 1 {
            1
        } else {
            (*m).val().clamp(0, nn as isize) as usize
        };
        for i in 0..nn {
            for j in 0..cols {
                *z.add(i * ldz as usize + j) = zt[i + j * nn];
            }
        }
    }
    info
}

macro_rules! stevx_exports {
    ($p:ident,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stevx $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,
            m:*mut $i,w:*mut $r,z:*mut $r,ldz:$i,ifail:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec((n as i64-1).max(0),e) {return -6;}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -7;}
                if bu {return -8;}
                if abstol.is_nan() {return -11;}
            }
            let nn=n.max(0) as usize;
            let mut work=vec![<$r>::z();(5*nn).max(1)];
            let mut iwork=vec![0 as $i;(5*nn).max(1)];
            [<LAPACKE_ $p stevx_work $($suffix)*>](layout,jobz,range,n,d,e,vl,vu,il,iu,abstol,m,w,z,ldz,work.as_mut_ptr(),iwork.as_mut_ptr(),ifail)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stevx_work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,
            m:*mut $i,w:*mut $r,z:*mut $r,ldz:$i,work:*mut $r,iwork:*mut $i,ifail:*mut $i,
        )->$i {
            tri_job::<$r,$i>(layout,jobz,range,n as i64,il as i64,iu as i64,m,z,ldz as i64,false,-15,|zz,lz|{
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p stevx $fort>](&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),work,iwork,ifail,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! stevr_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,
            m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,isuppz:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec((n as i64-1).max(0),e) {return -6;}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -7;}
                if bu {return -8;}
                if abstol.is_nan() {return -11;}
            }
            let (mut qw,mut qi)=(<$r>::z(),0 as $i);
            let r=[<LAPACKE_ $p $f _work $($suffix)*>](layout,jobz,range,n,d,e,vl,vu,il,iu,abstol,m,w,z,ldz,isuppz,&mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$r>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,jobz,range,n,d,e,vl,vu,il,iu,abstol,m,w,z,ldz,isuppz,work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,abstol:$r,
            m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,isuppz:*mut $i,work:*mut $r,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            tri_job::<$t,$i>(layout,jobz,range,n as i64,il as i64,iu as i64,m,z,ldz as i64,lwork== -1||liwork== -1,-15,|zz,lz|{
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p $f $fort>](&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,&abstol,m,w,zz,&(lz as $i),isuppz,work,&lwork,iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! stemr_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stemr $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,
            m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,nzc:$i,isuppz:*mut $i,tryrac:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if nan_vec(n as i64,d) {return -5;}
                if nan_vec(n as i64,e) {return -6;}
                let (bv,bu)=range_nan(range,vl,vu);
                if bv {return -7;}
                if bu {return -8;}
            }
            let (mut qw,mut qi)=(<$r>::z(),0 as $i);
            let r=[<LAPACKE_ $p stemr_work $($suffix)*>](layout,jobz,range,n,d,e,vl,vu,il,iu,m,w,z,ldz,nzc,isuppz,tryrac,&mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let li=(qi as usize).max(1);
            let mut work=vec![<$r>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p stemr_work $($suffix)*>](layout,jobz,range,n,d,e,vl,vu,il,iu,m,w,z,ldz,nzc,isuppz,tryrac,work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p stemr_work $($suffix)*>](
            layout:i32,jobz:u8,range:u8,n:$i,d:*mut $r,e:*mut $r,vl:$r,vu:$r,il:$i,iu:$i,
            m:*mut $i,w:*mut $r,z:*mut $t,ldz:$i,nzc:$i,isuppz:*mut $i,tryrac:*mut $i,
            work:*mut $r,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            tri_job::<$t,$i>(layout,jobz,range,n as i64,il as i64,iu as i64,m,z,ldz as i64,lwork== -1||liwork== -1,-14,|zz,lz|{
                let mut info:$i=0;
                crate::tridiagonal_evd::[<$p stemr $fort>](&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,m,w,zz,&(lz as $i),&nzc,isuppz,tryrac,work,&lwork,iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        stevx_exports!(s,f32,$i,[$($suffix)*],$fort);
        stevx_exports!(d,f64,$i,[$($suffix)*],$fort);
        stevr_exports!(s,stevr,f32,f32,$i,[$($suffix)*],$fort);
        stevr_exports!(d,stevr,f64,f64,$i,[$($suffix)*],$fort);
        stevr_exports!(s,stegr,f32,f32,$i,[$($suffix)*],$fort);
        stevr_exports!(d,stegr,f64,f64,$i,[$($suffix)*],$fort);
        stevr_exports!(c,stegr,colmat::c32,f32,$i,[$($suffix)*],$fort);
        stevr_exports!(z,stegr,colmat::c64,f64,$i,[$($suffix)*],$fort);
        stemr_exports!(s,f32,f32,$i,[$($suffix)*],$fort);
        stemr_exports!(d,f64,f64,$i,[$($suffix)*],$fort);
        stemr_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort);
        stemr_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
