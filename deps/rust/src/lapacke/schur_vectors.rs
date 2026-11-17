//! LAPACKE TREVC high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}trevc{,_work}.c.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

unsafe fn back<T: Scalar, I: crate::lapack::LapackInt>(
    layout: i32,
    side: u8,
    howmny: u8,
    n: usize,
    mm: usize,
    t: *mut T,
    ldt: usize,
    vl: *mut T,
    ldvl: usize,
    vr: *mut T,
    ldvr: usize,
    mut call: impl FnMut(*mut T, I, *mut T, I, *mut T, I) -> I,
) -> I {
    let left = matches!(side, b'L' | b'B');
    let right = matches!(side, b'R' | b'B');
    let back = howmny == b'B';
    if layout == COL_MAJOR {
        return call(
            t,
            I::from_i64(ldt as i64),
            vl,
            I::from_i64(ldvl as i64),
            vr,
            I::from_i64(ldvr as i64),
        );
    }
    if ldt < n {
        return I::from_i64(-7);
    }
    if ldvl < mm {
        return I::from_i64(-9);
    }
    if ldvr < mm {
        return I::from_i64(-11);
    }
    if n == 0 || mm == 0 {
        return I::from_i64(0);
    }
    let mut tt = match row_to_column(n, n, t, ldt) {
        Ok(v) => v,
        Err(e) => return I::from_i64(e),
    };
    let mut vlt = if left {
        if back {
            match row_to_column(n, mm, vl, ldvl) {
                Ok(v) => v,
                Err(e) => return I::from_i64(e),
            }
        } else {
            zeros::<T>(n * mm).unwrap_or_default()
        }
    } else {
        Vec::new()
    };
    let mut vrt = if right {
        if back {
            match row_to_column(n, mm, vr, ldvr) {
                Ok(v) => v,
                Err(e) => return I::from_i64(e),
            }
        } else {
            zeros::<T>(n * mm).unwrap_or_default()
        }
    } else {
        Vec::new()
    };
    let ld = I::from_i64(n.max(1) as i64);
    let info = call(
        tt.as_mut_ptr(),
        ld,
        vlt.as_mut_ptr(),
        ld,
        vrt.as_mut_ptr(),
        ld,
    );
    if info.val() >= 0 {
        if left {
            for i in 0..n {
                for j in 0..mm {
                    *vl.add(i * ldvl + j) = vlt[i + j * n];
                }
            }
        }
        if right {
            for i in 0..n {
                for j in 0..mm {
                    *vr.add(i * ldvr + j) = vrt[i + j * n];
                }
            }
        }
    }
    info
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trevc $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*mut $i,n:$i,t:*const $t,ldt:$i,
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,t,ldt as i64).unwrap_or(false) {return -6;}
                let side_u=side.to_ascii_uppercase();
                if matches!(side_u,b'L'|b'B') && ge_nancheck(layout,n as i64,mm as i64,vl,ldvl as i64).unwrap_or(false) {return -8;}
                if matches!(side_u,b'R'|b'B') && ge_nancheck(layout,n as i64,mm as i64,vr,ldvr as i64).unwrap_or(false) {return -10;}
            }
            let mut work=vec![<$t>::z(); (3*n.max(0) as usize).max(1)];
            [<LAPACKE_ $p trevc_work $($suffix)*>](layout,side,howmny,select,n,t,ldt,vl,ldvl,vr,ldvr,mm,m,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trevc_work $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*mut $i,n:$i,t:*const $t,ldt:$i,
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,_work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (side,howmny)=(side.to_ascii_uppercase(),howmny.to_ascii_uppercase());
            let nn=n.max(0) as usize;
            let info=back::<$t,$i>(layout,side,howmny,nn,mm.max(0) as usize,t as *mut $t,ldt.max(0) as usize,vl,ldvl.max(0) as usize,vr,ldvr.max(0) as usize,|t,ldt,vl,ldvl,vr,ldvr|{
                let mut info:$i=0;
                crate::schur_vectors::[<$p trevc $fortran>](&side,&howmny,select,&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,m,_work,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            });
            info
        }
    }};
}

macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trevc $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*mut $i,n:$i,t:*const $t,ldt:$i,
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,t,ldt as i64).unwrap_or(false) {return -6;}
                let side_u=side.to_ascii_uppercase();
                if matches!(side_u,b'L'|b'B') && ge_nancheck(layout,n as i64,mm as i64,vl,ldvl as i64).unwrap_or(false) {return -8;}
                if matches!(side_u,b'R'|b'B') && ge_nancheck(layout,n as i64,mm as i64,vr,ldvr as i64).unwrap_or(false) {return -10;}
            }
            let mut work=vec![<$t>::z(); (2*n.max(0) as usize).max(1)];
            let mut rwork=vec![<$r>::z(); (n.max(0) as usize).max(1)];
            [<LAPACKE_ $p trevc_work $($suffix)*>](layout,side,howmny,select,n,t,ldt,vl,ldvl,vr,ldvr,mm,m,work.as_mut_ptr(),rwork.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trevc_work $($suffix)*>](
            layout:i32,side:u8,howmny:u8,select:*mut $i,n:$i,t:*const $t,ldt:$i,
            vl:*mut $t,ldvl:$i,vr:*mut $t,ldvr:$i,mm:$i,m:*mut $i,_work:*mut $t,_rwork:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (side,howmny)=(side.to_ascii_uppercase(),howmny.to_ascii_uppercase());
            let nn=n.max(0) as usize;
            back::<$t,$i>(layout,side,howmny,nn,mm.max(0) as usize,t as *mut $t,ldt.max(0) as usize,vl,ldvl.max(0) as usize,vr,ldvr.max(0) as usize,|t,ldt,vl,ldvl,vr,ldvr|{
                let mut info:$i=0;
                crate::schur_vectors::[<$p trevc $fortran>](&side,&howmny,select,&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,m,_work,_rwork,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
    }};
}
real!(s, f32, i32, [], _);
real!(d, f64, i32, [], _);
complex!(c, colmat::c32, f32, i32, [], _);
complex!(z, colmat::c64, f64, i32, [], _);
real!(s, f32, i64, [64_], _64_);
real!(d, f64, i64, [64_], _64_);
complex!(c, colmat::c32, f32, i64, [64_], _64_);
complex!(z, colmat::c64, f64, i64, [64_], _64_);
