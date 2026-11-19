//! LAPACKE TGEXC high/work adapters for reordering a generalized Schur pair.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}tgexc{,_work}.c.
//! Real IFST/ILST are in/out pointers and the work entry takes WORK/LWORK;
//! complex IFST/ILST are by value and there is no workspace.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

/// Shared layout plumbing. `call(a, lda, b, ldb, q, ldq, z, ldz)` runs the
/// native solver and returns an already layout-shifted status.
#[allow(clippy::too_many_arguments)]
unsafe fn run<T: Scalar, I: LapackInt>(
    layout: i32,
    wq: bool,
    wz: bool,
    n: I,
    a: *mut T,
    lda: I,
    b: *mut T,
    ldb: I,
    q: *mut T,
    ldq: I,
    z: *mut T,
    ldz: I,
    query: bool,
    mut call: impl FnMut(*mut T, I, *mut T, I, *mut T, I, *mut T, I) -> I,
) -> I {
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return I::from_i64(-1);
    }
    let nv = n.val();
    if layout == ROW_MAJOR {
        if lda.val() < nv {
            return I::from_i64(-6);
        }
        if ldb.val() < nv {
            return I::from_i64(-8);
        }
        if wq && ldq.val() < nv {
            return I::from_i64(-10);
        }
        if wz && ldz.val() < nv {
            return I::from_i64(-12);
        }
    }
    if layout == COL_MAJOR || nv <= 0 || query {
        return call(a, lda, b, ldb, q, ldq, z, ldz);
    }
    let nn = nv as usize;
    let load = |m: *mut T, ld: I| row_to_column(nn, nn, m, ld.val() as usize);
    let mut at = match load(a, lda) {
        Ok(v) => v,
        Err(e) => return I::from_i64(e),
    };
    let mut bt = match load(b, ldb) {
        Ok(v) => v,
        Err(e) => return I::from_i64(e),
    };
    let mut qt = if wq {
        match load(q, ldq) {
            Ok(v) => v,
            Err(e) => return I::from_i64(e),
        }
    } else {
        Vec::new()
    };
    let mut zt = if wz {
        match load(z, ldz) {
            Ok(v) => v,
            Err(e) => return I::from_i64(e),
        }
    } else {
        Vec::new()
    };
    let ld = I::from_i64(nn.max(1) as i64);
    let info = call(
        at.as_mut_ptr(),
        ld,
        bt.as_mut_ptr(),
        ld,
        qt.as_mut_ptr(),
        ld,
        zt.as_mut_ptr(),
        ld,
    );
    if info.val() >= 0 {
        let store = |m: *mut T, ldm: I, t: &[T]| {
            for i in 0..nn {
                for j in 0..nn {
                    *m.add(i * ldm.val() as usize + j) = t[i + j * nn];
                }
            }
        };
        store(a, lda, &at);
        store(b, ldb, &bt);
        if wq {
            store(q, ldq, &qt);
        }
        if wz {
            store(z, ldz, &zt);
        }
    }
    info
}

#[allow(clippy::too_many_arguments)]
unsafe fn nan_inputs<T: Scalar, I: LapackInt>(
    layout: i32,
    wq: bool,
    wz: bool,
    n: I,
    a: *const T,
    lda: I,
    b: *const T,
    ldb: I,
    q: *const T,
    ldq: I,
    z: *const T,
    ldz: I,
) -> i64 {
    if LAPACKE_get_nancheck() == 0 {
        return 0;
    }
    let nv = n.val() as i64;
    let bad = |m: *const T, ld: I| ge_nancheck(layout, nv, nv, m, ld.val() as i64).unwrap_or(false);
    if bad(a, lda) {
        return -5;
    }
    if bad(b, ldb) {
        return -7;
    }
    if wq && bad(q, ldq) {
        return -9;
    }
    if wz && bad(z, ldz) {
        return -11;
    }
    0
}

macro_rules! real {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgexc $($suffix)*>](
            layout:i32,wantq:$i,wantz:$i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,ifst:*mut $i,ilst:*mut $i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_inputs(layout,wantq!=0,wantz!=0,n,a,lda,b,ldb,q,ldq,z,ldz);
            if e!=0 {return e as $i;}
            let mut wq=<$t>::z();
            let r=[<LAPACKE_ $p tgexc_work $($suffix)*>](layout,wantq,wantz,n,a,lda,b,ldb,q,ldq,z,ldz,ifst,ilst,&mut wq,-1);
            if r!=0 {return r;}
            let lw=wq.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p tgexc_work $($suffix)*>](layout,wantq,wantz,n,a,lda,b,ldb,q,ldq,z,ldz,ifst,ilst,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgexc_work $($suffix)*>](
            layout:i32,wantq:$i,wantz:$i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,ifst:*mut $i,ilst:*mut $i,work:*mut $t,lwork:$i,
        )->$i {
            run::<$t,$i>(layout,wantq!=0,wantz!=0,n,a,lda,b,ldb,q,ldq,z,ldz,lwork== -1,|a,la,b,lb,q,lq,z,lz|{
                let mut info:$i=0;
                crate::qz_reorder_driver::[<$p tgexc $fortran>](&wantq,&wantz,&n,a,&la,b,&lb,q,&lq,z,&lz,ifst,ilst,work,&lwork,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
    }};
}
macro_rules! complex {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgexc $($suffix)*>](
            layout:i32,wantq:$i,wantz:$i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,ifst:$i,ilst:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let e=nan_inputs(layout,wantq!=0,wantz!=0,n,a,lda,b,ldb,q,ldq,z,ldz);
            if e!=0 {return e as $i;}
            [<LAPACKE_ $p tgexc_work $($suffix)*>](layout,wantq,wantz,n,a,lda,b,ldb,q,ldq,z,ldz,ifst,ilst)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgexc_work $($suffix)*>](
            layout:i32,wantq:$i,wantz:$i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,ifst:$i,ilst:$i,
        )->$i {
            run::<$t,$i>(layout,wantq!=0,wantz!=0,n,a,lda,b,ldb,q,ldq,z,ldz,false,|a,la,b,lb,q,lq,z,lz|{
                let (mut fi,mut li)=(ifst,ilst);
                let mut info:$i=0;
                crate::qz_reorder_driver::[<$p tgexc $fortran>](&wantq,&wantz,&n,a,&la,b,&lb,q,&lq,z,&lz,&mut fi,&mut li,&mut info);
                if info<0 {<$i>::from_i64((info-1) as i64)} else {info}
            })
        }
    }};
}
real!(s, f32, i32, [], _);
real!(d, f64, i32, [], _);
complex!(c, colmat::c32, i32, [], _);
complex!(z, colmat::c64, i32, [], _);
real!(s, f32, i64, [64_], _64_);
real!(d, f64, i64, [64_], _64_);
complex!(c, colmat::c32, i64, [64_], _64_);
complex!(z, colmat::c64, i64, [64_], _64_);
