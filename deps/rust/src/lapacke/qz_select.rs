//! LAPACKE TGSEN high/work adapters for reordering a generalized Schur pair
//! so that a selected cluster of eigenvalues leads, with optional condition
//! estimates (IJOB).
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}tgsen{,_work}.c.
//! SELECT is a LOGICAL (integer) array; row-major A, B, Q and Z are transposed
//! and written back whole.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

#[allow(clippy::too_many_arguments)]
unsafe fn tgsen_job<T: Scalar, I: LapackInt>(
    layout: i32,
    wq: bool,
    wz: bool,
    n: i64,
    a: *mut T,
    lda: i64,
    b: *mut T,
    ldb: i64,
    q: *mut T,
    ldq: i64,
    z: *mut T,
    ldz: i64,
    query: bool,
    errs: (i64, i64, i64, i64),
    mut call: impl FnMut(*mut T, i64, *mut T, i64, *mut T, i64, *mut T, i64) -> i64,
) -> i64 {
    let _: Option<I> = None;
    if layout != ROW_MAJOR && layout != COL_MAJOR {
        return -1;
    }
    if layout == ROW_MAJOR {
        if lda < n {
            return errs.0;
        }
        if ldb < n {
            return errs.1;
        }
        if wq && ldq < n {
            return errs.2;
        }
        if wz && ldz < n {
            return errs.3;
        }
    }
    if layout == COL_MAJOR || n <= 0 || query {
        let ld = n.max(1);
        return if layout == ROW_MAJOR {
            call(a, ld, b, ld, q, ld, z, ld)
        } else {
            call(a, lda, b, ldb, q, ldq, z, ldz)
        };
    }
    let nn = n as usize;
    let load = |m: *mut T, ld: i64| row_to_column(nn, nn, m, ld as usize);
    let mut at = match load(a, lda) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut bt = match load(b, ldb) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut qt = if wq {
        match load(q, ldq) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let mut zt = if wz {
        match load(z, ldz) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        Vec::new()
    };
    let ld = n.max(1);
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
    if info >= 0 {
        let store = |m: *mut T, ldm: i64, t: &[T]| {
            for i in 0..nn {
                for j in 0..nn {
                    *m.add(i * ldm as usize + j) = t[i + j * nn];
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

macro_rules! tgsen_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,real)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgsen $($suffix)*>](
            layout:i32,ijob:$i,wantq:$i,wantz:$i,select:*const $i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
            m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nv=n as i64;
                if ge_nancheck(layout,nv,nv,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,nv,nv,b,ldb as i64).unwrap_or(false) {return -9;}
                if wantq!=0 && ge_nancheck(layout,nv,nv,q,ldq as i64).unwrap_or(false) {return -14;}
                if wantz!=0 && ge_nancheck(layout,nv,nv,z,ldz as i64).unwrap_or(false) {return -16;}
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            let r=[<LAPACKE_ $p tgsen_work $($suffix)*>](layout,ijob,wantq,wantz,select,n,a,lda,b,ldb,alphar,alphai,beta,q,ldq,z,ldz,m,pl,pr,dif,
                &mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            // LAPACK's TGSYL step needs LWORK-2*M*(N-M) >= 1, one more than the
            // reported minimum for the condition-estimate jobs, so pad the query.
            let lw=qw.re().max(1.0) as usize+2*(n.max(0) as usize)*(n.max(0) as usize)+1;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p tgsen_work $($suffix)*>](layout,ijob,wantq,wantz,select,n,a,lda,b,ldb,alphar,alphai,beta,q,ldq,z,ldz,m,pl,pr,dif,
                work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgsen_work $($suffix)*>](
            layout:i32,ijob:$i,wantq:$i,wantz:$i,select:*const $i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alphar:*mut $t,alphai:*mut $t,beta:*mut $t,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
            m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,work:*mut $t,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            tgsen_job::<$t,$i>(layout,wantq!=0,wantz!=0,n as i64,a,lda as i64,b,ldb as i64,q,ldq as i64,z,ldz as i64,
                lwork== -1||liwork== -1,(-8,-10,-15,-17),|aa,la,bb,lb,qq,lq,zz,lz|{
                let mut info:$i=0;
                crate::qz_reorder_driver::[<$p tgsen $fort>](&ijob,&wantq,&wantz,select,&n,aa,&(la as $i),bb,&(lb as $i),alphar,alphai,beta,
                    qq,&(lq as $i),zz,&(lz as $i),m,pl,pr,dif,work,&lwork,iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt,complex)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgsen $($suffix)*>](
            layout:i32,ijob:$i,wantq:$i,wantz:$i,select:*const $i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alpha:*mut $t,beta:*mut $t,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
            m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nv=n as i64;
                if ge_nancheck(layout,nv,nv,a,lda as i64).unwrap_or(false) {return -7;}
                if ge_nancheck(layout,nv,nv,b,ldb as i64).unwrap_or(false) {return -9;}
                if wantq!=0 && ge_nancheck(layout,nv,nv,q,ldq as i64).unwrap_or(false) {return -13;}
                if wantz!=0 && ge_nancheck(layout,nv,nv,z,ldz as i64).unwrap_or(false) {return -15;}
            }
            let (mut qw,mut qi)=(<$t>::z(),0 as $i);
            let r=[<LAPACKE_ $p tgsen_work $($suffix)*>](layout,ijob,wantq,wantz,select,n,a,lda,b,ldb,alpha,beta,q,ldq,z,ldz,m,pl,pr,dif,
                &mut qw,-1,&mut qi,-1);
            if r!=0 {return r;}
            // LAPACK's TGSYL step needs LWORK-2*M*(N-M) >= 1, one more than the
            // reported minimum for the condition-estimate jobs, so pad the query.
            let lw=qw.re().max(1.0) as usize+2*(n.max(0) as usize)*(n.max(0) as usize)+1;
            let li=(qi as usize).max(1);
            let mut work=vec![<$t>::z();lw];
            let mut iwork=vec![0 as $i;li];
            [<LAPACKE_ $p tgsen_work $($suffix)*>](layout,ijob,wantq,wantz,select,n,a,lda,b,ldb,alpha,beta,q,ldq,z,ldz,m,pl,pr,dif,
                work.as_mut_ptr(),lw as $i,iwork.as_mut_ptr(),li as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tgsen_work $($suffix)*>](
            layout:i32,ijob:$i,wantq:$i,wantz:$i,select:*const $i,n:$i,a:*mut $t,lda:$i,b:*mut $t,ldb:$i,
            alpha:*mut $t,beta:*mut $t,q:*mut $t,ldq:$i,z:*mut $t,ldz:$i,
            m:*mut $i,pl:*mut $r,pr:*mut $r,dif:*mut $r,work:*mut $t,lwork:$i,iwork:*mut $i,liwork:$i,
        )->$i {
            tgsen_job::<$t,$i>(layout,wantq!=0,wantz!=0,n as i64,a,lda as i64,b,ldb as i64,q,ldq as i64,z,ldz as i64,
                lwork== -1||liwork== -1,(-8,-10,-14,-16),|aa,la,bb,lb,qq,lq,zz,lz|{
                let mut info:$i=0;
                crate::qz_reorder_driver::[<$p tgsen $fort>](&ijob,&wantq,&wantz,select,&n,aa,&(la as $i),bb,&(lb as $i),alpha,beta,
                    qq,&(lq as $i),zz,&(lz as $i),m,pl,pr,dif,work,&lwork,iwork,&liwork,&mut info);
                if info<0 {(info-1) as i64} else {info as i64}
            }) as $i
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        tgsen_exports!(s,f32,f32,$i,[$($suffix)*],$fort,real);
        tgsen_exports!(d,f64,f64,$i,[$($suffix)*],$fort,real);
        tgsen_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort,complex);
        tgsen_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort,complex);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
