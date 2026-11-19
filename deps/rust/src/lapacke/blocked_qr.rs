//! LAPACKE GEQR/GELQ and GEMQR/GEMLQ high/work adapters for the blocked
//! reflector interfaces.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{geqr,gelq,gemqr,gemlq}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;
use crate::lapack::LapackInt;

macro_rules! factor {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident,$module:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,t:*mut $t,tsize:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -4;
            }
            if layout==ROW_MAJOR && lda<n {return -5;}
            // The reference high-level entry only allocates WORK; T is the
            // caller's responsibility and must not be a query size here.
            // The native query writes a small descriptor into T, so the
            // query buffer must hold more than one element.
            let mut tq=vec![<$t>::z();16];
            let mut wq=<$t>::z();
            let mut info=<$i>::from_i64(0);
            let neg1: $i = -1;
            crate::$module::[<$p $fname $fortran>](&m,&n,a,&lda,tq.as_mut_ptr(),&neg1,&mut wq,&neg1,&mut info);
            if info!=0 {return info;}
            let lw=wq.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            // Pass the caller's T buffer through so the packed reflectors are
            // visible for the matching GEMQR/GEMLQ entry, as in the reference.
            [<LAPACKE_ $p $fname _work $($suffix)*>](layout,m,n,a,lda,t,tsize,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname _work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*mut $t,lda:$i,t:*mut $t,tsize:$i,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || (tsize==-1||tsize==-2) || (lwork==-1||lwork==-2) {
                let mut info:$i=0;
                crate::$module::[<$p $fname $fortran>](&m,&n,a,&lda,t,&tsize,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=mm.max(1) as $i;
            let mut info:$i=0;
            crate::$module::[<$p $fname $fortran>](&m,&n,at.as_mut_ptr(),&ld,t,&tsize,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! apply {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident,$module:ident,$aorient:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,side:u8,trans:u8,m:$i,n:$i,k:$i,
            a:*const $t,lda:$i,t:*const $t,tsize:$i,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -10;}
                let r=if side.to_ascii_uppercase()==b'L' {m} else {n};
                // GEMQR stores an R-by-K reflector block, GEMLQ a K-by-R block.
                let (ar,ac)=if $aorient {(r as i64,k as i64)} else {(k as i64,r as i64)};
                if ge_nancheck(layout,ar,ac,a,lda as i64).unwrap_or(false) {return -7;}
                // T is a compact array of TSIZE entries, not a matrix.
                if (0..tsize.max(0) as usize).any(|i| {let v=*t.add(i); v!=v}) {return -9;}
            }
            let mut wq=<$t>::z();
            [<LAPACKE_ $p $fname _work $($suffix)*>](layout,side,trans,m,n,k,a,lda,t,tsize,c,ldc,&mut wq,-1)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname _work $($suffix)*>](
            layout:i32,side:u8,trans:u8,m:$i,n:$i,k:$i,
            a:*const $t,lda:$i,t:*const $t,tsize:$i,c:*mut $t,ldc:$i,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let left=side.to_ascii_uppercase()==b'L';
            let r=if left {m} else {n};
            if layout==ROW_MAJOR && lda<if $aorient {k} else {r} {return -8;}
            if layout==ROW_MAJOR && ldc<n {return -11;}
            if layout==COL_MAJOR || lwork==-1 || m==0 || n==0 || k==0 {
                let mut info:$i=0;
                crate::$module::[<$p $fname $fortran>](&side,&trans,&m,&n,&k,a,&lda,t,&tsize,c,&ldc,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,kk,rr)=(m as usize,n as usize,k as usize,r as usize);
            // GEMQR stores the reflectors in an R-by-K block; GEMLQ in a
            // K-by-R block, matching the reference row-major conventions.
            let (ar,ac)=if $aorient {(rr,kk)} else {(kk,rr)};
            let mut at=match row_to_column(ar,ac,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (lda_t,ldc_t)=(ar.max(1) as $i,mm.max(1) as $i);
            let mut info:$i=0;
            crate::$module::[<$p $fname $fortran>](&side,&trans,&m,&n,&k,at.as_ptr(),&lda_t,t,&tsize,ct.as_mut_ptr(),&ldc_t,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
factor!(s, f32, i32, [], _, geqr, geqr);
factor!(d, f64, i32, [], _, geqr, geqr);
factor!(c, colmat::c32, i32, [], _, geqr, geqr);
factor!(z, colmat::c64, i32, [], _, geqr, geqr);
factor!(s, f32, i64, [64_], _64_, geqr, geqr);
factor!(d, f64, i64, [64_], _64_, geqr, geqr);
factor!(c, colmat::c32, i64, [64_], _64_, geqr, geqr);
factor!(z, colmat::c64, i64, [64_], _64_, geqr, geqr);
factor!(s, f32, i32, [], _, gelq, lq);
factor!(d, f64, i32, [], _, gelq, lq);
factor!(c, colmat::c32, i32, [], _, gelq, lq);
factor!(z, colmat::c64, i32, [], _, gelq, lq);
factor!(s, f32, i64, [64_], _64_, gelq, lq);
factor!(d, f64, i64, [64_], _64_, gelq, lq);
factor!(c, colmat::c32, i64, [64_], _64_, gelq, lq);
factor!(z, colmat::c64, i64, [64_], _64_, gelq, lq);
apply!(s, f32, i32, [], _, gemqr, geqr, true);
apply!(d, f64, i32, [], _, gemqr, geqr, true);
apply!(c, colmat::c32, i32, [], _, gemqr, geqr, true);
apply!(z, colmat::c64, i32, [], _, gemqr, geqr, true);
apply!(s, f32, i64, [64_], _64_, gemqr, geqr, true);
apply!(d, f64, i64, [64_], _64_, gemqr, geqr, true);
apply!(c, colmat::c32, i64, [64_], _64_, gemqr, geqr, true);
apply!(z, colmat::c64, i64, [64_], _64_, gemqr, geqr, true);
apply!(s, f32, i32, [], _, gemlq, lq, false);
apply!(d, f64, i32, [], _, gemlq, lq, false);
apply!(c, colmat::c32, i32, [], _, gemlq, lq, false);
apply!(z, colmat::c64, i32, [], _, gemlq, lq, false);
apply!(s, f32, i64, [64_], _64_, gemlq, lq, false);
apply!(d, f64, i64, [64_], _64_, gemlq, lq, false);
apply!(c, colmat::c32, i64, [64_], _64_, gemlq, lq, false);
apply!(z, colmat::c64, i64, [64_], _64_, gemlq, lq, false);
