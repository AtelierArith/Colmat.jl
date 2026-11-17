//! LAPACKE ORGBR/ORMBR and ORGHR/ORMHR high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}
//! {orgbr,ormbr,orghr,ormhr}{,_work}.c.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

unsafe fn nan_vec<T: Scalar>(n: i64, a: *const T) -> bool {
    (0..n.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}
unsafe fn high_alloc<T: Scalar, I: crate::lapack::LapackInt + IntMax>(
    mut query: T,
    mut call: impl FnMut(*mut T, I) -> I,
) -> I {
    let info = call(&mut query, I::from_i64(-1));
    if info.val() != 0 {
        return info;
    }
    let length = query.re();
    if !length.is_finite() || length < 1.0 || length >= isize::MAX as f64 || length > I::MAX_F64 {
        return I::from_i64(-1010);
    }
    let mut work = match zeros::<T>(length as usize) {
        Ok(v) => v,
        Err(_) => return I::from_i64(-1010),
    };
    call(work.as_mut_ptr(), I::from_i64(length as i64))
}

trait IntMax {
    const MAX_F64: f64;
}
impl IntMax for i32 {
    const MAX_F64: f64 = i32::MAX as f64;
}
impl IntMax for i64 {
    const MAX_F64: f64 = i64::MAX as f64;
}

macro_rules! orgbr {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,vect:u8,m:$i,n:$i,k:$i,a:*mut $t,lda:$i,tau:*const $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -6;}
                if nan_vec((m.min(k)).max(0) as i64,tau) {return -8;}
            }
            high_alloc::<$t,$i>((<$t>::z()),|work,lwork|[
                <LAPACKE_ $p $lname _work $($suffix)*>](layout,vect,m,n,k,a,lda,tau,work,lwork))
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,vect:u8,m:$i,n:$i,k:$i,a:*mut $t,lda:$i,tau:*const $t,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            if layout==COL_MAJOR && lda<m.max(1) {return -7;}
            if !matches!(vect.to_ascii_uppercase(),b'Q'|b'P') {return -2;}
            if m<0 {return -3;}
            if n<0 {return -4;}
            if k<0 || k> if vect.to_ascii_uppercase()==b'Q' {n} else {m} {return -5;}
            if layout==COL_MAJOR || lwork==-1 || m==0 || n==0 {
                let mut info:$i=0;
                crate::bidiagonal_vectors::[<$p $fname $fortran>](&vect,&m,&n,&k,a,&lda,tau,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let mut at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut info:$i=0;let ld=mm.max(1) as $i;
            crate::bidiagonal_vectors::[<$p $fname $fortran>](&vect,&m,&n,&k,at.as_mut_ptr(),&ld,tau,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! orghr {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,tau:*const $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false) {return -5;}
                if nan_vec(n.saturating_sub(1) as i64,tau) {return -7;}
            }
            high_alloc::<$t,$i>((<$t>::z()),|work,lwork|[
                <LAPACKE_ $p $lname _work $($suffix)*>](layout,n,ilo,ihi,a,lda,tau,work,lwork))
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,n:$i,ilo:$i,ihi:$i,a:*mut $t,lda:$i,tau:*const $t,
            work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if n<0 {return -2;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==COL_MAJOR && lda<n.max(1) {return -6;}
            if layout==COL_MAJOR || lwork==-1 || n==0 {
                let mut info:$i=0;
                crate::hessenberg::[<$p $fname $fortran>](&n,&ilo,&ihi,a,&lda,tau,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut info:$i=0;let ld=nn.max(1) as $i;
            crate::hessenberg::[<$p $fname $fortran>](&n,&ilo,&ihi,at.as_mut_ptr(),&ld,tau,work,&lwork,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! ormbr {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,vect:u8,side:u8,trans:u8,m:$i,n:$i,k:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let nq=if side.to_ascii_uppercase()==b'L' {m} else {n};
                let (ar,ac)=if vect.to_ascii_uppercase()==b'Q' {(nq,k.min(nq))} else {(nq.min(k),nq)};
                if ge_nancheck(layout,ar as i64,ac as i64,a,lda as i64).unwrap_or(false) {return -8;}
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -11;}
                if nan_vec(nq.min(k).max(0) as i64,tau) {return -10;}
            }
            high_alloc::<$t,$i>((<$t>::z()),|work,lwork|[
                <LAPACKE_ $p $lname _work $($suffix)*>](layout,vect,side,trans,m,n,k,a,lda,tau,c,ldc,work,lwork))
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,vect:u8,side:u8,trans:u8,m:$i,n:$i,k:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (vect,side,trans)=(vect.to_ascii_uppercase(),side.to_ascii_uppercase(),trans.to_ascii_uppercase());
            let nq=if side==b'L' {m} else {n};
            let (ar,ac)=if vect==b'Q' {(nq,k.min(nq))} else {(nq.min(k),nq)};
            if layout==ROW_MAJOR && lda<ac {return -9;}
            if layout==COL_MAJOR && lda<ar.max(1) {return -9;}
            if layout==ROW_MAJOR && ldc<n {return -12;}
            if layout==COL_MAJOR && ldc<m.max(1) {return -12;}
            if !matches!(vect,b'Q'|b'P') {return -2;}
            if !matches!(side,b'L'|b'R') {return -3;}
            if trans!=b'N' && trans!=if $t::COMPLEX {b'C'} else {b'T'} {return -4;}
            if m<0 {return -5;}
            if n<0 {return -6;}
            if k<0 || k>nq {return -7;}
            if layout==COL_MAJOR || lwork==-1 || m==0 || n==0 || nq==0 {
                let mut info:$i=0;
                crate::bidiagonal_vectors::[<$p $fname $fortran>](&vect,&side,&trans,&m,&n,&k,a,&lda,tau,c,&ldc,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,rr)=(m as usize,n as usize,ar as usize);
            let cc=ac as usize;
            let mut at=match row_to_column(rr,cc,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (lda_t,ldc_t)=(rr.max(1) as $i,mm.max(1) as $i);
            let mut info:$i=0;
            crate::bidiagonal_vectors::[<$p $fname $fortran>](&vect,&side,&trans,&m,&n,&k,at.as_mut_ptr(),&lda_t,tau,ct.as_mut_ptr(),&ldc_t,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! ormhr {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,side:u8,trans:u8,m:$i,n:$i,ilo:$i,ihi:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let r=if side.to_ascii_uppercase()==b'L' {m.max(0)} else {n.max(0)};
                if ge_nancheck(layout,r as i64,r as i64,a,lda as i64).unwrap_or(false) {return -8;}
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -11;}
                if nan_vec(r.saturating_sub(1) as i64,tau) {return -10;}
            }
            high_alloc::<$t,$i>((<$t>::z()),|work,lwork|[
                <LAPACKE_ $p $lname _work $($suffix)*>](layout,side,trans,m,n,ilo,ihi,a,lda,tau,c,ldc,work,lwork))
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,side:u8,trans:u8,m:$i,n:$i,ilo:$i,ihi:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (side,trans)=(side.to_ascii_uppercase(),trans.to_ascii_uppercase());
            let r=if side==b'L' {m.max(0)} else {n.max(0)};
            if layout==ROW_MAJOR && lda<r {return -9;}
            if layout==COL_MAJOR && lda<r.max(1) {return -9;}
            if layout==ROW_MAJOR && ldc<n {return -12;}
            if layout==COL_MAJOR && ldc<m.max(1) {return -12;}
            if !matches!(side,b'L'|b'R') {return -2;}
            if trans!=b'N' && trans!=if $t::COMPLEX {b'C'} else {b'T'} {return -3;}
            if m<0 {return -4;}
            if n<0 {return -5;}
            if ilo<1 || ilo>r.max(1) {return -6;}
            if ihi<ilo.min(r) || ihi>r {return -7;}
            if layout==COL_MAJOR || lwork==-1 || m==0 || n==0 || r==0 {
                let mut info:$i=0;
                crate::hessenberg::[<$p $fname $fortran>](&side,&trans,&m,&n,&ilo,&ihi,a,&lda,tau,c,&ldc,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,rr)=(m as usize,n as usize,r as usize);
            let mut at=match row_to_column(rr,rr,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (lda_t,ldc_t)=(rr.max(1) as $i,mm.max(1) as $i);
            let mut info:$i=0;
            crate::hessenberg::[<$p $fname $fortran>](&side,&trans,&m,&n,&ilo,&ihi,at.as_mut_ptr(),&lda_t,tau,ct.as_mut_ptr(),&ldc_t,work,&lwork,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}

orgbr!(s, f32, i32, [], _, orgbr, orgbr);
orgbr!(d, f64, i32, [], _, orgbr, orgbr);
orgbr!(c, colmat::c32, i32, [], _, ungbr, ungbr);
orgbr!(z, colmat::c64, i32, [], _, ungbr, ungbr);
orgbr!(s, f32, i64, [64_], _64_, orgbr, orgbr);
orgbr!(d, f64, i64, [64_], _64_, orgbr, orgbr);
orgbr!(c, colmat::c32, i64, [64_], _64_, ungbr, ungbr);
orgbr!(z, colmat::c64, i64, [64_], _64_, ungbr, ungbr);
orghr!(s, f32, i32, [], _, orghr, orghr);
orghr!(d, f64, i32, [], _, orghr, orghr);
orghr!(c, colmat::c32, i32, [], _, unghr, unghr);
orghr!(z, colmat::c64, i32, [], _, unghr, unghr);
orghr!(s, f32, i64, [64_], _64_, orghr, orghr);
orghr!(d, f64, i64, [64_], _64_, orghr, orghr);
orghr!(c, colmat::c32, i64, [64_], _64_, unghr, unghr);
orghr!(z, colmat::c64, i64, [64_], _64_, unghr, unghr);
ormbr!(s, f32, i32, [], _, ormbr, ormbr);
ormbr!(d, f64, i32, [], _, ormbr, ormbr);
ormbr!(c, colmat::c32, i32, [], _, unmbr, unmbr);
ormbr!(z, colmat::c64, i32, [], _, unmbr, unmbr);
ormbr!(s, f32, i64, [64_], _64_, ormbr, ormbr);
ormbr!(d, f64, i64, [64_], _64_, ormbr, ormbr);
ormbr!(c, colmat::c32, i64, [64_], _64_, unmbr, unmbr);
ormbr!(z, colmat::c64, i64, [64_], _64_, unmbr, unmbr);
ormhr!(s, f32, i32, [], _, ormhr, ormhr);
ormhr!(d, f64, i32, [], _, ormhr, ormhr);
ormhr!(c, colmat::c32, i32, [], _, unmhr, unmhr);
ormhr!(z, colmat::c64, i32, [], _, unmhr, unmhr);
ormhr!(s, f32, i64, [64_], _64_, ormhr, ormhr);
ormhr!(d, f64, i64, [64_], _64_, ormhr, ormhr);
ormhr!(c, colmat::c32, i64, [64_], _64_, unmhr, unmhr);
ormhr!(z, colmat::c64, i64, [64_], _64_, unmhr, unmhr);
