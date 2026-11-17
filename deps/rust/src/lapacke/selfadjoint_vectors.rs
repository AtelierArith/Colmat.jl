//! LAPACKE ORGTR/UNGTR and ORMTR/UNMTR high/work adapters.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{s,d,c,z}
//! {orgtr,ungtr,ormtr,unmtr}{,_work}.c. Row-major storage is converted over the
//! logical rectangle; only the selected triangle is inspected for NaN.
use super::{
    ge_nancheck, row_to_column, self_adjoint::zeros, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR,
};
use crate::kernels::Scalar;

fn triangle_selected(u: u8, i: usize, j: usize) -> bool {
    u.to_ascii_uppercase() == b'L' && i >= j || u.to_ascii_uppercase() == b'U' && i <= j
}
unsafe fn triangle_nan<T: Scalar>(layout: i32, u: u8, n: usize, a: *const T, ld: usize) -> bool {
    for j in 0..n {
        for i in 0..n {
            if !triangle_selected(u, i, j) {
                continue;
            }
            let value = if layout == ROW_MAJOR {
                *a.add(i * ld + j)
            } else {
                *a.add(i + j * ld)
            };
            if value != value {
                return true;
            }
        }
    }
    false
}
unsafe fn tau_nan<T: Scalar>(n: usize, tau: *const T) -> bool {
    (0..n.saturating_sub(1)).any(|i| {
        let v = *tau.add(i);
        v != v
    })
}

macro_rules! generator {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,tau:*const $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if triangle_nan(layout,uplo,n.max(0) as usize,a,lda as usize) {return -4;}
                if tau_nan(n.max(0) as usize,tau) {return -6;}
            }
            let mut query=<$t>::z();
            let info=[<LAPACKE_ $p $lname _work $($suffix)*>](layout,uplo,n,a,lda,tau,&mut query,-1);
            if info!=0 {return info;}
            let length=query.re();
            if !length.is_finite()||length<1.||length>=isize::MAX as f64||length><$i>::MAX as f64 {return -1010;}
            let mut work=match zeros::<$t>(length as usize) {Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p $lname _work $($suffix)*>](layout,uplo,n,a,lda,tau,work.as_mut_ptr(),length as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,tau:*const $t,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if !matches!(uplo.to_ascii_uppercase(),b'L'|b'U') {return -2;}
            if n<0 {return -3;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR && lda<n.max(1) {return -5;}
            if layout==COL_MAJOR || lwork==-1 || n==0 {
                let mut info:$i=0;
                crate::selfadjoint_vectors::[<$p $fname $fortran>](&uplo,&n,a,&lda,tau,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize) {Ok(v)=>v,Err(info)=>return info as $i};
            let mut info:$i=0;
            let ld=nn.max(1) as $i;
            crate::selfadjoint_vectors::[<$p $fname $fortran>](&uplo,&n,at.as_mut_ptr(),&ld,tau,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! applicator {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$lname:ident,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname $($suffix)*>](
            layout:i32,side:u8,uplo:u8,trans:u8,m:$i,n:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let r=if side.to_ascii_uppercase()==b'L' {m.max(0) as usize} else {n.max(0) as usize};
                if triangle_nan(layout,uplo,r,a,lda as usize) {return -7;}
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -10;}
                if tau_nan(r,tau) {return -9;}
            }
            let mut query=<$t>::z();
            let info=[<LAPACKE_ $p $lname _work $($suffix)*>](layout,side,uplo,trans,m,n,a,lda,tau,c,ldc,&mut query,-1);
            if info!=0 {return info;}
            let length=query.re();
            if !length.is_finite()||length<1.||length>=isize::MAX as f64||length><$i>::MAX as f64 {return -1010;}
            let mut work=match zeros::<$t>(length as usize) {Ok(v)=>v,Err(_)=>return -1010};
            [<LAPACKE_ $p $lname _work $($suffix)*>](layout,side,uplo,trans,m,n,a,lda,tau,c,ldc,work.as_mut_ptr(),length as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $lname _work $($suffix)*>](
            layout:i32,side:u8,uplo:u8,trans:u8,m:$i,n:$i,a:*const $t,lda:$i,
            tau:*const $t,c:*mut $t,ldc:$i,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (side,trans)=(side.to_ascii_uppercase(),trans.to_ascii_uppercase());
            if !matches!(side,b'L'|b'R') {return -2;}
            if !matches!(uplo.to_ascii_uppercase(),b'L'|b'U') {return -3;}
            if trans!=b'N' && trans!=if $t::COMPLEX {b'C'} else {b'T'} {return -4;}
            if m<0 {return -5;}
            if n<0 {return -6;}
            let r=if side==b'L' {m.max(0)} else {n.max(0)};
            if layout==ROW_MAJOR && lda<r {return -8;}
            if layout==COL_MAJOR && lda<r.max(1) {return -8;}
            if layout==ROW_MAJOR && ldc<n {return -11;}
            if layout==COL_MAJOR && ldc<m.max(1) {return -11;}
            if layout==COL_MAJOR || lwork==-1 || m==0 || n==0 || r==0 {
                let mut info:$i=0;
                crate::selfadjoint_vectors::[<$p $fname $fortran>](&side,&uplo,&trans,&m,&n,a,&lda,tau,c,&ldc,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn,rr)=(m as usize,n as usize,r as usize);
            let mut at=match row_to_column(rr,rr,a,lda as usize) {Ok(v)=>v,Err(info)=>return info as $i};
            let mut ct=match row_to_column(mm,nn,c,ldc as usize) {Ok(v)=>v,Err(info)=>return info as $i};
            let (lda_t,ldc_t)=(rr.max(1) as $i,mm.max(1) as $i);
            let mut info:$i=0;
            crate::selfadjoint_vectors::[<$p $fname $fortran>](&side,&uplo,&trans,&m,&n,at.as_mut_ptr(),&lda_t,tau,ct.as_mut_ptr(),&ldc_t,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}
            }
            if info<0 {info-1} else {info}
        }
    }};
}

generator!(s, f32, i32, [], _, orgtr, orgtr);
generator!(d, f64, i32, [], _, orgtr, orgtr);
generator!(c, colmat::c32, i32, [], _, ungtr, ungtr);
generator!(z, colmat::c64, i32, [], _, ungtr, ungtr);
generator!(s, f32, i64, [64_], _64_, orgtr, orgtr);
generator!(d, f64, i64, [64_], _64_, orgtr, orgtr);
generator!(c, colmat::c32, i64, [64_], _64_, ungtr, ungtr);
generator!(z, colmat::c64, i64, [64_], _64_, ungtr, ungtr);
applicator!(s, f32, i32, [], _, ormtr, ormtr);
applicator!(d, f64, i32, [], _, ormtr, ormtr);
applicator!(c, colmat::c32, i32, [], _, unmtr, unmtr);
applicator!(z, colmat::c64, i32, [], _, unmtr, unmtr);
applicator!(s, f32, i64, [64_], _64_, ormtr, ormtr);
applicator!(d, f64, i64, [64_], _64_, ormtr, ormtr);
applicator!(c, colmat::c32, i64, [64_], _64_, unmtr, unmtr);
applicator!(z, colmat::c64, i64, [64_], _64_, unmtr, unmtr);
