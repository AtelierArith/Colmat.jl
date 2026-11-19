//! LAPACKE SYTRD/HETRD and SYEQUB/HEEQUB high/work adapters for dense
//! symmetric/Hermitian tridiagonal reduction and scaling.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sytrd,hetrd}{,_work}.c and
//! lapacke_{s,d,c,z}{syequb,heequb}{,_work}.c.
//! Only the UPLO triangle is scanned for NaN and written back in row-major.
use super::cholesky::po_nancheck;
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn in_triangle(uplo: u8, i: usize, j: usize) -> bool {
    if uplo.to_ascii_uppercase() == b'L' {
        i >= j
    } else {
        i <= j
    }
}

macro_rules! trd_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,d:*mut $r,e:*mut $r,tau:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -4,Err(e)=>return e as $i,Ok(false)=>{}}
            }
            let mut qw=<$t>::z();
            let r=[<LAPACKE_ $p $f _work $($suffix)*>](layout,uplo,n,a,lda,d,e,tau,&mut qw,-1);
            if r!=0 {return r;}
            let lw=qw.re().max(1.0) as usize;
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,uplo,n,a,lda,d,e,tau,work.as_mut_ptr(),lw as $i)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,d:*mut $r,e:*mut $r,tau:*mut $t,work:*mut $t,lwork:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n<=0 || lwork== -1 {
                let mut info:$i=0;
                let ld=if layout==ROW_MAJOR {n.max(1)} else {lda};
                crate::selfadjoint_reduction::[<$p $f $fort>](&uplo,&n,a,&ld,d,e,tau,work,&lwork,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(err)=>return err as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::selfadjoint_reduction::[<$p $f $fort>](&uplo,&n,at.as_mut_ptr(),&ld,d,e,tau,work,&lwork,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    if in_triangle(uplo,i,j) {*a.add(i*lda as usize+j)=at[i+j*nn];}
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! equb_exports {
    ($p:ident,$f:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*const $t,lda:$i,s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -4,Err(e)=>return e as $i,Ok(false)=>{}}
            }
            let mut work=vec![<$t>::z();(3*n.max(0) as usize).max(1)];
            [<LAPACKE_ $p $f _work $($suffix)*>](layout,uplo,n,a,lda,s,scond,amax,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $f _work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*const $t,lda:$i,s:*mut $r,scond:*mut $r,amax:*mut $r,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n<=0 {
                let mut info:$i=0;
                crate::symmetric_equilibration::[<$p $f $fort>](&uplo,&n,a,&lda,s,scond,amax,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(err)=>return err as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::symmetric_equilibration::[<$p $f $fort>](&uplo,&n,at.as_ptr(),&ld,s,scond,amax,work,&mut info);
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        trd_exports!(s,sytrd,f32,f32,$i,[$($suffix)*],$fort);
        trd_exports!(d,sytrd,f64,f64,$i,[$($suffix)*],$fort);
        trd_exports!(c,hetrd,colmat::c32,f32,$i,[$($suffix)*],$fort);
        trd_exports!(z,hetrd,colmat::c64,f64,$i,[$($suffix)*],$fort);
        equb_exports!(s,syequb,f32,f32,$i,[$($suffix)*],$fort);
        equb_exports!(d,syequb,f64,f64,$i,[$($suffix)*],$fort);
        equb_exports!(c,syequb,colmat::c32,f32,$i,[$($suffix)*],$fort);
        equb_exports!(z,syequb,colmat::c64,f64,$i,[$($suffix)*],$fort);
        equb_exports!(c,heequb,colmat::c32,f32,$i,[$($suffix)*],$fort);
        equb_exports!(z,heequb,colmat::c64,f64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
