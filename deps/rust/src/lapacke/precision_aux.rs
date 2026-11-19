//! LAPACKE LAG2C/LAG2D/LAG2S/LAG2Z, LACRM/LARCM and PSTRF high/work adapters
//! for mixed-precision conversion, complex-by-real products and pivoted
//! Cholesky.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{dlag2s,slag2d,zlag2c,clag2z,clacrm,zlacrm,clarcm,zlarcm}{,_work}.c
//! and lapacke_{s,d,c,z}pstrf{,_work}.c. LACRM/LARCM and LAG2* return no INFO
//! for invalid column-major arguments (the native checker reports them), and
//! row-major reports an undersized leading dimension.
use super::cholesky::po_nancheck;
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn in_triangle(uplo: u8, i: usize, j: usize) -> bool {
    if uplo.to_ascii_uppercase() == b'L' {
        i >= j
    } else {
        i <= j
    }
}

macro_rules! lag2_exports {
    ($lname:ident,$src:ty,$dst:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $lname $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $src,lda:$i,b:*mut $dst,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {
                return -4;
            }
            [<LAPACKE_ $lname _work $($suffix)*>](layout,m,n,a,lda,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $lname _work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $src,lda:$i,b:*mut $dst,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==ROW_MAJOR && ldb<n {return -7;}
            if layout==COL_MAJOR || m<=0 || n<=0 {
                let mut info:$i=0;
                crate::precision_conversion::[<$lname $fort>](&m,&n,a,&lda,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut bt=vec![<$dst>::z();mm*nn];
            let (la,lb)=(mm.max(1) as $i,mm.max(1) as $i);
            let mut info:$i=0;
            crate::precision_conversion::[<$lname $fort>](&m,&n,at.as_ptr(),&la,bt.as_mut_ptr(),&lb,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*b.add(i*ldb as usize+j)=bt[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! cr_exports {
    // lacrm: A complex m-by-n, B real n-by-n.  larcm: A real m-by-m, B complex m-by-n.
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacrm $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $t,lda:$i,b:*const $r,ldb:$i,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,n as i64,a,lda as i64).unwrap_or(false) {return -4;}
                if ge_nancheck(layout,n as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -6;}
            }
            let mut work=vec![<$r>::z();(2*(m.max(0) as usize)*(n.max(0) as usize)).max(1)];
            [<LAPACKE_ $p lacrm_work $($suffix)*>](layout,m,n,a,lda,b,ldb,c,ldc,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacrm_work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $t,lda:$i,b:*const $r,ldb:$i,c:*mut $t,ldc:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==ROW_MAJOR && ldb<n {return -7;}
            if layout==ROW_MAJOR && ldc<n {return -9;}
            if layout==COL_MAJOR || m<=0 || n<=0 {
                crate::complex_auxiliary::[<$p lacrm $fort>](&m,&n,a,&lda,b,&ldb,c,&ldc,work);
                return 0;
            }
            let (mm,nn)=(m as usize,n as usize);
            let at=match row_to_column(mm,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let bt=match row_to_column(nn,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=vec![<$t>::z();mm*nn];
            let (la,lb,lc)=(mm.max(1) as $i,nn.max(1) as $i,mm.max(1) as $i);
            crate::complex_auxiliary::[<$p lacrm $fort>](&m,&n,at.as_ptr(),&la,bt.as_ptr(),&lb,ct.as_mut_ptr(),&lc,work);
            for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larcm $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $r,lda:$i,b:*const $t,ldb:$i,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,m as i64,m as i64,a,lda as i64).unwrap_or(false) {return -4;}
                if ge_nancheck(layout,m as i64,n as i64,b,ldb as i64).unwrap_or(false) {return -6;}
            }
            let mut work=vec![<$r>::z();(2*(m.max(0) as usize)*(n.max(0) as usize)).max(1)];
            [<LAPACKE_ $p larcm_work $($suffix)*>](layout,m,n,a,lda,b,ldb,c,ldc,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p larcm_work $($suffix)*>](
            layout:i32,m:$i,n:$i,a:*const $r,lda:$i,b:*const $t,ldb:$i,c:*mut $t,ldc:$i,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<m {return -5;}
            if layout==ROW_MAJOR && ldb<n {return -7;}
            if layout==ROW_MAJOR && ldc<n {return -9;}
            if layout==COL_MAJOR || m<=0 || n<=0 {
                crate::complex_auxiliary::[<$p larcm $fort>](&m,&n,a,&lda,b,&ldb,c,&ldc,work);
                return 0;
            }
            let (mm,nn)=(m as usize,n as usize);
            let at=match row_to_column(mm,mm,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let bt=match row_to_column(mm,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=vec![<$t>::z();mm*nn];
            let (la,lb,lc)=(mm.max(1) as $i,mm.max(1) as $i,mm.max(1) as $i);
            crate::complex_auxiliary::[<$p larcm $fort>](&m,&n,at.as_ptr(),&la,bt.as_ptr(),&lb,ct.as_mut_ptr(),&lc,work);
            for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}
            0
        }
    }};
}
macro_rules! pstrf_exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fort:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pstrf $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,piv:*mut $i,rank:*mut $i,tol:$r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match po_nancheck(layout,uplo,n as i64,a,lda as i64) {Ok(true)=>return -4,Err(e)=>return e as $i,Ok(false)=>{}}
                if tol.is_nan() {return -8;}
            }
            let mut work=vec![<$r>::z();(2*n.max(0) as usize).max(1)];
            [<LAPACKE_ $p pstrf_work $($suffix)*>](layout,uplo,n,a,lda,piv,rank,tol,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pstrf_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*mut $t,lda:$i,piv:*mut $i,rank:*mut $i,tol:$r,work:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n<=0 {
                let mut info:$i=0;
                crate::pivoted_cholesky::[<$p pstrf $fort>](&uplo,&n,a,&lda,piv,rank,&tol,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::pivoted_cholesky::[<$p pstrf $fort>](&uplo,&n,at.as_mut_ptr(),&ld,piv,rank,&tol,work,&mut info);
            if info>=0 {
                for i in 0..nn {for j in 0..nn {
                    if in_triangle(uplo,i,j) {*a.add(i*lda as usize+j)=at[i+j*nn];}
                }}
            }
            if info<0 {info-1} else {info}
        }
    }};
}
macro_rules! all_abi {
    ($i:ty,[$($suffix:tt)*],$fort:tt)=>{
        lag2_exports!(dlag2s,f64,f32,$i,[$($suffix)*],$fort);
        lag2_exports!(slag2d,f32,f64,$i,[$($suffix)*],$fort);
        lag2_exports!(zlag2c,colmat::c64,colmat::c32,$i,[$($suffix)*],$fort);
        lag2_exports!(clag2z,colmat::c32,colmat::c64,$i,[$($suffix)*],$fort);
        cr_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort);
        cr_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort);
        pstrf_exports!(s,f32,f32,$i,[$($suffix)*],$fort);
        pstrf_exports!(d,f64,f64,$i,[$($suffix)*],$fort);
        pstrf_exports!(c,colmat::c32,f32,$i,[$($suffix)*],$fort);
        pstrf_exports!(z,colmat::c64,f64,$i,[$($suffix)*],$fort);
    };
}
all_abi!(i32, [], _);
all_abi!(i64, [64_], _64_);
