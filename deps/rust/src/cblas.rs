//! CBLAS ABI adapters over the existing Fortran/native operations.
//! Signatures: OpenBLAS v0.3.30 cblas.h (CBLAS_INDEX is size_t for both ABIs).
//! https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/cblas.h
use crate::kernels::Scalar;
use colmat::{c32, c64};
use paste::paste;

// Validate the complete referenced byte span before the Fortran layer computes
// signed offsets. Zero increments retain the underlying Level-1 semantics.
fn vector_ok<T>(n: i64, inc: i64, argument: i32) -> bool {
    if n <= 0 {
        return true;
    }
    let last = (n as u128 - 1) * inc.unsigned_abs() as u128;
    if n > isize::MAX as i64
        || last >= isize::MAX as u128
        || (last + 1) * core::mem::size_of::<T>() as u128 > isize::MAX as u128
    {
        crate::invalid(argument);
        return false;
    }
    true
}
macro_rules! vectors {
    ($p:ident,$t:ty,$i:ty,$fs:tt,$cs:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p copy $cs>](n:$i,x:*const $t,ix:$i,y:*mut $t,iy:$i) {
            if !vector_ok::<$t>(n as i64,ix as i64,3)||!vector_ok::<$t>(n as i64,iy as i64,5){return;}
            crate::abi::[<$p copy $fs>](&n,x,&ix,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p swap $cs>](n:$i,x:*mut $t,ix:$i,y:*mut $t,iy:$i) {
            if !vector_ok::<$t>(n as i64,ix as i64,3)||!vector_ok::<$t>(n as i64,iy as i64,5){return;}
            crate::abi::[<$p swap $fs>](&n,x,&ix,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_i $p amax $cs>](n:$i,x:*const $t,inc:$i)->usize {
            if n<=0||inc<=0{return 0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0;}
            let index=crate::abi::[<i $p amax $fs>](&n,x,&inc);
            if index>0 {index as usize-1} else {0}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_i $p amin $cs>](n:$i,x:*const $t,inc:$i)->usize {
            if n<=0||inc<=0{return 0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0;}
            let index=crate::reduction_extensions::[<i $p amin $fs>](&n,x,&inc);
            if index>0 {index as usize-1} else {0}
        }
    }};
}
macro_rules! real {
    ($p:ident,$t:ty,$i:ty,$fs:tt,$cs:tt) => {paste! {
        vectors!($p,$t,$i,$fs,$cs);
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p scal $cs>](n:$i,alpha:$t,x:*mut $t,inc:$i) {
            if n<=0||inc<=0{return;}
            if !vector_ok::<$t>(n as i64,inc as i64,4){return;}
            crate::abi::[<$p scal $fs>](&n,&alpha,x,&inc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p axpy $cs>](n:$i,alpha:$t,x:*const $t,ix:$i,y:*mut $t,iy:$i) {
            if !vector_ok::<$t>(n as i64,ix as i64,4)||!vector_ok::<$t>(n as i64,iy as i64,6){return;}
            crate::abi::[<$p axpy $fs>](&n,&alpha,x,&ix,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p axpby $cs>](n:$i,alpha:$t,x:*const $t,ix:$i,beta:$t,y:*mut $t,iy:$i) {
            if !vector_ok::<$t>(n as i64,ix as i64,4)||!vector_ok::<$t>(n as i64,iy as i64,7){return;}
            crate::abi::[<$p axpby $fs>](&n,&alpha,x,&ix,&beta,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p nrm2 $cs>](n:$i,x:*const $t,inc:$i)->$t {
            if n<=0||inc<=0{return 0.0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0.0;}
            crate::abi::[<$p nrm2 $fs>](&n,x,&inc)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p asum $cs>](n:$i,x:*const $t,inc:$i)->$t {
            if n<=0||inc<=0{return 0.0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0.0;}
            crate::abi::[<$p asum $fs>](&n,x,&inc)
        }
        value_reduce!($p,$t,$i,$fs,$cs,amax);
        value_reduce!($p,$t,$i,$fs,$cs,amin);
        value_reduce!($p,$t,$i,$fs,$cs,sum);
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p rot $cs>](n:$i,x:*mut $t,ix:$i,y:*mut $t,iy:$i,c:$t,s:$t) {
            if !vector_ok::<$t>(n as i64,ix as i64,3)||!vector_ok::<$t>(n as i64,iy as i64,5){return;}
            crate::abi::[<$p rot $fs>](&n,x,&ix,y,&iy,&c,&s);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p rotg $cs>](a:*mut $t,b:*mut $t,c:*mut $t,s:*mut $t) {
            crate::givens::[<$p rotg $fs>](a,b,c,s);
        }
    }};
}
macro_rules! value_reduce {
    ($p:ident,$t:ty,$i:ty,$fs:tt,$cs:tt,$op:ident) => {
        paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<cblas_ $p $op $cs>](n:$i,x:*const $t,inc:$i)->$t {
                if !vector_ok::<$t>(n as i64,inc as i64,3){return 0.0;}
                crate::reduction_extensions::[<$p $op $fs>](&n,x,&inc)
            }
        }
    };
}
macro_rules! complex {
    ($p:ident,$rp:ident,$t:ty,$r:ty,$i:ty,$fs:tt,$cs:tt) => {paste! {
        vectors!($p,$t,$i,$fs,$cs);
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p scal $cs>](n:$i,alpha:*const $t,x:*mut $t,inc:$i) {
            if n<=0||inc<=0{return;}
            if !vector_ok::<$t>(n as i64,inc as i64,4){return;}
            crate::abi::[<$p scal $fs>](&n,alpha,x,&inc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p axpy $cs>](n:$i,alpha:*const $t,x:*const $t,ix:$i,y:*mut $t,iy:$i) {
            if n<=0{return;}
            if !vector_ok::<$t>(n as i64,ix as i64,4)||!vector_ok::<$t>(n as i64,iy as i64,6){return;}
            crate::abi::[<$p axpy $fs>](&n,alpha,x,&ix,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p axpby $cs>](n:$i,alpha:*const $t,x:*const $t,ix:$i,beta:*const $t,y:*mut $t,iy:$i) {
            if n<=0{return;}
            if !vector_ok::<$t>(n as i64,ix as i64,4)||!vector_ok::<$t>(n as i64,iy as i64,7){return;}
            crate::abi::[<$p axpby $fs>](&n,alpha,x,&ix,beta,y,&iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $rp $p nrm2 $cs>](n:$i,x:*const $t,inc:$i)->$r {
            if n<=0||inc<=0{return 0.0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0.0;}
            crate::abi::[<$rp $p nrm2 $fs>](&n,x,&inc)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $rp $p asum $cs>](n:$i,x:*const $t,inc:$i)->$r {
            if n<=0||inc<=0{return 0.0;}
            if !vector_ok::<$t>(n as i64,inc as i64,3){return 0.0;}
            crate::abi::[<$rp $p asum $fs>](&n,x,&inc)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $rp scal $cs>](n:$i,alpha:$r,x:*mut $t,inc:$i) {
            if n<=0||inc<=0{return;}
            if !vector_ok::<$t>(n as i64,inc as i64,4){return;}
            crate::abi::[<$p $rp scal $fs>](&n,&alpha,x,&inc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $rp rot $cs>](n:$i,x:*mut $t,ix:$i,y:*mut $t,iy:$i,c:$r,s:$r) {
            if !vector_ok::<$t>(n as i64,ix as i64,3)||!vector_ok::<$t>(n as i64,iy as i64,5){return;}
            crate::abi::[<$p $rp rot $fs>](&n,x,&ix,y,&iy,&c,&s);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p rotg $cs>](a:*mut $t,b:*mut $t,c:*mut $r,s:*mut $t) {
            crate::givens::[<$p rotg $fs>](a,b,c,s);
        }
        complex_dot!($p,$t,$i,$cs,dotu,false);
        complex_dot!($p,$t,$i,$cs,dotc,true);
    }};
}
macro_rules! complex_dot {
    ($p:ident,$t:ty,$i:ty,$cs:tt,$op:ident,$conj:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](n:$i,x:*const $t,ix:$i,y:*const $t,iy:$i)->$t {
            if !vector_ok::<$t>(n as i64,ix as i64,3)||!vector_ok::<$t>(n as i64,iy as i64,5){return <$t>::z();}
            crate::kernels::dot(n as isize,x,ix as isize,y,iy as isize,$conj)
        }
    }};
}

fn cblas_trans(value: i32) -> Option<u8> {
    match value {
        111 => Some(b'N'),
        112 => Some(b'T'),
        113 => Some(b'C'),
        114 => Some(b'R'),
        _ => None,
    }
}
fn matrix_ok<T>(rows: i64, cols: i64, ld: i64, row_major: bool, argument: i32) -> bool {
    let (rows, cols) = if row_major {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if ld < rows.max(1)
        || !crate::kernels::matrix_addressable::<T>(rows as isize, cols as isize, ld as isize)
    {
        crate::invalid(argument);
        return false;
    }
    true
}
macro_rules! gemm_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p gemm $cs>](order:i32,ta:i32,tb:i32,m:$i,n:$i,k:$i,alpha:$coefficient,a:*const $t,lda:$i,b:*const $t,ldb:$i,beta:$coefficient,c:*mut $t,ldc:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let Some(ta)=cblas_trans(ta) else {crate::invalid(2);return;};
            let Some(tb)=cblas_trans(tb) else {crate::invalid(3);return;};
            if m<0 {crate::invalid(4);return;}
            if n<0 {crate::invalid(5);return;}
            if k<0 {crate::invalid(6);return;}
            let row=order==101;
            let (ar,ac)=if matches!(ta,b'N'|b'R'){(m,k)}else{(k,m)};
            let (br,bc)=if matches!(tb,b'N'|b'R'){(k,n)}else{(n,k)};
            if !matrix_ok::<$t>(ar as i64,ac as i64,lda as i64,row,9)
                || !matrix_ok::<$t>(br as i64,bc as i64,ldb as i64,row,11)
                || !matrix_ok::<$t>(m as i64,n as i64,ldc as i64,row,14) {return;}
            if m==0||n==0 {return;}
            let alpha=($read)(alpha);
            let beta=($read)(beta);
            if row {
                crate::kernels::gemm(tb,ta,n as isize,m as isize,k as isize,alpha,b,ldb as isize,a,lda as isize,beta,c,ldc as isize);
            } else {
                crate::kernels::gemm(ta,tb,m as isize,n as isize,k as isize,alpha,a,lda as isize,b,ldb as isize,beta,c,ldc as isize);
            }
        }
    }};
}

macro_rules! gemv_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p gemv $cs>](order:i32,t:i32,m:$i,n:$i,alpha:$coefficient,a:*const $t,lda:$i,x:*const $t,ix:$i,beta:$coefficient,y:*mut $t,iy:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let Some(t)=cblas_trans(t) else {crate::invalid(2);return;};
            if m<0 {crate::invalid(3);return;}
            if n<0 {crate::invalid(4);return;}
            if !matrix_ok::<$t>(m as i64,n as i64,lda as i64,order==101,7) {return;}
            if ix==0 {crate::invalid(9);return;}
            if iy==0 {crate::invalid(12);return;}
            let (nx,ny)=if matches!(t,b'N'|b'R'){(n,m)}else{(m,n)};
            if !vector_ok::<$t>(nx as i64,ix as i64,9)||!vector_ok::<$t>(ny as i64,iy as i64,12){return;}
            if m==0||n==0 {return;}
            let alpha=($read)(alpha);
            let beta=($read)(beta);
            if order==101 {
                let t=match t {b'N'=>b'T',b'T'=>b'N',b'C'=>b'R',_=>b'C'};
                crate::kernels::gemv(t,n as isize,m as isize,alpha,a,lda as isize,x,ix as isize,beta,y,iy as isize);
            } else {
                crate::kernels::gemv(t,m as isize,n as isize,alpha,a,lda as isize,x,ix as isize,beta,y,iy as isize);
            }
        }
    }};
}

fn cblas_uplo(value: i32) -> Option<u8> {
    match value {
        121 => Some(b'U'),
        122 => Some(b'L'),
        _ => None,
    }
}
fn cblas_diag(value: i32) -> Option<u8> {
    match value {
        131 => Some(b'N'),
        132 => Some(b'U'),
        _ => None,
    }
}
fn flip_uplo(u: u8) -> u8 {
    if u == b'U' {
        b'L'
    } else {
        b'U'
    }
}
fn transpose_operation(t: u8) -> u8 {
    match t {
        b'N' => b'T',
        b'T' => b'N',
        b'C' => b'R',
        _ => b'C',
    }
}
macro_rules! triangular_vector_export {
    ($p:ident,$t:ty,$i:ty,$cs:tt,$op:ident,$solve:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,t:i32,d:i32,n:$i,a:*const $t,lda:$i,x:*mut $t,inc:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            let Some(mut t)=cblas_trans(t) else {crate::invalid(3);return;};
            let Some(d)=cblas_diag(d) else {crate::invalid(4);return;};
            if n<0 {crate::invalid(5);return;}
            if !matrix_ok::<$t>(n as i64,n as i64,lda as i64,false,7){return;}
            if inc==0 {crate::invalid(9);return;}
            if !vector_ok::<$t>(n as i64,inc as i64,9){return;}
            if order==101 {u=flip_uplo(u);t=transpose_operation(t);}
            crate::kernels::trmv(u,t,d,n as isize,a,lda as isize,x,inc as isize,$solve);
        }
    }};
}
macro_rules! triangular_matrix_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$solve:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,side:i32,u:i32,t:i32,d:i32,m:$i,n:$i,alpha:$coefficient,a:*const $t,lda:$i,b:*mut $t,ldb:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let mut side=match side {141=>b'L',142=>b'R',_=>{crate::invalid(2);return;}};
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(3);return;};
            let Some(t)=cblas_trans(t) else {crate::invalid(4);return;};
            let Some(d)=cblas_diag(d) else {crate::invalid(5);return;};
            if m<0 {crate::invalid(6);return;}
            if n<0 {crate::invalid(7);return;}
            let dim=if side==b'L'{m}else{n};
            if !matrix_ok::<$t>(dim as i64,dim as i64,lda as i64,false,10)
                || !matrix_ok::<$t>(m as i64,n as i64,ldb as i64,order==101,12){return;}
            if m==0||n==0{return;}
            let alpha=($read)(alpha);
            if order==101 {
                side=if side==b'L'{b'R'}else{b'L'};u=flip_uplo(u);
                crate::kernels::trmatrix(side,u,t,d,n as isize,m as isize,alpha,a,lda as isize,b,ldb as isize,$solve);
            }else{
                crate::kernels::trmatrix(side,u,t,d,m as isize,n as isize,alpha,a,lda as isize,b,ldb as isize,$solve);
            }
        }
    }};
}
macro_rules! triangular_exports {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt) => {
        triangular_vector_export!($p, $t, $i, $cs, trmv, false);
        triangular_vector_export!($p, $t, $i, $cs, trsv, true);
        triangular_matrix_export!($p, $t, $coefficient, $read, $i, $cs, trmm, false);
        triangular_matrix_export!($p, $t, $coefficient, $read, $i, $cs, trsm, true);
    };
}

macro_rules! symmetric_matrix_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$hermitian:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,side:i32,u:i32,m:$i,n:$i,alpha:$coefficient,a:*const $t,lda:$i,b:*const $t,ldb:$i,beta:$coefficient,c:*mut $t,ldc:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let mut side=match side {141=>b'L',142=>b'R',_=>{crate::invalid(2);return;}};
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(3);return;};
            if m<0 {crate::invalid(4);return;}
            if n<0 {crate::invalid(5);return;}
            let dim=if side==b'L'{m}else{n};
            if !matrix_ok::<$t>(dim as i64,dim as i64,lda as i64,false,8)
                || !matrix_ok::<$t>(m as i64,n as i64,ldb as i64,order==101,10)
                || !matrix_ok::<$t>(m as i64,n as i64,ldc as i64,order==101,13){return;}
            if m==0||n==0{return;}
            let alpha=($read)(alpha);
            let beta=($read)(beta);
            if order==101 {
                side=if side==b'L'{b'R'}else{b'L'};u=flip_uplo(u);
                crate::kernels::symm(side,u,n as isize,m as isize,alpha,a,lda as isize,b,ldb as isize,beta,c,ldc as isize,$hermitian);
            }else{
                crate::kernels::symm(side,u,m as isize,n as isize,alpha,a,lda as isize,b,ldb as isize,beta,c,ldc as isize,$hermitian);
            }
        }
    }};
}
macro_rules! ger_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$conjugate:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,m:$i,n:$i,alpha:$coefficient,x:*const $t,ix:$i,y:*const $t,iy:$i,a:*mut $t,lda:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            if m<0 {crate::invalid(2);return;}
            if n<0 {crate::invalid(3);return;}
            if ix==0 {crate::invalid(6);return;}
            if iy==0 {crate::invalid(8);return;}
            if !matrix_ok::<$t>(m as i64,n as i64,lda as i64,order==101,10)
                ||!vector_ok::<$t>(m as i64,ix as i64,6)||!vector_ok::<$t>(n as i64,iy as i64,8){return;}
            if m==0||n==0{return;}
            let alpha=($read)(alpha);
            if alpha==<$t>::z(){return;}
            if order==101 {
                if $conjugate {
                    let start=crate::kernels::start(n as isize,iy as isize);
                    let conjugated:Vec<$t>=(0..n as isize).map(|j|(*y.offset(start+j*iy as isize)).cj()).collect();
                    crate::kernels::ger(n as isize,m as isize,alpha,conjugated.as_ptr(),1,x,ix as isize,a,lda as isize,false);
                } else {
                    crate::kernels::ger(n as isize,m as isize,alpha,y,iy as isize,x,ix as isize,a,lda as isize,false);
                }
            }else{
                crate::kernels::ger(m as isize,n as isize,alpha,x,ix as isize,y,iy as isize,a,lda as isize,$conjugate);
            }
        }
    }};
}

macro_rules! symmetric_vector_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$hermitian:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,n:$i,alpha:$coefficient,a:*const $t,lda:$i,x:*const $t,ix:$i,beta:$coefficient,y:*mut $t,iy:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            if n<0 {crate::invalid(3);return;}
            if !matrix_ok::<$t>(n as i64,n as i64,lda as i64,false,6){return;}
            if ix==0 {crate::invalid(8);return;}
            if iy==0 {crate::invalid(11);return;}
            if !vector_ok::<$t>(n as i64,ix as i64,8)||!vector_ok::<$t>(n as i64,iy as i64,11){return;}
            if n==0{return;}
            let alpha=($read)(alpha);
            let beta=($read)(beta);
            if order==101 {u=flip_uplo(u);}
            crate::kernels::symv_with_conjugation(u,n as isize,alpha,a,lda as isize,x,ix as isize,beta,y,iy as isize,$hermitian,$hermitian&&order==101);
        }
    }};
}

macro_rules! rank_one_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$hermitian:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,n:$i,alpha:$coefficient,x:*const $t,inc:$i,a:*mut $t,lda:$i) {
            if !matches!(order,101|102) {crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            if n<0 {crate::invalid(3);return;}
            if inc==0 {crate::invalid(6);return;}
            if !matrix_ok::<$t>(n as i64,n as i64,lda as i64,false,8)||!vector_ok::<$t>(n as i64,inc as i64,6){return;}
            if n==0{return;}
            let alpha=($read)(alpha);
            if alpha==<$t>::z(){return;}
            if order==101 {
                u=flip_uplo(u);
                if $hermitian {
                    let start=crate::kernels::start(n as isize,inc as isize);
                    let conjugated:Vec<$t>=(0..n as isize).map(|j|(*x.offset(start+j*inc as isize)).cj()).collect();
                    crate::kernels::rank1(u,n as isize,alpha,conjugated.as_ptr(),1,a,lda as isize,true);
                    return;
                }
            }
            crate::kernels::rank1(u,n as isize,alpha,x,inc as isize,a,lda as isize,$hermitian);
        }
    }};
}

// OpenBLAS v0.3.30 interface/syrk.c shares HERK under HEMM.
fn rankk_header<T: Scalar>(
    order: i32,
    u: i32,
    t: i32,
    n: i64,
    k: i64,
    lda: i64,
    ldb: Option<i64>,
    ldc: i64,
    h: bool,
) -> Option<(u8, u8)> {
    if !matches!(order, 101 | 102) {
        crate::invalid(1);
        return None;
    }
    let Some(mut u) = cblas_uplo(u) else {
        crate::invalid(2);
        return None;
    };
    let t = if !T::COMPLEX {
        match t {
            111 | 114 => b'N',
            112 | 113 => b'T',
            _ => {
                crate::invalid(3);
                return None;
            }
        }
    } else if h {
        match t {
            111 => b'N',
            113 => b'C',
            _ => {
                crate::invalid(3);
                return None;
            }
        }
    } else {
        match t {
            111 => b'N',
            112 => b'T',
            _ => {
                crate::invalid(3);
                return None;
            }
        }
    };
    if n < 0 {
        crate::invalid(4);
        return None;
    }
    if k < 0 {
        crate::invalid(5);
        return None;
    }
    let (rows, cols) = if t == b'N' { (n, k) } else { (k, n) };
    if !matrix_ok::<T>(rows, cols, lda, order == 101, 8) {
        return None;
    }
    if let Some(ldb) = ldb {
        if !matrix_ok::<T>(rows, cols, ldb, order == 101, 10) {
            return None;
        }
    }
    if !matrix_ok::<T>(n, n, ldc, false, if ldb.is_some() { 13 } else { 11 }) {
        return None;
    }
    let t = if order == 101 {
        u = flip_uplo(u);
        if t == b'N' {
            if h {
                b'C'
            } else {
                b'T'
            }
        } else {
            b'N'
        }
    } else {
        t
    };
    Some((u, t))
}
macro_rules! rank_k_export {
    ($p:ident,$t:ty,$alpha:ty,$read_alpha:expr,$beta:ty,$read_beta:expr,$i:ty,$cs:tt,$op:ident,$h:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,t:i32,n:$i,k:$i,alpha:$alpha,a:*const $t,lda:$i,beta:$beta,c:*mut $t,ldc:$i) {
            let Some((u,t))=rankk_header::<$t>(order,u,t,n as i64,k as i64,lda as i64,None,ldc as i64,$h) else{return;};
            if n==0{return;}
            crate::kernels::rankk(u,t,n as isize,k as isize,($read_alpha)(alpha),a,lda as isize,a,lda as isize,($read_beta)(beta),c,ldc as isize,$h,false);
        }
    }};
}
macro_rules! rank_two_k_export {
    ($p:ident,$t:ty,$alpha:ty,$read_alpha:expr,$beta:ty,$read_beta:expr,$i:ty,$cs:tt,$op:ident,$h:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,t:i32,n:$i,k:$i,alpha:$alpha,a:*const $t,lda:$i,b:*const $t,ldb:$i,beta:$beta,c:*mut $t,ldc:$i) {
            let Some((u,t))=rankk_header::<$t>(order,u,t,n as i64,k as i64,lda as i64,Some(ldb as i64),ldc as i64,$h) else{return;};
            if n==0{return;}
            let (a,lda,b,ldb)=if $h&&order==101 {(b,ldb,a,lda)}else{(a,lda,b,ldb)};
            crate::kernels::rankk(u,t,n as isize,k as isize,($read_alpha)(alpha),a,lda as isize,b,ldb as isize,($read_beta)(beta),c,ldc as isize,$h,true);
        }
    }};
}
macro_rules! rank_two_export {
    ($p:ident,$t:ty,$coefficient:ty,$read:expr,$i:ty,$cs:tt,$op:ident,$h:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $op $cs>](order:i32,u:i32,n:$i,alpha:$coefficient,x:*const $t,ix:$i,y:*const $t,iy:$i,a:*mut $t,lda:$i) {
            if !matches!(order,101|102){crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            if n<0 {crate::invalid(3);return;}
            if ix==0 {crate::invalid(6);return;}
            if iy==0 {crate::invalid(8);return;}
            if !matrix_ok::<$t>(n as i64,n as i64,lda as i64,false,10)||!vector_ok::<$t>(n as i64,ix as i64,6)||!vector_ok::<$t>(n as i64,iy as i64,8){return;}
            if n==0{return;}
            let alpha=($read)(alpha);
            if alpha==<$t>::z(){return;}
            if order==101 {
                u=flip_uplo(u);
                if $h {
                    let xs=crate::kernels::start(n as isize,ix as isize);
                    let ys=crate::kernels::start(n as isize,iy as isize);
                    let xc:Vec<$t>=(0..n as isize).map(|j|(*x.offset(xs+j*ix as isize)).cj()).collect();
                    let yc:Vec<$t>=(0..n as isize).map(|j|(*y.offset(ys+j*iy as isize)).cj()).collect();
                    crate::dense_extensions::rank2(u,n as isize,alpha.cj(),xc.as_ptr(),1,yc.as_ptr(),1,a,lda as isize,true);
                    return;
                }
            }
            crate::dense_extensions::rank2(u,n as isize,alpha,x,ix as isize,y,iy as isize,a,lda as isize,$h);
        }
    }};
}

fn packed_ok<T>(n: i64, argument: i32) -> bool {
    let count = n as u128 * (n as u128 + 1) / 2;
    if count > isize::MAX as u128 / core::mem::size_of::<T>() as u128 {
        crate::invalid(argument);
        return false;
    }
    true
}
unsafe fn conjugate_vector<T: Scalar>(n: isize, x: *mut T, inc: isize) {
    let start = crate::kernels::start(n, inc);
    for j in 0..n {
        let p = x.offset(start + j * inc);
        *p = (*p).cj();
    }
}
macro_rules! compact_triangle_export {
    ($p:ident,$t:ty,$i:ty,$fs:tt,$cs:tt,$band:ident,$packed:ident) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $band $cs>](order:i32,u:i32,t:i32,d:i32,n:$i,k:$i,a:*const $t,lda:$i,x:*mut $t,inc:$i) {
            if !matches!(order,101|102){crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            let Some(mut t)=cblas_trans(t) else {crate::invalid(3);return;};
            let Some(d)=cblas_diag(d) else {crate::invalid(4);return;};
            if n<0 {crate::invalid(5);return;}
            if k<0 {crate::invalid(6);return;}
            let Some(rows)=(k as i64).checked_add(1) else {crate::invalid(8);return;};
            if !matrix_ok::<$t>(rows,n as i64,lda as i64,false,8){return;}
            if inc==0 {crate::invalid(10);return;}
            if !vector_ok::<$t>(n as i64,inc as i64,10){return;}
            if n==0||(d==b'U'&&(n==1||k==0)){return;}
            if order==101 {u=flip_uplo(u);t=transpose_operation(t);}
            let conjugate=t==b'R';
            if conjugate {conjugate_vector(n as isize,x,inc as isize);t=b'N';}
            crate::structured_triangular::[<$p $band $fs>](&u,&t,&d,&n,&k,a,&lda,x,&inc);
            if conjugate {conjugate_vector(n as isize,x,inc as isize);}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<cblas_ $p $packed $cs>](order:i32,u:i32,t:i32,d:i32,n:$i,a:*const $t,x:*mut $t,inc:$i) {
            if !matches!(order,101|102){crate::invalid(1);return;}
            let Some(mut u)=cblas_uplo(u) else {crate::invalid(2);return;};
            let Some(mut t)=cblas_trans(t) else {crate::invalid(3);return;};
            let Some(d)=cblas_diag(d) else {crate::invalid(4);return;};
            if n<0 {crate::invalid(5);return;}
            if inc==0 {crate::invalid(8);return;}
            if !packed_ok::<$t>(n as i64,5)||!vector_ok::<$t>(n as i64,inc as i64,8){return;}
            if n==0||(d==b'U'&&n==1){return;}
            if order==101 {u=flip_uplo(u);t=transpose_operation(t);}
            let conjugate=t==b'R';
            if conjugate {conjugate_vector(n as isize,x,inc as isize);t=b'N';}
            crate::structured_triangular::[<$p $packed $fs>](&u,&t,&d,&n,a,x,&inc);
            if conjugate {conjugate_vector(n as isize,x,inc as isize);}
        }
    }};
}
macro_rules! interface {
    ($i:ty,$fs:tt,$cs:tt) => {
        compact_triangle_export!(s, f32, $i, $fs, $cs, tbmv, tpmv);
        compact_triangle_export!(s, f32, $i, $fs, $cs, tbsv, tpsv);
        compact_triangle_export!(d, f64, $i, $fs, $cs, tbmv, tpmv);
        compact_triangle_export!(d, f64, $i, $fs, $cs, tbsv, tpsv);
        compact_triangle_export!(c, c32, $i, $fs, $cs, tbmv, tpmv);
        compact_triangle_export!(c, c32, $i, $fs, $cs, tbsv, tpsv);
        compact_triangle_export!(z, c64, $i, $fs, $cs, tbmv, tpmv);
        compact_triangle_export!(z, c64, $i, $fs, $cs, tbsv, tpsv);
        rank_k_export!(
            s,
            f32,
            f32,
            |x: f32| x,
            f32,
            |x: f32| x,
            $i,
            $cs,
            syrk,
            false
        );
        rank_two_k_export!(
            s,
            f32,
            f32,
            |x: f32| x,
            f32,
            |x: f32| x,
            $i,
            $cs,
            syr2k,
            false
        );
        rank_two_export!(s, f32, f32, |x: f32| x, $i, $cs, syr2, false);
        rank_k_export!(
            d,
            f64,
            f64,
            |x: f64| x,
            f64,
            |x: f64| x,
            $i,
            $cs,
            syrk,
            false
        );
        rank_two_k_export!(
            d,
            f64,
            f64,
            |x: f64| x,
            f64,
            |x: f64| x,
            $i,
            $cs,
            syr2k,
            false
        );
        rank_two_export!(d, f64, f64, |x: f64| x, $i, $cs, syr2, false);
        rank_k_export!(
            c,
            c32,
            *const c32,
            |x: *const c32| *x,
            *const c32,
            |x: *const c32| *x,
            $i,
            $cs,
            syrk,
            false
        );
        rank_two_k_export!(
            c,
            c32,
            *const c32,
            |x: *const c32| *x,
            *const c32,
            |x: *const c32| *x,
            $i,
            $cs,
            syr2k,
            false
        );
        rank_k_export!(
            c,
            c32,
            f32,
            |x: f32| c32::new(x, 0.0),
            f32,
            |x: f32| c32::new(x, 0.0),
            $i,
            $cs,
            herk,
            true
        );
        rank_two_k_export!(
            c,
            c32,
            *const c32,
            |x: *const c32| *x,
            f32,
            |x: f32| c32::new(x, 0.0),
            $i,
            $cs,
            her2k,
            true
        );
        rank_two_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, her2, true);
        rank_k_export!(
            z,
            c64,
            *const c64,
            |x: *const c64| *x,
            *const c64,
            |x: *const c64| *x,
            $i,
            $cs,
            syrk,
            false
        );
        rank_two_k_export!(
            z,
            c64,
            *const c64,
            |x: *const c64| *x,
            *const c64,
            |x: *const c64| *x,
            $i,
            $cs,
            syr2k,
            false
        );
        rank_k_export!(
            z,
            c64,
            f64,
            |x: f64| c64::new(x, 0.0),
            f64,
            |x: f64| c64::new(x, 0.0),
            $i,
            $cs,
            herk,
            true
        );
        rank_two_k_export!(
            z,
            c64,
            *const c64,
            |x: *const c64| *x,
            f64,
            |x: f64| c64::new(x, 0.0),
            $i,
            $cs,
            her2k,
            true
        );
        rank_two_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, her2, true);
        rank_one_export!(s, f32, f32, |x: f32| x, $i, $cs, syr, false);
        rank_one_export!(d, f64, f64, |x: f64| x, $i, $cs, syr, false);
        rank_one_export!(c, c32, f32, |x: f32| c32::new(x, 0.0), $i, $cs, her, true);
        rank_one_export!(z, c64, f64, |x: f64| c64::new(x, 0.0), $i, $cs, her, true);
        symmetric_vector_export!(s, f32, f32, |x: f32| x, $i, $cs, symv, false);
        symmetric_vector_export!(d, f64, f64, |x: f64| x, $i, $cs, symv, false);
        symmetric_vector_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, hemv, true);
        symmetric_vector_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, hemv, true);
        symmetric_matrix_export!(s, f32, f32, |x: f32| x, $i, $cs, symm, false);
        symmetric_matrix_export!(d, f64, f64, |x: f64| x, $i, $cs, symm, false);
        symmetric_matrix_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, symm, false);
        symmetric_matrix_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, symm, false);
        symmetric_matrix_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, hemm, true);
        symmetric_matrix_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, hemm, true);
        ger_export!(s, f32, f32, |x: f32| x, $i, $cs, ger, false);
        ger_export!(d, f64, f64, |x: f64| x, $i, $cs, ger, false);
        ger_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, geru, false);
        ger_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, geru, false);
        ger_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs, gerc, true);
        ger_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs, gerc, true);
        triangular_exports!(s, f32, f32, |x: f32| x, $i, $cs);
        triangular_exports!(d, f64, f64, |x: f64| x, $i, $cs);
        triangular_exports!(c, c32, *const c32, |x: *const c32| *x, $i, $cs);
        triangular_exports!(z, c64, *const c64, |x: *const c64| *x, $i, $cs);
        gemv_export!(s, f32, f32, |x: f32| x, $i, $cs);
        gemv_export!(d, f64, f64, |x: f64| x, $i, $cs);
        gemv_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs);
        gemv_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs);
        gemm_export!(s, f32, f32, |x: f32| x, $i, $cs);
        gemm_export!(d, f64, f64, |x: f64| x, $i, $cs);
        gemm_export!(c, c32, *const c32, |x: *const c32| *x, $i, $cs);
        gemm_export!(z, c64, *const c64, |x: *const c64| *x, $i, $cs);
        real!(s, f32, $i, $fs, $cs);
        real!(d, f64, $i, $fs, $cs);
        complex!(c, s, c32, f32, $i, $fs, $cs);
        complex!(z, d, c64, f64, $i, $fs, $cs);
    };
}
interface!(i32, _, "");
interface!(i64, _64_, "64_");

#[cfg(test)]
mod triangular_tests;

#[cfg(test)]
mod symmetric_tests;

#[cfg(test)]
mod rank_tests;

#[cfg(test)]
mod compact_tests;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! gemm_cases {
        ($name:ident,$p:ident,$t:ty,$coefficient:expr,$imag:expr,$tolerance:expr) => {paste! {
            #[test]
            fn $name() {
                let value=|i:usize,j:usize| <$t>::real((1+i+3*j) as f64)*<$t>::real(0.125)+($imag)*<$t>::real((i as f64-j as f64)*0.25);
                let alpha=<$t>::real(0.75)+($imag)*<$t>::real(0.25);
                let beta=<$t>::real(-0.5)+($imag)*<$t>::real(0.125);
                let (m,n,k)=(2usize,3usize,4usize);
                for order in [101,102] { for ta in 111..=114 { for tb in 111..=114 {
                    let transpose=|t|t==112||t==113;
                    let (ar,ac)=if transpose(ta){(k,m)}else{(m,k)};
                    let (br,bc)=if transpose(tb){(n,k)}else{(k,n)};
                    let lda=if order==101 {ac+2}else{ar+2};
                    let ldb=if order==101 {bc+2}else{br+2};
                    let ldc=if order==101 {n+2}else{m+2};
                    let offset=|i,j,ld|if order==101 {i*ld+j}else{i+j*ld};
                    let sentinel=<$t>::real(97.0);
                    let mut a=vec![sentinel;lda*ar.max(ac)];
                    let mut b=vec![sentinel;ldb*br.max(bc)];
                    let mut initial=vec![sentinel;ldc*m.max(n)];
                    for i in 0..ar {for j in 0..ac {a[offset(i,j,lda)]=value(i,j);}}
                    for i in 0..br {for j in 0..bc {b[offset(i,j,ldb)]=value(i+1,j);}}
                    for i in 0..m {for j in 0..n {initial[offset(i,j,ldc)]=value(i,j+2);}}
                    let mut expected=initial.clone();
                    let op=|data:&[$t],ld,i,j,t| {
                        let v=if transpose(t){data[offset(j,i,ld)]}else{data[offset(i,j,ld)]};
                        if t==113||t==114 {v.cj()}else{v}
                    };
                    for i in 0..m {for j in 0..n {
                        let mut sum=<$t>::z();
                        for l in 0..k {sum+=op(&a,lda,i,l,ta)*op(&b,ldb,l,j,tb);}
                        expected[offset(i,j,ldc)]=alpha*sum+beta*initial[offset(i,j,ldc)];
                    }}
                    let mut c=initial.clone();
                    unsafe {[<cblas_ $p gemm>](order,ta,tb,m as i32,n as i32,k as i32,($coefficient)(&alpha),a.as_ptr(),lda as i32,b.as_ptr(),ldb as i32,($coefficient)(&beta),c.as_mut_ptr(),ldc as i32);}
                    for (actual,expected) in c.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tolerance,"LP64 order={order} ta={ta} tb={tb}");}
                    let mut c=initial.clone();
                    unsafe {[<cblas_ $p gemm64_>](order,ta,tb,m as i64,n as i64,k as i64,($coefficient)(&alpha),a.as_ptr(),lda as i64,b.as_ptr(),ldb as i64,($coefficient)(&beta),c.as_mut_ptr(),ldc as i64);}
                    for (actual,expected) in c.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tolerance,"ILP64 order={order} ta={ta} tb={tb}");}
                }}}
            }
        }};
    }
    gemm_cases!(gemm_real32, s, f32, |x: &f32| *x, 0.0f32, 1e-5);
    gemm_cases!(gemm_real64, d, f64, |x: &f64| *x, 0.0f64, 1e-12);
    gemm_cases!(
        gemm_complex32,
        c,
        c32,
        |x: &c32| x as *const c32,
        c32::new(0.0, 1.0),
        1e-5
    );
    gemm_cases!(
        gemm_complex64,
        z,
        c64,
        |x: &c64| x as *const c64,
        c64::new(0.0, 1.0),
        1e-12
    );

    macro_rules! gemv_cases {
        ($name:ident,$p:ident,$t:ty,$coefficient:expr,$imag:expr,$tolerance:expr) => {paste! {
            #[test]
            fn $name() {
                let value=|i:usize,j:usize| <$t>::real((1+i+3*j) as f64)*<$t>::real(0.125)+($imag)*<$t>::real((i as f64-j as f64)*0.25);
                let alpha=<$t>::real(0.75)+($imag)*<$t>::real(0.25);
                let beta=<$t>::real(-0.5)+($imag)*<$t>::real(0.125);
                let (m,n)=(2usize,3usize);
                for order in [101,102] {for t in 111..=114 {for ix in [-2i32,1,3] {for iy in [-3i32,1,2] {
                    let transpose=t==112||t==113;
                    let (nx,ny)=if transpose {(m,n)}else{(n,m)};
                    let ld=if order==101 {n+2}else{m+2};
                    let offset=|i,j|if order==101 {i*ld+j}else{i+j*ld};
                    let sentinel=<$t>::real(97.0);
                    let mut a=vec![sentinel;ld*m.max(n)];
                    for i in 0..m {for j in 0..n {a[offset(i,j)]=value(i,j);}}
                    let index=|i:usize,len:usize,inc:i32| if inc<0 {(len-1-i)*(-inc) as usize}else{i*inc as usize};
                    let mut x=vec![sentinel;1+(nx-1)*ix.unsigned_abs() as usize];
                    let mut initial=vec![sentinel;1+(ny-1)*iy.unsigned_abs() as usize];
                    for i in 0..nx {x[index(i,nx,ix)]=value(i,1);}
                    for i in 0..ny {initial[index(i,ny,iy)]=value(i,2);}
                    let mut expected=initial.clone();
                    for i in 0..ny {
                        let mut sum=<$t>::z();
                        for j in 0..nx {
                            let v=if transpose {a[offset(j,i)]}else{a[offset(i,j)]};
                            let v=if t==113||t==114 {v.cj()}else{v};
                            sum+=v*x[index(j,nx,ix)];
                        }
                        let out=index(i,ny,iy);
                        expected[out]=alpha*sum+beta*initial[out];
                    }
                    let mut y=initial.clone();
                    unsafe {[<cblas_ $p gemv>](order,t,m as i32,n as i32,($coefficient)(&alpha),a.as_ptr(),ld as i32,x.as_ptr(),ix,($coefficient)(&beta),y.as_mut_ptr(),iy);}
                    for (actual,expected) in y.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tolerance,"LP64 order={order} t={t} ix={ix} iy={iy}");}
                    let mut y=initial.clone();
                    unsafe {[<cblas_ $p gemv64_>](order,t,m as i64,n as i64,($coefficient)(&alpha),a.as_ptr(),ld as i64,x.as_ptr(),ix as i64,($coefficient)(&beta),y.as_mut_ptr(),iy as i64);}
                    for (actual,expected) in y.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tolerance,"ILP64 order={order} t={t} ix={ix} iy={iy}");}
                }}}}
            }
        }};
    }
    gemv_cases!(gemv_real32, s, f32, |x: &f32| *x, 0.0f32, 1e-5);
    gemv_cases!(gemv_real64, d, f64, |x: &f64| *x, 0.0f64, 1e-12);
    gemv_cases!(
        gemv_complex32,
        c,
        c32,
        |x: &c32| x as *const c32,
        c32::new(0.0, 1.0),
        1e-5
    );
    gemv_cases!(
        gemv_complex64,
        z,
        c64,
        |x: &c64| x as *const c64,
        c64::new(0.0, 1.0),
        1e-12
    );

    #[test]
    fn matrix_headers_precede_scalar_and_array_access() {
        unsafe {
            cblas_zgemm64_(
                102,
                111,
                111,
                0,
                3,
                4,
                core::ptr::null(),
                core::ptr::null(),
                1,
                core::ptr::null(),
                4,
                core::ptr::null(),
                core::ptr::null_mut(),
                1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 0);
            cblas_zgemv64_(
                101,
                113,
                0,
                3,
                core::ptr::null(),
                core::ptr::null(),
                3,
                core::ptr::null(),
                1,
                core::ptr::null(),
                core::ptr::null_mut(),
                1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 0);
            cblas_zgemm64_(
                101,
                111,
                111,
                2,
                3,
                4,
                core::ptr::null(),
                core::ptr::null(),
                3,
                core::ptr::null(),
                3,
                core::ptr::null(),
                core::ptr::null_mut(),
                3,
            );
            assert_eq!(crate::colmat_blas_last_error(), 9);
            cblas_zgemv64_(
                101,
                113,
                2,
                3,
                core::ptr::null(),
                core::ptr::null(),
                3,
                core::ptr::null(),
                0,
                core::ptr::null(),
                core::ptr::null_mut(),
                1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 9);
            cblas_zgemm64_(
                102,
                111,
                111,
                2,
                3,
                4,
                core::ptr::null(),
                core::ptr::null(),
                i64::MAX,
                core::ptr::null(),
                4,
                core::ptr::null(),
                core::ptr::null_mut(),
                2,
            );
            assert_eq!(crate::colmat_blas_last_error(), 9);
        }
    }

    #[test]
    fn complex_index_uses_single_precision_abs1_and_first_tie() {
        let x = [c32::new(1.0, 0.0), c32::new(1.0, 2.0f32.powi(-25))];
        // BLAS SCABS1 adds the components in single precision.
        assert_eq!(x[0].re.abs() + x[0].im.abs(), x[1].re.abs() + x[1].im.abs());
        unsafe {
            assert_eq!(cblas_icamax(2, x.as_ptr(), 1), 0);
            assert_eq!(cblas_icamax64_(2, x.as_ptr(), 1), 0);
        }
    }

    #[test]
    fn ilp64_span_rejection_precedes_pointer_access() {
        unsafe {
            cblas_dcopy64_(2, core::ptr::null(), i64::MAX, core::ptr::null_mut(), 1);
            assert_eq!(crate::colmat_blas_last_error(), 3);
            assert_eq!(crate::colmat_blas_last_error(), 0);
            assert_eq!(cblas_idamax64_(0, core::ptr::null(), i64::MAX), 0);
        }
    }

    #[test]
    fn complex_dot_conjugation_and_negative_strides_both_abis() {
        let x = [c64::new(1.0, 2.0), c64::new(3.0, -1.0)];
        let y = [c64::new(2.0, -1.0), c64::new(-1.0, 3.0)];
        let expected_u = x[1] * y[0] + x[0] * y[1];
        let expected_c = x[1].cj() * y[0] + x[0].cj() * y[1];
        unsafe {
            assert_eq!(cblas_zdotu(2, x.as_ptr(), -1, y.as_ptr(), 1), expected_u);
            assert_eq!(cblas_zdotu64_(2, x.as_ptr(), -1, y.as_ptr(), 1), expected_u);
            assert_eq!(cblas_zdotc(2, x.as_ptr(), -1, y.as_ptr(), 1), expected_c);
            assert_eq!(cblas_zdotc64_(2, x.as_ptr(), -1, y.as_ptr(), 1), expected_c);
        }
    }
}
