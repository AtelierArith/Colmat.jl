//! LAPACKE SFRK/HFRK high/work adapters for the RFP symmetric/Hermitian
//! rank-k update C := alpha*op(A)*op(A)^T + beta*C.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{sfrk,hfrk}{,_work}.c. The update returns no
//! INFO, so column-major argument errors are reported by the native checker
//! only and the LAPACKE status stays zero; row-major reports LDA < KA (-9).
use super::rfp_cholesky::{rfp_from_column, rfp_nan, rfp_to_column};
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

macro_rules! exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,trans:u8,n:$i,k:$i,alpha:$r,a:*const $t,lda:$i,beta:$r,c:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let notrans=trans.to_ascii_uppercase()==b'N';
                let (nrowa,ka)=if notrans {(n as i64,k as i64)} else {(k as i64,n as i64)};
                if alpha.is_nan() {return -7;}
                if ge_nancheck(layout,nrowa,ka,a,lda as i64).unwrap_or(false) {return -8;}
                if beta.is_nan() {return -10;}
                if rfp_nan(layout,transr,n.max(0) as usize,c) {return -11;}
            }
            [<LAPACKE_ $p $family _work $($suffix)*>](layout,transr,uplo,trans,n,k,alpha,a,lda,beta,c)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $family _work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,trans:u8,n:$i,k:$i,alpha:$r,a:*const $t,lda:$i,beta:$r,c:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let notrans=trans.to_ascii_uppercase()==b'N';
            let (nrowa,ka)=if notrans {(n,k)} else {(k,n)};
            if layout==ROW_MAJOR && lda<ka {return -9;}
            if layout==COL_MAJOR || n<=0 {
                crate::rfp_rank::[<$p $family $fortran>](&transr,&uplo,&trans,&n,&k,&alpha,a,&lda,&beta,c);
                return 0;
            }
            let (rr,cc,nn)=(nrowa as usize,ka as usize,n as usize);
            let at=match row_to_column(rr,cc,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let mut ct=rfp_to_column(layout,transr,nn,c);
            let lda_t=rr.max(1) as $i;
            crate::rfp_rank::[<$p $family $fortran>](&transr,&uplo,&trans,&n,&k,&alpha,at.as_ptr(),&lda_t,&beta,ct.as_mut_ptr());
            rfp_from_column(layout,transr,nn,c,&ct);
            0
        }
    }};
}
exports!(s, sfrk, f32, f32, i32, [], _);
exports!(d, sfrk, f64, f64, i32, [], _);
exports!(c, hfrk, colmat::c32, f32, i32, [], _);
exports!(z, hfrk, colmat::c64, f64, i32, [], _);
exports!(s, sfrk, f32, f32, i64, [64_], _64_);
exports!(d, sfrk, f64, f64, i64, [64_], _64_);
exports!(c, hfrk, colmat::c32, f32, i64, [64_], _64_);
exports!(z, hfrk, colmat::c64, f64, i64, [64_], _64_);
