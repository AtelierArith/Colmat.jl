//! LAPACKE PFTRF/PFTRI/PFTRS high/work adapters for the rectangular full
//! packed (RFP) Cholesky family.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{pftrf,pftri,pftrs}{,_work}.c.
//! The RFP array is a ROW-by-COL rectangle whose transpose follows the
//! reference dtf_trans mapping.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn rfp_dims(transr: u8, n: usize) -> (usize, usize) {
    if transr.to_ascii_uppercase() == b'N' {
        if n % 2 == 0 {
            (n + 1, n / 2)
        } else {
            (n, (n + 1) / 2)
        }
    } else if n % 2 == 0 {
        (n / 2, n + 1)
    } else {
        ((n + 1) / 2, n)
    }
}
unsafe fn rfp_to_column<T: Scalar>(layout: i32, transr: u8, n: usize, a: *const T) -> Vec<T> {
    let (row, col) = rfp_dims(transr, n);
    let mut out = vec![T::z(); row * col];
    if layout == COL_MAJOR {
        for k in 0..row * col {
            out[k] = *a.add(k);
        }
        return out;
    }
    for i in 0..row {
        for j in 0..col {
            out[i + j * row] = *a.add(i * col + j);
        }
    }
    out
}
unsafe fn rfp_from_column<T: Scalar>(layout: i32, transr: u8, n: usize, a: *mut T, source: &[T]) {
    let (row, col) = rfp_dims(transr, n);
    if layout == COL_MAJOR {
        for k in 0..row * col {
            *a.add(k) = source[k];
        }
        return;
    }
    for i in 0..row {
        for j in 0..col {
            *a.add(i * col + j) = source[i + j * row];
        }
    }
}
unsafe fn rfp_nan<T: Scalar>(layout: i32, transr: u8, n: usize, a: *const T) -> bool {
    let (row, col) = rfp_dims(transr, n);
    for k in 0..row * col {
        let v = *a.add(k);
        if v != v {
            return true;
        }
    }
    false
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftrf $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if rfp_nan(layout,transr,n.max(0) as usize,a) {return -5;}
            [<LAPACKE_ $p pftrf_work $($suffix)*>](layout,transr,uplo,n,a)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftrf_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::packed_cholesky::[<$p pftrf $fortran>](&transr,&uplo,&n,a,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tmp=rfp_to_column(layout,transr,nn,a);
            let mut info:$i=0;
            crate::packed_cholesky::[<$p pftrf $fortran>](&transr,&uplo,&n,tmp.as_mut_ptr(),&mut info);
            rfp_from_column(layout,transr,nn,a,&tmp);
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftri $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if rfp_nan(layout,transr,n.max(0) as usize,a) {return -5;}
            [<LAPACKE_ $p pftri_work $($suffix)*>](layout,transr,uplo,n,a)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftri_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::packed_cholesky::[<$p pftri $fortran>](&transr,&uplo,&n,a,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut tmp=rfp_to_column(layout,transr,nn,a);
            let mut info:$i=0;
            crate::packed_cholesky::[<$p pftri $fortran>](&transr,&uplo,&n,tmp.as_mut_ptr(),&mut info);
            if info>=0 {rfp_from_column(layout,transr,nn,a,&tmp);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftrs $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,nrhs:$i,a:*const $t,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if rfp_nan(layout,transr,n.max(0) as usize,a) {return -6;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -7;}
            }
            [<LAPACKE_ $p pftrs_work $($suffix)*>](layout,transr,uplo,n,nrhs,a,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pftrs_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,nrhs:$i,a:*const $t,
            b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -8;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::packed_cholesky::[<$p pftrs $fortran>](&transr,&uplo,&n,&nrhs,a,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let tmp=rfp_to_column(layout,transr,nn,a);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::packed_cholesky::[<$p pftrs $fortran>](&transr,&uplo,&n,&nrhs,tmp.as_ptr(),bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(s, f32, i32, [], _);
exports!(d, f64, i32, [], _);
exports!(c, colmat::c32, i32, [], _);
exports!(z, colmat::c64, i32, [], _);
exports!(s, f32, i64, [64_], _64_);
exports!(d, f64, i64, [64_], _64_);
exports!(c, colmat::c32, i64, [64_], _64_);
exports!(z, colmat::c64, i64, [64_], _64_);
