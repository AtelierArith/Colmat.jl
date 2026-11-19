//! LAPACKE TFSM, TFTRI and PTSV high/work adapters for the RFP triangular
//! operations and the positive-definite tridiagonal solver.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{tfsm,tftri,ptsv}{,_work}.c.
//! Row-major RFP storage is the dense transpose of the ROW-by-COL rectangle
//! (the reference dtf_trans mapping).
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
unsafe fn rfp_nan<T: Scalar>(transr: u8, n: usize, a: *const T) -> bool {
    let (row, col) = rfp_dims(transr, n);
    (0..row * col).any(|k| {
        let v = *a.add(k);
        v != v
    })
}
unsafe fn nan_vec<T: Scalar>(count: i64, a: *const T) -> bool {
    (0..count.max(0) as usize).any(|i| {
        let v = *a.add(i);
        v != v
    })
}

macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfsm $($suffix)*>](
            layout:i32,transr:u8,side:u8,uplo:u8,trans:u8,diag:u8,
            m:$i,n:$i,alpha:$t,a:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                let mn=if side.to_ascii_uppercase()==b'R' {n} else {m};
                if alpha!=<$t>::z() && rfp_nan(transr,mn.max(0) as usize,a) {return -10;}
                if nan_scalar::<$t>(alpha) {return -9;}
                if alpha!=<$t>::z()
                    && ge_nancheck(layout,m as i64,n as i64,b,ldb as i64).unwrap_or(false)
                {
                    return -11;
                }
            }
            [<LAPACKE_ $p tfsm_work $($suffix)*>](layout,transr,side,uplo,trans,diag,m,n,alpha,a,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfsm_work $($suffix)*>](
            layout:i32,transr:u8,side:u8,uplo:u8,trans:u8,diag:u8,
            m:$i,n:$i,alpha:$t,a:*const $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<m {return -12;}
            if layout==COL_MAJOR || (m==0&&n==0) {
                crate::structured_triangular_lapack::[<$p tfsm $fortran>](&transr,&side,&uplo,&trans,&diag,&m,&n,&alpha,a,b,&ldb);
                return 0;
            }
            let (mm,nn)=(m as usize,n as usize);
            let mn=if side.to_ascii_uppercase()==b'R' {nn} else {mm};
            let at=rfp_to_column(layout,transr,mn,a);
            let mut bt=match row_to_column(mm,nn,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=mm.max(1) as $i;
            crate::structured_triangular_lapack::[<$p tfsm $fortran>](&transr,&side,&uplo,&trans,&diag,&m,&n,&alpha,at.as_ptr(),bt.as_mut_ptr(),&ld);
            for i in 0..mm {for j in 0..nn {*b.add(i*ldb as usize+j)=bt[i+j*mm];}}
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tftri $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,diag:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && rfp_nan(transr,n.max(0) as usize,a) {
                return -6;
            }
            [<LAPACKE_ $p tftri_work $($suffix)*>](layout,transr,uplo,diag,n,a)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tftri_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,diag:u8,n:$i,a:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::structured_triangular_lapack::[<$p tftri $fortran>](&transr,&uplo,&diag,&n,a,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut at=rfp_to_column(layout,transr,nn,a);
            let mut info:$i=0;
            crate::structured_triangular_lapack::[<$p tftri $fortran>](&transr,&uplo,&diag,&n,at.as_mut_ptr(),&mut info);
            if info>=0 {rfp_from_column(layout,transr,nn,a,&at);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ptsv $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,d:*mut $r,e:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -6;}
                if (0..n.max(0) as usize).any(|i| {let v=*d.add(i); v!=v}) {return -4;}
                if nan_vec((n-1).max(0) as i64,e) {return -5;}
            }
            [<LAPACKE_ $p ptsv_work $($suffix)*>](layout,n,nrhs,d,e,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ptsv_work $($suffix)*>](
            layout:i32,n:$i,nrhs:$i,d:*mut $r,e:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -7;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::positive_tridiagonal::[<$p ptsv $fortran>](&n,&nrhs,d,e,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::positive_tridiagonal::[<$p ptsv $fortran>](&n,&nrhs,d,e,bt.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
unsafe fn nan_scalar<T: Scalar>(v: T) -> bool {
    v != v
}
exports!(s, f32, f32, i32, [], _);
exports!(d, f64, f64, i32, [], _);
exports!(c, colmat::c32, f32, i32, [], _);
exports!(z, colmat::c64, f64, i32, [], _);
exports!(s, f32, f32, i64, [64_], _64_);
exports!(d, f64, f64, i64, [64_], _64_);
exports!(c, colmat::c32, f32, i64, [64_], _64_);
exports!(z, colmat::c64, f64, i64, [64_], _64_);
