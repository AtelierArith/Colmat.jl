//! LAPACKE UPGTR/UPMTR high/work adapters for packed tridiagonal Q.
//! Reference: Reference LAPACK v3.12.1 LAPACKE/src/lapacke_{c,z}{upgtr,upmtr}{,_work}.c.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}

// Row-major packed index for the logical entry (i, j) inside the triangle.
fn row_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        i * n - i * (i + 1) / 2 + j
    } else {
        i * (i + 1) / 2 + j + n - 1 - (n - 1)
    }
}

// Column-major packed index for the logical entry (i, j) inside the triangle.
fn col_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        j * (j + 1) / 2 + i
    } else {
        j * n - j * (j + 1) / 2 + i
    }
}

unsafe fn packed_to_column<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> Vec<T> {
    let mut out = vec![T::z(); packed_len(n)];
    if layout == COL_MAJOR {
        for k in 0..packed_len(n) {
            out[k] = *ap.add(k);
        }
        return out;
    }
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            out[col_index(upper, n, i, j)] = *ap.add(row_index(upper, n, i, j));
        }
    }
    out
}

unsafe fn packed_nan<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> bool {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper { (0, j) } else { (j, n - 1) };
        for i in lo..=hi {
            let k = if layout == ROW_MAJOR {
                row_index(upper, n, i, j)
            } else {
                col_index(upper, n, i, j)
            };
            let v = *ap.add(k);
            if v != v {
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

macro_rules! exports {
    ($p:ident,$g:ident,$m:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $g $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,tau:*const $t,q:*mut $t,ldq:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -4;}
                if tau_nan(n.max(0) as usize,tau) {return -5;}
            }
            let mut work=vec![<$t>::z();(n.max(0) as usize).saturating_sub(1).max(1)];
            [<LAPACKE_ $p $g _work $($suffix)*>](layout,uplo,n,ap,tau,q,ldq,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $g _work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,tau:*const $t,q:*mut $t,ldq:$i,
            work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldq<n {return -7;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::structured_tridiagonal::[<$p $g $fortran>](&uplo,&n,ap,tau,q,&ldq,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let apt=packed_to_column(layout,uplo,nn,ap);
            let mut qt=match row_to_column(nn,nn,q,ldq as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldq_t,)=((nn.max(1)) as $i,);
            let mut info:$i=0;
            crate::structured_tridiagonal::[<$p $g $fortran>](&uplo,&n,apt.as_ptr(),tau,qt.as_mut_ptr(),&ldq_t,work,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*q.add(i*ldq as usize+j)=qt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $m $($suffix)*>](
            layout:i32,side:u8,uplo:u8,trans:u8,m:$i,n:$i,
            ap:*const $t,tau:*const $t,c:*mut $t,ldc:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            let (sd,tr)=(side.to_ascii_uppercase(),trans.to_ascii_uppercase());
            let r=if sd==b'L' {m.max(0) as usize} else {n.max(0) as usize};
            if LAPACKE_get_nancheck()!=0 {
                if packed_nan(layout,uplo,r,ap) {return -7;}
                if ge_nancheck(layout,m as i64,n as i64,c,ldc as i64).unwrap_or(false) {return -9;}
                if tau_nan(r,tau) {return -8;}
            }
            let _=tr;
            let lw=if sd==b'L' {n.max(1) as usize} else {m.max(1) as usize};
            let mut work=vec![<$t>::z();lw];
            [<LAPACKE_ $p $m _work $($suffix)*>](layout,side,uplo,trans,m,n,ap,tau,c,ldc,work.as_mut_ptr())
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $m _work $($suffix)*>](
            layout:i32,side:u8,uplo:u8,trans:u8,m:$i,n:$i,
            ap:*const $t,tau:*const $t,c:*mut $t,ldc:$i,work:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldc<n {return -10;}
            if layout==COL_MAJOR || m==0 || n==0 {
                let mut info:$i=0;
                crate::structured_tridiagonal::[<$p $m $fortran>](&side,&uplo,&trans,&m,&n,ap,tau,c,&ldc,work,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (mm,nn)=(m as usize,n as usize);
            let r=if side.to_ascii_uppercase()==b'L' {mm} else {nn};
            let apt=packed_to_column(layout,uplo,r,ap);
            let mut ct=match row_to_column(mm,nn,c,ldc as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let (ldc_t,)=(mm.max(1) as $i,);
            let mut info:$i=0;
            crate::structured_tridiagonal::[<$p $m $fortran>](&side,&uplo,&trans,&m,&n,apt.as_ptr(),tau,ct.as_mut_ptr(),&ldc_t,work,&mut info);
            if info>=0 {for i in 0..mm {for j in 0..nn {*c.add(i*ldc as usize+j)=ct[i+j*mm];}}}
            if info<0 {info-1} else {info}
        }
    }};
}
exports!(c, upgtr, upmtr, colmat::c32, f32, i32, [], _);
exports!(z, upgtr, upmtr, colmat::c64, f64, i32, [], _);
exports!(c, upgtr, upmtr, colmat::c32, f32, i64, [64_], _64_);
exports!(z, upgtr, upmtr, colmat::c64, f64, i64, [64_], _64_);
exports!(s, opgtr, opmtr, f32, f32, i32, [], _);
exports!(d, opgtr, opmtr, f64, f64, i32, [], _);
exports!(s, opgtr, opmtr, f32, f32, i64, [64_], _64_);
exports!(d, opgtr, opmtr, f64, f64, i64, [64_], _64_);
