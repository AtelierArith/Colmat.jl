//! LAPACKE TRTTF/TFTTR/TFTTP/TPTTF/TPTTR/TRTTP high/work adapters for the
//! triangular storage conversions between full, RFP and packed formats.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{trttf,tfttr,tfttp,tpttf,tpttr,trttp}{,_work}.c.
//! Row-major RFP storage is the dense transpose of the ROW-by-COL rectangle
//! (the reference dtf_trans mapping) and row-major packed AP uses the
//! reference pp_trans permutation.
use super::{row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn packed_len(n: usize) -> usize {
    n * (n + 1) / 2
}
fn row_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        i * n - i * (i + 1) / 2 + j
    } else {
        i * (i + 1) / 2 + j
    }
}
fn col_index(upper: bool, n: usize, i: usize, j: usize) -> usize {
    if upper {
        j * (j + 1) / 2 + i
    } else {
        j * n - j * (j + 1) / 2 + i
    }
}
unsafe fn packed_to_column<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> Vec<T> {
    let upper = uplo.to_ascii_uppercase() == b'U';
    let mut out = vec![T::z(); packed_len(n)];
    if layout == COL_MAJOR {
        for k in 0..packed_len(n) {
            out[k] = *ap.add(k);
        }
        return out;
    }
    for j in 0..n {
        let (lo, hi) = if upper {
            (0, j)
        } else {
            (j, n.saturating_sub(1))
        };
        for i in lo..=hi {
            out[col_index(upper, n, i, j)] = *ap.add(row_index(upper, n, i, j));
        }
    }
    out
}
unsafe fn packed_from_column<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *mut T, source: &[T]) {
    if layout == COL_MAJOR {
        for k in 0..packed_len(n) {
            *ap.add(k) = source[k];
        }
        return;
    }
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper {
            (0, j)
        } else {
            (j, n.saturating_sub(1))
        };
        for i in lo..=hi {
            *ap.add(row_index(upper, n, i, j)) = source[col_index(upper, n, i, j)];
        }
    }
}
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
unsafe fn nan_count<T: Scalar>(count: usize, a: *const T) -> bool {
    (0..count).any(|k| {
        let v = *a.add(k);
        v != v
    })
}
unsafe fn triangle_nan<T: Scalar>(
    layout: i32,
    uplo: u8,
    n: usize,
    a: *const T,
    lda: usize,
) -> bool {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper {
            (0, j)
        } else {
            (j, n.saturating_sub(1))
        };
        for i in lo..=hi {
            let v = if layout == ROW_MAJOR {
                *a.add(i * lda + j)
            } else {
                *a.add(i + j * lda)
            };
            if v != v {
                return true;
            }
        }
    }
    false
}

macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trttf $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*const $t,lda:$i,arf:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && triangle_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize)
            {
                return -5;
            }
            [<LAPACKE_ $p trttf_work $($suffix)*>](layout,transr,uplo,n,a,lda,arf)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trttf_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,a:*const $t,lda:$i,arf:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p trttf $fortran>](&transr,&uplo,&n,a,&lda,arf,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut arft=vec![<$t>::z();packed_len(nn)];
            let mut info:$i=0;
            crate::storage_conversions::[<$p trttf $fortran>](&transr,&uplo,&n,at.as_ptr(),&ld,arft.as_mut_ptr(),&mut info);
            if info>=0 {rfp_from_column(layout,transr,nn,arf,&arft);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfttr $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,arf:*const $t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && nan_count(rfp_dims(transr,n.max(0) as usize).0 * rfp_dims(transr,n.max(0) as usize).1,arf)
            {
                return -5;
            }
            [<LAPACKE_ $p tfttr_work $($suffix)*>](layout,transr,uplo,n,arf,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfttr_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,arf:*const $t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -7;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p tfttr $fortran>](&transr,&uplo,&n,arf,a,&lda,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let arft=rfp_to_column(layout,transr,nn,arf);
            let mut at=vec![<$t>::z();nn*nn];
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::storage_conversions::[<$p tfttr $fortran>](&transr,&uplo,&n,arft.as_ptr(),at.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfttp $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,arf:*const $t,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && nan_count(rfp_dims(transr,n.max(0) as usize).0 * rfp_dims(transr,n.max(0) as usize).1,arf)
            {
                return -5;
            }
            [<LAPACKE_ $p tfttp_work $($suffix)*>](layout,transr,uplo,n,arf,ap)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tfttp_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,arf:*const $t,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p tfttp $fortran>](&transr,&uplo,&n,arf,ap,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let arft=rfp_to_column(layout,transr,nn,arf);
            let mut apt=vec![<$t>::z();packed_len(nn)];
            let mut info:$i=0;
            crate::storage_conversions::[<$p tfttp $fortran>](&transr,&uplo,&n,arft.as_ptr(),apt.as_mut_ptr(),&mut info);
            if info>=0 {packed_from_column(layout,uplo,nn,ap,&apt);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tpttf $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,ap:*const $t,arf:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && nan_count(packed_len(n.max(0) as usize),ap) {
                return -5;
            }
            [<LAPACKE_ $p tpttf_work $($suffix)*>](layout,transr,uplo,n,ap,arf)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tpttf_work $($suffix)*>](
            layout:i32,transr:u8,uplo:u8,n:$i,ap:*const $t,arf:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p tpttf $fortran>](&transr,&uplo,&n,ap,arf,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let apt=packed_to_column(layout,uplo,nn,ap);
            let mut arft=vec![<$t>::z();packed_len(nn)];
            let mut info:$i=0;
            crate::storage_conversions::[<$p tpttf $fortran>](&transr,&uplo,&n,apt.as_ptr(),arft.as_mut_ptr(),&mut info);
            if info>=0 {rfp_from_column(layout,transr,nn,arf,&arft);}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tpttr $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && nan_count(packed_len(n.max(0) as usize),ap) {
                return -4;
            }
            [<LAPACKE_ $p tpttr_work $($suffix)*>](layout,uplo,n,ap,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p tpttr_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -6;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p tpttr $fortran>](&uplo,&n,ap,a,&lda,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let apt=packed_to_column(layout,uplo,nn,ap);
            let mut at=vec![<$t>::z();nn*nn];
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::storage_conversions::[<$p tpttr $fortran>](&uplo,&n,apt.as_ptr(),at.as_mut_ptr(),&ld,&mut info);
            if info>=0 {for i in 0..nn {for j in 0..nn {*a.add(i*lda as usize+j)=at[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trttp $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*const $t,lda:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && triangle_nan(layout,uplo,n.max(0) as usize,a,lda.max(0) as usize)
            {
                return -4;
            }
            [<LAPACKE_ $p trttp_work $($suffix)*>](layout,uplo,n,a,lda,ap)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p trttp_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,a:*const $t,lda:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -5;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::storage_conversions::[<$p trttp $fortran>](&uplo,&n,a,&lda,ap,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut apt=vec![<$t>::z();packed_len(nn)];
            let mut info:$i=0;
            crate::storage_conversions::[<$p trttp $fortran>](&uplo,&n,at.as_ptr(),&ld,apt.as_mut_ptr(),&mut info);
            if info>=0 {packed_from_column(layout,uplo,nn,ap,&apt);}
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
