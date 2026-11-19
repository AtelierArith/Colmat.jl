//! LAPACKE POEQU/POEQUB/PPEQU/PPSV/PPTRI high/work adapters for the
//! positive-definite equilibration and packed extras.
//! Reference: Reference LAPACK v3.12.1
//! LAPACKE/src/lapacke_{s,d,c,z}{poequ,poequb,ppequ,ppsv,pptri}{,_work}.c.
//! Row-major packed AP uses the reference pp_trans permutation.
use super::{ge_nancheck, row_to_column, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
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
unsafe fn packed_nan<T: Scalar>(layout: i32, uplo: u8, n: usize, ap: *const T) -> bool {
    let upper = uplo.to_ascii_uppercase() == b'U';
    for j in 0..n {
        let (lo, hi) = if upper {
            (0, j)
        } else {
            (j, n.saturating_sub(1))
        };
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

macro_rules! dense_equil {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt,$fname:ident)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname $($suffix)*>](
            layout:i32,n:$i,a:*const $t,lda:$i,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0
                && ge_nancheck(layout,n as i64,n as i64,a,lda as i64).unwrap_or(false)
            {
                return -3;
            }
            [<LAPACKE_ $p $fname _work $($suffix)*>](layout,n,a,lda,s,scond,amax)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p $fname _work $($suffix)*>](
            layout:i32,n:$i,a:*const $t,lda:$i,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && lda<n {return -4;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::positive_equilibration::[<$p $fname $fortran>](&n,a,&lda,s,scond,amax,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let at=match row_to_column(nn,nn,a,lda as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::positive_equilibration::[<$p $fname $fortran>](&n,at.as_ptr(),&ld,s,scond,amax,&mut info);
            if info<0 {info-1} else {info}
        }
    }};
}

macro_rules! packed {
    ($p:ident,$t:ty,$r:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ppequ $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 && packed_nan(layout,uplo,n.max(0) as usize,ap) {
                return -4;
            }
            [<LAPACKE_ $p ppequ_work $($suffix)*>](layout,uplo,n,ap,s,scond,amax)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ppequ_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*const $t,
            s:*mut $r,scond:*mut $r,amax:*mut $r,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::positive_equilibration::[<$p ppequ $fortran>](&uplo,&n,ap,s,scond,amax,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let apt=packed_to_column(layout,uplo,nn,ap);
            let mut info:$i=0;
            crate::positive_equilibration::[<$p ppequ $fortran>](&uplo,&n,apt.as_ptr(),s,scond,amax,&mut info);
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ppsv $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -5;}
                if ge_nancheck(layout,n as i64,nrhs as i64,b,ldb as i64).unwrap_or(false) {return -6;}
            }
            [<LAPACKE_ $p ppsv_work $($suffix)*>](layout,uplo,n,nrhs,ap,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p ppsv_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,nrhs:$i,ap:*mut $t,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR && ldb<nrhs {return -7;}
            if layout==COL_MAJOR || n==0 || nrhs==0 {
                let mut info:$i=0;
                crate::packed_cholesky::[<$p ppsv $fortran>](&uplo,&n,&nrhs,ap,b,&ldb,&mut info);
                return if info<0 {info-1} else {info};
            }
            let (nn,rr)=(n as usize,nrhs as usize);
            let mut apt=packed_to_column(layout,uplo,nn,ap);
            let mut bt=match row_to_column(nn,rr,b,ldb as usize){Ok(v)=>v,Err(e)=>return e as $i};
            let ld=nn.max(1) as $i;
            let mut info:$i=0;
            crate::packed_cholesky::[<$p ppsv $fortran>](&uplo,&n,&nrhs,apt.as_mut_ptr(),bt.as_mut_ptr(),&ld,&mut info);
            packed_from_column(layout,uplo,nn,ap,&apt);
            if info>=0 {for i in 0..nn {for j in 0..rr {*b.add(i*ldb as usize+j)=bt[i+j*nn];}}}
            if info<0 {info-1} else {info}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pptri $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if packed_nan(layout,uplo,n.max(0) as usize,ap) {return -4;}
            [<LAPACKE_ $p pptri_work $($suffix)*>](layout,uplo,n,ap)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p pptri_work $($suffix)*>](
            layout:i32,uplo:u8,n:$i,ap:*mut $t,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==COL_MAJOR || n==0 {
                let mut info:$i=0;
                crate::packed_cholesky::[<$p pptri $fortran>](&uplo,&n,ap,&mut info);
                return if info<0 {info-1} else {info};
            }
            let nn=n as usize;
            let mut apt=packed_to_column(layout,uplo,nn,ap);
            let mut info:$i=0;
            crate::packed_cholesky::[<$p pptri $fortran>](&uplo,&n,apt.as_mut_ptr(),&mut info);
            if info>=0 {packed_from_column(layout,uplo,nn,ap,&apt);}
            if info<0 {info-1} else {info}
        }
    }};
}
dense_equil!(s, f32, f32, i32, [], _, poequ);
dense_equil!(d, f64, f64, i32, [], _, poequ);
dense_equil!(c, colmat::c32, f32, i32, [], _, poequ);
dense_equil!(z, colmat::c64, f64, i32, [], _, poequ);
dense_equil!(s, f32, f32, i64, [64_], _64_, poequ);
dense_equil!(d, f64, f64, i64, [64_], _64_, poequ);
dense_equil!(c, colmat::c32, f32, i64, [64_], _64_, poequ);
dense_equil!(z, colmat::c64, f64, i64, [64_], _64_, poequ);
dense_equil!(s, f32, f32, i32, [], _, poequb);
dense_equil!(d, f64, f64, i32, [], _, poequb);
dense_equil!(c, colmat::c32, f32, i32, [], _, poequb);
dense_equil!(z, colmat::c64, f64, i32, [], _, poequb);
dense_equil!(s, f32, f32, i64, [64_], _64_, poequb);
dense_equil!(d, f64, f64, i64, [64_], _64_, poequb);
dense_equil!(c, colmat::c32, f32, i64, [64_], _64_, poequb);
dense_equil!(z, colmat::c64, f64, i64, [64_], _64_, poequb);
packed!(s, f32, f32, i32, [], _);
packed!(d, f64, f64, i32, [], _);
packed!(c, colmat::c32, f32, i32, [], _);
packed!(z, colmat::c64, f64, i32, [], _);
packed!(s, f32, f32, i64, [64_], _64_);
packed!(d, f64, f64, i64, [64_], _64_);
packed!(c, colmat::c32, f32, i64, [64_], _64_);
packed!(z, colmat::c64, f64, i64, [64_], _64_);
