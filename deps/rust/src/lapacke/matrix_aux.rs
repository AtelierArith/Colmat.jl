//! LAPACKE matrix copy/fill/row-swap adapters.
//! Reference contracts: Reference LAPACK v3.12.1, LAPACKE/src/
//! lapacke_{s,d,c,z}{lacpy,laset,laswp}{,_work}.c.
//!
//! LACPY and LASET convert only the logical M-by-N rectangle between row- and
//! column-major storage. LAPACK's corresponding wrappers derive the destination
//! row count from UPLO and stop at the selected triangle, so the direct loops
//! below do the same and never read or write caller padding. LASWP swaps whole
//! rows of the row-major view, which is the transpose of the stored matrix.
use super::{ge_nancheck, LAPACKE_get_nancheck, COL_MAJOR, ROW_MAJOR};
use crate::kernels::Scalar;

fn selected(u: u8, i: usize, j: usize) -> bool {
    match u {
        b'U' => i <= j,
        b'L' => i >= j,
        _ => true,
    }
}

unsafe fn copy<T: Scalar>(
    layout: i32,
    u: u8,
    m: usize,
    n: usize,
    a: *const T,
    lda: usize,
    b: *mut T,
    ldb: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            if !selected(u, i, j) {
                continue;
            }
            let source = if layout == ROW_MAJOR {
                *a.add(i * lda + j)
            } else {
                *a.add(i + j * lda)
            };
            if layout == ROW_MAJOR {
                *b.add(i * ldb + j) = source;
            } else {
                *b.add(i + j * ldb) = source;
            }
        }
    }
}

unsafe fn set<T: Scalar>(
    layout: i32,
    u: u8,
    m: usize,
    n: usize,
    alpha: T,
    beta: T,
    a: *mut T,
    lda: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            if !selected(u, i, j) {
                continue;
            }
            let value = if i == j { beta } else { alpha };
            if layout == ROW_MAJOR {
                *a.add(i * lda + j) = value;
            } else {
                *a.add(i + j * lda) = value;
            }
        }
    }
}

unsafe fn swap_rows<T: Scalar>(n: usize, a: *mut T, lda: usize, swaps: &[(i64, i64)]) {
    for &(i, pivot) in swaps {
        if pivot == i {
            continue;
        }
        let (r0, r1) = ((i - 1) as usize, (pivot - 1) as usize);
        for j in 0..n {
            let (x, y) = (a.add(r0 * lda + j), a.add(r1 * lda + j));
            core::ptr::swap(x, y);
        }
    }
}

macro_rules! aux {
    ($p:ident,$t:ty,$i:ty,[$($suffix:tt)*],$fortran:tt)=>{paste::paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacpy $($suffix)*>](
            layout:i32,u:u8,m:$i,n:$i,a:*const $t,lda:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                match ge_nancheck(layout,m as i64,n as i64,a,lda as i64) {
                    Ok(true)=>return -5,Err(info)=>return info as $i,Ok(false)=>{},
                }
            }
            [<LAPACKE_ $p lacpy_work $($suffix)*>](layout,u,m,n,a,lda,b,ldb)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p lacpy_work $($suffix)*>](
            layout:i32,u:u8,m:$i,n:$i,a:*const $t,lda:$i,b:*mut $t,ldb:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR {
                if lda<n {return -6;}
                if ldb<n {return -8;}
                if m<0 || n<0 {return 0;}
                copy(layout,u.to_ascii_uppercase(),m as usize,n as usize,a,lda as usize,b,ldb as usize);
                return 0;
            }
            crate::matrix_copy::[<$p lacpy $fortran>](
                &u,&m,&n,a,&lda,b,&ldb,
            );
            0
        }

        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p laset $($suffix)*>](
            layout:i32,u:u8,m:$i,n:$i,alpha:$t,beta:$t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if LAPACKE_get_nancheck()!=0 {
                if alpha!=alpha {return -5;}
                if beta!=beta {return -6;}
            }
            [<LAPACKE_ $p laset_work $($suffix)*>](layout,u,m,n,alpha,beta,a,lda)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p laset_work $($suffix)*>](
            layout:i32,u:u8,m:$i,n:$i,alpha:$t,beta:$t,a:*mut $t,lda:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR {
                if lda<n {return -8;}
                if m<0 || n<0 {return 0;}
                set(layout,u.to_ascii_uppercase(),m as usize,n as usize,alpha,beta,a,lda as usize);
                return 0;
            }
            crate::matrix_copy::[<$p laset $fortran>](
                &u,&m,&n,&alpha,&beta,a,&lda,
            );
            0
        }

        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p laswp $($suffix)*>](
            layout:i32,n:$i,a:*mut $t,lda:$i,k1:$i,k2:$i,ipiv:*const $i,incx:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            [<LAPACKE_ $p laswp_work $($suffix)*>](layout,n,a,lda,k1,k2,ipiv,incx)
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<LAPACKE_ $p laswp_work $($suffix)*>](
            layout:i32,n:$i,a:*mut $t,lda:$i,k1:$i,k2:$i,ipiv:*const $i,incx:$i,
        )->$i {
            if layout!=ROW_MAJOR && layout!=COL_MAJOR {return -1;}
            if layout==ROW_MAJOR {
                if lda<n {return -4;}
                if n<=0 || incx==0 || ipiv.is_null() {return 0;}
                // The reference wrapper transposes as many rows as the largest
                // pivot index and then swaps the resulting rows. A direct swap
                // of N entries has the same effect without a temporary matrix.
                let mut swaps:Vec<(i64,i64)>=Vec::new();
                if incx>0 {
                    let mut ix=k1;
                    let mut i=k1;
                    while i<=k2 {
                        swaps.push((i as i64,*ipiv.offset((ix-1) as isize) as i64));
                        ix+=incx;
                        i+=1;
                    }
                } else {
                    let ix0=k1+(k1-k2)*incx;
                    let mut i=k2;
                    while i>=k1 {
                        let ix=ix0+(k2-i)*incx;
                        swaps.push((i as i64,*ipiv.offset((ix-1) as isize) as i64));
                        i-=1;
                    }
                }
                if swaps.iter().any(|&(_,p)| p<1) {return 0;}
                swap_rows(n as usize,a,lda as usize,&swaps);
                return 0;
            }
            crate::matrix_permutation::[<$p laswp $fortran>](
                &n,a,&lda,&k1,&k2,ipiv,&incx,
            );
            0
        }
    }};
}
aux!(s, f32, i32, [], _);
aux!(d, f64, i32, [], _);
aux!(c, colmat::c32, i32, [], _);
aux!(z, colmat::c64, i32, [], _);
aux!(s, f32, i64, [64_], _64_);
aux!(d, f64, i64, [64_], _64_);
aux!(c, colmat::c32, i64, [64_], _64_);
aux!(z, colmat::c64, i64, [64_], _64_);
