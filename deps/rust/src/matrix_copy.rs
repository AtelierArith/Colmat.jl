//! LACPY/LASET: only the selected entries are borrowed by native copy/fill views.
use crate::kernels::{flag, Scalar};
use colmat::{MatMut, MatRef};
use paste::paste;

fn selected(u: u8, m: usize, j: usize) -> (usize, usize) {
    match u {
        b'U' => (0, m.min(j + 1)),
        b'L' => (m.min(j), m),
        _ => (0, m),
    }
}

unsafe fn copy<T: Scalar>(
    u: u8,
    m: usize,
    n: usize,
    a: *const T,
    lda: usize,
    b: *mut T,
    ldb: usize,
) {
    for j in 0..n {
        let (start, end) = selected(u, m, j);
        if start == end {
            continue;
        }
        let src = MatRef::from_raw_parts(a.add(start + j * lda), end - start, 1, 1, 0);
        let mut dst = MatMut::from_raw_parts_mut(b.add(start + j * ldb), end - start, 1, 1, 0);
        dst.copy_from(src);
    }
}

unsafe fn set<T: Scalar>(u: u8, m: usize, n: usize, alpha: T, beta: T, a: *mut T, lda: usize) {
    for j in 0..n {
        let (start, end) = selected(u, m, j);
        // Split at the diagonal so even a transient off-diagonal fill never
        // touches an entry excluded by UPLO or reads the previous destination.
        let left_end = end.min(j);
        if start < left_end {
            MatMut::from_raw_parts_mut(a.add(start + j * lda), left_end - start, 1, 1, 0)
                .fill(alpha);
        }
        let right_start = start.max(j + 1);
        if right_start < end {
            MatMut::from_raw_parts_mut(a.add(right_start + j * lda), end - right_start, 1, 1, 0)
                .fill(alpha);
        }
        if j < m {
            *a.add(j + j * lda) = beta;
        }
    }
}

macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$suffix:tt) => {
        paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<$p lacpy $suffix>](u:*const u8,m:*const $i,n:*const $i,
                a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i) {
                let (m,n,la,lb)=(*m as isize,*n as isize,*lda as isize,*ldb as isize);
                check!(2=>m>=0,3=>n>=0,5=>la>=m.max(1),7=>lb>=m.max(1));
                copy(flag(u),m as usize,n as usize,a,la as usize,b,lb as usize);
            }
            #[no_mangle]
            pub unsafe extern "C" fn [<$p laset $suffix>](u:*const u8,m:*const $i,n:*const $i,
                alpha:*const $t,beta:*const $t,a:*mut $t,lda:*const $i) {
                let (m,n,ld)=(*m as isize,*n as isize,*lda as isize);
                check!(2=>m>=0,3=>n>=0,7=>ld>=m.max(1));
                if m==0 || n==0 {return;}
                set(flag(u),m as usize,n as usize,*alpha,*beta,a,ld as usize);
            }
        }
    };
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);
