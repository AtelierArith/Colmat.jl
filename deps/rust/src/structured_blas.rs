//! Band BLAS composed from native products over the stored column intervals.
use crate::kernels::{flag, Scalar};
use colmat::{Accum, Conj, MatMut, MatRef};
use paste::paste;

fn vector_start(n: isize, inc: isize) -> Option<isize> {
    let span = (n - 1).checked_mul(inc)?;
    if inc < 0 {
        span.checked_neg()
    } else {
        Some(0)
    }
}
unsafe fn gbmv<T: Scalar>(
    tr: u8,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: T,
    a: *const T,
    ld: usize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
) {
    let (nx, ny) = if tr == b'N' { (n, m) } else { (m, n) };
    if alpha == T::z() && beta == T::o() {
        return;
    }
    let Some(y_start) = vector_start(ny as isize, iy) else {
        crate::invalid(13);
        return;
    };
    let x_start = if alpha == T::z() {
        0
    } else {
        let Some(start) = vector_start(nx as isize, ix) else {
            crate::invalid(10);
            return;
        };
        start
    };
    let ix = if nx == 1 { 1 } else { ix };
    let iy = if ny == 1 { 1 } else { iy };
    let y = y.offset(y_start);
    if beta != T::o() {
        let mut output = MatMut::from_raw_parts_mut(y, ny, 1, iy, 0);
        if beta == T::z() {
            output.fill(T::z());
        } else {
            output.for_each_mut(|v| *v = beta * *v);
        }
    }
    if alpha == T::z() {
        return;
    }
    let x = x.offset(x_start);
    for j in 0..n {
        let first = j.saturating_sub(ku).min(m);
        let end = j.saturating_add(kl).saturating_add(1).min(m);
        if first == end {
            if tr != b'N' {
                let increment = alpha * T::z();
                MatMut::from_raw_parts_mut(y.offset(j as isize * iy), 1, 1, 1, 0)
                    .for_each_mut(|v| *v = *v + increment);
            }
            continue;
        }
        if tr == b'N' {
            let value = *x.offset(j as isize * ix);
            if value == T::z() {
                continue;
            }
            let column = a.add(ku + first - j + j * ld);
            let coefficient = alpha * value;
            let lhs = MatRef::from_raw_parts(column, end - first, 1, 1, 0);
            let rhs = MatRef::from_raw_parts(&coefficient, 1, 1, 1, 0);
            let dst =
                MatMut::from_raw_parts_mut(y.offset(first as isize * iy), end - first, 1, iy, 0);
            crate::threading::with_par(|par| {
                colmat::linalg::matmul::matmul_with_conj(
                    dst,
                    Accum::Add,
                    lhs,
                    Conj::No,
                    rhs,
                    Conj::No,
                    T::o(),
                    par,
                )
            });
        } else {
            let column = a.add(ku + first - j + j * ld);
            let lhs = MatRef::from_raw_parts(column, 1, end - first, 0, 1);
            let rhs = MatRef::from_raw_parts(x.offset(first as isize * ix), end - first, 1, ix, 0);
            let dst = MatMut::from_raw_parts_mut(y.offset(j as isize * iy), 1, 1, 1, 0);
            crate::threading::with_par(|par| {
                colmat::linalg::matmul::matmul_with_conj(
                    dst,
                    Accum::Add,
                    lhs,
                    if tr == b'C' { Conj::Yes } else { Conj::No },
                    rhs,
                    Conj::No,
                    alpha,
                    par,
                )
            });
        }
    }
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p gbmv $suffix>](tr:*const u8,m:*const $i,n:*const $i,kl:*const $i,ku:*const $i,
    alpha:*const $t,a:*const $t,ld:*const $i,x:*const $t,ix:*const $i,beta:*const $t,y:*mut $t,iy:*const $i) {
    let(tr,m,n,kl,ku,ld,ix,iy)=(flag(tr),*m as isize,*n as isize,*kl as isize,*ku as isize,*ld as isize,*ix as isize,*iy as isize);
    check!(1=>matches!(tr,b'N'|b'T'|b'C'),2=>m>=0,3=>n>=0,4=>kl>=0,5=>ku>=0,
        8=>kl.checked_add(ku).and_then(|v|v.checked_add(1)).is_some_and(|required|ld>=required),10=>ix!=0,13=>iy!=0);
    if m==0||n==0 {return;}
    gbmv(tr,m as usize,n as usize,kl as usize,ku as usize,*alpha,a,ld as usize,x,ix,*beta,y,iy);
  }
 }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_and_zero_coefficient_calls_need_no_unused_arrays() {
        unsafe {
            dgbmv_(
                b"N".as_ptr(),
                &3,
                &0,
                &0,
                &0,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
                &1,
            );
            dgbmv_(
                b"N".as_ptr(),
                &3,
                &2,
                &1,
                &1,
                &0.0,
                std::ptr::null(),
                &3,
                std::ptr::null(),
                &1,
                &1.0,
                std::ptr::null_mut(),
                &1,
            );
            let mut y = [777.0; 3];
            dgbmv_(
                b"N".as_ptr(),
                &3,
                &2,
                &1,
                &1,
                &0.0,
                std::ptr::null(),
                &3,
                std::ptr::null(),
                &1,
                &0.0,
                y.as_mut_ptr(),
                &1,
            );
            assert_eq!(y, [0.0; 3]);
            y.fill(777.0);
            dgbmv_(
                b"N".as_ptr(),
                &3,
                &2,
                &1,
                &1,
                &1.0,
                std::ptr::null(),
                &3,
                [0.0, 0.0].as_ptr(),
                &1,
                &1.0,
                y.as_mut_ptr(),
                &1,
            );
            assert_eq!(y, [777.0; 3]);
        }
    }
    #[test]
    fn empty_transposed_columns_still_evaluate_alpha_times_zero() {
        unsafe {
            let mut y = [777.0; 3];
            dgbmv_(
                b"T".as_ptr(),
                &1,
                &3,
                &0,
                &0,
                &f64::INFINITY,
                [1.0, f64::NAN, f64::NAN].as_ptr(),
                &1,
                [1.0].as_ptr(),
                &1,
                &0.0,
                y.as_mut_ptr(),
                &1,
            );
            assert_eq!(y[0], f64::INFINITY);
            assert!(y[1].is_nan() && y[2].is_nan());
        }
    }
    #[test]
    fn integer_overflow_is_diagnosed_before_output_changes() {
        unsafe {
            let mut y = [777.0; 2];
            dgbmv_64_(
                b"N".as_ptr(),
                &2,
                &3,
                &0,
                &0,
                &1.0,
                std::ptr::null(),
                &1,
                std::ptr::null(),
                &i64::MAX,
                &0.0,
                y.as_mut_ptr(),
                &1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 10);
            assert_eq!(y, [777.0; 2]);
            dgbmv_64_(
                b"N".as_ptr(),
                &0,
                &0,
                &i64::MAX,
                &0,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
                &1,
            );
            assert_eq!(crate::colmat_blas_last_error(), 8);
        }
    }
    #[test]
    fn singleton_vectors_accept_the_smallest_integer_increment() {
        unsafe {
            let mut y = [777.0];
            dgbmv_64_(
                b"C".as_ptr(),
                &1,
                &1,
                &0,
                &0,
                &1.0,
                [2.0].as_ptr(),
                &1,
                [3.0].as_ptr(),
                &i64::MIN,
                &0.0,
                y.as_mut_ptr(),
                &i64::MIN,
            );
            assert_eq!(y, [6.0]);
        }
    }
}
