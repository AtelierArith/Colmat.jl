//! Packed rank updates over native column products, without dense expansion.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, Scalar};
use crate::storage_layout::TriangleStorage;
use crate::structured_blas::vector_start;
use colmat::{Accum, Conj, MatMut, MatRef};
use paste::paste;

unsafe fn add_column<T: Scalar>(a: *mut T, length: usize, v: *const T, inc: isize, coefficient: T) {
    if length == 0 {
        return;
    }
    let lhs = MatRef::from_raw_parts(v, length, 1, inc, 0);
    let rhs = MatRef::from_raw_parts(&coefficient, 1, 1, 1, 0);
    let dst = MatMut::from_raw_parts_mut(a, length, 1, 1, 0);
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
}
pub(crate) unsafe fn update<T: GeneralizedScalar + ReflectorScalar>(
    lower: bool,
    n: usize,
    alpha: T,
    x: *const T,
    ix: isize,
    second: Option<(*const T, isize)>,
    a: *mut T,
    hermitian: bool,
) {
    if alpha == T::z() {
        return;
    }
    let Some(xstart) = vector_start(n as isize, ix) else {
        crate::invalid(5);
        return;
    };
    let second = match second {
        Some((y, iy)) => {
            let Some(start) = vector_start(n as isize, iy) else {
                crate::invalid(7);
                return;
            };
            Some((y.offset(start), if n == 1 { 1 } else { iy }))
        }
        None => None,
    };
    let x = x.offset(xstart);
    let ix = if n == 1 { 1 } else { ix };
    for j in 0..n {
        let (base, _, _) = TriangleStorage::Packed.segment(n, lower, j);
        let diagonal = base + if lower { 0 } else { j };
        let xj = *x.offset(j as isize * ix);
        let yj = match second {
            Some((y, iy)) => *y.offset(j as isize * iy),
            None => T::z(),
        };
        if xj == T::z() && yj == T::z() {
            if hermitian {
                let p = a.add(diagonal);
                *p = T::diagonal(p);
            }
            continue;
        }
        let first_coefficient = if second.is_some() {
            alpha * if hermitian { yj.cj() } else { yj }
        } else if hermitian {
            // HPR has a real alpha. Scale the two components independently
            // before forming the real diagonal update, preserving its range.
            T::parts(alpha.re() * xj.re(), -alpha.re() * xj.im())
        } else {
            alpha * xj
        };
        let second_coefficient = if second.is_none() {
            T::z()
        } else if hermitian {
            (alpha * xj).cj()
        } else {
            alpha * xj
        };
        let (first, end, offset) = if hermitian {
            if lower {
                (j + 1, n, base + 1)
            } else {
                (0, j, base)
            }
        } else if lower {
            (j, n, base)
        } else {
            (0, j + 1, base)
        };
        if first < end {
            add_column(
                a.add(offset),
                end - first,
                x.offset(first as isize * ix),
                ix,
                first_coefficient,
            );
            if let Some((y, iy)) = second {
                add_column(
                    a.add(offset),
                    end - first,
                    y.offset(first as isize * iy),
                    iy,
                    second_coefficient,
                );
            }
        }
        if hermitian {
            let mut increment = xj * first_coefficient;
            if second.is_some() {
                increment = increment + yj * second_coefficient;
            }
            let p = a.add(diagonal);
            *p = T::diagonal(p) + T::real(increment.re());
        }
    }
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! symmetric_exports {
 ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p spr $suffix>](u:*const u8,n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,a:*mut $t) {
    let(u,n,ix)=(flag(u),*n as isize,*ix as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>n>=0,5=>ix!=0);
    if n==0 {return;}
    update(u==b'L',n as usize,*alpha,x,ix,None,a,false);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p spr2 $suffix>](u:*const u8,n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i,a:*mut $t) {
    let(u,n,ix,iy)=(flag(u),*n as isize,*ix as isize,*iy as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>n>=0,5=>ix!=0,7=>iy!=0);
    if n==0 {return;}
    update(u==b'L',n as usize,*alpha,x,ix,Some((y,iy)),a,false);
  }
 }};
}
macro_rules! hermitian_exports {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p hpr $suffix>](u:*const u8,n:*const $i,alpha:*const $r,x:*const $t,ix:*const $i,a:*mut $t) {
    let(u,n,ix)=(flag(u),*n as isize,*ix as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>n>=0,5=>ix!=0);
    if n==0 {return;}
    update(u==b'L',n as usize,<$t>::real(*alpha as f64),x,ix,None,a,true);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p hpr2 $suffix>](u:*const u8,n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i,a:*mut $t) {
    let(u,n,ix,iy)=(flag(u),*n as isize,*ix as isize,*iy as isize);
    check!(1=>matches!(u,b'U'|b'L'),2=>n>=0,5=>ix!=0,7=>iy!=0);
    if n==0 {return;}
    update(u==b'L',n as usize,*alpha,x,ix,Some((y,iy)),a,true);
  }
 }};
}
symmetric_exports!(s, f32, i32, _);
symmetric_exports!(d, f64, i32, _);
symmetric_exports!(c, colmat::c32, i32, _);
symmetric_exports!(z, colmat::c64, i32, _);
symmetric_exports!(s, f32, i64, _64_);
symmetric_exports!(d, f64, i64, _64_);
symmetric_exports!(c, colmat::c32, i64, _64_);
symmetric_exports!(z, colmat::c64, i64, _64_);
hermitian_exports!(c, colmat::c32, f32, i32, _);
hermitian_exports!(z, colmat::c64, f64, i32, _);
hermitian_exports!(c, colmat::c32, f32, i64, _64_);
hermitian_exports!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_zero_alpha_and_zero_columns_need_no_unused_arrays() {
        unsafe {
            sspr_(
                b"U".as_ptr(),
                &0,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
            );
            dspr_64_(
                b"U".as_ptr(),
                &i64::MAX,
                &0.0,
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
            );
            zhpr_64_(
                b"L".as_ptr(),
                &2,
                &0.0,
                std::ptr::null(),
                &i64::MIN,
                std::ptr::null_mut(),
            );
            dspr_(
                b"U".as_ptr(),
                &2,
                &1.0,
                [0.0, 0.0].as_ptr(),
                &1,
                std::ptr::null_mut(),
            );
        }
    }
    #[test]
    fn hermitian_zero_columns_clear_only_diagonal_imaginary_parts() {
        unsafe {
            use colmat::c64;
            let mut a = [
                c64::new(3.0, f64::NAN),
                c64::new(f64::NAN, f64::NAN),
                c64::new(5.0, f64::NAN),
            ];
            zhpr_64_(
                b"L".as_ptr(),
                &2,
                &1.0,
                [c64::new(0.0, 0.0); 2].as_ptr(),
                &1,
                a.as_mut_ptr(),
            );
            assert_eq!(a[0], c64::new(3.0, 0.0));
            assert_eq!(a[2], c64::new(5.0, 0.0));
            assert!(a[1].re.is_nan() && a[1].im.is_nan());
        }
    }
    #[test]
    fn diagonal_conjugation_and_real_alpha_contracts() {
        unsafe {
            use colmat::{c32, c64};
            let mut single = [c32::new(3.0, f32::NAN)];
            chpr_(
                b"U".as_ptr(),
                &1,
                &0.5,
                [c32::new(2.0, 1.0)].as_ptr(),
                &1,
                single.as_mut_ptr(),
            );
            assert_eq!(single, [c32::new(5.5, 0.0)]);
            let mut hermitian = [c64::new(3.0, f64::NAN)];
            zhpr2_64_(
                b"L".as_ptr(),
                &1,
                &c64::new(1.0, 2.0),
                [c64::new(2.0, 1.0)].as_ptr(),
                &i64::MIN,
                [c64::new(3.0, -1.0)].as_ptr(),
                &i64::MIN,
                hermitian.as_mut_ptr(),
            );
            assert_eq!(hermitian, [c64::new(-7.0, 0.0)]);
            let mut symmetric = [c64::new(3.0, 4.0)];
            zspr2_64_(
                b"L".as_ptr(),
                &1,
                &c64::new(1.0, 2.0),
                [c64::new(2.0, 1.0)].as_ptr(),
                &1,
                [c64::new(3.0, -1.0)].as_ptr(),
                &1,
                symmetric.as_mut_ptr(),
            );
            assert_eq!(symmetric, [c64::new(13.0, 34.0)]);
        }
    }
    #[test]
    fn integer_width_and_stride_failures_precede_any_update() {
        unsafe {
            let mut a = [777.0; 6];
            dspr2_64_(
                b"U".as_ptr(),
                &3,
                &1.0,
                [1.0; 3].as_ptr(),
                &1,
                std::ptr::null(),
                &i64::MAX,
                a.as_mut_ptr(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 7);
            assert_eq!(a, [777.0; 6]);
            dspr_64_(
                b"U".as_ptr(),
                &(-(1_i64 << 32) + 3),
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 2);
        }
    }
}
