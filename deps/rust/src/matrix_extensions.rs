//! OpenBLAS Fortran matrix extensions using native strided copy/map views.
use crate::kernels::{flag, Scalar};
use colmat::{MatMut, MatRef};
use paste::paste;
pub(crate) unsafe fn matcopy<T: Scalar>(
    order: u8,
    tr: u8,
    m: usize,
    n: usize,
    alpha: T,
    a: *const T,
    la: isize,
    b: *mut T,
    lb: isize,
    inplace: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    let transpose = matches!(tr, b'T' | b'C');
    let conjugate = T::COMPLEX && matches!(tr, b'R' | b'C');
    let (rows, cols) = if transpose { (n, m) } else { (m, n) };
    let (br, bc) = if order == b'C' { (1, lb) } else { (lb, 1) };
    if !T::COMPLEX && alpha == T::z() {
        MatMut::from_raw_parts_mut(b, rows, cols, br, bc).fill(T::z());
        return;
    }
    let (ar, ac) = if order == b'C' { (1, la) } else { (la, 1) };
    let (ar, ac) = if transpose { (ac, ar) } else { (ar, ac) };
    let source = MatRef::from_raw_parts(a, rows, cols, ar, ac);
    // An owned native snapshot handles overlapping source/destination layouts,
    // including rectangular transposes and different leading dimensions.
    let owned = if inplace {
        Some(source.to_owned())
    } else {
        None
    };
    let src = match &owned {
        Some(matrix) => matrix.as_ref(),
        None => source,
    };
    let mut dst = MatMut::from_raw_parts_mut(b, rows, cols, br, bc);
    if !T::COMPLEX && alpha == T::o() {
        dst.copy_from(src);
    } else {
        colmat::zip!(dst, src).for_each(|colmat::unzip!(out, input)| {
            *out = alpha * if conjugate { input.cj() } else { *input }
        });
    }
}
pub(crate) unsafe fn geadd<T: Scalar>(
    m: usize,
    n: usize,
    alpha: T,
    a: *const T,
    la: isize,
    beta: T,
    b: *mut T,
    lb: isize,
) {
    if m == 0 || n == 0 || (alpha == T::z() && beta == T::o()) {
        return;
    }
    let mut dst = MatMut::from_raw_parts_mut(b, m, n, 1, lb);
    if alpha == T::z() {
        if beta == T::z() {
            dst.fill(T::z());
        } else if beta != T::o() {
            dst.for_each_mut(|x| *x = beta * *x);
        }
        return;
    }
    let src = MatRef::from_raw_parts(a, m, n, 1, la);
    if beta == T::z() {
        colmat::zip!(dst, src).for_each(|colmat::unzip!(out, input)| *out = alpha * *input);
    } else {
        colmat::zip!(dst, src)
            .for_each(|colmat::unzip!(out, input)| *out = alpha * *input + beta * *out);
    }
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! copy_check {($order:expr,$tr:expr,$m:expr,$n:expr,$la:expr,$lb:expr,$lbarg:expr)=>{
    check!(1=>matches!($order,b'C'|b'R'),2=>matches!($tr,b'N'|b'T'|b'R'|b'C'),3=>$m>=0,4=>$n>=0);
    let minor=if $order==b'C' {$m}else{$n};
    let output_minor=if matches!($tr,b'T'|b'C') {if $order==b'C' {$n}else{$m}}else{minor};
    check!(7=>$la>=minor.max(1),$lbarg=>$lb>=output_minor.max(1));
};}
macro_rules! exports {
 ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p omatcopy $suffix>](order:*const u8,tr:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,la:*const $i,b:*mut $t,lb:*const $i) {
    let(order,tr,m,n,la,lb)=(flag(order),flag(tr),*m as isize,*n as isize,*la as isize,*lb as isize);
    copy_check!(order,tr,m,n,la,lb,9);
    if m==0||n==0{return;}
    matcopy(order,tr,m as usize,n as usize,*alpha,a,la,b,lb,false);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p imatcopy $suffix>](order:*const u8,tr:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*mut $t,la:*const $i,lb:*const $i) {
    let(order,tr,m,n,la,lb)=(flag(order),flag(tr),*m as isize,*n as isize,*la as isize,*lb as isize);
    copy_check!(order,tr,m,n,la,lb,8);
    if m==0||n==0{return;}
    matcopy(order,tr,m as usize,n as usize,*alpha,a,la,a,lb,true);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p geadd $suffix>](m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,la:*const $i,beta:*const $t,b:*mut $t,lb:*const $i) {
    let(m,n,la,lb)=(*m as isize,*n as isize,*la as isize,*lb as isize);
    check!(1=>m>=0,2=>n>=0,5=>la>=m.max(1),8=>lb>=m.max(1));
    if m==0||n==0{return;}
    geadd(m as usize,n as usize,*alpha,a,la,*beta,b,lb);
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
    fn zero_coefficients_and_empty_layouts_need_no_input_storage() {
        unsafe {
            let mut output = [f64::NAN; 6];
            domatcopy_(
                b"C".as_ptr(),
                b"T".as_ptr(),
                &2,
                &3,
                &0.0,
                std::ptr::null(),
                &2,
                output.as_mut_ptr(),
                &3,
            );
            assert_eq!(output, [0.0; 6]);
            zgeadd_64_(
                &2,
                &3,
                &colmat::c64::new(0.0, 0.0),
                std::ptr::null(),
                &2,
                &colmat::c64::new(1.0, 0.0),
                std::ptr::null_mut(),
                &2,
            );
            simatcopy_(
                b"R".as_ptr(),
                b"N".as_ptr(),
                &0,
                &4,
                std::ptr::null(),
                std::ptr::null_mut(),
                &4,
                &4,
            );
        }
    }
}
