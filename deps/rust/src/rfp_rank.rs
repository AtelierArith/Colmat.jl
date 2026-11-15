//! SFRK/HFRK: RFP adaptation around native selected-triangle rank-k products.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, Scalar};
use crate::storage_layout::TriangleStorage;
use colmat::{Accum, Conj, Mat, MatMut, MatRef};
use paste::paste;

unsafe fn read_triangle<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    c: *const T,
    work: &mut Mat<T>,
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let (first, length, skip) = if lower {
            (j + 1, n - j - 1, 1)
        } else {
            (0, j, 0)
        };
        if length > 0 {
            let source = MatRef::from_raw_parts(
                c.offset(offset as isize + skip * stride),
                length,
                1,
                stride,
                0,
            );
            let mut target = work.as_mut().submatrix_mut(first, j, length, 1);
            if T::COMPLEX && conjugated {
                target.copy_from(source.conjugate());
            } else {
                target.copy_from(source);
            }
        }
        let diagonal = offset as isize + if lower { 0 } else { j as isize * stride };
        work[(j, j)] = if T::COMPLEX {
            T::diagonal(c.offset(diagonal))
        } else {
            *c.offset(diagonal)
        };
    }
}
unsafe fn write_triangle<T: Scalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    c: *mut T,
    work: &Mat<T>,
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let (first, length, skip) = if lower {
            (j + 1, n - j - 1, 1)
        } else {
            (0, j, 0)
        };
        if length > 0 {
            let source = work.as_ref().submatrix(first, j, length, 1);
            let mut target = MatMut::from_raw_parts_mut(
                c.offset(offset as isize + skip * stride),
                length,
                1,
                stride,
                0,
            );
            if T::COMPLEX && conjugated {
                target.copy_from(source.conjugate());
            } else {
                target.copy_from(source);
            }
        }
        let diagonal = offset as isize + if lower { 0 } else { j as isize * stride };
        // Every physical Hermitian diagonal is explicitly real, independent
        // of the RFP folded block's conjugation.
        *c.offset(diagonal) = if T::COMPLEX {
            T::real(work[(j, j)].re())
        } else {
            work[(j, j)]
        };
    }
}
unsafe fn rank<T: GeneralizedScalar + ReflectorScalar>(
    transposed_rfp: bool,
    lower: bool,
    tr: u8,
    n: usize,
    k: usize,
    alpha: f64,
    a: *const T,
    ld: isize,
    beta: f64,
    c: *mut T,
) {
    let zero_product = alpha == 0.0 || k == 0;
    if zero_product && beta == 1.0 {
        return;
    }
    let Some(next) = n.checked_add(1) else {
        crate::invalid(4);
        return;
    };
    let count = if n % 2 == 0 {
        (n / 2).checked_mul(next)
    } else {
        n.checked_mul(next / 2)
    };
    let Some(count) = count.filter(|count| {
        count
            .checked_mul(std::mem::size_of::<T>())
            .is_some_and(|bytes| bytes <= isize::MAX as usize)
    }) else {
        crate::invalid(4);
        return;
    };
    if zero_product && beta == 0.0 {
        MatMut::from_raw_parts_mut(c, count, 1, 1, 0).fill(T::z());
        return;
    }
    let layout = TriangleStorage::Rfp {
        transposed: transposed_rfp,
    };
    let mut work = Mat::<T>::zeros(n, n);
    if beta != 0.0 {
        read_triangle(n, lower, layout, c, &mut work);
        if beta != 1.0 {
            for j in 0..n {
                let (first, length) = if lower { (j, n - j) } else { (0, j + 1) };
                work.as_mut()
                    .submatrix_mut(first, j, length, 1)
                    .for_each_mut(|v| *v = T::parts(v.re() * beta, v.im() * beta));
            }
        }
    }
    if !zero_product {
        let lhs = MatRef::from_raw_parts(
            a,
            n,
            k,
            if tr == b'N' { 1 } else { ld },
            if tr == b'N' { ld } else { 1 },
        );
        let conj_l = T::COMPLEX && tr == b'C';
        let conj_r = T::COMPLEX && !conj_l;
        use colmat::linalg::matmul::triangular::{matmul_with_conj, BlockStructure::*};
        crate::threading::with_par(|par| {
            matmul_with_conj(
                work.as_mut(),
                if lower {
                    TriangularLower
                } else {
                    TriangularUpper
                },
                if beta == 0.0 {
                    Accum::Replace
                } else {
                    Accum::Add
                },
                lhs,
                Rectangular,
                if conj_l { Conj::Yes } else { Conj::No },
                lhs.transpose(),
                Rectangular,
                if conj_r { Conj::Yes } else { Conj::No },
                T::real(alpha),
                par,
            )
        });
    }
    write_triangle(n, lower, layout, c, &work);
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$name:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p $name $suffix>](trr:*const u8,u:*const u8,tr:*const u8,n:*const $i,k:*const $i,
      alpha:*const $r,a:*const $t,ld:*const $i,beta:*const $r,c:*mut $t) {
    let(trr,u,tr,n,k,ld)=(flag(trr),flag(u),flag(tr),*n as isize,*k as isize,*ld as isize);
    check!(1=>(trr==b'N'||trr==if <$t>::COMPLEX {b'C'}else{b'T'}),2=>matches!(u,b'U'|b'L'),
        3=>(tr==b'N'||tr==if <$t>::COMPLEX {b'C'}else{b'T'}),4=>n>=0,5=>k>=0,8=>ld>=(if tr==b'N' {n}else{k}).max(1));
    if n==0 {return;}
    rank(trr!=b'N',u==b'L',tr,n as usize,k as usize,*alpha as f64,a,ld,*beta as f64,c);
  }
 }};
}
exports!(s, sfrk, f32, f32, i32, _);
exports!(d, sfrk, f64, f64, i32, _);
exports!(c, hfrk, colmat::c32, f32, i32, _);
exports!(z, hfrk, colmat::c64, f64, i32, _);
exports!(s, sfrk, f32, f32, i64, _64_);
exports!(d, sfrk, f64, f64, i64, _64_);
exports!(c, hfrk, colmat::c32, f32, i64, _64_);
exports!(z, hfrk, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_and_noop_calls_need_no_array_storage() {
        unsafe {
            ssfrk_(
                b"N".as_ptr(),
                b"L".as_ptr(),
                b"N".as_ptr(),
                &0,
                &3,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            let large = 1_i64 << 40;
            zhfrk_64_(
                b"C".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &large,
                &1,
                &0.0,
                std::ptr::null(),
                &large,
                &1.0,
                std::ptr::null_mut(),
            );
        }
    }
    #[test]
    fn zero_beta_never_reads_old_rfp_and_zero_product_never_reads_a() {
        unsafe {
            use colmat::c64;
            let mut c = [c64::new(f64::NAN, f64::NAN); 3];
            zhfrk_64_(
                b"N".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &2,
                &1,
                &0.0,
                std::ptr::null(),
                &2,
                &0.0,
                c.as_mut_ptr(),
            );
            assert!(c.iter().all(|z| z.re.to_bits() == 0 && z.im.to_bits() == 0));
            c[0] = c64::new(3.0, f64::NAN);
            zhfrk_64_(
                b"C".as_ptr(),
                b"L".as_ptr(),
                b"N".as_ptr(),
                &1,
                &0,
                &1.0,
                std::ptr::null(),
                &1,
                &0.5,
                c.as_mut_ptr(),
            );
            assert_eq!(c[0], c64::new(1.5, 0.0));
            let mut real = [f64::NAN; 3];
            dsfrk_64_(
                b"N".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &2,
                &1,
                &1.0,
                [1.0, 2.0].as_ptr(),
                &2,
                &0.0,
                real.as_mut_ptr(),
            );
            assert_eq!(real, [2.0, 4.0, 1.0]);
        }
    }
    #[test]
    fn flags_width_and_leading_dimensions_are_checked_before_arrays() {
        unsafe {
            dsfrk_64_(
                b"C".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &2,
                &1,
                std::ptr::null(),
                std::ptr::null(),
                &2,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 1);
            zhfrk_64_(
                b"N".as_ptr(),
                b"L".as_ptr(),
                b"T".as_ptr(),
                &2,
                &1,
                std::ptr::null(),
                std::ptr::null(),
                &2,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 3);
            dsfrk_64_(
                b"N".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &(-(1_i64 << 32) + 3),
                &0,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 4);
            dsfrk_64_(
                b"N".as_ptr(),
                b"U".as_ptr(),
                b"N".as_ptr(),
                &2,
                &1,
                std::ptr::null(),
                std::ptr::null(),
                &1,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 8);
        }
    }
}
