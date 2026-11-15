//! Structured storage around native generalized self-adjoint congruences.
use crate::generalized_selfadjoint::{self as reduction, GeneralizedScalar};
use crate::kernels::flag;
use crate::lapack::LapackInt;
use crate::storage_layout::TriangleStorage;
use colmat::Mat;

unsafe fn unpack<T: GeneralizedScalar>(
    n: usize,
    lower: bool,
    a: *const T,
    hermitian: bool,
) -> Mat<T> {
    let mut result = Mat::<T>::zeros(n, n);
    for j in 0..n {
        let (offset, stride, _) = TriangleStorage::Packed.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { n - j } else { j + 1 };
        for k in 0..count {
            let i = first + k;
            let p = a.offset(offset as isize + k as isize * stride);
            let value = if i == j { T::diagonal(p) } else { *p };
            result[(i, j)] = value;
            if hermitian && i != j {
                result[(j, i)] = value.cj();
            }
        }
    }
    result
}
unsafe fn packed_standardize<T: GeneralizedScalar>(
    kind: u8,
    lower: bool,
    n: usize,
    a: *mut T,
    b: *const T,
) {
    if n == 0 {
        return;
    }
    let matrix = unpack(n, lower, a, true);
    let factor = unpack(n, lower, b, false);
    let u = if lower { b'L' } else { b'U' };
    let result = reduction::with_lower(
        u,
        n,
        factor.as_ref().as_ptr(),
        factor.as_ref().col_stride(),
        |l| reduction::standardize(kind, matrix, l),
    );
    for j in 0..n {
        let (offset, stride, _) = TriangleStorage::Packed.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let count = if lower { n - j } else { j + 1 };
        for k in 0..count {
            *a.offset(offset as isize + k as isize * stride) = result[(first + k, j)];
        }
    }
}
unsafe fn packed_entry<T: GeneralizedScalar, I: LapackInt>(
    kind: isize,
    u: u8,
    n: isize,
    a: *mut T,
    b: *const T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let addressable = n >= 0
        && (n as usize)
            .checked_mul(n as usize)
            .and_then(|x| x.checked_mul(std::mem::size_of::<T>()))
            .is_some_and(|bytes| bytes <= isize::MAX as usize);
    for (arg, ok) in [
        (1, (1..=3).contains(&kind)),
        (2, matches!(u, b'U' | b'L')),
        (3, addressable),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    packed_standardize(kind as u8, u == b'L', n as usize, a, b);
}
macro_rules! packed_exports {($p:ident,$name:ident,$t:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p $name $s>](kind:*const $i,u:*const u8,n:*const $i,a:*mut $t,b:*const $t,info:*mut $i) {
        packed_entry((*kind).val(),flag(u),(*n).val(),a,b,info);
    }
}};}
packed_exports!(s, spgst, f32, i32, _);
packed_exports!(d, spgst, f64, i32, _);
packed_exports!(c, hpgst, colmat::c32, i32, _);
packed_exports!(z, hpgst, colmat::c64, i32, _);
packed_exports!(s, spgst, f32, i64, _64_);
packed_exports!(d, spgst, f64, i64, _64_);
packed_exports!(c, hpgst, colmat::c32, i64, _64_);
packed_exports!(z, hpgst, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn check<T: GeneralizedScalar, I: LapackInt>(phase: T, tol: f64) {
        for n in [0usize, 1, 4] {
            for lower in [false, true] {
                for kind in 1..=3 {
                    let l = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(2.0 + i as f64 * 0.2)
                        } else if i > j {
                            phase
                        } else {
                            T::z()
                        }
                    });
                    let a = Mat::<T>::from_fn(n, n, |i, j| {
                        if i == j {
                            T::real(3.0 + i as f64)
                        } else if i > j {
                            phase * T::real(0.3)
                        } else {
                            phase.cj() * T::real(0.3)
                        }
                    });
                    let mut ap = vec![T::real(777.0); n * (n + 1) / 2 + 2];
                    let mut bp = ap.clone();
                    for j in 0..n {
                        let (offset, _, _) = TriangleStorage::Packed.segment(n, lower, j);
                        let first = if lower { j } else { 0 };
                        let count = if lower { n - j } else { j + 1 };
                        for k in 0..count {
                            let i = first + k;
                            let imaginary = (phase - phase.cj()) * T::real(777.0);
                            ap[1 + offset + k] =
                                a[(i, j)] + if i == j { imaginary } else { T::z() };
                            bp[1 + offset + k] = (if lower { l[(i, j)] } else { l[(j, i)].cj() })
                                + if i == j { imaginary } else { T::z() };
                        }
                    }
                    let oldb = bp.clone();
                    let mut info = I::from_i64(-777);
                    unsafe {
                        packed_entry(
                            kind,
                            if lower { b'L' } else { b'U' },
                            n as isize,
                            ap.as_mut_ptr().add(1),
                            bp.as_ptr().add(1),
                            &mut info,
                        );
                    }
                    assert_eq!(info.val(), 0);
                    assert_eq!(crate::colmat_blas_last_error(), 0);
                    assert_eq!(bp, oldb);
                    assert_eq!(ap[0].re(), 777.0);
                    assert_eq!(ap[ap.len() - 1].re(), 777.0);
                    let c = unsafe { unpack(n, lower, ap.as_ptr().add(1), true) };
                    for j in 0..n {
                        for i in 0..n {
                            let mut value = T::z();
                            for k in 0..n {
                                for m in 0..n {
                                    value = value
                                        + if kind == 1 {
                                            l[(i, k)] * c[(k, m)] * l[(j, m)].cj()
                                        } else {
                                            l[(k, i)].cj() * a[(k, m)] * l[(m, j)]
                                        };
                                }
                            }
                            let expected = if kind == 1 { a[(i, j)] } else { c[(i, j)] };
                            assert!((value - expected).magnitude() < tol);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn packed_congruences_all_scalar_integer_types() {
        check::<f32, i32>(0.2, 1e-4);
        check::<f32, i64>(0.2, 1e-4);
        check::<f64, i32>(0.2, 1e-11);
        check::<f64, i64>(0.2, 1e-11);
        check::<colmat::c32, i32>(colmat::c32::new(0.2, 0.3), 1e-4);
        check::<colmat::c32, i64>(colmat::c32::new(0.2, 0.3), 1e-4);
        check::<colmat::c64, i32>(colmat::c64::new(0.2, 0.3), 1e-11);
        check::<colmat::c64, i64>(colmat::c64::new(0.2, 0.3), 1e-11);
    }
}
