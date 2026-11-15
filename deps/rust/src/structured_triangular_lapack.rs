//! Owned triangle conversion for packed/RFP LAPACK triangular operations.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::storage_layout::TriangleStorage;
use colmat::{Conj, MatMut, MatRef};
use paste::paste;

unsafe fn read<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *const T,
) -> Vec<T> {
    let mut dense = vec![T::z(); n * n];
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            dense[i + j * n] = if unit && i == j {
                T::o()
            } else {
                let value = *a.offset(offset as isize + k as isize * stride);
                if conjugated {
                    value.cj()
                } else {
                    value
                }
            };
        }
    }
    dense
}
unsafe fn write<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *mut T,
    dense: &[T],
) {
    for j in 0..n {
        let (offset, stride, conjugated) = layout.segment(n, lower, j);
        let first = if lower { j } else { 0 };
        let length = if lower { n - j } else { j + 1 };
        for k in 0..length {
            let i = first + k;
            if unit && i == j {
                continue;
            }
            let value = dense[i + j * n];
            *a.offset(offset as isize + k as isize * stride) =
                if conjugated { value.cj() } else { value };
        }
    }
}
unsafe fn inverse<T: Scalar>(
    n: usize,
    lower: bool,
    unit: bool,
    layout: TriangleStorage,
    a: *mut T,
) -> usize {
    if n == 0 {
        return 0;
    }
    let mut dense = read(n, lower, unit, layout, a);
    if !unit {
        for j in 0..n {
            if dense[j + j * n] == T::z() {
                return j + 1;
            }
        }
    }
    crate::triangular_lapack::inverse(
        if lower { b'L' } else { b'U' },
        unit,
        n,
        dense.as_mut_ptr(),
        n as isize,
    );
    write(n, lower, unit, layout, a, &dense);
    0
}
fn fits<T>(n: isize) -> bool {
    n >= 0
        && n.checked_mul(n)
            .and_then(|x| x.checked_mul(core::mem::size_of::<T>() as isize))
            .is_some()
}
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
fn transr<T: Scalar>(r: u8) -> bool {
    r == b'N' || r == if T::COMPLEX { b'C' } else { b'T' }
}
unsafe fn diagonal_zero<T: Scalar>(
    n: usize,
    lower: bool,
    layout: TriangleStorage,
    a: *const T,
) -> usize {
    for j in 0..n {
        let (offset, stride, _) = layout.segment(n, lower, j);
        let k = if lower { 0 } else { j };
        if *a.offset(offset as isize + k as isize * stride) == T::z() {
            return j + 1;
        }
    }
    0
}
unsafe fn solve_dense<T: Scalar>(
    left: bool,
    lower: bool,
    tr: u8,
    unit: bool,
    m: usize,
    n: usize,
    alpha: T,
    dense: &[T],
    b: *mut T,
    ldb: isize,
) {
    use colmat::linalg::triangular_solve::*;
    let order = if left { m } else { n };
    let matrix = MatRef::from_raw_parts(dense.as_ptr(), order, order, 1, order as isize);
    let transpose = if left { tr != b'N' } else { tr == b'N' };
    let matrix = if transpose {
        matrix.transpose()
    } else {
        matrix
    };
    let upper = (!lower) ^ transpose;
    let conj = if tr == b'C' { Conj::Yes } else { Conj::No };
    crate::kernels::scale(m as isize, n as isize, alpha, b, ldb);
    let rhs = MatMut::from_raw_parts_mut(b, m, n, 1, if n == 1 { 0 } else { ldb });
    let rhs = if left { rhs } else { rhs.transpose_mut() };
    crate::threading::with_par(|par| match (upper, unit) {
        (false, false) => solve_lower_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (false, true) => solve_unit_lower_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (true, false) => solve_upper_triangular_in_place_with_conj(matrix, conj, rhs, par),
        (true, true) => solve_unit_upper_triangular_in_place_with_conj(matrix, conj, rhs, par),
    });
}
unsafe fn packed_solve<T: Scalar, I: LapackInt>(
    u: u8,
    tr: u8,
    d: u8,
    n: isize,
    nrhs: isize,
    a: *const T,
    b: *mut T,
    ldb: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(u, b'U' | b'L')),
        (2, matches!(tr, b'N' | b'T' | b'C')),
        (3, matches!(d, b'U' | b'N')),
        (4, n >= 0),
        (5, nrhs >= 0),
        (8, ldb >= n.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !fits::<T>(n) {
        error(info, 4);
        return;
    }
    let lower = u == b'L';
    let unit = d == b'U';
    if unit && n == 1 {
        return;
    }
    if !unit {
        let singular = diagonal_zero(n as usize, lower, TriangleStorage::Packed, a);
        if singular != 0 {
            *info = I::from_i64(singular as i64);
            return;
        }
    }
    if nrhs == 0 {
        return;
    }
    if !matrix_addressable::<T>(n, nrhs, ldb) {
        error(info, 8);
        return;
    }
    let dense = read(n as usize, lower, unit, TriangleStorage::Packed, a);
    solve_dense(
        true,
        lower,
        tr,
        unit,
        n as usize,
        nrhs as usize,
        T::o(),
        &dense,
        b,
        ldb,
    );
}
unsafe fn band_solve<T: Scalar, I: LapackInt>(
    u: u8,
    tr: u8,
    d: u8,
    n: isize,
    kd: isize,
    nrhs: isize,
    a: *const T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(u, b'U' | b'L')),
        (2, matches!(tr, b'N' | b'T' | b'C')),
        (3, matches!(d, b'U' | b'N')),
        (4, n >= 0),
        (5, kd >= 0),
        (6, nrhs >= 0),
        (8, kd.checked_add(1).is_some_and(|k| lda >= k)),
        (10, ldb >= n.max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if n == 0 {
        return;
    }
    let lower = u == b'L';
    let unit = d == b'U';
    if unit && (n == 1 || kd == 0) {
        return;
    }
    let rows = if lower { kd.min(n - 1) + 1 } else { kd + 1 };
    if !unit && !matrix_addressable::<T>(if lower { 1 } else { kd + 1 }, n, lda) {
        error(info, 8);
        return;
    }
    if !unit {
        for j in 0..n {
            if *a.offset(j * lda + if lower { 0 } else { kd }) == T::z() {
                *info = I::from_i64((j + 1) as i64);
                return;
            }
        }
    }
    if nrhs == 0 {
        return;
    }
    if !matrix_addressable::<T>(n, nrhs, ldb) {
        error(info, 10);
        return;
    }
    if !matrix_addressable::<T>(rows, n, lda) {
        error(info, 8);
        return;
    }
    if !fits::<T>(n) {
        error(info, 4);
        return;
    }
    let mut dense = vec![T::z(); n as usize * n as usize];
    for j in 0..n {
        let (first, end) = if lower {
            (j, j.saturating_add(kd).saturating_add(1).min(n))
        } else {
            ((j - kd).max(0), j + 1)
        };
        for i in first..end {
            dense[(i + j * n) as usize] = if unit && i == j {
                T::o()
            } else {
                *a.offset(j * lda + if lower { i - j } else { kd + i - j })
            };
        }
    }
    solve_dense(
        true,
        lower,
        tr,
        unit,
        n as usize,
        nrhs as usize,
        T::o(),
        &dense,
        b,
        ldb,
    );
}
unsafe fn inverse_driver<T: Scalar, I: LapackInt>(
    r: Option<u8>,
    u: u8,
    d: u8,
    n: isize,
    a: *mut T,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let shift = i64::from(r.is_some());
    if let Some(r) = r {
        if !transr::<T>(r) {
            error(info, 1);
            return;
        }
    }
    for (arg, ok) in [
        (1 + shift, matches!(u, b'U' | b'L')),
        (2 + shift, matches!(d, b'U' | b'N')),
        (3 + shift, n >= 0),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    if n == 0 {
        return;
    }
    if !fits::<T>(n) {
        error(info, 3 + shift);
        return;
    }
    let layout = r.map_or(TriangleStorage::Packed, |r| TriangleStorage::Rfp {
        transposed: r != b'N',
    });
    *info = I::from_i64(inverse(n as usize, u == b'L', d == b'U', layout, a) as i64);
}
unsafe fn rfp_solve<T: Scalar>(
    r: u8,
    side: u8,
    u: u8,
    tr: u8,
    d: u8,
    m: isize,
    n: isize,
    alpha: *const T,
    a: *const T,
    b: *mut T,
    ldb: isize,
) {
    for (arg, ok) in [
        (1, transr::<T>(r)),
        (2, matches!(side, b'L' | b'R')),
        (3, matches!(u, b'U' | b'L')),
        (4, matches!(tr, b'N' | b'T' | b'C')),
        (5, matches!(d, b'U' | b'N')),
        (6, m >= 0),
        (7, n >= 0),
        (11, ldb >= m.max(1)),
    ] {
        if !ok {
            crate::invalid(arg);
            return;
        }
    }
    if m == 0 || n == 0 {
        return;
    }
    let alpha = *alpha;
    let order = if side == b'L' { m } else { n };
    if d == b'U' && order == 1 && alpha == T::o() {
        return;
    }
    if !matrix_addressable::<T>(m, n, ldb) {
        crate::invalid(11);
        return;
    }
    if alpha == T::z() {
        crate::kernels::scale(m, n, T::z(), b, ldb);
        return;
    }
    if !fits::<T>(order) {
        crate::invalid(if side == b'L' { 6 } else { 7 });
        return;
    }
    let dense = read(
        order as usize,
        u == b'L',
        d == b'U',
        TriangleStorage::Rfp {
            transposed: r != b'N',
        },
        a,
    );
    solve_dense(
        side == b'L',
        u == b'L',
        tr,
        d == b'U',
        m as usize,
        n as usize,
        alpha,
        &dense,
        b,
        ldb,
    );
}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tptri $s>](u:*const u8,d:*const u8,n:*const $i,a:*mut $t,info:*mut $i) {inverse_driver(None,flag(u),flag(d),*n as isize,a,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p tftri $s>](r:*const u8,u:*const u8,d:*const u8,n:*const $i,a:*mut $t,info:*mut $i) {inverse_driver(Some(flag(r)),flag(u),flag(d),*n as isize,a,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p tptrs $s>](u:*const u8,tr:*const u8,d:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,b:*mut $t,ldb:*const $i,info:*mut $i) {packed_solve(flag(u),flag(tr),flag(d),*n as isize,*nrhs as isize,a,b,*ldb as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p tbtrs $s>](u:*const u8,tr:*const u8,d:*const u8,n:*const $i,kd:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {band_solve(flag(u),flag(tr),flag(d),*n as isize,*kd as isize,*nrhs as isize,a,*lda as isize,b,*ldb as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p tfsm $s>](r:*const u8,side:*const u8,u:*const u8,tr:*const u8,d:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,b:*mut $t,ldb:*const $i) {rfp_solve(flag(r),flag(side),flag(u),flag(tr),flag(d),*m as isize,*n as isize,alpha,a,b,*ldb as isize);}
}};}
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
    fn verify<T: Scalar>(imag: T, tolerance: f64) {
        unsafe {
            for u in [b'L', b'U'] {
                let mut info = 99i64;
                packed_solve::<T, i64>(
                    u,
                    b'C',
                    b'U',
                    1,
                    2,
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    isize::MAX,
                    &mut info,
                );
                assert_eq!(info, 0);
                band_solve::<T, i64>(
                    u,
                    b'T',
                    b'U',
                    5,
                    0,
                    2,
                    core::ptr::null(),
                    isize::MAX,
                    core::ptr::null_mut(),
                    5,
                    &mut info,
                );
                assert_eq!(info, 0);
                rfp_solve::<T>(
                    b'N',
                    b'L',
                    u,
                    b'C',
                    b'U',
                    1,
                    2,
                    &T::o(),
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    isize::MAX,
                );
            }

            for n in [0usize, 1, 2, 5, 6] {
                for lower in [false, true] {
                    for unit in [false, true] {
                        for layout in [
                            TriangleStorage::Packed,
                            TriangleStorage::Rfp { transposed: false },
                            TriangleStorage::Rfp { transposed: true },
                        ] {
                            let len = n * (n + 1) / 2;
                            let mut data = vec![T::real(777.0); len + 2];
                            let a = data.as_mut_ptr().add(1);
                            let mut original = vec![T::z(); n * n];
                            for j in 0..n {
                                let (offset, stride, conjugated) = layout.segment(n, lower, j);
                                let first = if lower { j } else { 0 };
                                let length = if lower { n - j } else { j + 1 };
                                for k in 0..length {
                                    let i = first + k;
                                    let value = if i == j {
                                        if unit {
                                            T::o()
                                        } else {
                                            T::real((i + 2) as f64) + imag * T::real(0.2)
                                        }
                                    } else {
                                        T::real((i + j + 1) as f64 / 19.0)
                                            + imag * T::real((i + 1) as f64 / 29.0)
                                    };
                                    original[i + j * n] = value;
                                    *a.offset(offset as isize + k as isize * stride) =
                                        if unit && i == j {
                                            T::real(f64::NAN)
                                        } else if conjugated {
                                            value.cj()
                                        } else {
                                            value
                                        };
                                }
                            }
                            if n > 0 {
                                for left in [false, true] {
                                    for tr in [b'N', b'T', b'C'] {
                                        let (rows, cols) = if left { (n, 3) } else { (3, n) };
                                        let ld = rows + 2;
                                        let alpha = T::real(1.3) + imag * T::real(0.2);
                                        let want: Vec<T> = (0..rows * cols)
                                            .map(|k| {
                                                T::real((k + 1) as f64 / 17.0)
                                                    + imag * T::real((k % 5) as f64 / 23.0)
                                            })
                                            .collect();
                                        let op = |i: usize, j: usize| {
                                            if tr == b'N' {
                                                original[i + j * n]
                                            } else if tr == b'C' {
                                                original[j + i * n].cj()
                                            } else {
                                                original[j + i * n]
                                            }
                                        };
                                        let mut rhs = vec![T::real(777.0); ld * cols];
                                        for j in 0..cols {
                                            for i in 0..rows {
                                                rhs[i + j * ld] = if left {
                                                    (0..n).fold(T::z(), |v, k| {
                                                        v + op(i, k) * want[k + j * rows]
                                                    }) / alpha
                                                } else {
                                                    (0..n).fold(T::z(), |v, k| {
                                                        v + want[i + k * rows] * op(k, j)
                                                    }) / alpha
                                                };
                                            }
                                        }
                                        match layout {
                                            TriangleStorage::Rfp { transposed } => rfp_solve(
                                                if transposed {
                                                    if T::COMPLEX {
                                                        b'C'
                                                    } else {
                                                        b'T'
                                                    }
                                                } else {
                                                    b'N'
                                                },
                                                if left { b'L' } else { b'R' },
                                                if lower { b'L' } else { b'U' },
                                                tr,
                                                if unit { b'U' } else { b'N' },
                                                rows as isize,
                                                cols as isize,
                                                &alpha,
                                                a,
                                                rhs.as_mut_ptr(),
                                                ld as isize,
                                            ),
                                            _ => solve_dense(
                                                left,
                                                lower,
                                                tr,
                                                unit,
                                                rows,
                                                cols,
                                                alpha,
                                                &original,
                                                rhs.as_mut_ptr(),
                                                ld as isize,
                                            ),
                                        }
                                        for j in 0..cols {
                                            for i in 0..rows {
                                                assert!(
                                                    (rhs[i + j * ld] - want[i + j * rows])
                                                        .magnitude()
                                                        < tolerance * 10.0
                                                );
                                            }
                                            for i in rows..ld {
                                                assert_eq!(rhs[i + j * ld], T::real(777.0));
                                            }
                                        }
                                    }
                                }
                            }
                            assert_eq!(inverse(n, lower, unit, layout, a), 0);
                            let result = read(n, lower, unit, layout, a);
                            for j in 0..n {
                                for i in 0..n {
                                    let product = (0..n).fold(T::z(), |s, k| {
                                        s + original[i + k * n] * result[k + j * n]
                                    });
                                    assert!(
                                        (product - if i == j { T::o() } else { T::z() })
                                            .magnitude()
                                            < tolerance
                                    );
                                }
                            }
                            assert_eq!(data[0], T::real(777.0));
                            assert_eq!(data[len + 1], T::real(777.0));
                            if unit {
                                for j in 0..n {
                                    let (offset, stride, _) = layout.segment(n, lower, j);
                                    let k = if lower { 0 } else { j };
                                    assert!((*a.offset(offset as isize + k as isize * stride))
                                        .re()
                                        .is_nan());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        verify(0.0f32, 1e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 1e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 1e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 1e-13);
    }
}
