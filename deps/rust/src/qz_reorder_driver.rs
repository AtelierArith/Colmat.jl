//! Fortran integer ABIs for generalized Schur block movement.
use crate::kernels::{matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::qz_reorder::ReorderScalar;
use paste::paste;
unsafe fn invalid<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn first<T: Scalar>(n: isize, index: isize, a: *const T, ld: isize) -> Option<isize> {
    if !T::COMPLEX && index > 0 {
        if !matrix_addressable::<T>(index + 1, index, ld) {
            return None;
        }
        if *a.offset(index + (index - 1) * ld) != T::z() {
            return Some(index - 1);
        }
    }
    debug_assert!(index < n);
    Some(index)
}
unsafe fn exchange<T: ReorderScalar, I: LapackInt>(
    wantq: bool,
    wantz: bool,
    n: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    ifst: *mut I,
    ilst: *mut I,
    work: Option<(*mut T, isize)>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let from = (*ifst).val();
    let to = (*ilst).val();
    for (arg, ok) in [
        (3, n >= 0),
        (5, lda >= n.max(1)),
        (7, ldb >= n.max(1)),
        (9, ldq >= if wantq { n.max(1) } else { 1 }),
        (11, ldz >= if wantz { n.max(1) } else { 1 }),
        (12, from >= 1 && from <= n),
        (13, to >= 1 && to <= n),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    if let Some((pointer, length)) = work {
        let Some(need) = (if n <= 1 {
            Some(1)
        } else {
            n.checked_mul(4).and_then(|x| x.checked_add(16))
        }) else {
            invalid(info, 15);
            return;
        };
        *pointer = T::real(need as f64);
        if length == -1 {
            return;
        }
        if length < need {
            invalid(info, 15);
            return;
        }
    }
    if n <= 1 {
        return;
    }
    let (from, to) = (from - 1, to - 1);
    if T::COMPLEX && from == to {
        return;
    }
    let Some(start) = first(n, from, a, lda) else {
        invalid(info, 5);
        return;
    };
    let Some(target) = first(n, to, a, lda) else {
        invalid(info, 5);
        return;
    };
    if start == target {
        *ifst = I::from_i64(start as i64 + 1);
        *ilst = I::from_i64(target as i64 + 1);
        return;
    }
    for (arg, ok) in [
        (5, matrix_addressable::<T>(n, n, lda)),
        (7, matrix_addressable::<T>(n, n, ldb)),
        (9, !wantq || matrix_addressable::<T>(n, n, ldq)),
        (11, !wantz || matrix_addressable::<T>(n, n, ldz)),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    let result = crate::qz_reorder::move_pair(
        n as usize,
        from as usize,
        to as usize,
        a,
        lda,
        b,
        ldb,
        wantq.then_some((q, ldq)),
        wantz.then_some((z, ldz)),
    );
    *ifst = I::from_i64(result.first as i64 + 1);
    // ZTGEXC's upward failure reports the attempted adjacent-swap start;
    // real TGEXC reports the moving block's current first row instead.
    let position = if T::COMPLEX && result.rejected && from > to {
        result.position as i64
    } else {
        result.position as i64 + 1
    };
    *ilst = I::from_i64(position);
    if result.rejected {
        *info = I::from_i64(1);
    }
}
// TGEX2 exchanges exactly two adjacent diagonal blocks. Unlike TGEXC,
// its real workspace argument has no query mode.
unsafe fn adjacent<T: ReorderScalar, I: LapackInt>(
    wantq: bool,
    wantz: bool,
    n: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    q: *mut T,
    ldq: isize,
    z: *mut T,
    ldz: isize,
    j1: isize,
    n1: isize,
    n2: isize,
    work: Option<(*mut T, isize)>,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n <= 1 || n1 <= 0 || n2 <= 0 {
        return;
    }
    if !T::COMPLEX && (n1 > n || j1.checked_add(n1).is_some_and(|x| x > n)) {
        return;
    }
    let Some(m) = n1.checked_add(n2) else {
        invalid(info, 14);
        return;
    };
    if let Some((pointer, length)) = work {
        let need = n
            .checked_mul(m)
            .zip(m.checked_mul(m).and_then(|x| x.checked_mul(2)))
            .map(|(x, y)| x.max(y).max(1));
        let Some(need) = need else {
            invalid(info, 16);
            return;
        };
        if length < need {
            *pointer = T::real(need as f64);
            invalid(info, 16);
            return;
        }
    }
    for (arg, ok) in [
        (5, lda >= n && matrix_addressable::<T>(n, n, lda)),
        (7, ldb >= n && matrix_addressable::<T>(n, n, ldb)),
        (
            9,
            ldq >= 1 && (!wantq || (ldq >= n && matrix_addressable::<T>(n, n, ldq))),
        ),
        (
            11,
            ldz >= 1 && (!wantz || (ldz >= n && matrix_addressable::<T>(n, n, ldz))),
        ),
        (12, j1 >= 1 && j1.checked_add(m - 1).is_some_and(|x| x <= n)),
        (13, n1 <= 2),
        (14, n2 <= 2),
    ] {
        if !ok {
            invalid(info, arg);
            return;
        }
    }
    let start = j1 - 1;
    // The explicit real block sizes must describe the actual Schur blocks;
    // otherwise a multi-step move could partially mutate a requested single swap.
    if !T::COMPLEX {
        let width = |index: isize| -> isize {
            if index + 1 < n && *a.offset(index + 1 + index * lda) != T::z() {
                2
            } else {
                1
            }
        };
        if first(n, start, a, lda) != Some(start) || width(start) != n1 || width(start + n1) != n2 {
            *info = I::from_i64(1);
            return;
        }
    }
    let result = crate::qz_reorder::move_pair(
        n as usize,
        (start + n1) as usize,
        start as usize,
        a,
        lda,
        b,
        ldb,
        wantq.then_some((q, ldq)),
        wantz.then_some((z, ldz)),
    );
    if result.rejected {
        *info = I::from_i64(1);
    }
}
macro_rules! adjacent_real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgex2 $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,j1:*const $i,n1:*const $i,n2:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
        adjacent((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),(*j1).val(),(*n1).val(),(*n2).val(),Some((work,(*lwork).val())),info);
    }
}};}
macro_rules! adjacent_complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgex2 $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,j1:*const $i,info:*mut $i) {
        adjacent((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),(*j1).val(),1,1,None,info);
    }
}};}
adjacent_real_exports!(s, f32, i32, _);
adjacent_real_exports!(d, f64, i32, _);
adjacent_real_exports!(s, f32, i64, _64_);
adjacent_real_exports!(d, f64, i64, _64_);
adjacent_complex_exports!(c, colmat::c32, i32, _);
adjacent_complex_exports!(z, colmat::c64, i32, _);
adjacent_complex_exports!(c, colmat::c32, i64, _64_);
adjacent_complex_exports!(z, colmat::c64, i64, _64_);

macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgexc $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,ifst:*mut $i,ilst:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
        exchange((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),ifst,ilst,Some((work,(*lwork).val())),info);
    }
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p tgexc $s>](wq:*const $i,wz:*const $i,n:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,q:*mut $t,ldq:*const $i,z:*mut $t,ldz:*const $i,ifst:*mut $i,ilst:*mut $i,info:*mut $i) {
        exchange((*wq).val()!=0,(*wz).val()!=0,(*n).val(),a,(*lda).val(),b,(*ldb).val(),q,(*ldq).val(),z,(*ldz).val(),ifst,ilst,None,info);
    }
}};}
real_exports!(s, f32, i32, _);
real_exports!(d, f64, i32, _);
real_exports!(s, f32, i64, _64_);
real_exports!(d, f64, i64, _64_);
complex_exports!(c, colmat::c32, i32, _);
complex_exports!(z, colmat::c64, i32, _);
complex_exports!(c, colmat::c32, i64, _64_);
complex_exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn abi<T: ReorderScalar, I: LapackInt>(tol: f64) {
        let n = 3;
        let ld = 5;
        let mut a = vec![T::real(777.0); ld * n];
        let mut b = a.clone();
        for j in 0..n {
            for i in 0..n {
                a[i + j * ld] = if i == j {
                    T::real((i + 1) as f64)
                } else {
                    T::z()
                };
                b[i + j * ld] = if i == j { T::o() } else { T::z() };
            }
        }
        let mut q = vec![T::real(777.0); 1];
        let mut z = q.clone();
        let mut from = [I::from_i64(-777), I::from_i64(3), I::from_i64(-777)];
        let mut to = [I::from_i64(-777), I::from_i64(1), I::from_i64(-777)];
        let mut info = [I::from_i64(-777); 3];
        let mut work = [T::real(777.0); 30];
        unsafe {
            exchange(
                false,
                false,
                n as isize,
                a.as_mut_ptr(),
                ld as isize,
                b.as_mut_ptr(),
                ld as isize,
                q.as_mut_ptr(),
                1,
                z.as_mut_ptr(),
                1,
                &mut from[1],
                &mut to[1],
                if T::COMPLEX {
                    None
                } else {
                    Some((work.as_mut_ptr(), 28))
                },
                &mut info[1],
            );
        }
        assert_eq!(info[1].val(), 0);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        assert_eq!(from[1].val(), 3);
        assert_eq!(to[1].val(), 1);
        assert!((a[0] / b[0] - T::real(3.0)).magnitude() < tol);
        for array in [from, to, info] {
            assert_eq!(array[0].val(), -777);
            assert_eq!(array[2].val(), -777);
        }
        assert_eq!(q[0].re(), 777.0);
        assert_eq!(z[0].re(), 777.0);
        for j in 0..n {
            for i in n..ld {
                assert_eq!(a[i + j * ld].re(), 777.0);
                assert_eq!(b[i + j * ld].re(), 777.0);
            }
        }
    }
    fn adjacent_abi<T: ReorderScalar, I: LapackInt>(tol: f64) {
        let mut a = [T::real(1.0), T::z(), T::real(0.25), T::real(3.0)];
        let mut b = [T::o(), T::z(), T::z(), T::o()];
        let mut dummy = T::real(777.0);
        let mut work = [T::real(777.0); 8];
        let mut info = I::from_i64(-777);
        unsafe {
            adjacent(
                false,
                false,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                &mut dummy,
                1,
                &mut dummy,
                1,
                1,
                1,
                1,
                if T::COMPLEX {
                    None
                } else {
                    Some((work.as_mut_ptr(), 8))
                },
                &mut info,
            );
        }
        assert_eq!(info.val(), 0);
        assert_eq!(crate::colmat_blas_last_error(), 0);
        assert!((a[0] / b[0] - T::real(3.0)).magnitude() < tol);
        assert!((a[3] / b[3] - T::real(1.0)).magnitude() < tol);
        assert_eq!(dummy.re(), 777.0);
    }
    #[test]
    fn adjacent_all_scalar_and_integer_widths() {
        adjacent_abi::<f32, i32>(1e-4);
        adjacent_abi::<f32, i64>(1e-4);
        adjacent_abi::<f64, i32>(1e-11);
        adjacent_abi::<f64, i64>(1e-11);
        adjacent_abi::<colmat::c32, i32>(1e-4);
        adjacent_abi::<colmat::c32, i64>(1e-4);
        adjacent_abi::<colmat::c64, i32>(1e-11);
        adjacent_abi::<colmat::c64, i64>(1e-11);
    }
    #[test]
    fn adjacent_real_workspace_is_not_a_query() {
        let mut scalar = 777.0f64;
        let mut work = 777.0;
        let mut info = -777i64;
        unsafe {
            adjacent(
                false,
                false,
                2,
                &mut scalar,
                2,
                &mut scalar,
                2,
                &mut scalar,
                1,
                &mut scalar,
                1,
                1,
                1,
                1,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, -16);
        assert_eq!(crate::colmat_blas_last_error(), 16);
        assert_eq!(work, 8.0);
        assert_eq!(scalar, 777.0);
    }
    #[test]
    fn all_scalar_and_integer_widths() {
        abi::<f32, i32>(1e-4);
        abi::<f32, i64>(1e-4);
        abi::<f64, i32>(1e-11);
        abi::<f64, i64>(1e-11);
        abi::<colmat::c32, i32>(1e-4);
        abi::<colmat::c32, i64>(1e-4);
        abi::<colmat::c64, i32>(1e-11);
        abi::<colmat::c64, i64>(1e-11);
    }
    #[test]
    fn real_query_and_empty_index_validation() {
        let mut scalar = 777.0f64;
        let mut from = 2i64;
        let mut to = 1i64;
        let mut info = -777i64;
        let mut work = 777.0;
        unsafe {
            exchange(
                false,
                false,
                3,
                &mut scalar,
                3,
                &mut scalar,
                3,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut from,
                &mut to,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, 0);
        assert_eq!(work, 28.0);
        assert_eq!(scalar, 777.0);
        from = 1;
        to = 1;
        work = 777.0;
        unsafe {
            exchange(
                false,
                false,
                0,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut scalar,
                1,
                &mut from,
                &mut to,
                Some((&mut work, -1)),
                &mut info,
            );
        }
        assert_eq!(info, -12);
        assert_eq!(crate::colmat_blas_last_error(), 12);
        assert_eq!(work, 777.0);
    }
}
