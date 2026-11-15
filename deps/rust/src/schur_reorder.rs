//! Ordinary Schur block moves with LAPACK index and rejection contracts.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use crate::schur::SchurScalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Mat, MatMut};
use paste::paste;
trait MoveScalar: SchurScalar {
    fn move_block(
        t: MatMut<'_, Self>,
        q: Option<MatMut<'_, Self>>,
        from: usize,
        to: usize,
    ) -> Result<usize, usize>;
}
macro_rules! real_move {
    ($t:ty) => {
        impl MoveScalar for $t {
            fn move_block(
                t: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                from: usize,
                to: usize,
            ) -> Result<usize, usize> {
                use colmat::linalg::evd::{reorder_schur_real, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut memory = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    reorder_schur_real(t, q, from, to, par, MemStack::new(&mut memory))
                        .map_err(|e| e.position)
                })
            }
        }
    };
}
macro_rules! complex_move {
    ($t:ty) => {
        impl MoveScalar for $t {
            fn move_block(
                t: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                from: usize,
                to: usize,
            ) -> Result<usize, usize> {
                use colmat::linalg::evd::{reorder_schur_cplx, reorder_schur_scratch};
                crate::threading::with_par(|par| {
                    let mut memory = MemBuffer::new(reorder_schur_scratch::<Self>(t.nrows()));
                    Ok(reorder_schur_cplx(
                        t,
                        q,
                        from,
                        to,
                        par,
                        MemStack::new(&mut memory),
                    ))
                })
            }
        }
    };
}
real_move!(f32);
real_move!(f64);
complex_move!(colmat::c32);
complex_move!(colmat::c64);
unsafe fn fail<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn stage<T: Scalar>(n: usize, t: *const T, ldt: isize) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i <= j || (!T::COMPLEX && i == j + 1) {
            *t.offset(i as isize + j as isize * ldt)
        } else {
            T::z()
        }
    })
}
unsafe fn perform<T: MoveScalar>(
    n: usize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    want: bool,
    from: usize,
    to: usize,
) -> (usize, bool) {
    let mut tcopy = stage(n, t, ldt);
    let mut qcopy = if want {
        Some(crate::lapack::input_ref(q, n, n, ldq).to_owned())
    } else {
        None
    };
    let result = T::move_block(tcopy.as_mut(), qcopy.as_mut().map(|x| x.as_mut()), from, to);
    // A failed move commits exactly the state of the last successful swap.
    crate::lapack::output(&tcopy, t, ldt);
    if let Some(qcopy) = qcopy {
        crate::lapack::output(&qcopy, q, ldq);
    }
    match result {
        Ok(position) => (position, false),
        Err(position) => (position, true),
    }
}
unsafe fn trexc<T: MoveScalar, I: LapackInt>(
    comp: u8,
    n: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    first: *mut I,
    last: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let want = comp == b'V';
    for (arg, ok) in [
        (1, matches!(comp, b'V' | b'N')),
        (2, n >= 0),
        (4, ldt >= n.max(1)),
        (6, ldq >= 1 && (!want || ldq >= n)),
        (7, n == 0 || ((*first).val() >= 1 && (*first).val() <= n)),
        (8, n == 0 || ((*last).val() >= 1 && (*last).val() <= n)),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if n <= 1 {
        return;
    }
    let (mut from, mut to) = (((*first).val() - 1) as usize, ((*last).val() - 1) as usize);
    if T::COMPLEX && from == to {
        return;
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 4);
        return;
    }
    if !T::COMPLEX {
        if from > 0 && *t.offset(from as isize + (from - 1) as isize * ldt) != T::z() {
            from -= 1;
        }
        if to > 0 && *t.offset(to as isize + (to - 1) as isize * ldt) != T::z() {
            to -= 1;
        }
        *first = I::from_i64(from as i64 + 1);
        *last = I::from_i64(to as i64 + 1);
    }
    if from == to {
        return;
    }
    if want && !matrix_addressable::<T>(n, n, ldq) {
        fail(info, 6);
        return;
    }
    let (position, rejected) = perform(n as usize, t, ldt, q, ldq, want, from, to);
    if !T::COMPLEX {
        *last = I::from_i64(position as i64 + 1);
    }
    *info = I::from_i64(rejected as i64);
}
unsafe fn laexc<T: MoveScalar, I: LapackInt>(
    want: bool,
    n: isize,
    t: *mut T,
    ldt: isize,
    q: *mut T,
    ldq: isize,
    j: isize,
    n1: isize,
    n2: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    if n == 0 || n1 == 0 || n2 == 0 {
        return;
    }
    for (arg, ok) in [
        (2, n >= 0),
        (4, ldt >= n.max(1)),
        (6, ldq >= 1 && (!want || ldq >= n)),
        (8, (1..=2).contains(&n1)),
        (9, (1..=2).contains(&n2)),
        (
            7,
            j >= 1
                && j.checked_add(n1)
                    .and_then(|v| v.checked_add(n2))
                    .is_some_and(|v| v - 1 <= n),
        ),
    ] {
        if !ok {
            fail(info, arg);
            return;
        }
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        fail(info, 4);
        return;
    }
    if want && !matrix_addressable::<T>(n, n, ldq) {
        fail(info, 6);
        return;
    }
    let (_, rejected) = perform(
        n as usize,
        t,
        ldt,
        q,
        ldq,
        want,
        (j - 1) as usize,
        (j - 1 + n1) as usize,
    );
    *info = I::from_i64(rejected as i64);
}
macro_rules! real_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trexc $s>](comp:*const u8,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,first:*mut $i,last:*mut $i,_work:*mut $t,info:*mut $i) {trexc(flag(comp),*n as isize,t,*ldt as isize,q,*ldq as isize,first,last,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p laexc $s>](want:*const $i,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,j:*const $i,n1:*const $i,n2:*const $i,_work:*mut $t,info:*mut $i) {laexc((*want).val()!=0,*n as isize,t,*ldt as isize,q,*ldq as isize,*j as isize,*n1 as isize,*n2 as isize,info);}
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trexc $s>](comp:*const u8,n:*const $i,t:*mut $t,ldt:*const $i,q:*mut $t,ldq:*const $i,first:*mut $i,last:*mut $i,info:*mut $i) {trexc(flag(comp),*n as isize,t,*ldt as isize,q,*ldq as isize,first,last,info);}
}};}
macro_rules! abi {
    ($i:ty,$s:tt) => {
        real_exports!(s, f32, $i, $s);
        real_exports!(d, f64, $i, $s);
        complex_exports!(c, colmat::c32, $i, $s);
        complex_exports!(z, colmat::c64, $i, $s);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: MoveScalar, I: LapackInt>(imag: T, tol: f64) {
        unsafe {
            for n in [0usize, 1, 5] {
                for from in 0..n.max(1) {
                    for to in 0..n.max(1) {
                        let mut a = Mat::from_fn(n, n, |i, j| {
                            if i > j {
                                T::z()
                            } else if i == j {
                                T::real((i + 2) as f64) + imag * T::real(i as f64 / 7.0)
                            } else {
                                T::real((i + j + 1) as f64 / 11.0)
                                    + imag * T::real((i + 2 * j) as f64 / 13.0)
                            }
                        });
                        if !T::COMPLEX && n == 5 {
                            a[(0, 0)] = T::real(2.0);
                            a[(1, 1)] = T::real(2.0);
                            a[(0, 1)] = T::real(3.0);
                            a[(1, 0)] = T::real(-2.0);
                            a[(3, 3)] = T::real(7.0);
                            a[(4, 4)] = T::real(7.0);
                            a[(3, 4)] = T::real(-4.0);
                            a[(4, 3)] = T::real(1.0);
                        }
                        let canonical = |j: usize| {
                            if !T::COMPLEX && j > 0 && a[(j, j - 1)] != T::z() {
                                j - 1
                            } else {
                                j
                            }
                        };
                        let (first, destination) = if n > 0 {
                            (canonical(from), canonical(to))
                        } else {
                            (0, 0)
                        };
                        let block_size = |j: usize| {
                            if !T::COMPLEX && j + 1 < n && a[(j + 1, j)] != T::z() {
                                2
                            } else {
                                1
                            }
                        };
                        let expected = if first < destination {
                            destination + block_size(destination) - block_size(first)
                        } else {
                            destination
                        };
                        for want in [false, true] {
                            let ld = n + 2;
                            let mut t = vec![T::real(777.0); ld * n];
                            let mut q = t.clone();
                            for j in 0..n {
                                for i in 0..n {
                                    t[i + j * ld] = a[(i, j)];
                                    q[i + j * ld] = if i == j { T::o() } else { T::z() };
                                }
                            }
                            let old_q = q.clone();
                            let mut ifst = I::from_i64(if n == 0 { -777 } else { from as i64 + 1 });
                            let mut ilst = I::from_i64(if n == 0 { 888 } else { to as i64 + 1 });
                            let mut info = I::from_i64(-777);
                            let old_first = ifst.val();
                            let old_last = ilst.val();
                            trexc(
                                if want { b'V' } else { b'N' },
                                n as isize,
                                t.as_mut_ptr(),
                                ld as isize,
                                if want {
                                    q.as_mut_ptr()
                                } else {
                                    core::ptr::null_mut()
                                },
                                if want { ld as isize } else { 1 },
                                &mut ifst,
                                &mut ilst,
                                &mut info,
                            );
                            assert_eq!(info.val(), 0);
                            if !T::COMPLEX && n > 1 {
                                assert_eq!(ifst.val(), first as isize + 1);
                                assert_eq!(ilst.val(), expected as isize + 1);
                            } else {
                                assert_eq!(ifst.val(), old_first);
                                assert_eq!(ilst.val(), old_last);
                            }
                            for j in 0..n {
                                for i in n..ld {
                                    assert_eq!(t[i + j * ld], T::real(777.0));
                                    assert_eq!(q[i + j * ld], T::real(777.0));
                                }
                            }
                            if !want {
                                assert_eq!(q, old_q);
                                continue;
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let mut rebuilt = T::z();
                                    let mut gram = T::z();
                                    for k in 0..n {
                                        gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                        for l in 0..n {
                                            rebuilt = rebuilt
                                                + q[i + k * ld]
                                                    * t[k + l * ld]
                                                    * q[j + l * ld].cj();
                                        }
                                    }
                                    assert!(
                                        (rebuilt - a[(i, j)]).abs1() < tol,
                                        "n={n} from={from} to={to}"
                                    );
                                    assert!(
                                        (gram - if i == j { T::o() } else { T::z() }).abs1() < tol
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn trexc_positions_similarity_and_ignored_q_all_scalar_integer_types() {
        verify::<f32, i32>(0.0, 3e-4);
        verify::<f32, i64>(0.0, 3e-4);
        verify::<f64, i32>(0.0, 3e-11);
        verify::<f64, i64>(0.0, 3e-11);
        verify::<colmat::c32, i32>(colmat::c32::new(0.0, 1.0), 3e-4);
        verify::<colmat::c32, i64>(colmat::c32::new(0.0, 1.0), 3e-4);
        verify::<colmat::c64, i32>(colmat::c64::new(0.0, 1.0), 3e-11);
        verify::<colmat::c64, i64>(colmat::c64::new(0.0, 1.0), 3e-11);
    }
    fn auxiliary<T: MoveScalar, I: LapackInt>(tol: f64) {
        unsafe {
            for n1 in [1usize, 2] {
                for n2 in [1usize, 2] {
                    for first in [0usize, 1] {
                        for want in [false, true] {
                            let n = first + n1 + n2 + 1;
                            let ld = n + 2;
                            let mut a = Mat::from_fn(n, n, |i, j| {
                                if i > j {
                                    T::z()
                                } else if i == j {
                                    T::real((i + 2) as f64)
                                } else {
                                    T::real((i + j + 1) as f64 / 13.0)
                                }
                            });
                            for (start, size, value) in [(first, n1, 10.0), (first + n1, n2, 20.0)]
                            {
                                for i in start..start + size {
                                    a[(i, i)] = T::real(value);
                                }
                                if size == 2 {
                                    a[(start, start + 1)] = T::real(3.0);
                                    a[(start + 1, start)] = T::real(-2.0);
                                }
                            }
                            let mut t = vec![T::real(777.0); ld * n];
                            let mut q = t.clone();
                            for j in 0..n {
                                for i in 0..n {
                                    t[i + j * ld] = a[(i, j)];
                                    q[i + j * ld] = if i == j { T::o() } else { T::z() };
                                }
                            }
                            let old_q = q.clone();
                            let mut info = I::from_i64(-777);
                            laexc(
                                want,
                                n as isize,
                                t.as_mut_ptr(),
                                ld as isize,
                                if want {
                                    q.as_mut_ptr()
                                } else {
                                    core::ptr::null_mut()
                                },
                                if want { ld as isize } else { 1 },
                                first as isize + 1,
                                n1 as isize,
                                n2 as isize,
                                &mut info,
                            );
                            assert_eq!(info.val(), 0);
                            assert!((t[first + first * ld] - T::real(20.0)).abs1() < tol);
                            for j in 0..n {
                                for i in n..ld {
                                    assert_eq!(t[i + j * ld], T::real(777.0));
                                    assert_eq!(q[i + j * ld], T::real(777.0));
                                }
                            }
                            if !want {
                                assert_eq!(q, old_q);
                                continue;
                            }
                            for j in 0..n {
                                for i in 0..n {
                                    let mut rebuilt = T::z();
                                    let mut gram = T::z();
                                    for k in 0..n {
                                        gram = gram + q[k + i * ld].cj() * q[k + j * ld];
                                        for l in 0..n {
                                            rebuilt = rebuilt
                                                + q[i + k * ld]
                                                    * t[k + l * ld]
                                                    * q[j + l * ld].cj();
                                        }
                                    }
                                    assert!((rebuilt - a[(i, j)]).abs1() < tol);
                                    assert!(
                                        (gram - if i == j { T::o() } else { T::z() }).abs1() < tol
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn adjacent_all_real_block_sizes_and_integer_abis() {
        auxiliary::<f32, i32>(4e-4);
        auxiliary::<f32, i64>(4e-4);
        auxiliary::<f64, i32>(4e-11);
        auxiliary::<f64, i64>(4e-11);
    }
    #[test]
    fn auxiliary_quick_returns_do_not_read_matrices() {
        unsafe {
            let mut info = 777i64;
            let nil = core::ptr::null_mut::<f64>();
            for (n, n1, n2) in [(0, 1, 2), (4, 0, 2), (4, 2, 0)] {
                laexc::<f64, i64>(true, n, nil, 0, nil, 0, -777, n1, n2, &mut info);
                assert_eq!(info, 0);
            }
        }
    }
}

#[cfg(test)]
mod rejection_tests {
    use super::*;
    fn check<T: MoveScalar, I: LapackInt>(rows: [[f64; 4]; 4]) {
        unsafe {
            for want in [false, true] {
                let ld = 6;
                let mut t = vec![T::real(777.0); ld * 4];
                let mut q = t.clone();
                for j in 0..4 {
                    for i in 0..4 {
                        t[i + j * ld] = T::real(rows[i][j]);
                        q[i + j * ld] = if i == j { T::o() } else { T::z() };
                    }
                }
                let old_t = t.clone();
                let old_q = q.clone();
                let mut info = [I::from_i64(-777); 3];
                laexc(
                    want,
                    4,
                    t.as_mut_ptr(),
                    ld as isize,
                    if want {
                        q.as_mut_ptr()
                    } else {
                        core::ptr::null_mut()
                    },
                    if want { ld as isize } else { 1 },
                    1,
                    2,
                    2,
                    info.as_mut_ptr().add(1),
                );
                assert_eq!(info[1].val(), 1);
                assert_eq!(info[0].val(), -777);
                assert_eq!(info[2].val(), -777);
                assert_eq!(
                    t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                    old_t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                    old_q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                );
                for from in [1, 2] {
                    for to in [3, 4] {
                        let mut first = I::from_i64(from);
                        let mut last = I::from_i64(to);
                        trexc(
                            if want { b'V' } else { b'N' },
                            4,
                            t.as_mut_ptr(),
                            ld as isize,
                            if want {
                                q.as_mut_ptr()
                            } else {
                                core::ptr::null_mut()
                            },
                            if want { ld as isize } else { 1 },
                            &mut first,
                            &mut last,
                            info.as_mut_ptr().add(1),
                        );
                        assert_eq!((info[1].val(), first.val(), last.val()), (1, 1, 1));
                        assert_eq!(
                            t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                            old_t.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(
                            q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>(),
                            old_q.iter().map(|x| x.re().to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn finite_rejected_swaps_preserve_t_q_and_return_current_position() {
        // These finite standardized windows were independently executed with
        // original OpenBLAS DLAEXC/SLAEXC: INFO=1 and bitwise unchanged T/Q.
        let real64 = [
            [
                1.0,
                7.1362384635298e+44,
                -3.7414441915671115e+50,
                -3.622271631530685e-71,
            ],
            [
                -7.24454326306137e-71,
                1.0,
                -4.7477838728798994e-66,
                -5.795634610449096e-70,
            ],
            [0.0, 0.0, 1.5, 2.535301200456459e+30],
            [0.0, 0.0, -2.8698592549372254e-42, 1.5],
        ];
        let real32 = [
            [1.0, 0.5, 8.881784e-16, 536870900.0],
            [-3.3881318e-21, 1.0, 128.0, 5.421011e-20],
            [0.0, 0.0, 1.000061, 5.684342e-14],
            [0.0, 0.0, -67108864.0, 1.000061],
        ];
        check::<f64, i32>(real64);
        check::<f64, i64>(real64);
        check::<f32, i32>(real32);
        check::<f32, i64>(real32);
    }
}
