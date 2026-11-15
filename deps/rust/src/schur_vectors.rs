//! Eigenvectors of an existing Schur form, without recomputing its Schur factors.
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Mat, MatRef};

trait SchurVectors: ReflectorScalar {
    fn right(t: MatRef<'_, Self>) -> Result<Mat<Self>, ()>;
}
macro_rules! native_vectors {
    ($t:ty,$function:ident) => {
        impl SchurVectors for $t {
            fn right(t: MatRef<'_, Self>) -> Result<Mat<Self>, ()> {
                use colmat::linalg::evd::{eigenvectors_from_schur_scratch, $function};
                let n = t.nrows();
                let mut x = Mat::<Self>::zeros(n, n);
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(eigenvectors_from_schur_scratch::<Self>(n));
                    $function(
                        t,
                        None,
                        x.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(x)
            }
        }
    };
}
native_vectors!(f32, eigenvectors_from_schur_real);
native_vectors!(f64, eigenvectors_from_schur_real);
native_vectors!(colmat::c32, eigenvectors_from_schur_cplx);
native_vectors!(colmat::c64, eigenvectors_from_schur_cplx);

fn blocks<T: Scalar>(t: MatRef<'_, T>) -> Result<Vec<(usize, usize)>, ()> {
    let n = t.nrows();
    let mut result = Vec::new();
    let mut j = 0;
    while j < n {
        let pair = !T::COMPLEX && j + 1 < n && t[(j + 1, j)] != T::z();
        if pair {
            // Public native helper requires standardized real Schur blocks.
            let (a, b, c, d) = (
                t[(j, j)].re(),
                t[(j, j + 1)].re(),
                t[(j + 1, j)].re(),
                t[(j + 1, j + 1)].re(),
            );
            if a != d
                || !((b > 0.0 && c < 0.0) || (b < 0.0 && c > 0.0))
                || (j + 2 < n && t[(j + 2, j + 1)] != T::z())
            {
                return Err(());
            }
        }
        let len = if pair { 2 } else { 1 };
        result.push((j, len));
        j += len;
    }
    Ok(result)
}
fn vectors<T: SchurVectors>(t: MatRef<'_, T>, left: bool) -> Result<Mat<T>, ()> {
    let n = t.nrows();
    let layout = blocks(t)?;
    if !left {
        return T::right(t);
    }
    // Reversal changes lower triangular T^H back into an upper Schur form.
    let reverse = Mat::from_fn(n, n, |i, j| t[(n - 1 - j, n - 1 - i)].cj());
    let x = T::right(reverse.as_ref())?;
    let mut output = Mat::<T>::zeros(n, n);
    for (j, len) in layout {
        if len == 2 {
            let k = n - 2 - j;
            for i in 0..n {
                output[(i, j)] = x[(n - 1 - i, k)];
                output[(i, j + 1)] = -x[(n - 1 - i, k + 1)];
            }
        } else {
            for i in 0..n {
                output[(i, j)] = x[(n - 1 - i, n - 1 - j)];
            }
        }
    }
    Ok(output)
}
fn normalize<T: Scalar>(x: &mut Mat<T>, layout: &[(usize, usize)]) -> Result<(), ()> {
    let n = x.nrows();
    for &(j, len) in layout {
        let max = (0..n).fold(0.0f64, |value, i| {
            value.max(if len == 2 {
                x[(i, j)].abs1() + x[(i, j + 1)].abs1()
            } else {
                x[(i, j)].abs1()
            })
        });
        if !(max > 0.0 && max.is_finite()) {
            return Err(());
        }
        for k in j..j + len {
            for i in 0..n {
                x[(i, k)] = x[(i, k)] / T::real(max);
            }
        }
    }
    if x.as_ref().is_all_finite() {
        Ok(())
    } else {
        Err(())
    }
}

unsafe fn failure<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
unsafe fn compute<T: SchurVectors, R: Scalar, I: LapackInt>(
    blocked: bool,
    side: u8,
    how: u8,
    select: *mut I,
    n: isize,
    t: *const T,
    ldt: isize,
    vl: *mut T,
    ldvl: isize,
    vr: *mut T,
    ldvr: isize,
    mm: isize,
    m: *mut I,
    work: *mut T,
    lwork: isize,
    rwork: *mut R,
    lrwork: isize,
    info: *mut I,
) -> bool {
    *info = I::from_i64(0);
    let left = matches!(side, b'L' | b'B');
    let right = matches!(side, b'R' | b'B');
    let minimum = match n.max(0).checked_mul(if T::COMPLEX { 2 } else { 3 }) {
        Some(v) if core::mem::size_of::<I>() != 4 || v <= i32::MAX as isize => v.max(1),
        _ => {
            failure(info, 4);
            return true;
        }
    };
    let real_min = n.max(1);
    let query = blocked && (lwork == -1 || (T::COMPLEX && lrwork == -1));
    if blocked {
        *work = crate::lapack::workspace::<T>(minimum);
        if T::COMPLEX {
            *rwork = crate::lapack::workspace::<R>(real_min);
        }
    }
    for (arg, ok) in [
        (1, left || right),
        (2, matches!(how, b'A' | b'B' | b'S')),
        (4, n >= 0),
        (6, ldt >= n.max(1)),
        (8, ldvl >= 1 && (!left || ldvl >= n)),
        (10, ldvr >= 1 && (!right || ldvr >= n)),
    ] {
        if !ok {
            failure(info, arg);
            return true;
        }
    }
    // Real TREVC3 checks LWORK before writing M/SELECT; complex TREVC3
    // checks it after MM, as do the original LAPACK routines.
    if blocked && !T::COMPLEX && !query && lwork < minimum {
        failure(info, 14);
        return true;
    }
    let mut chosen = Vec::new();
    let mut count = 0usize;
    if how == b'S' {
        if !T::COMPLEX && !matrix_addressable::<T>(n, n, ldt) {
            failure(info, 6);
            return true;
        }
        let mut j = 0usize;
        while j < n as usize {
            let pair = !T::COMPLEX
                && j + 1 < n as usize
                && *t.offset((j + 1) as isize + j as isize * ldt) != T::z();
            if pair {
                let selected = (*select.add(j)).val() != 0 || (*select.add(j + 1)).val() != 0;
                if selected {
                    *select.add(j) = I::from_i64(1);
                    count += 2;
                    if !query {
                        chosen.extend([j, j + 1]);
                    }
                }
                *select.add(j + 1) = I::from_i64(0);
                j += 2;
            } else {
                if (*select.add(j)).val() != 0 {
                    count += 1;
                    if !query {
                        chosen.push(j);
                    }
                }
                j += 1;
            }
        }
    } else {
        count = n as usize;
        if !query {
            chosen.extend(0..n as usize);
        }
    }
    *m = I::from_i64(count as i64);
    if mm < count as isize {
        failure(info, 11);
        return true;
    }
    if blocked && T::COMPLEX && !query {
        if lwork < minimum {
            failure(info, 14);
            return true;
        }
        if lrwork < real_min {
            failure(info, 16);
            return true;
        }
    }
    if query || n == 0 || chosen.is_empty() {
        return true;
    }
    if !matrix_addressable::<T>(n, n, ldt) {
        failure(info, 6);
        return true;
    }
    let columns = if how == b'B' {
        n
    } else {
        chosen.len() as isize
    };
    if left && !matrix_addressable::<T>(n, columns, ldvl) {
        failure(info, 8);
        return true;
    }
    if right && !matrix_addressable::<T>(n, columns, ldvr) {
        failure(info, 10);
        return true;
    }
    let size = n as usize;
    let schur = Mat::from_fn(size, size, |i, j| {
        if i <= j || (!T::COMPLEX && i == j + 1) {
            *t.offset(i as isize + j as isize * ldt)
        } else {
            T::z()
        }
    });
    let Ok(layout) = blocks(schur.as_ref()) else {
        return false;
    };
    let prepare = |left_side: bool, p: *const T, ld: isize| -> Result<Mat<T>, ()> {
        let mut x = vectors(schur.as_ref(), left_side)?;
        if how == b'B' {
            let q = crate::lapack::input_ref(p, size, size, ld);
            x = crate::schur::multiply(q, b'N', x.as_ref(), b'N');
        }
        normalize(&mut x, &layout)?;
        Ok(x)
    };
    let left_output = if left {
        match prepare(true, vl, ldvl) {
            Ok(x) => Some(x),
            Err(()) => return false,
        }
    } else {
        None
    };
    let right_output = if right {
        match prepare(false, vr, ldvr) {
            Ok(x) => Some(x),
            Err(()) => return false,
        }
    } else {
        None
    };
    for (output, p, ld) in [(left_output, vl, ldvl), (right_output, vr, ldvr)] {
        if let Some(output) = output {
            for (column, &source) in chosen.iter().enumerate() {
                for i in 0..size {
                    *p.offset(i as isize + column as isize * ld) = output[(i, source)];
                }
            }
        }
    }
    true
}

use core::ffi::c_void;
use core::sync::atomic::{AtomicPtr, Ordering};
static FALLBACKS: [AtomicPtr<c_void>; 8] = [const { AtomicPtr::new(core::ptr::null_mut()) }; 8];
#[no_mangle]
pub extern "C" fn colmat_schur_vectors_set_fallback(id: i32, p: *mut c_void) {
    if let Some(slot) = FALLBACKS.get(id as usize) {
        slot.store(p, Ordering::Release);
    }
}
#[no_mangle]
pub extern "C" fn colmat_schur_vectors_has_fallback(id: i32) -> i32 {
    FALLBACKS
        .get(id as usize)
        .is_some_and(|slot| !slot.load(Ordering::Acquire).is_null()) as i32
}
fn fallback(id: usize) -> *mut c_void {
    let p = FALLBACKS[id].load(Ordering::Acquire);
    assert!(
        !p.is_null(),
        "initialize Schur eigenvector fallback before use"
    );
    p
}
unsafe fn selection64<I: LapackInt>(how: u8, n: isize, select: *const I) -> Vec<i64> {
    if how == b'S' {
        (0..n).map(|i| (*select.offset(i)).val() as i64).collect()
    } else {
        Vec::new()
    }
}
unsafe fn copy_selection<I: LapackInt>(how: u8, source: &[i64], select: *mut I) {
    if how == b'S' {
        for (i, &v) in source.iter().enumerate() {
            *select.add(i) = I::from_i64(v);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: SchurVectors>(imag: T, tol: f64) {
        for n in [0usize, 1, 5] {
            let mut t = Mat::from_fn(n, n, |i, j| {
                if i > j {
                    T::z()
                } else if i == j {
                    T::real((i + 1) as f64) + imag * T::real((i + 2) as f64 / 7.0)
                } else {
                    T::real((i + j + 1) as f64 / 13.0) + imag * T::real((j - i) as f64 / 17.0)
                }
            });
            if !T::COMPLEX && n == 5 {
                t[(0, 0)] = T::real(2.0);
                t[(1, 1)] = T::real(2.0);
                t[(0, 1)] = T::real(3.0);
                t[(1, 0)] = T::real(-2.0);
                t[(3, 3)] = T::real(7.0);
                t[(4, 4)] = T::real(7.0);
                t[(3, 4)] = T::real(-4.0);
                t[(4, 3)] = T::real(1.0);
            }
            let layout = blocks(t.as_ref()).unwrap();
            for left in [false, true] {
                let mut x = vectors(t.as_ref(), left).unwrap();
                normalize(&mut x, &layout).unwrap();
                for &(j, len) in &layout {
                    let max = (0..n).fold(0.0f64, |v, i| {
                        v.max(if len == 2 {
                            x[(i, j)].abs1() + x[(i, j + 1)].abs1()
                        } else {
                            x[(i, j)].abs1()
                        })
                    });
                    assert!((max - 1.0).abs() < tol);
                    for i in 0..n {
                        let apply = |column| {
                            (0..n).fold(T::z(), |sum, k| {
                                sum + if left {
                                    t[(k, i)].cj() * x[(k, column)]
                                } else {
                                    t[(i, k)] * x[(k, column)]
                                }
                            })
                        };
                        if len == 1 {
                            let value = if left { t[(j, j)].cj() } else { t[(j, j)] };
                            assert!((apply(j) - value * x[(i, j)]).abs1() < tol);
                        } else {
                            let real = t[(j, j)];
                            let beta =
                                T::real(t[(j, j + 1)].abs1().sqrt() * t[(j + 1, j)].abs1().sqrt());
                            let sign = if left { T::real(-1.0) } else { T::o() };
                            assert!(
                                (apply(j) - real * x[(i, j)] + sign * beta * x[(i, j + 1)]).abs1()
                                    < tol,
                                "left={left}, n={n}, block={j}, row={i}"
                            );
                            assert!(
                                (apply(j + 1) - sign * beta * x[(i, j)] - real * x[(i, j + 1)])
                                    .abs1()
                                    < tol
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real_selection_query_and_workspace_order() {
        unsafe {
            let nil = core::ptr::null_mut::<f64>();
            let t = [f64::NAN, -2.0, f64::NAN, f64::NAN];
            let mut select = [0i32, -3];
            let (mut m, mut info) = (-777i32, -777i32);
            let mut work = 777.0;
            assert!(compute::<f64, f64, i32>(
                true,
                b'B',
                b'S',
                select.as_mut_ptr(),
                2,
                t.as_ptr(),
                2,
                nil,
                2,
                nil,
                2,
                2,
                &mut m,
                &mut work,
                -1,
                nil,
                0,
                &mut info
            ));
            assert_eq!((info, m), (0, 2));
            assert_eq!(select, [1, 0]);
            assert_eq!(work, 6.0);
            select = [0, -3];
            m = -777;
            assert!(compute::<f64, f64, i32>(
                true,
                b'B',
                b'S',
                select.as_mut_ptr(),
                2,
                t.as_ptr(),
                2,
                nil,
                2,
                nil,
                2,
                0,
                &mut m,
                &mut work,
                5,
                nil,
                0,
                &mut info
            ));
            assert_eq!((info, m), (-14, -777));
            assert_eq!(select, [0, -3]);
            assert_eq!(crate::colmat_blas_last_error(), 14);
            assert!(compute::<f64, f64, i32>(
                false,
                b'B',
                b'S',
                select.as_mut_ptr(),
                2,
                t.as_ptr(),
                2,
                nil,
                2,
                nil,
                2,
                1,
                &mut m,
                nil,
                0,
                nil,
                0,
                &mut info
            ));
            assert_eq!((info, m), (-11, 2));
            assert_eq!(select, [1, 0]);
            assert_eq!(crate::colmat_blas_last_error(), 11);
            select = [0, 0];
            m = -777;
            assert!(compute::<f64, f64, i32>(
                false,
                b'B',
                b'S',
                select.as_mut_ptr(),
                2,
                t.as_ptr(),
                2,
                nil,
                2,
                nil,
                2,
                0,
                &mut m,
                nil,
                0,
                nil,
                0,
                &mut info
            ));
            assert_eq!((info, m), (0, 0));
            // A full-spectrum query does not allocate n columns or read T/Q.
            assert!(compute::<f64, f64, i64>(
                true,
                b'R',
                b'A',
                core::ptr::null_mut(),
                100_000_000,
                nil,
                100_000_000,
                nil,
                1,
                nil,
                100_000_000,
                100_000_000,
                &mut 0i64,
                &mut work,
                -1,
                nil,
                0,
                &mut 0i64
            ));
        }
    }
    #[test]
    fn left_right_schur_vectors_preserve_real_pair_order() {
        verify::<f32>(0.0, 3e-4);
        verify::<f64>(0.0, 3e-11);
        verify::<colmat::c32>(colmat::c32::new(0.0, 1.0), 3e-4);
        verify::<colmat::c64>(colmat::c64::new(0.0, 1.0), 3e-11);
    }
}

use paste::paste;
macro_rules! real_exports {($p:ident,$t:ty,$id:expr,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trevc $s>](side:*const u8,how:*const u8,select:*mut $i,n:*const $i,t:*mut $t,ldt:*const $i,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,mm:*const $i,m:*mut $i,work:*mut $t,info:*mut $i) {
        if compute::<$t,$t,$i>(false,flag(side),flag(how),select,*n as isize,t,*ldt as isize,vl,*ldvl as isize,vr,*ldvr as isize,*mm as isize,m,work,0,core::ptr::null_mut(),0,info) {return;}
        let mut selected=selection64(flag(how),*n as isize,select);
        let (n,ldt,ldvl,ldvr,mm)=(*n as i64,*ldt as i64,*ldvl as i64,*ldvr as i64,*mm as i64);
        let(mut count,mut status)=((*m).val() as i64,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*mut i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*const i64,*const i64,*mut i64,*mut $t,*mut i64,usize,usize)=core::mem::transmute(fallback($id));
        f(side,how,selected.as_mut_ptr(),&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,&mut count,work,&mut status,1,1);
        copy_selection(flag(how),&selected,select);*m=count as $i;*info=status as $i;
    }
    #[no_mangle] pub unsafe extern "C" fn [<$p trevc3 $s>](side:*const u8,how:*const u8,select:*mut $i,n:*const $i,t:*mut $t,ldt:*const $i,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,mm:*const $i,m:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
        if compute::<$t,$t,$i>(true,flag(side),flag(how),select,*n as isize,t,*ldt as isize,vl,*ldvl as isize,vr,*ldvr as isize,*mm as isize,m,work,*lwork as isize,core::ptr::null_mut(),0,info) {return;}
        let mut selected=selection64(flag(how),*n as isize,select);
        let(n,ldt,ldvl,ldvr,mm,lwork)=(*n as i64,*ldt as i64,*ldvl as i64,*ldvr as i64,*mm as i64,*lwork as i64);
        let(mut count,mut status)=((*m).val() as i64,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*mut i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*const i64,*const i64,*mut i64,*mut $t,*const i64,*mut i64,usize,usize)=core::mem::transmute(fallback(4+$id));
        f(side,how,selected.as_mut_ptr(),&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,&mut count,work,&lwork,&mut status,1,1);
        copy_selection(flag(how),&selected,select);*m=count as $i;*info=status as $i;
    }
}};}
macro_rules! complex_exports {($p:ident,$t:ty,$r:ty,$id:expr,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p trevc $s>](side:*const u8,how:*const u8,select:*mut $i,n:*const $i,t:*mut $t,ldt:*const $i,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,mm:*const $i,m:*mut $i,work:*mut $t,rwork:*mut $r,info:*mut $i) {
        if compute::<$t,$r,$i>(false,flag(side),flag(how),select,*n as isize,t,*ldt as isize,vl,*ldvl as isize,vr,*ldvr as isize,*mm as isize,m,work,0,rwork,0,info) {return;}
        let mut selected=selection64(flag(how),*n as isize,select);
        let(n,ldt,ldvl,ldvr,mm)=(*n as i64,*ldt as i64,*ldvl as i64,*ldvr as i64,*mm as i64);
        let(mut count,mut status)=((*m).val() as i64,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*mut i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*const i64,*const i64,*mut i64,*mut $t,*mut $r,*mut i64,usize,usize)=core::mem::transmute(fallback($id));
        f(side,how,selected.as_mut_ptr(),&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,&mut count,work,rwork,&mut status,1,1);
        // Complex SELECT is input only, including noncanonical true values.
        *m=count as $i;*info=status as $i;
    }
    #[no_mangle] pub unsafe extern "C" fn [<$p trevc3 $s>](side:*const u8,how:*const u8,select:*mut $i,n:*const $i,t:*mut $t,ldt:*const $i,vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,mm:*const $i,m:*mut $i,work:*mut $t,lwork:*const $i,rwork:*mut $r,lrwork:*const $i,info:*mut $i) {
        if compute::<$t,$r,$i>(true,flag(side),flag(how),select,*n as isize,t,*ldt as isize,vl,*ldvl as isize,vr,*ldvr as isize,*mm as isize,m,work,*lwork as isize,rwork,*lrwork as isize,info) {return;}
        let mut selected=selection64(flag(how),*n as isize,select);
        let(n,ldt,ldvl,ldvr,mm,lwork,lrwork)=(*n as i64,*ldt as i64,*ldvl as i64,*ldvr as i64,*mm as i64,*lwork as i64,*lrwork as i64);
        let(mut count,mut status)=((*m).val() as i64,0i64);
        let f:unsafe extern "C" fn(*const u8,*const u8,*mut i64,*const i64,*mut $t,*const i64,*mut $t,*const i64,*mut $t,*const i64,*const i64,*mut i64,*mut $t,*const i64,*mut $r,*const i64,*mut i64,usize,usize)=core::mem::transmute(fallback(4+$id));
        f(side,how,selected.as_mut_ptr(),&n,t,&ldt,vl,&ldvl,vr,&ldvr,&mm,&mut count,work,&lwork,rwork,&lrwork,&mut status,1,1);
        *m=count as $i;*info=status as $i;
    }
}};}
macro_rules! abi {
    ($i:ty,$s:tt) => {
        real_exports!(s, f32, 0, $i, $s);
        real_exports!(d, f64, 1, $i, $s);
        complex_exports!(c, colmat::c32, f32, 2, $i, $s);
        complex_exports!(z, colmat::c64, f64, 3, $i, $s);
    };
}
abi!(i32, "_");
abi!(i64, "_64_");
