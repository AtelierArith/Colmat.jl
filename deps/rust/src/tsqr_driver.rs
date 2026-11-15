//! Public LATSQR and TSQR explicit-Q workspace/ABI contracts.
use crate::householder::ReflectorScalar;
use crate::kernels::{matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;
unsafe fn error<I: LapackInt>(info: *mut I, arg: i64) {
    *info = I::from_i64(-arg);
    crate::invalid(arg as i32);
}
fn slots(m: isize, n: isize, mb: isize) -> isize {
    if mb > n && mb < m {
        (m - n) / (mb - n) + isize::from((m - n) % (mb - n) != 0)
    } else {
        1
    }
}
unsafe fn factor<T: ReflectorScalar, I: LapackInt>(
    m: isize,
    n: isize,
    mb: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && m >= n),
        (3, mb >= 1),
        (4, nb >= 1 && (nb <= n || n == 0)),
        (6, lda >= m.max(1)),
        (8, ldt >= nb),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let Some(minimum) = (if n == 0 { Some(1) } else { n.checked_mul(nb) }) else {
        error(info, 10);
        return;
    };
    if lwork != -1 && lwork < minimum {
        error(info, 10);
        return;
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork == -1 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 6);
        return;
    }
    let Some(columns) = n.checked_mul(slots(m, n, mb)) else {
        error(info, 8);
        return;
    };
    if !matrix_addressable::<T>(nb, columns, ldt) {
        error(info, 8);
        return;
    }
    crate::tsqr::factor(
        m as usize,
        n as usize,
        mb as usize,
        nb as usize,
        a,
        lda,
        t,
        ldt,
    );
}
unsafe fn generate<T: Scalar, I: LapackInt>(
    row: bool,
    m: isize,
    n: isize,
    mb: isize,
    nb: isize,
    a: *mut T,
    lda: isize,
    t: *const T,
    ldt: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && m >= n),
        (3, mb > n),
        (4, nb >= 1),
        (6, lda >= m.max(1)),
        (8, ldt >= nb.min(n).max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let block = nb.min(n);
    let Some(optimum) = (if row {
        block.checked_mul(block.max(n - block))
    } else {
        m.checked_add(block).and_then(|x| x.checked_mul(n))
    }) else {
        error(info, 10);
        return;
    };
    let minimum = if row { 1 } else { optimum.max(2) };
    if lwork != -1 && lwork < minimum {
        error(info, 10);
        return;
    }
    *work = crate::lapack::workspace::<T>(optimum.max(if row { 1 } else { 2 }));
    if lwork == -1 || n == 0 {
        return;
    }
    if !matrix_addressable::<T>(m, n, lda) {
        error(info, 6);
        return;
    }
    let Some(columns) = n.checked_mul(slots(m, n, mb)) else {
        error(info, 8);
        return;
    };
    if !matrix_addressable::<T>(block, columns, ldt) {
        error(info, 8);
        return;
    }
    let q = crate::tsqr::thin_q(
        m as usize,
        n as usize,
        mb as usize,
        block as usize,
        a,
        lda,
        t,
        ldt,
    );
    for j in 0..n {
        for i in 0..m {
            *a.offset(i + j * lda) = q[(i + j * m) as usize];
        }
    }
}
unsafe fn getsqrhrt<T: ReflectorScalar, I: LapackInt>(
    m: isize,
    n: isize,
    mb: isize,
    nb1: isize,
    nb2: isize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, m >= 0),
        (2, n >= 0 && m >= n),
        (3, mb > n),
        (4, nb1 >= 1),
        (5, nb2 >= 1),
        (7, lda >= m.max(1)),
        (9, ldt >= nb2.min(n).max(1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let b1 = nb1.min(n);
    let b2 = nb2.min(n);
    let sizes = (|| {
        let columns = n.checked_mul(slots(m, n, mb))?;
        let lwt = b1.checked_mul(columns)?;
        let square = n.checked_mul(n)?;
        let lw1 = b1.checked_mul(n)?;
        let lw2 = b1.checked_mul(b1.max(n - b1))?;
        let minimum = lwt
            .checked_add(lw1)?
            .max(lwt.checked_add(square)?.checked_add(lw2.max(n))?)
            .max(1);
        Some((columns, lwt, minimum))
    })();
    let Some((columns, lwt, minimum)) = sizes else {
        error(info, 11);
        return;
    };
    if lwork != -1 && lwork < minimum {
        error(info, 11);
        return;
    }
    *work = crate::lapack::workspace::<T>(minimum);
    if lwork == -1 || n == 0 {
        return;
    }
    for (arg, ok) in [
        (7, matrix_addressable::<T>(m, n, lda)),
        (9, matrix_addressable::<T>(b2, n, ldt)),
        (11, matrix_addressable::<T>(b1, columns, b1)),
    ] {
        if !ok {
            error(info, arg);
            return;
        }
    }
    let mut tsqr_t = vec![T::z(); lwt as usize];
    crate::tsqr::factor(
        m as usize,
        n as usize,
        mb as usize,
        b1 as usize,
        a,
        lda,
        tsqr_t.as_mut_ptr(),
        b1,
    );
    let mut r = vec![T::z(); (n * n) as usize];
    for j in 0..n {
        for i in 0..=j {
            r[(i + j * n) as usize] = *a.offset(i + j * lda);
        }
    }
    let q = crate::tsqr::thin_q(
        m as usize,
        n as usize,
        mb as usize,
        b1 as usize,
        a,
        lda,
        tsqr_t.as_ptr(),
        b1,
    );
    for j in 0..n {
        for i in 0..m {
            *a.offset(i + j * lda) = q[(i + j * m) as usize];
        }
    }
    let mut d = vec![T::z(); n as usize];
    crate::householder_reconstruction::reconstruct(
        m as usize,
        n as usize,
        b2 as usize,
        a,
        lda,
        t,
        ldt,
        d.as_mut_ptr(),
    );
    for j in 0..n {
        for i in 0..=j {
            *a.offset(i + j * lda) = d[i as usize] * r[(i + j * n) as usize];
        }
    }
}
macro_rules! exports {($p:ident,$org:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p getsqrhrt $s>](m:*const $i,n:*const $i,mb:*const $i,nb1:*const $i,nb2:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {getsqrhrt(*m as isize,*n as isize,*mb as isize,*nb1 as isize,*nb2 as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p latsqr $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*mut $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {factor(*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $org $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*const $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {generate(false,*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
    #[no_mangle] pub unsafe extern "C" fn [<$p $org _row $s>](m:*const $i,n:*const $i,mb:*const $i,nb:*const $i,a:*mut $t,lda:*const $i,t:*const $t,ldt:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {generate(true,*m as isize,*n as isize,*mb as isize,*nb as isize,a,*lda as isize,t,*ldt as isize,work,*lwork as isize,info);}
}};}
exports!(s, orgtsqr, f32, i32, _);
exports!(d, orgtsqr, f64, i32, _);
exports!(c, ungtsqr, colmat::c32, i32, _);
exports!(z, ungtsqr, colmat::c64, i32, _);
exports!(s, orgtsqr, f32, i64, _64_);
exports!(d, orgtsqr, f64, i64, _64_);
exports!(c, ungtsqr, colmat::c32, i64, _64_);
exports!(z, ungtsqr, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for (m, n) in [(0usize, 0usize), (1, 1), (9, 3), (18, 4)] {
                for nb1 in [1, 3] {
                    for nb2 in [1, 2, 5] {
                        let mb = n + 3;
                        let lda = m + 2;
                        let ldt = nb2 + 2;
                        let mut a = vec![T::real(777.0); lda * n];
                        let mut original = vec![T::z(); m * n];
                        for j in 0..n {
                            for i in 0..m {
                                let x =
                                    T::real(((i * 13 + j * 17 + i * j) % 29) as f64 / 19.0 - 0.7)
                                        + imag * T::real(((i * 7 + j * 11) % 13) as f64 / 23.0);
                                a[i + j * lda] = x;
                                original[i + j * m] = x;
                            }
                        }
                        let mut t = vec![T::real(777.0); ldt * n];
                        let mut work = vec![T::z(); 4096];
                        let mut info = 99i64;
                        getsqrhrt(
                            m as isize,
                            n as isize,
                            mb as isize,
                            nb1,
                            nb2 as isize,
                            a.as_mut_ptr(),
                            lda as isize,
                            t.as_mut_ptr(),
                            ldt as isize,
                            work.as_mut_ptr(),
                            4096,
                            &mut info,
                        );
                        assert_eq!(info, 0);
                        let q = crate::tsqr::thin_q(
                            m,
                            n,
                            m + 1,
                            nb2,
                            a.as_ptr(),
                            lda as isize,
                            t.as_ptr(),
                            ldt as isize,
                        );
                        for j in 0..n {
                            for i in 0..m {
                                let qr =
                                    (0..=j).fold(T::z(), |s, k| s + q[i + k * m] * a[k + j * lda]);
                                assert!((qr - original[i + j * m]).magnitude() < tolerance);
                            }
                            for i in m..lda {
                                assert_eq!(a[i + j * lda], T::real(777.0));
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        verify(0.0f32, 5e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 5e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 5e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 5e-13);
    }
}
