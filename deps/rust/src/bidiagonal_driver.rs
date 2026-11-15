//! Checked bidiagonal SVD ABI and accumulation contracts.
use crate::kernels::{flag, matrix_addressable, Scalar};
use crate::lapack::LapackInt;
use paste::paste;

unsafe fn bdsqr<R: Scalar, T: Scalar, I: LapackInt>(
    uplo: u8,
    n: isize,
    ncvt: isize,
    nru: isize,
    ncc: isize,
    d: *mut R,
    e: *mut R,
    vt: *mut T,
    ldvt: isize,
    u: *mut T,
    ldu: isize,
    c: *mut T,
    ldc: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    for (arg, ok) in [
        (1, matches!(uplo, b'U' | b'L')),
        (2, n >= 0),
        (3, ncvt >= 0),
        (4, nru >= 0),
        (5, ncc >= 0),
        (9, ldvt >= if ncvt == 0 { 1 } else { n.max(1) }),
        (11, ldu >= nru.max(1)),
        (13, ldc >= if ncc == 0 { 1 } else { n.max(1) }),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    if n == 0 {
        return;
    }
    for (arg, ok) in [
        (2, matrix_addressable::<R>(n, 1, n)),
        (9, matrix_addressable::<T>(n, ncvt, ldvt)),
        (11, matrix_addressable::<T>(nru, n, ldu)),
        (13, matrix_addressable::<T>(n, ncc, ldc)),
        (2, matrix_addressable::<R>(n, n, n)),
    ] {
        if !ok {
            *info = I::from_i64(-arg);
            crate::invalid(arg as i32);
            return;
        }
    }
    let n = n as usize;
    if n == 1 {
        if (*d).re() < 0.0 {
            *d = -*d;
            for j in 0..ncvt as usize {
                *vt.offset(j as isize * ldvt) = -*vt.offset(j as isize * ldvt);
            }
        }
        return;
    }
    let diagonal = std::slice::from_raw_parts(d, n).to_vec();
    let offdiagonal = std::slice::from_raw_parts(e, n - 1).to_vec();
    let Ok((values, left, right)) =
        crate::bidiagonal_svd::decompose(&diagonal, &offdiagonal, uplo == b'U', false)
    else {
        *info = I::from_i64(1);
        return;
    };
    let mut out_vt = vec![T::z(); n * ncvt as usize];
    let mut out_u = vec![T::z(); nru as usize * n];
    let mut out_c = vec![T::z(); n * ncc as usize];
    for j in 0..ncvt as usize {
        for i in 0..n {
            for k in 0..n {
                out_vt[i + j * n] = out_vt[i + j * n]
                    + T::real(right[(k, i)].re()) * *vt.offset(k as isize + j as isize * ldvt);
            }
        }
    }
    for j in 0..n {
        for i in 0..nru as usize {
            for k in 0..n {
                out_u[i + j * nru as usize] = out_u[i + j * nru as usize]
                    + *u.offset(i as isize + k as isize * ldu) * T::real(left[(k, j)].re());
            }
        }
    }
    for j in 0..ncc as usize {
        for i in 0..n {
            for k in 0..n {
                out_c[i + j * n] = out_c[i + j * n]
                    + T::real(left[(k, i)].re()) * *c.offset(k as isize + j as isize * ldc);
            }
        }
    }
    for i in 0..n {
        *d.add(i) = values[i];
    }
    for i in 0..n - 1 {
        *e.add(i) = R::z();
    }
    for j in 0..ncvt as usize {
        for i in 0..n {
            *vt.offset(i as isize + j as isize * ldvt) = out_vt[i + j * n];
        }
    }
    for j in 0..n {
        for i in 0..nru as usize {
            *u.offset(i as isize + j as isize * ldu) = out_u[i + j * nru as usize];
        }
    }
    for j in 0..ncc as usize {
        for i in 0..n {
            *c.offset(i as isize + j as isize * ldc) = out_c[i + j * n];
        }
    }
}
macro_rules! exports {($p:ident,$r:ty,$t:ty,$i:ty,$s:tt)=>{paste!{
    #[no_mangle] pub unsafe extern "C" fn [<$p bdsqr $s>](uplo:*const u8,n:*const $i,ncvt:*const $i,nru:*const $i,ncc:*const $i,d:*mut $r,e:*mut $r,vt:*mut $t,ldvt:*const $i,u:*mut $t,ldu:*const $i,c:*mut $t,ldc:*const $i,_work:*mut $r,info:*mut $i) {
        bdsqr(flag(uplo),*n as isize,*ncvt as isize,*nru as isize,*ncc as isize,d,e,vt,*ldvt as isize,u,*ldu as isize,c,*ldc as isize,info);
    }
}};}
exports!(s, f32, f32, i32, _);
exports!(d, f64, f64, i32, _);
exports!(c, f32, colmat::c32, i32, _);
exports!(z, f64, colmat::c64, i32, _);
exports!(s, f32, f32, i64, _64_);
exports!(d, f64, f64, i64, _64_);
exports!(c, f32, colmat::c32, i64, _64_);
exports!(z, f64, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    fn accumulation<R: Scalar, T: Scalar, I: LapackInt>(imaginary: T, tol: f64) {
        for n in [1usize, 3, 7] {
            for upper in [false, true] {
                let ld = n + 2;
                let diagonal: Vec<R> = (0..n)
                    .map(|i| {
                        R::real(if i % 2 == 0 {
                            -(i as f64 + 1.0)
                        } else {
                            i as f64 + 1.0
                        })
                    })
                    .collect();
                let off: Vec<R> = (0..n.saturating_sub(1))
                    .map(|i| R::real(if i == 2 { 0.0 } else { 0.3 }))
                    .collect();
                let mut d = diagonal.clone();
                let mut e = off.clone();
                let mut u = vec![T::real(777.0); ld * n];
                let mut vt = u.clone();
                let mut c = vec![T::real(777.0); ld * 2];
                for j in 0..n {
                    for i in 0..n {
                        u[i + j * ld] = if i == j { T::o() } else { T::z() };
                        vt[i + j * ld] = u[i + j * ld];
                    }
                }
                for j in 0..2 {
                    for i in 0..n {
                        c[i + j * ld] = T::real((i + j + 1) as f64) / T::real(7.0) + imaginary;
                    }
                }
                let before = c.clone();
                let mut info = I::from_i64(-777);
                unsafe {
                    bdsqr(
                        if upper { b'U' } else { b'L' },
                        n as isize,
                        n as isize,
                        n as isize,
                        2,
                        d.as_mut_ptr(),
                        e.as_mut_ptr(),
                        vt.as_mut_ptr(),
                        ld as isize,
                        u.as_mut_ptr(),
                        ld as isize,
                        c.as_mut_ptr(),
                        ld as isize,
                        &mut info,
                    );
                }
                assert_eq!(info.val(), 0);
                assert_eq!(crate::colmat_blas_last_error(), 0);
                for j in 0..n {
                    for i in 0..n {
                        let mut product = T::z();
                        let mut dot = T::z();
                        for k in 0..n {
                            product = product + u[i + k * ld] * T::real(d[k].re()) * vt[k + j * ld];
                            dot = dot + u[k + i * ld].cj() * u[k + j * ld];
                        }
                        let expected = if i == j {
                            diagonal[i]
                        } else if (upper && j == i + 1) || (!upper && i == j + 1) {
                            off[i.min(j)]
                        } else {
                            R::z()
                        };
                        assert!((product - T::real(expected.re())).magnitude() < tol * n as f64);
                        assert!((dot - if i == j { T::o() } else { T::z() }).magnitude() < tol);
                    }
                }
                for j in 0..2 {
                    for i in 0..n {
                        let mut expected = T::z();
                        for k in 0..n {
                            expected = expected + u[k + i * ld].cj() * before[k + j * ld];
                        }
                        assert!((c[i + j * ld] - expected).magnitude() < tol);
                    }
                }
                for j in 0..n {
                    for i in n..ld {
                        assert_eq!(u[i + j * ld].re(), 777.0);
                        assert_eq!(vt[i + j * ld].re(), 777.0);
                    }
                }
                for j in 0..2 {
                    for i in n..ld {
                        assert_eq!(c[i + j * ld].re(), 777.0);
                    }
                }
                assert!(e.iter().all(|x| x.re() == 0.0));
            }
        }
    }
    #[test]
    fn accumulation_all_scalar_and_integer_abis() {
        accumulation::<f32, f32, i32>(0.0, 1e-4);
        accumulation::<f32, f32, i64>(0.0, 1e-4);
        accumulation::<f64, f64, i32>(0.0, 1e-11);
        accumulation::<f64, f64, i64>(0.0, 1e-11);
        accumulation::<f32, colmat::c32, i32>(colmat::c32::new(0.0, 0.25), 1e-4);
        accumulation::<f32, colmat::c32, i64>(colmat::c32::new(0.0, 0.25), 1e-4);
        accumulation::<f64, colmat::c64, i32>(colmat::c64::new(0.0, 0.25), 1e-11);
        accumulation::<f64, colmat::c64, i64>(colmat::c64::new(0.0, 0.25), 1e-11);
    }
}
