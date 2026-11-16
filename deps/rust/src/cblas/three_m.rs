//! Three real native products: P=Ar*Br, Q=Ai*Bi, R=(Ar+Ai)*(Br+Bi).
//! AB=(P-Q)+i(R-P-Q). This composition stages logical operands and makes
//! no performance claim against ordinary native complex GEMM. As with 3M
//! generally, intermediate sums can overflow or lose precision independently
//! of the final complex product.
use super::*;

macro_rules! implementation {
    ($name:ident,$t:ty,$r:ty) => {
        pub(super) unsafe fn $name(
            ta: u8,
            tb: u8,
            m: isize,
            n: isize,
            k: isize,
            alpha: $t,
            a: *const $t,
            la: isize,
            b: *const $t,
            lb: isize,
            beta: $t,
            c: *mut $t,
            lc: isize,
        ) {
            if m == 0 || n == 0 {
                return;
            }
            if alpha == <$t>::z() || k == 0 {
                crate::kernels::scale(m, n, beta, c, lc);
                return;
            }
            let mut ar = vec![0.0 as $r; (m * k) as usize];
            let mut ai = ar.clone();
            let mut asum = ar.clone();
            let mut br = vec![0.0 as $r; (k * n) as usize];
            let mut bi = br.clone();
            let mut bsum = br.clone();
            for j in 0..k {
                for i in 0..m {
                    let v = crate::kernels::get(a, la, i, j, ta);
                    let p = (i + j * m) as usize;
                    ar[p] = v.re;
                    ai[p] = v.im;
                    asum[p] = v.re + v.im;
                }
            }
            for j in 0..n {
                for i in 0..k {
                    let v = crate::kernels::get(b, lb, i, j, tb);
                    let p = (i + j * k) as usize;
                    br[p] = v.re;
                    bi[p] = v.im;
                    bsum[p] = v.re + v.im;
                }
            }
            let mut p = vec![0.0 as $r; (m * n) as usize];
            let mut q = p.clone();
            let mut r = p.clone();
            for (lhs, rhs, out) in [
                (&ar, &br, &mut p),
                (&ai, &bi, &mut q),
                (&asum, &bsum, &mut r),
            ] {
                crate::kernels::gemm(
                    b'N',
                    b'N',
                    m,
                    n,
                    k,
                    1.0 as $r,
                    lhs.as_ptr(),
                    m,
                    rhs.as_ptr(),
                    k,
                    0.0 as $r,
                    out.as_mut_ptr(),
                    m,
                );
            }
            for j in 0..n {
                for i in 0..m {
                    let index = (i + j * m) as usize;
                    let product = <$t>::new(p[index] - q[index], (r[index] - p[index]) - q[index]);
                    let dst = c.offset(i + j * lc);
                    *dst = alpha * product
                        + if beta == <$t>::z() {
                            <$t>::z()
                        } else {
                            beta * *dst
                        };
                }
            }
        }
    };
}
implementation!(single, c32, f32);
implementation!(double, c64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_and_nonbinary_operands() {
        let (m, n, k) = (3usize, 4usize, 17usize);
        let a: Vec<c64> = (0..m * k)
            .map(|i| c64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let b: Vec<c64> = (0..k * n)
            .map(|i| c64::new((i as f64 * 0.4).cos(), -(i as f64 * 0.4).cos() + 1e-6))
            .collect();
        let a32: Vec<c32> = a
            .iter()
            .map(|v| c32::new(v.re as f32, v.im as f32))
            .collect();
        let b32: Vec<c32> = b
            .iter()
            .map(|v| c32::new(v.re as f32, v.im as f32))
            .collect();
        let mut c = vec![c64::new(f64::NAN, f64::NAN); m * n];
        let mut c32out = vec![c32::new(f32::NAN, f32::NAN); m * n];
        unsafe {
            super::double(
                b'N',
                b'N',
                m as isize,
                n as isize,
                k as isize,
                c64::new(1.0, 0.0),
                a.as_ptr(),
                m as isize,
                b.as_ptr(),
                k as isize,
                c64::new(0.0, 0.0),
                c.as_mut_ptr(),
                m as isize,
            );
            super::single(
                b'N',
                b'N',
                m as isize,
                n as isize,
                k as isize,
                c32::new(1.0, 0.0),
                a32.as_ptr(),
                m as isize,
                b32.as_ptr(),
                k as isize,
                c32::new(0.0, 0.0),
                c32out.as_mut_ptr(),
                m as isize,
            );
        }
        for j in 0..n {
            for i in 0..m {
                let mut reference = c64::new(0.0, 0.0);
                let mut reference32 = c64::new(0.0, 0.0);
                for l in 0..k {
                    reference += a[i + l * m] * b[l + j * k];
                    let (x, y) = (a32[i + l * m], b32[l + j * k]);
                    reference32 +=
                        c64::new(x.re as f64, x.im as f64) * c64::new(y.re as f64, y.im as f64);
                }
                assert!((c[i + j * m] - reference).magnitude() < 1e-11);
                let actual = c32out[i + j * m];
                assert!(
                    (c64::new(actual.re as f64, actual.im as f64) - reference32).magnitude() < 1e-4
                );
            }
        }
    }
    #[test]
    fn c_headers_and_zero_coefficients() {
        unsafe {
            cblas_zgemm3m64_(
                101,
                111,
                111,
                0,
                3,
                4,
                core::ptr::null(),
                core::ptr::null(),
                4,
                core::ptr::null(),
                3,
                core::ptr::null(),
                core::ptr::null_mut(),
                3,
            );
            assert_eq!(crate::colmat_blas_last_error(), 0);
            cblas_zgemm3m64_(
                101,
                111,
                111,
                2,
                3,
                4,
                core::ptr::null(),
                core::ptr::null(),
                3,
                core::ptr::null(),
                3,
                core::ptr::null(),
                core::ptr::null_mut(),
                3,
            );
            assert_eq!(crate::colmat_blas_last_error(), 9);
            let zero = c64::new(0.0, 0.0);
            let mut c = [c64::new(f64::NAN, f64::NAN); 6];
            cblas_zgemm3m64_(
                101,
                111,
                111,
                2,
                3,
                4,
                &zero,
                core::ptr::null(),
                4,
                core::ptr::null(),
                3,
                &zero,
                c.as_mut_ptr(),
                3,
            );
            assert_eq!(c, [zero; 6]);
        }
    }
}
