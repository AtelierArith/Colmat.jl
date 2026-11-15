//! LAPACK TSQR row-block representation over native compact-WY QR panels.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;

fn groups(m: usize, n: usize, mb: usize) -> Vec<(usize, usize)> {
    if m == 0 || n == 0 {
        return vec![];
    }
    let first = if mb <= n || mb >= m { m } else { mb };
    let mut groups = vec![(0, first)];
    let mut start = first;
    while start < m {
        let end = start.saturating_add(mb - n).min(m);
        groups.push((start, end));
        start = end;
    }
    groups
}
pub(crate) unsafe fn factor<T: ReflectorScalar>(
    m: usize,
    n: usize,
    mb: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    t: *mut T,
    ldt: isize,
) {
    for (slot, (start, end)) in groups(m, n, mb).into_iter().enumerate() {
        let first = slot == 0;
        let height = if first { end } else { n + end - start };
        // Later row blocks factor [R; B], with an explicit zero strict
        // lower triangle in R. Earlier stored reflectors are never read
        // as if they were entries of this triangular R.
        let mut panel = vec![T::z(); height * n];
        for j in 0..n {
            if first {
                for i in 0..height {
                    panel[i + j * height] = *a.offset(i as isize + j as isize * lda);
                }
            } else {
                for i in 0..=j {
                    panel[i + j * height] = *a.offset(i as isize + j as isize * lda);
                }
                for i in start..end {
                    panel[n + i - start + j * height] = *a.offset(i as isize + j as isize * lda);
                }
            }
        }
        crate::qr::compact_qr(
            height,
            n,
            nb,
            panel.as_mut_ptr(),
            height as isize,
            t.offset((slot * n) as isize * ldt),
            ldt,
        );
        for j in 0..n {
            if first {
                for i in 0..height {
                    *a.offset(i as isize + j as isize * lda) = panel[i + j * height];
                }
            } else {
                for i in 0..=j {
                    *a.offset(i as isize + j as isize * lda) = panel[i + j * height];
                }
                for i in start..end {
                    *a.offset(i as isize + j as isize * lda) = panel[n + i - start + j * height];
                }
            }
        }
    }
}
struct Block<T: Scalar> {
    indices: Vec<usize>,
    v: Vec<T>,
    t: Vec<T>,
    width: usize,
}
unsafe fn gather<T: Scalar>(
    order: usize,
    k: usize,
    mb: usize,
    nb: usize,
    a: *const T,
    lda: isize,
    t: *const T,
    ldt: isize,
) -> Vec<Block<T>> {
    let mut blocks = vec![];
    for (slot, (start, end)) in groups(order, k, mb).into_iter().enumerate() {
        for j in (0..k).step_by(nb) {
            let width = nb.min(k - j);
            let indices: Vec<usize> = if slot == 0 {
                (j..end).collect()
            } else {
                (j..j + width).chain(start..end).collect()
            };
            let height = indices.len();
            let mut v = vec![T::z(); height * width];
            let mut coefficients = vec![T::z(); width * width];
            for col in 0..width {
                for i in 0..height {
                    v[i + col * height] = if i == col {
                        T::o()
                    } else if i < col || (slot != 0 && i < width) {
                        T::z()
                    } else {
                        *a.offset(indices[i] as isize + (j + col) as isize * lda)
                    };
                }
                for i in 0..=col {
                    coefficients[i + col * width] =
                        *t.offset(i as isize + (slot * k + j + col) as isize * ldt);
                }
            }
            blocks.push(Block {
                indices,
                v,
                t: coefficients,
                width,
            });
        }
    }
    blocks
}
pub(crate) unsafe fn apply<T: Scalar>(
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    mb: usize,
    nb: usize,
    a: *const T,
    lda: isize,
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    let mut blocks = gather(order, k, mb, nb, a, lda, t, ldt);
    if (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N') {
        blocks.reverse();
    }
    for block in blocks {
        let height = block.indices.len();
        let (rows, cols) = if side == b'L' {
            (height, n)
        } else {
            (m, height)
        };
        let mut active = vec![T::z(); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                let (row, col) = if side == b'L' {
                    (block.indices[i], j)
                } else {
                    (i, block.indices[j])
                };
                active[i + j * rows] = *c.offset(row as isize + col as isize * ldc);
            }
        }
        crate::householder::apply_block(
            side,
            tr,
            b'F',
            b'C',
            rows,
            cols,
            block.width,
            block.v.as_ptr(),
            height as isize,
            block.t.as_ptr(),
            block.width as isize,
            active.as_mut_ptr(),
            rows as isize,
        );
        for j in 0..cols {
            for i in 0..rows {
                let (row, col) = if side == b'L' {
                    (block.indices[i], j)
                } else {
                    (i, block.indices[j])
                };
                *c.offset(row as isize + col as isize * ldc) = active[i + j * rows];
            }
        }
    }
}
pub(crate) unsafe fn thin_q<T: Scalar>(
    m: usize,
    n: usize,
    mb: usize,
    nb: usize,
    a: *const T,
    lda: isize,
    t: *const T,
    ldt: isize,
) -> Vec<T> {
    let mut q = vec![T::z(); m * n];
    for j in 0..n {
        q[j + j * m] = T::o();
    }
    apply(
        b'L',
        b'N',
        m,
        n,
        n,
        mb,
        nb,
        a,
        lda,
        t,
        ldt,
        q.as_mut_ptr(),
        m as isize,
    );
    q
}
#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for (m, n) in [(0usize, 0usize), (1, 1), (5, 2), (12, 3), (17, 4)] {
                for mb in [1, n + 1, n + 4, m + 2] {
                    for nb in [1, n.max(1).min(2), n.max(1)] {
                        let lda = m + 2;
                        let ldt = nb + 1;
                        let slots = groups(m, n, mb).len().max(1);
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
                        let mut t = vec![T::real(777.0); ldt * n * slots];
                        factor(
                            m,
                            n,
                            mb,
                            nb,
                            a.as_mut_ptr(),
                            lda as isize,
                            t.as_mut_ptr(),
                            ldt as isize,
                        );
                        let q = thin_q(
                            m,
                            n,
                            mb,
                            nb,
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
                        for j in 0..n {
                            for i in 0..n {
                                let dot = (0..m)
                                    .fold(T::z(), |s, k| s + q[k + i * m].cj() * q[k + j * m]);
                                assert!(
                                    (dot - if i == j { T::o() } else { T::z() }).magnitude()
                                        < tolerance
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        verify(0.0f32, 3e-5);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 3e-13);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 4e-5);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 4e-13);
    }
}
