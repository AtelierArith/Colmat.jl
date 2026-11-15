//! Triangular-pentagonal factors and block reflectors over native kernels.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;

fn active(forward: bool, tail: usize, k: usize, l: usize, row: usize, col: usize) -> bool {
    if forward {
        row < tail - l || row - (tail - l) <= col
    } else {
        row >= l || col <= k - l + row
    }
}
pub(crate) unsafe fn factor<T: ReflectorScalar>(
    lq: bool,
    m: usize,
    n: usize,
    l: usize,
    nb: usize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    t: *mut T,
    ldt: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (k, tail) = if lq { (m, n) } else { (n, m) };
    let order = k + tail;
    let mut panel = vec![T::z(); order * k];
    let mut tau = vec![T::z(); k];
    for j in 0..k {
        for i in 0..=j {
            panel[i + j * order] = if lq {
                (*a.offset(j as isize + i as isize * lda)).cj()
            } else {
                *a.offset(i as isize + j as isize * lda)
            };
        }
        for i in 0..tail {
            if active(true, tail, k, l, i, j) {
                panel[k + i + j * order] = if lq {
                    (*b.offset(j as isize + i as isize * ldb)).cj()
                } else {
                    *b.offset(i as isize + j as isize * ldb)
                };
            }
        }
    }
    crate::qr::standard_qr(
        order,
        k,
        panel.as_mut_ptr(),
        order as isize,
        tau.as_mut_ptr(),
        false,
    );
    for j in (0..k).step_by(nb) {
        crate::householder::factor(
            b'F',
            b'C',
            order - j,
            nb.min(k - j),
            panel.as_ptr().add(j + j * order),
            order as isize,
            tau.as_ptr().add(j),
            t.offset(j as isize * ldt),
            ldt,
        );
    }
    for j in 0..k {
        for i in 0..=j {
            if lq {
                *a.offset(j as isize + i as isize * lda) = panel[i + j * order].cj();
            } else {
                *a.offset(i as isize + j as isize * lda) = panel[i + j * order];
            }
        }
        for i in 0..tail {
            if active(true, tail, k, l, i, j) {
                if lq {
                    *b.offset(j as isize + i as isize * ldb) = panel[k + i + j * order].cj();
                } else {
                    *b.offset(i as isize + j as isize * ldb) = panel[k + i + j * order];
                }
            }
        }
    }
}
pub(crate) unsafe fn block<T: Scalar>(
    side: u8,
    tr: u8,
    direct: u8,
    store: u8,
    m: usize,
    n: usize,
    k: usize,
    l: usize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let left = side == b'L';
    let forward = direct == b'F';
    let tail = if left { m } else { n };
    let order = k + tail;
    let mut basis = vec![T::z(); order * k];
    for j in 0..k {
        basis[(if forward { j } else { tail + j }) + j * order] = T::o();
        for i in 0..tail {
            if active(forward, tail, k, l, i, j) {
                basis[(if forward { k + i } else { i }) + j * order] = if store == b'C' {
                    *v.offset(i as isize + j as isize * ldv)
                } else {
                    (*v.offset(j as isize + i as isize * ldv)).cj()
                };
            }
        }
    }
    let (rows, cols) = if left { (order, n) } else { (m, order) };
    let mut c = vec![T::z(); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            let axis = if left { i } else { j };
            let head = if forward { axis < k } else { axis >= tail };
            let local = if head {
                if forward {
                    axis
                } else {
                    axis - tail
                }
            } else if forward {
                axis - k
            } else {
                axis
            };
            let (r, col) = if left { (local, j) } else { (i, local) };
            c[i + j * rows] = if head {
                *a.offset(r as isize + col as isize * lda)
            } else {
                *b.offset(r as isize + col as isize * ldb)
            };
        }
    }
    crate::householder::apply_block(
        side,
        tr,
        direct,
        b'C',
        rows,
        cols,
        k,
        basis.as_ptr(),
        order as isize,
        t,
        ldt,
        c.as_mut_ptr(),
        rows as isize,
    );
    for j in 0..cols {
        for i in 0..rows {
            let axis = if left { i } else { j };
            let head = if forward { axis < k } else { axis >= tail };
            let local = if head {
                if forward {
                    axis
                } else {
                    axis - tail
                }
            } else if forward {
                axis - k
            } else {
                axis
            };
            let (r, col) = if left { (local, j) } else { (i, local) };
            if head {
                *a.offset(r as isize + col as isize * lda) = c[i + j * rows];
            } else {
                *b.offset(r as isize + col as isize * ldb) = c[i + j * rows];
            }
        }
    }
}
pub(crate) unsafe fn apply<T: Scalar>(
    lq: bool,
    side: u8,
    tr: u8,
    m: usize,
    n: usize,
    k: usize,
    l: usize,
    nb: usize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    a: *mut T,
    lda: isize,
    b: *mut T,
    ldb: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let tail = if side == b'L' { m } else { n };
    let mut starts: Vec<usize> = (0..k).step_by(nb).collect();
    let reverse = (side == b'L' && tr == b'N') || (side == b'R' && tr != b'N');
    if reverse != lq {
        starts.reverse();
    }
    let block_tr = if lq {
        if tr == b'N' {
            if T::COMPLEX {
                b'C'
            } else {
                b'T'
            }
        } else {
            b'N'
        }
    } else {
        tr
    };
    for j in starts {
        let width = nb.min(k - j);
        let span = tail - l + l.min(j + width);
        let overlap = if j < l { width.min(l - j) } else { 0 };
        let (rows, cols) = if side == b'L' { (span, n) } else { (m, span) };
        block(
            side,
            block_tr,
            b'F',
            if lq { b'R' } else { b'C' },
            rows,
            cols,
            width,
            overlap,
            v.offset(if lq { j as isize } else { j as isize * ldv }),
            ldv,
            t.offset(j as isize * ldt),
            ldt,
            a.offset(if side == b'L' {
                j as isize
            } else {
                j as isize * lda
            }),
            lda,
            b,
            ldb,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn verify<T: ReflectorScalar>(imag: T, tolerance: f64) {
        unsafe {
            for lq in [false, true] {
                for k in [1usize, 3, 5] {
                    for tail in [1usize, 2, 4] {
                        for l in [0, k.min(tail)] {
                            for nb in [1, k.min(2), k] {
                                let (m, n) = if lq { (k, tail) } else { (tail, k) };
                                let lda = k + 2;
                                let ldb = m + 2;
                                let ldt = nb + 2;
                                let order = k + tail;
                                let mut a = vec![T::real(777.0); lda * k];
                                let mut b = vec![T::real(777.0); ldb * n];
                                let mut t = vec![T::real(777.0); ldt * k];
                                let mut original = vec![T::z(); order * k];
                                for j in 0..k {
                                    for i in 0..k {
                                        let value =
                                            T::real(((i * 7 + j * 11) % 19) as f64 / 17.0 - 0.3)
                                                + imag * T::real(((i + j * 3) % 11) as f64 / 23.0);
                                        let (row, col) = if lq { (j, i) } else { (i, j) };
                                        a[row + col * lda] = if i <= j {
                                            if lq {
                                                value.cj()
                                            } else {
                                                value
                                            }
                                        } else {
                                            T::real(f64::NAN)
                                        };
                                        if i <= j {
                                            original[i + j * order] = value;
                                        }
                                    }
                                }
                                for j in 0..k {
                                    for i in 0..tail {
                                        let value =
                                            T::real(((i * 13 + j * 17) % 29) as f64 / 19.0 - 0.7)
                                                + imag
                                                    * T::real(
                                                        ((i * 7 + j * 11) % 13) as f64 / 23.0,
                                                    );
                                        let exists = i < tail - l || i - (tail - l) <= j;
                                        let (row, col) = if lq { (j, i) } else { (i, j) };
                                        b[row + col * ldb] = if exists {
                                            if lq {
                                                value.cj()
                                            } else {
                                                value
                                            }
                                        } else {
                                            T::real(f64::NAN)
                                        };
                                        if exists {
                                            original[k + i + j * order] = value;
                                        }
                                    }
                                }
                                let before_a = a.clone();
                                let before_b = b.clone();
                                factor(
                                    lq,
                                    m,
                                    n,
                                    l,
                                    nb,
                                    a.as_mut_ptr(),
                                    lda as isize,
                                    b.as_mut_ptr(),
                                    ldb as isize,
                                    t.as_mut_ptr(),
                                    ldt as isize,
                                );
                                let mut q = vec![T::z(); order * order];
                                for j in 0..order {
                                    q[j + j * order] = T::o();
                                }
                                // QR Q, or the adjoint of the LQ Q, reconstructs the common
                                // stacked [R;B] representation used to build this fixture.
                                apply(
                                    lq,
                                    b'L',
                                    if lq {
                                        if T::COMPLEX {
                                            b'C'
                                        } else {
                                            b'T'
                                        }
                                    } else {
                                        b'N'
                                    },
                                    tail,
                                    order,
                                    k,
                                    l,
                                    nb,
                                    b.as_ptr(),
                                    ldb as isize,
                                    t.as_ptr(),
                                    ldt as isize,
                                    q.as_mut_ptr(),
                                    order as isize,
                                    q.as_mut_ptr().add(k),
                                    order as isize,
                                );
                                for j in 0..k {
                                    for i in 0..order {
                                        let value = (0..=j).fold(T::z(), |s, c| {
                                            s + q[i + c * order]
                                                * if lq {
                                                    a[j + c * lda].cj()
                                                } else {
                                                    a[c + j * lda]
                                                }
                                        });
                                        assert!(
                                            (value - original[i + j * order]).magnitude()
                                                < tolerance
                                        );
                                    }
                                }
                                for j in 0..order {
                                    for i in 0..order {
                                        let dot = (0..order).fold(T::z(), |s, c| {
                                            s + q[c + i * order].cj() * q[c + j * order]
                                        });
                                        assert!(
                                            (dot - if i == j { T::o() } else { T::z() })
                                                .magnitude()
                                                < tolerance
                                        );
                                    }
                                }
                                for j in 0..k {
                                    for i in 0..k {
                                        if (lq && i < j) || (!lq && i > j) {
                                            assert!((a[i + j * lda]).re().is_nan());
                                        }
                                    }
                                    for i in k..lda {
                                        assert_eq!(a[i + j * lda], before_a[i + j * lda]);
                                    }
                                }
                                for j in 0..n {
                                    for i in 0..m {
                                        let (row, col) = if lq { (j, i) } else { (i, j) };
                                        if !(row < tail - l || row - (tail - l) <= col) {
                                            assert!(b[i + j * ldb].re().is_nan());
                                        }
                                    }
                                    for i in m..ldb {
                                        assert_eq!(b[i + j * ldb], before_b[i + j * ldb]);
                                    }
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
        verify(0.0f32, 1e-4);
    }
    #[test]
    fn real64() {
        verify(0.0f64, 1e-12);
    }
    #[test]
    fn complex32() {
        verify(colmat::c32::new(0.0, 1.0), 1e-4);
    }
    #[test]
    fn complex64() {
        verify(colmat::c64::new(0.0, 1.0), 1e-12);
    }
}
