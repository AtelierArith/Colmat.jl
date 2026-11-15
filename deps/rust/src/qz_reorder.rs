//! Checked generalized Schur moves over the public native real/complex APIs.
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Mat, MatMut};

pub(crate) trait ReorderScalar: Scalar {
    fn native_move(
        s: MatMut<'_, Self>,
        t: MatMut<'_, Self>,
        q: Option<MatMut<'_, Self>>,
        z: Option<MatMut<'_, Self>>,
        from: usize,
        to: usize,
    ) -> Result<usize, usize>;
}
macro_rules! implement {
    ($t:ty,$entry:ident) => {
        impl ReorderScalar for $t {
            fn native_move(
                s: MatMut<'_, Self>,
                t: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                z: Option<MatMut<'_, Self>>,
                from: usize,
                to: usize,
            ) -> Result<usize, usize> {
                use colmat::linalg::gevd::{reorder_schur_scratch, $entry};
                crate::threading::with_par(|par| {
                    let mut memory = MemBuffer::new(reorder_schur_scratch::<Self>(s.nrows()));
                    $entry(s, t, q, z, from, to, par, MemStack::new(&mut memory))
                        .map_err(|error| error.position)
                })
            }
        }
    };
}
implement!(f32, reorder_schur_real);
implement!(f64, reorder_schur_real);
implement!(colmat::c32, reorder_schur_cplx);
implement!(colmat::c64, reorder_schur_cplx);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MoveResult {
    pub first: usize,
    pub position: usize,
    pub rejected: bool,
}

pub(crate) unsafe fn move_pair<T: ReorderScalar>(
    n: usize,
    from: usize,
    to: usize,
    s: *mut T,
    lds: isize,
    t: *mut T,
    ldt: isize,
    q: Option<(*mut T, isize)>,
    z: Option<(*mut T, isize)>,
) -> MoveResult {
    assert!((n == 0 && from == 0 && to == 0) || (from < n && to < n));
    let mut ss = Mat::<T>::from_fn(n, n, |i, j| {
        if i <= j + usize::from(!T::COMPLEX) {
            *s.offset(i as isize + j as isize * lds)
        } else {
            T::z()
        }
    });
    let mut tt = Mat::<T>::from_fn(n, n, |i, j| {
        if i <= j {
            *t.offset(i as isize + j as isize * ldt)
        } else {
            T::z()
        }
    });
    let mut qq =
        q.map(|(p, ld)| Mat::<T>::from_fn(n, n, |i, j| *p.offset(i as isize + j as isize * ld)));
    let mut zz =
        z.map(|(p, ld)| Mat::<T>::from_fn(n, n, |i, j| *p.offset(i as isize + j as isize * ld)));
    let first = if !T::COMPLEX && from > 0 && ss[(from, from - 1)] != T::z() {
        from - 1
    } else {
        from
    };
    let moved = T::native_move(
        ss.as_mut(),
        tt.as_mut(),
        qq.as_mut().map(|m| m.as_mut()),
        zz.as_mut().map(|m| m.as_mut()),
        from,
        to,
    );
    // Native failures retain the state after the last accepted adjacent move.
    // Copy that state back as well as successful moves, preserving padding.
    for j in 0..n {
        for i in 0..n {
            if i <= j + usize::from(!T::COMPLEX) {
                *s.offset(i as isize + j as isize * lds) = ss[(i, j)];
            }
            if i <= j {
                *t.offset(i as isize + j as isize * ldt) = tt[(i, j)];
            }
            if let (Some((p, ld)), Some(matrix)) = (q, qq.as_ref()) {
                *p.offset(i as isize + j as isize * ld) = matrix[(i, j)];
            }
            if let (Some((p, ld)), Some(matrix)) = (z, zz.as_ref()) {
                *p.offset(i as isize + j as isize * ld) = matrix[(i, j)];
            }
        }
    }
    match moved {
        Ok(position) => MoveResult {
            first,
            position,
            rejected: false,
        },
        Err(position) => MoveResult {
            first,
            position,
            rejected: true,
        },
    }
}

/// Stable partition of whole Schur blocks, using the original selection mask.
/// Either selected row of a real conjugate pair selects both rows. On refusal,
/// the caller retains the factors after the last accepted adjacent exchange.
pub(crate) unsafe fn select_blocks<T: ReorderScalar>(
    n: usize,
    selected: &[bool],
    s: *mut T,
    lds: isize,
    t: *mut T,
    ldt: isize,
    q: Option<(*mut T, isize)>,
    z: Option<(*mut T, isize)>,
) -> (usize, bool) {
    assert_eq!(selected.len(), n);
    let mut widths = Vec::new();
    let mut total = 0;
    let mut k = 0;
    while k < n {
        let width =
            if !T::COMPLEX && k + 1 < n && *s.offset(k as isize + 1 + k as isize * lds) != T::z() {
                2
            } else {
                1
            };
        let take = selected[k..k + width].iter().any(|&x| x);
        if take {
            total += width;
        }
        widths.push((k, width, take));
        k += width;
    }
    let mut target = 0;
    for (from, width, take) in widths {
        if take {
            if from != target && move_pair(n, from, target, s, lds, t, ldt, q, z).rejected {
                return (total, true);
            }
            target += width;
        }
    }
    (total, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selection<T: ReorderScalar>(tol: f64) {
        let n = 4;
        let mut s = vec![T::z(); n * n];
        let mut t = s.clone();
        for i in 0..n {
            s[i + i * n] = T::real((i + 1) as f64);
            t[i + i * n] = T::o();
        }
        let result = unsafe {
            select_blocks(
                n,
                &[false, true, false, true],
                s.as_mut_ptr(),
                n as isize,
                t.as_mut_ptr(),
                n as isize,
                None,
                None,
            )
        };
        assert_eq!(result, (2, false));
        for (i, value) in [2.0, 4.0, 1.0, 3.0].into_iter().enumerate() {
            assert!((s[i + i * n] / t[i + i * n] - T::real(value)).magnitude() < tol);
        }
    }
    #[test]
    fn stable_selected_partition() {
        selection::<f32>(1e-4);
        selection::<f64>(1e-11);
        selection::<colmat::c32>(1e-4);
        selection::<colmat::c64>(1e-11);
    }
    #[test]
    fn selection_of_second_conjugate_row_moves_whole_pair() {
        let mut s = [4.0f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let mut t = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            unsafe {
                select_blocks(
                    3,
                    &[false, false, true],
                    s.as_mut_ptr(),
                    3,
                    t.as_mut_ptr(),
                    3,
                    None,
                    None,
                )
            },
            (2, false)
        );
        assert!(s[1].abs() > 0.5);
        assert!((s[8] / t[8] - 4.0).abs() < 1e-11);
    }
    fn check<T: ReorderScalar>(imaginary: T, tolerance: f64) {
        for (from, to) in [(0, 2), (2, 0), (1, 1)] {
            let n = 3;
            let ld = 5;
            let mut s = vec![T::real(777.0); ld * n];
            let mut t = s.clone();
            let mut q = s.clone();
            let mut z = s.clone();
            for j in 0..n {
                for i in 0..n {
                    s[i + j * ld] = if i == j {
                        T::real((i + 1) as f64) + imaginary * T::real((i + 1) as f64 / 4.0)
                    } else if i < j {
                        T::real(0.125)
                    } else {
                        T::z()
                    };
                    t[i + j * ld] = if i == j { T::o() } else { T::z() };
                    q[i + j * ld] = t[i + j * ld];
                    z[i + j * ld] = t[i + j * ld];
                }
            }
            let original = s.clone();
            let result = unsafe {
                move_pair(
                    n,
                    from,
                    to,
                    s.as_mut_ptr(),
                    ld as isize,
                    t.as_mut_ptr(),
                    ld as isize,
                    Some((q.as_mut_ptr(), ld as isize)),
                    Some((z.as_mut_ptr(), ld as isize)),
                )
            };
            assert!(!result.rejected);
            assert_eq!(result.position, to);
            let expected =
                T::real((from + 1) as f64) + imaginary * T::real((from + 1) as f64 / 4.0);
            assert!((s[to + to * ld] / t[to + to * ld] - expected).magnitude() < tolerance);
            for j in 0..n {
                for i in 0..n {
                    let mut a = T::z();
                    let mut b = T::z();
                    let mut unit = T::z();
                    let mut zunit = T::z();
                    for k in 0..n {
                        unit = unit + q[k + i * ld].cj() * q[k + j * ld];
                        zunit = zunit + z[k + i * ld].cj() * z[k + j * ld];
                        for l in 0..n {
                            a = a + q[i + k * ld] * s[k + l * ld] * z[j + l * ld].cj();
                            b = b + q[i + k * ld] * t[k + l * ld] * z[j + l * ld].cj();
                        }
                    }
                    assert!((a - original[i + j * ld]).magnitude() < tolerance);
                    let identity = if i == j { T::o() } else { T::z() };
                    assert!((b - identity).magnitude() < tolerance);
                    assert!((unit - identity).magnitude() < tolerance);
                    assert!((zunit - identity).magnitude() < tolerance);
                }
            }
            for j in 0..n {
                for i in n..ld {
                    assert_eq!(s[i + j * ld].re(), 777.0);
                    assert_eq!(t[i + j * ld].re(), 777.0);
                    assert_eq!(q[i + j * ld].re(), 777.0);
                    assert_eq!(z[i + j * ld].re(), 777.0);
                }
            }
        }
    }
    #[test]
    fn real32() {
        check::<f32>(0.0, 2e-4);
    }
    #[test]
    fn real64() {
        check::<f64>(0.0, 1e-11);
    }
    #[test]
    fn complex32() {
        check::<colmat::c32>(colmat::c32::new(0.0, 1.0), 2e-4);
    }
    #[test]
    fn complex64() {
        check::<colmat::c64>(colmat::c64::new(0.0, 1.0), 1e-11);
    }
    fn real_pair<T: ReorderScalar>(tol: f64) {
        for from in [0usize, 1] {
            let n = 3;
            let ld = 5;
            let mut s = vec![T::real(777.0); ld * n];
            let mut t = s.clone();
            let mut q = s.clone();
            let mut z = s.clone();
            for j in 0..n {
                for i in 0..n {
                    s[i + j * ld] = T::z();
                    t[i + j * ld] = if i == j { T::o() } else { T::z() };
                    q[i + j * ld] = t[i + j * ld];
                    z[i + j * ld] = t[i + j * ld];
                }
            }
            s[0] = T::o();
            s[1] = T::real(-2.0);
            s[ld] = T::real(2.0);
            s[1 + ld] = T::o();
            s[2 + 2 * ld] = T::real(4.0);
            s[2 * ld] = T::real(0.25);
            s[1 + 2 * ld] = T::real(0.125);
            let original = s.clone();
            let r = unsafe {
                move_pair(
                    n,
                    from,
                    2,
                    s.as_mut_ptr(),
                    ld as isize,
                    t.as_mut_ptr(),
                    ld as isize,
                    Some((q.as_mut_ptr(), ld as isize)),
                    Some((z.as_mut_ptr(), ld as isize)),
                )
            };
            assert!(!r.rejected);
            assert_eq!(r.first, 0);
            assert_eq!(r.position, 1);
            assert!((s[0] / t[0] - T::real(4.0)).magnitude() < tol);
            assert_ne!(s[2 + ld], T::z());
            for j in 0..n {
                for i in 0..n {
                    let mut a = T::z();
                    let mut b = T::z();
                    for k in 0..n {
                        for l in 0..n {
                            a = a + q[i + k * ld] * s[k + l * ld] * z[j + l * ld].cj();
                            b = b + q[i + k * ld] * t[k + l * ld] * z[j + l * ld].cj();
                        }
                    }
                    assert!((a - original[i + j * ld]).magnitude() < tol);
                    assert!((b - if i == j { T::o() } else { T::z() }).magnitude() < tol);
                }
            }
        }
    }
    fn reject_and_empty<T: ReorderScalar>() {
        let mut s = vec![T::z(); 4];
        let mut t = s.clone();
        s[3] = T::real(2.0);
        t[3] = T::o();
        let old_s = s.clone();
        let old_t = t.clone();
        let r = unsafe { move_pair(2, 1, 0, s.as_mut_ptr(), 2, t.as_mut_ptr(), 2, None, None) };
        assert!(r.rejected);
        assert_eq!(r.position, 1);
        assert!(s == old_s && t == old_t);
        let r = unsafe {
            move_pair::<T>(
                0,
                0,
                0,
                std::ptr::null_mut(),
                1,
                std::ptr::null_mut(),
                1,
                None,
                None,
            )
        };
        assert_eq!(
            r,
            MoveResult {
                first: 0,
                position: 0,
                rejected: false
            }
        );
    }
    #[test]
    fn real_conjugate_block_positions() {
        real_pair::<f32>(2e-4);
        real_pair::<f64>(1e-11);
    }
    #[test]
    fn rejected_and_empty_moves() {
        reject_and_empty::<f32>();
        reject_and_empty::<f64>();
        reject_and_empty::<colmat::c32>();
        reject_and_empty::<colmat::c64>();
    }
}
