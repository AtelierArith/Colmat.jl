//! One Hessenberg/Schur reduction shared by general LAPACK eigenvalue outputs.
use crate::householder::ReflectorScalar;
use colmat::dyn_stack::{MemBuffer, MemStack, StackReq};
use colmat::linalg::{evd::*, householder::*, qr::no_pivoting::factor::recommended_block_size};
use colmat::{Col, ColMut, Conj, Mat, MatMut, MatRef, Par};

pub(crate) trait SharedEvd: ReflectorScalar {
    fn schur(
        h: MatMut<'_, Self>,
        q: Option<MatMut<'_, Self>>,
        w: ColMut<'_, Self>,
        wi: &mut [f64],
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<(), ()>;
    fn vectors(
        t: MatRef<'_, Self>,
        q: MatRef<'_, Self>,
        out: MatMut<'_, Self>,
        par: Par<'_>,
        stack: &mut MemStack,
    ) -> Result<(), ()>;
}
macro_rules! real {
    ($t:ty) => {
        impl SharedEvd for $t {
            fn schur(
                h: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                w: ColMut<'_, Self>,
                wi: &mut [f64],
                par: Par<'_>,
                stack: &mut MemStack,
            ) -> Result<(), ()> {
                let mut imag = Col::<Self>::zeros(wi.len());
                hessenberg_schur_real(h, q, w, imag.as_mut(), par, stack, Default::default())
                    .map_err(|_| ())?;
                for (i, x) in wi.iter_mut().enumerate() {
                    *x = imag[i] as f64;
                }
                Ok(())
            }
            fn vectors(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                out: MatMut<'_, Self>,
                par: Par<'_>,
                stack: &mut MemStack,
            ) -> Result<(), ()> {
                eigenvectors_from_schur_real(t, Some(q), out, par, stack, Default::default())
                    .map_err(|_| ())
            }
        }
    };
}
macro_rules! complex {
    ($t:ty) => {
        impl SharedEvd for $t {
            fn schur(
                h: MatMut<'_, Self>,
                q: Option<MatMut<'_, Self>>,
                w: ColMut<'_, Self>,
                _wi: &mut [f64],
                par: Par<'_>,
                stack: &mut MemStack,
            ) -> Result<(), ()> {
                hessenberg_schur_cplx(h, q, w, par, stack, Default::default()).map_err(|_| ())
            }
            fn vectors(
                t: MatRef<'_, Self>,
                q: MatRef<'_, Self>,
                out: MatMut<'_, Self>,
                par: Par<'_>,
                stack: &mut MemStack,
            ) -> Result<(), ()> {
                eigenvectors_from_schur_cplx(t, Some(q), out, par, stack, Default::default())
                    .map_err(|_| ())
            }
        }
    };
}
real!(f32);
real!(f64);
complex!(colmat::c32);
complex!(colmat::c64);

pub(crate) fn compute<T: SharedEvd>(
    mut a: MatMut<'_, T>,
    left: bool,
    right: bool,
) -> Result<crate::lapack::GeneralResult<T>, ()> {
    let n = a.nrows();
    let mut max = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let z = a[(i, j)];
            if !z.re().is_finite() || !z.im().is_finite() {
                return Err(());
            }
            max = max.max(z.re().abs()).max(z.im().abs());
        }
    }
    // A positive power of two protects public low-level kernels from extremes.
    // Eigenvectors are computed in these units, before restoring T and W.
    let small = (T::SAFE * (T::EPS / 2.0)).sqrt() / T::EPS;
    let large = 1.0 / small;
    let mut factor = 1.0;
    if max > 0.0 {
        while max < small {
            max *= 2.0;
            factor *= 2.0;
        }
        while max > large {
            max *= 0.5;
            factor *= 0.5;
        }
    }
    if factor != 1.0 {
        for j in 0..n {
            for i in 0..n {
                let z = a[(i, j)];
                a[(i, j)] = T::parts(z.re() * factor, z.im() * factor);
            }
        }
    }
    let want_q = left || right;
    let mut q = Mat::<T>::zeros(n, if want_q { n } else { 0 });
    let mut vl = Mat::<T>::zeros(n, if left { n } else { 0 });
    let mut vr = Mat::<T>::zeros(n, if right { n } else { 0 });
    let mut w = Col::<T>::zeros(n);
    let mut wi = vec![0.0; n];
    crate::threading::with_par_if(
        crate::threading::real_work(n, n, n, T::COMPLEX),
        crate::threading::GENERAL_EVD_WORK,
        |mut par| {
            let bs = if n > 1 {
                recommended_block_size::<T>(n - 1, n - 1)
            } else {
                1
            };
            let mut coeff = Mat::<T>::zeros(bs, n.saturating_sub(1));
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                if n > 1 {
                    hessenberg::hessenberg_in_place_scratch::<T>(
                        n,
                        bs,
                        par.shape(),
                        Default::default(),
                    )
                } else {
                    StackReq::EMPTY
                },
                hessenberg_schur_scratch::<T>(n, want_q, par.shape(), Default::default()),
                if want_q {
                    eigenvectors_from_schur_scratch::<T>(n)
                } else {
                    StackReq::EMPTY
                },
                if n > 1 && want_q {
                    apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(
                        n - 1,
                        bs,
                        n - 1,
                    )
                } else {
                    StackReq::EMPTY
                },
            ]));
            if n > 1 {
                {
                    let _phase = crate::evd_profile::Scope::new(3);
                    hessenberg::hessenberg_in_place(
                        a.as_mut(),
                        coeff.as_mut(),
                        par.reborrow(),
                        MemStack::new(&mut mem),
                        Default::default(),
                    );
                }
            }
            if want_q {
                let _phase = crate::evd_profile::Scope::new(13);
                for i in 0..n {
                    q[(i, i)] = T::o();
                }
                if n > 1 {
                    apply_block_householder_sequence_on_the_right_in_place_with_conj(
                        a.as_ref().submatrix(1, 0, n - 1, n - 1),
                        coeff.as_ref(),
                        Conj::No,
                        q.as_mut().submatrix_mut(1, 1, n - 1, n - 1),
                        par.reborrow(),
                        MemStack::new(&mut mem),
                    );
                }
            }
            for j in 0..n {
                for i in j + 2..n {
                    a[(i, j)] = T::z();
                }
            }
            {
                let _phase = crate::evd_profile::Scope::new(4);
                T::schur(
                    a.as_mut(),
                    if want_q { Some(q.as_mut()) } else { None },
                    w.as_mut(),
                    &mut wi,
                    par.reborrow(),
                    MemStack::new(&mut mem),
                )?;
            }
            let _phase = crate::evd_profile::Scope::new(12);
            if right {
                T::vectors(
                    a.as_ref(),
                    q.as_ref(),
                    vr.as_mut(),
                    par.reborrow(),
                    MemStack::new(&mut mem),
                )?;
            }
            if left {
                // U = J T^T J is upper Schur. With Q_l = J conj(Q) J,
                // conjugating/reversing the result gives left vectors of Q T Q^H.
                for j in 0..n {
                    for i in 0..n {
                        q[(i, j)] = q[(i, j)].cj();
                    }
                }
                T::vectors(
                    a.as_ref().reverse_rows().reverse_cols().transpose(),
                    q.as_ref().reverse_rows().reverse_cols(),
                    vl.as_mut().reverse_rows_mut().reverse_cols_mut(),
                    par.reborrow(),
                    MemStack::new(&mut mem),
                )?;
                if T::COMPLEX {
                    for j in 0..n {
                        for i in 0..n {
                            vl[(i, j)] = vl[(i, j)].cj();
                        }
                    }
                } else {
                    let mut j = 0;
                    while j < n {
                        if wi[j] > 0.0 && j + 1 < n {
                            for i in 0..n {
                                let x = vl[(i, j)];
                                vl[(i, j)] = vl[(i, j + 1)];
                                vl[(i, j + 1)] = -x;
                            }
                            j += 2;
                        } else {
                            j += 1;
                        }
                    }
                }
            }
            Ok::<(), ()>(())
        },
    )?;
    if factor != 1.0 {
        for j in 0..n {
            for i in 0..n {
                let z = a[(i, j)];
                a[(i, j)] = T::parts(z.re() / factor, z.im() / factor);
            }
        }
        for i in 0..n {
            let z = w[i];
            w[i] = T::parts(z.re() / factor, z.im() / factor);
            wi[i] /= factor;
        }
    }
    Ok(crate::lapack::GeneralResult {
        w: (0..n).map(|i| w[i]).collect(),
        wi,
        left: vl,
        right: vr,
    })
}
