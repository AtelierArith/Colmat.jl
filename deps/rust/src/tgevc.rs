//! Generalized eigenvectors of an existing upper (quasi-)triangular pencil.
//! The back-substitution and protected small solves follow Reference LAPACK
//! v3.12.1 SRC/{s,d,c,z}tgevc.f and SRC/dlaln2.f. The numerical kernels are a
//! local adaptation of colmat-rs's private `qz_to_gevd_{real,cplx}` helpers
//! (revision 4e6ed0e34659147a5c744ab6431f150e78db6f82), so TGEVC consumes the
//! caller's Schur pencil instead of repeating a QZ factorization.
use crate::householder::ReflectorScalar;
use crate::lapack::LapackInt;
use colmat::dyn_stack::{MemBuffer, MemStack, StackReq};
use colmat::linalg::temp_mat_scratch;
use colmat::mat::AsMatMut;
use colmat::prelude::*;
use colmat::traits::ext::*;
use colmat::traits::math_utils;
use colmat::traits::{ComplexField, RealField};
use colmat::Accum;
use num_complex::Complex;

fn solve_shifted_1x1<T: ComplexField>(smin: T::Real, ca: T::Real, A: T, d0: T, B: &mut T, w: T) {
    let safmin = &math_utils::min_positive::<T::Real>();
    let smlnum = &(safmin + safmin);
    let smin = &smin.fmax(smlnum);
    let CR = A.mul_real(ca) - w * d0;
    let cmax = CR.abs();
    if cmax < *smin {
        // protected denominator: `d = max(|d|, smin)` instead of dividing by
        // the shift twice (LAPACK `dtgevc`: `IF( ABS( D ).LE.DMIN ) D = DMIN`)
        *B = B.mul_real(smin.recip());
        return;
    }
    *B *= CR.recip();
}
#[allow(clippy::toplevel_ref_arg)] // `ref` parameter keeps the upstream signature; the body wants `&T` without derefing each use
fn solve_complex_shifted_1x1<T: RealField>(
    smin: T,
    ca: T,
    A: MatRef<'_, T>,
    ref d0: T,
    mut B: MatMut<'_, T>,
    wr: T,
    wi: T,
) {
    let nw = B.ncols();
    let safmin = &math_utils::min_positive::<T>();
    let smlnum = &(safmin + safmin);
    let smin = &smin.fmax(smlnum);
    let CR = ca * &A[(0, 0)] - wr * d0;
    let CI = -wi * d0;
    let cmax = CR.abs().fmax(CI.abs());
    if cmax < *smin {
        // protected denominator, as in `solve_shifted_1x1` (LAPACK `ztgevc`:
        // `IF( CABS1( D ).LE.DMIN ) D = CMPLX( DMIN )`); dividing by the
        // complex shift afterwards would scale it twice
        let smin_inv = &smin.recip();
        zip!(B.rb_mut()).for_each(|unzip!(x)| *x *= smin_inv);
        return;
    }
    if nw == 1 {
        let C = &CR.recip();
        zip!(B.rb_mut()).for_each(|unzip!(x)| *x *= C);
    } else {
        let (Br, Bi) = B.two_cols_mut(0, 1);
        let C = Complex { re: CR, im: CI }.recip();
        zip!(Br, Bi).for_each(|unzip!(re, im)| {
            (*re, *im) = (&*re * &C.re - &*im * &C.im, &*re * &C.im + &*im * &C.re);
        });
    }
}
// Reserve room for the two elimination/substitution steps, including complex
// products. The caller protects both pivots before using this bound.
//
// What it guarantees: with `limit = min(pivot_min, 1) / (16 * min_positive)`
// and the right hand side scaled to `|b|_max <= limit`, no intermediate of
// `solve_scaled_2x2` overflows. Complete pivoting gives `|L10| <= 1` and
// `|U01 / U00| <= 1`, so `|b1 - L10 b0| <= 2 |b|_max`, `|x1| <= 2 |b|_max /
// pivot_min` and `|x0| <= 3 |b|_max / pivot_min`, all below `3 / (16 *
// min_positive)`; the factor 16 covers this and the growth of complex
// products. `pivot_min` is capped at 1 because only a pivot below 1 can
// enlarge the solution (`dlaln2` likewise scales only when `|U11| < 1`).
//
// How it differs from `dlaln2`: Reference LAPACK bounds the eliminated
// right hand side, `BBND = max(|b0 U11 / U00|, |b1 - L10 b0|)`, against
// `BIGNUM * |U11|` with `BIGNUM = 1 / (2 * min_positive)`, after the `SMIN`
// clamp, and rescales a second time when `|x| > BIGNUM / max|C|` so that the
// caller's next update `C x` cannot overflow. Here the bound is taken on the
// unreduced `b` against the smaller pivot, which is sufficient but can scale
// where `dlaln2` would not; the second rescale is left to the caller, which
// normalizes the whole working vector after every block
// (`rescale_gevd_rhs`). A different scale only changes the working vector by
// a positive factor, which the final normalization removes; the vector is
// not bit-identical to `dtgevc`'s.
//
// Out of scope: this is an overflow guard for the 2x2 solve, nothing more.
// - Underflow: scaling down may flush small components to subnormals or zero
//   earlier than `dlaln2` would.
// - The complex divisions in `solve_scaled_2x2` (and the complex reciprocal
//   in `solve_complex_shifted_1x1`) form `|d|^2` without scaling, unlike
//   `dlaln2`'s `DLADIV`, so they lose the result for `|d|` outside about
//   `[sqrt(min_positive), sqrt(max_positive)]` (`1e-154 .. 1e154` in f64,
//   `1e-19 .. 1e19` in f32), whatever this scale is. A complex pair's
//   vectors are therefore only finite for pencils within that range: in f64,
//   `(s A, s B)` for the faer#220 pencils gives non-finite vectors for `s`
//   outside `[1e-146, 1e154]`, measured at powers of ten (faer#234). Reference LAPACK's `dggev` avoids this
//   range by scaling the pencil into `[sqrt(min_positive) / eps, eps /
//   sqrt(min_positive)]` before the QZ; `gevd_real` does not.
// - With `dtgevc`'s coefficient normalization the right hand side is about
//   `|coef| |(A, B)| xmax` and `smin >= eps |coef| |(A, B)|`, so `|b|_max /
//   pivot_min` stays below about `2 / eps`: this scale can only engage when
//   `pivot_min > 1` and `|b|_max > 1 / (16 min_positive)`, that is for
//   `|(A, B)|` near `max_positive`, beyond the range where the complex
//   divisions above already fail. No GEVD test, reference fixture or graded
//   random pencil (entries up to `1e+-300`) reaches it (faer#234).
fn shifted_2x2_scale<T: RealField>(bnorm: T, pivot_min: T) -> T {
    let limit = (math_utils::min_positive::<T>() * math_utils::from_f64::<T>(16.0)).recip()
        * pivot_min.fmin(math_utils::one::<T>());
    if bnorm > limit {
        limit / bnorm
    } else {
        math_utils::one()
    }
}

// `(a + i b) / (c + i d)` without forming `c^2 + d^2` (LAPACK `dladiv`,
// Baudin and Smith's robust complex division). The quotient by `num_complex`'s
// `Div` squares the denominator, which underflows to zero for `|c + i d|`
// below about `sqrt(min_positive)` and overflows above `sqrt(max_positive)`, so
// a finite quotient came out as `0/0 = NaN` or `inf` (tensor4all/faer#370).
fn ladiv<T: RealField>(a: &T, b: &T, c: &T, d: &T) -> (T, T) {
    fn ladiv2<T: RealField>(a: &T, b: &T, c: &T, d: &T, r: &T, t: &T) -> T {
        if *r != math_utils::zero() {
            let br = b * r;
            if br != math_utils::zero() {
                (a + br) * t
            } else {
                a * t + (b * t) * r
            }
        } else {
            (a + d * (b / c)) * t
        }
    }
    fn ladiv1<T: RealField>(a: &T, b: &T, c: &T, d: &T) -> (T, T) {
        let r = &(d / c);
        let t = &(c + d * r).recip();
        let p = ladiv2(a, b, c, d, r, t);
        let q = ladiv2(b, &-a, c, d, r, t);
        (p, q)
    }
    let half = &math_utils::from_f64::<T>(0.5);
    let two = &math_utils::from_f64::<T>(2.0);
    let (mut a, mut b, mut c, mut d) = (a.copy(), b.copy(), c.copy(), d.copy());
    let ab = a.abs().fmax(b.abs());
    let cd = c.abs().fmax(d.abs());
    let mut s = math_utils::one::<T>();
    let ov = math_utils::max_positive::<T>();
    let un = math_utils::min_positive::<T>();
    let eps = math_utils::eps::<T>();
    let be = two / (&eps * &eps);
    let tiny = un * two / eps;
    if ab >= half * &ov {
        (a, b, s) = (half * a, half * b, two * s);
    }
    if cd >= half * &ov {
        (c, d, s) = (half * c, half * d, half * s);
    }
    if ab <= tiny {
        (a, b, s) = (&be * a, &be * b, s / &be);
    }
    if cd <= tiny {
        (c, d, s) = (&be * c, &be * d, s * &be);
    }
    let (p, q) = if d.abs() <= c.abs() {
        ladiv1(&a, &b, &c, &d)
    } else {
        let (p, q) = ladiv1(&b, &a, &d, &c);
        (p, -q)
    };
    (p * &s, q * &s)
}

fn ladiv_complex<T: RealField>(x: &Complex<T>, y: &Complex<T>) -> Complex<T> {
    let (re, im) = ladiv(&x.re, &x.im, &y.re, &y.im);
    Complex { re, im }
}

// Solve the pivoted 2x2 system with a scaled RHS. Form U01/U00 before
// multiplying by x1: U01*x1 itself can overflow even when x0 is finite. `div`
// is the scalar quotient: `/` for a real system, `ladiv_complex` for a complex
// one.
fn solve_scaled_2x2<T: ComplexField>(
    LU: MatRef<'_, T>,
    row_perm: colmat::perm::PermRef<'_, usize>,
    col_perm: colmat::perm::PermRef<'_, usize>,
    mut b: ColMut<'_, T>,
    div: impl Fn(&T, &T) -> T,
) -> T::Real {
    let bnorm = b[0]
        .real()
        .abs()
        .fmax(b[0].imag().abs())
        .fmax(b[1].real().abs())
        .fmax(b[1].imag().abs());
    let scale = shifted_2x2_scale(bnorm, LU[(0, 0)].abs().fmin(LU[(1, 1)].abs()));
    let rows = row_perm.arrays().0;
    let cols = col_perm.arrays().0;
    let b0 = b[rows[0]].mul_real(&scale);
    let b1 = b[rows[1]].mul_real(&scale) - &LU[(1, 0)] * &b0;
    let x1 = div(&b1, &LU[(1, 1)]);
    let x0 = div(&b0, &LU[(0, 0)]) - div(&LU[(0, 1)], &LU[(0, 0)]) * &x1;
    b[cols[0]] = x0;
    b[cols[1]] = x1;
    scale
}

#[allow(clippy::too_many_arguments)] // faer kernel shape, kept flat to match upstream
#[allow(clippy::toplevel_ref_arg)] // `ref` parameter keeps the upstream signature; the body wants `&T` without derefing each use
fn solve_complex_shifted_2x2<T: RealField>(
    smin: T,
    ref ca: T,
    A: MatRef<'_, T>,
    ref d0: T,
    ref d1: T,
    mut B: MatMut<'_, T>,
    ref wr: T,
    ref wi: T,
    stack: &mut MemStack,
) -> T {
    let nw = B.ncols();
    let safmin = &math_utils::min_positive::<T>();
    let smlnum = &(safmin + safmin);
    let smin = &smin.fmax(smlnum);
    if nw == 1 {
        let (mut C, stack) = unsafe { colmat::linalg::temp_mat_uninit::<T, _, _>(2, 2, stack) };
        let mut C = C.as_mat_mut();
        let mut row_perm_fwd = [0usize; 2];
        let mut row_perm_inv = [0usize; 2];
        let mut col_perm_fwd = [0usize; 2];
        let mut col_perm_inv = [0usize; 2];
        C[(0, 0)] = ca * &A[(0, 0)] - wr * d0;
        C[(1, 0)] = ca * &A[(1, 0)];
        C[(0, 1)] = ca * &A[(0, 1)];
        C[(1, 1)] = ca * &A[(1, 1)] - wr * d1;
        let cmax = C.norm_max();
        if cmax < *smin {
            // `C` is replaced by `smin * I` (LAPACK `dlaln2`), so the solve is
            // this scaling alone
            let smin_inv = &smin.recip();
            let scale = shifted_2x2_scale(B.norm_max(), smin.copy());
            zip!(B.rb_mut()).for_each(|unzip!(x)| {
                *x = (&*x * &scale) * smin_inv;
            });
            return scale;
        }
        let (_, row_perm, col_perm) = colmat::linalg::lu::full_pivoting::factor::lu_in_place(
            C.rb_mut(),
            &mut row_perm_fwd,
            &mut row_perm_inv,
            &mut col_perm_fwd,
            &mut col_perm_inv,
            Par::Seq,
            stack,
            Default::default(),
        );
        // a second pivot below `smin` is replaced by `smin` (LAPACK `dlaln2`).
        // `C` is exactly singular when the block holds a copy of the
        // eigenvalue being computed, and the zero pivot would divide the right
        // hand side into `inf`/`NaN` (faer#220)
        //
        // The clamp is applied to the factorized `U11`, after `lu_in_place`
        // has formed `L10 = C10 / U00`, and `L10` is not recomputed. For a 2x2
        // this is the same perturbation as `dlaln2`'s, not an approximation of
        // it: `dlaln2` also forms `LR21 = CR21 / UR11` and `UR22 = CR22 - UR12 *
        // LR21` first and then clamps `UR22` (v3.12.1, SRC/dlaln2.f, real 2x2
        // and complex 2x2 branches). The multiplier depends only on the first
        // pivot, which complete pivoting makes the largest entry, `>= cmax >=
        // smin`, so it never needs protection; the clamp replaces the solved
        // system by `L (U + (smin - U11) e1 e1^T)`, which perturbs only the
        // pivoted (1, 1) entry of `C`, by less than `2 smin`. What can differ
        // from `dlaln2` is the first pivot itself: faer's LU may rank complex
        // entries by another magnitude than `dlaln2`'s `|re| + |im|` and break
        // ties differently. Any largest entry gives `|L10| <= 1` and the same
        // argument.
        if C[(1, 1)].abs() < *smin {
            C[(1, 1)] = smin.copy();
        }
        solve_scaled_2x2(
            C.rb(),
            row_perm,
            col_perm,
            B.rb_mut().col_mut(0),
            |x: &T, y: &T| x / y,
        )
    } else {
        let (mut C, stack) =
            unsafe { colmat::linalg::temp_mat_uninit::<Complex<T>, _, _>(2, 2, stack) };
        let mut C = C.as_mat_mut();
        let mut row_perm_fwd = [0usize; 2];
        let mut row_perm_inv = [0usize; 2];
        let mut col_perm_fwd = [0usize; 2];
        let mut col_perm_inv = [0usize; 2];
        C[(0, 0)] = Complex {
            re: ca * &A[(0, 0)] - wr * d0,
            im: -wi * d0,
        };
        C[(1, 0)] = Complex {
            re: ca * &A[(1, 0)],
            im: math_utils::zero(),
        };
        C[(0, 1)] = Complex {
            re: ca * &A[(0, 1)],
            im: math_utils::zero(),
        };
        C[(1, 1)] = Complex {
            re: ca * &A[(1, 1)] - wr * d1,
            im: -wi * d1,
        };
        let cmax = C.norm_max();
        if cmax < *smin {
            // `C` is replaced by `smin * I`, as in the real branch
            let smin_inv = &smin.recip();
            let scale = shifted_2x2_scale(B.norm_max(), smin.copy());
            zip!(B.rb_mut()).for_each(|unzip!(x)| {
                *x = (&*x * &scale) * smin_inv;
            });
            return scale;
        }
        let (_, row_perm, col_perm) = colmat::linalg::lu::full_pivoting::factor::lu_in_place(
            C.rb_mut(),
            &mut row_perm_fwd,
            &mut row_perm_inv,
            &mut col_perm_fwd,
            &mut col_perm_inv,
            Par::Seq,
            stack,
            Default::default(),
        );
        // the second pivot is protected as in the real branch, measured by
        // `|re| + |im|` (LAPACK `dlaln2`), and for the same reason it is
        // equivalent to `dlaln2`'s clamp after elimination (see the real
        // branch above)
        if C[(1, 1)].re.abs() + C[(1, 1)].im.abs() < *smin {
            C[(1, 1)] = Complex {
                re: smin.copy(),
                im: math_utils::zero(),
            };
        }
        let n = B.nrows();
        let (Br, Bi) = B.two_cols_mut(0, 1);
        let (mut B, _) =
            unsafe { colmat::linalg::temp_mat_uninit::<Complex<T>, _, _>(n, 1, stack) };
        let mut B = B.as_mat_mut().col_mut(0);
        zip!(&mut B, &Br, &Bi).for_each(|unzip!(z, re, im)| {
            *z = Complex {
                re: re.copy(),
                im: im.copy(),
            }
        });
        let scale = solve_scaled_2x2(C.rb(), row_perm, col_perm, B.rb_mut(), ladiv_complex::<T>);
        zip!(Br, Bi, &B).for_each(|unzip!(re, im, z)| (*re, *im) = (z.re.copy(), z.im.copy()));
        scale
    }
}

// Apply the solve's scale to the remaining entries, then bound the whole
// working vector before the next matrix update or the back transformation.
#[allow(clippy::toplevel_ref_arg)] // `ref` parameter keeps the upstream signature; the body wants `&T` without derefing each use
fn rescale_gevd_rhs<T: RealField>(mut rhs: MatMut<'_, T>, j: usize, na: usize, ref scale: T) -> T {
    if *scale != math_utils::one() {
        zip!(rhs.rb_mut().get_mut(..j, ..)).for_each(|unzip!(x)| *x *= scale);
        zip!(rhs.rb_mut().get_mut(j + na.., ..)).for_each(|unzip!(x)| *x *= scale);
    }
    let limit = if rhs.ncols() == 2 {
        math_utils::from_f64::<T>(0.5)
    } else {
        math_utils::one()
    };
    let xmax = rhs.norm_max();
    if xmax > limit {
        let scale = &(&limit / &xmax);
        zip!(rhs).for_each(|unzip!(x)| *x *= scale);
        math_utils::one()
    } else {
        xmax / limit
    }
}
fn qz_to_gevd_real<T: RealField>(
    A: MatRef<'_, T>,
    B: MatRef<'_, T>,
    Q: Option<MatMut<'_, T>>,
    Z: Option<MatMut<'_, T>>,
    mut par: Par,
    stack: &mut MemStack,
) {
    let n = A.nrows();
    if n == 0 {
        return;
    }
    let ulp = &math_utils::eps::<T>();
    let safmin = &math_utils::min_positive::<T>();
    let smallnum = &(safmin * math_utils::from_f64::<T::Real>(n as f64));
    let small = &(smallnum * ulp.recip());
    let bignum = &smallnum.recip();
    let big = &small.recip();
    let (mut acolnorm, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
    let acolnorm = acolnorm.as_mat_mut();
    let (mut bcolnorm, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
    let bcolnorm = bcolnorm.as_mat_mut();
    let mut acolnorm = acolnorm.col_mut(0);
    let mut bcolnorm = bcolnorm.col_mut(0);
    let mut anorm = math_utils::zero::<T>();
    let mut bnorm = math_utils::zero::<T>();
    let mut j = 0;
    while j < n {
        let cplx = j + 1 < n && A[(j + 1, j)] != math_utils::zero();
        if !cplx {
            let a = A.rb().col(j).get(..j).norm_l1();
            acolnorm[j] = a.copy();
            anorm = anorm.fmax(a + A[(j, j)].abs());
            j += 1;
        } else {
            for jc in j..j + 2 {
                let a = A.rb().col(jc).get(..j).norm_l1();
                acolnorm[jc] = a.copy();
                anorm = anorm.fmax(a + A[(j, jc)].abs() + A[(j + 1, jc)].abs());
            }
            j += 2;
        }
    }
    for j in 0..n {
        let b = B.rb().col(j).get(..j).norm_l1();
        bcolnorm[j] = b.copy();
        bnorm = bnorm.fmax(b + B[(j, j)].abs());
    }
    let anorm = &anorm;
    let bnorm = &bnorm;
    let ascale = &anorm.fmax(safmin).recip();
    let bscale = &bnorm.fmax(safmin).recip();
    if let Some(mut u) = Q {
        let mut je = 0usize;
        while je < n {
            let cplx_eigval = je + 1 < n && A[(je + 1, je)] != math_utils::zero();
            let nw = if cplx_eigval { 2 } else { 1 };
            if !cplx_eigval && A[(je, je)].abs().fmax(B[(je, je)].abs()) < *safmin {
                u.rb_mut().col_mut(je).fill(math_utils::zero::<T>());
                u[(je, je)] = math_utils::one();
                je += 1;
                continue;
            }
            let mut acoef;
            let mut acoefa;
            let mut bcoefa;
            let mut bcoefr;
            let mut bcoefi;
            let mut xmax;
            let (mut rhs, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
            let mut rhs = rhs.as_mat_mut();
            if !cplx_eigval {
                let temp =
                    &safmin.fmax((A[(je, je)].abs() * ascale).fmax(B[(je, je)].abs() * bscale));
                let salfar = &A[(je, je)].mul_real(temp).mul_real(ascale);
                let sbeta = &(B[(je, je)].real() * temp * bscale);
                acoef = sbeta * ascale;
                bcoefr = salfar * bscale;
                bcoefi = math_utils::zero();
                let mut scale = math_utils::one();
                let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
                let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.fmin(big);
                }
                if lsb {
                    scale = scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
                }
                if lsa || lsb {
                    scale = scale.fmin(
                        math_utils::one::<T>()
                            / (safmin
                                * math_utils::one::<T>().fmax(acoef.abs()).fmax(bcoefr.abs())),
                    );
                    if lsa {
                        acoef = ascale * (&scale * sbeta)
                    } else {
                        acoef = &scale * acoef
                    }
                    if lsb {
                        bcoefr = bscale * (&scale * salfar)
                    } else {
                        bcoefr = &scale * bcoefr
                    }
                }
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs();
                rhs[(je, 0)] = math_utils::one();
                xmax = math_utils::one::<T::Real>();
            } else {
                let (ref scale, _, wr, _, wi) = generalized_eigval_2x2(
                    (
                        A[(je, je)].copy(),
                        A[(je, je + 1)].copy(),
                        A[(je + 1, je)].copy(),
                        A[(je + 1, je + 1)].copy(),
                    ),
                    (
                        B[(je, je)].copy(),
                        B[(je, je + 1)].copy(),
                        B[(je + 1, je)].copy(),
                        B[(je + 1, je + 1)].copy(),
                    ),
                );
                acoef = scale.copy();
                bcoefr = wr;
                bcoefi = wi;
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs() + bcoefi.abs();
                let mut scale = math_utils::one();
                if &acoefa * ulp < *safmin && acoefa >= *safmin {
                    scale = (safmin / ulp) / &acoefa;
                }
                if &bcoefa * ulp < *safmin && bcoefa >= *safmin {
                    scale = scale.fmax((safmin / ulp) / &bcoefa);
                }
                if safmin * &acoefa > *ascale {
                    scale = ascale / (safmin * &acoefa);
                }
                if safmin * &bcoefa > *bscale {
                    scale = scale.fmin(bscale / (safmin * &bcoefa));
                }
                if scale != math_utils::one() {
                    acoef = &scale * &acoef;
                    acoefa = acoef.abs();
                    bcoefr = &scale * &bcoefr;
                    bcoefi = &scale * &bcoefi;
                    bcoefa = bcoefr.abs() + bcoefi.abs();
                }
                let temp = &(&acoef * &A[(je + 1, je)]);
                let temp2r = &acoef * &A[(je, je)] - &bcoefr * &B[(je, je)];
                let temp2i = -&bcoefi * &B[(je, je)];
                if temp.abs() > temp2r.abs() + temp2i.abs() {
                    rhs[(je, 0)] = math_utils::one();
                    rhs[(je, 1)] = math_utils::zero();
                    rhs[(je + 1, 0)] = -temp2r / temp;
                    rhs[(je + 1, 1)] = -temp2i / temp;
                } else {
                    rhs[(je + 1, 0)] = math_utils::one();
                    rhs[(je + 1, 1)] = math_utils::zero();
                    let temp = &(&acoef * &A[(je, je + 1)]);
                    rhs[(je, 0)] =
                        (&bcoefr * &B[(je + 1, je + 1)] - &acoef * &A[(je + 1, je + 1)]) / temp;
                    rhs[(je, 1)] = &bcoefi * &B[(je + 1, je + 1)] / temp;
                }
                xmax = (rhs[(je, 0)].abs() + rhs[(je, 1)].abs())
                    .fmax(rhs[(je + 1, 0)].abs() + rhs[(je + 1, 1)].abs());
            }
            let dmin = safmin
                .fmax(ulp * &acoefa * anorm)
                .fmax(ulp * &bcoefa * bnorm);
            let mut j = je + nw;
            while j < n {
                let cplx = j + 1 < n && A[(j + 1, j)] != math_utils::zero();
                let na = if cplx { 2 } else { 1 };
                // the guard below may only shrink the vector (LAPACK `dtgevc`:
                // `XSCALE = ONE / MAX( ONE, XMAX )`)
                let xscale = &xmax.fmax(math_utils::one::<T>()).recip();
                let mut temp = acolnorm[j]
                    .fmax(&bcolnorm[j])
                    .fmax(&acoefa * &acolnorm[j] + &bcoefa * &bcolnorm[j]);
                let b0 = B[(j, j)].copy();
                let b1;
                if cplx {
                    temp = temp
                        .fmax(&acoefa * &acolnorm[j + 1] + &bcoefa * &bcolnorm[j + 1])
                        .fmax(&acolnorm[j + 1])
                        .fmax(&bcolnorm[j + 1]);
                    b1 = B[(j + 1, j + 1)].copy();
                } else {
                    b1 = math_utils::zero();
                }
                if temp > bignum * xscale {
                    for jw in 0..nw {
                        for jr in je..j {
                            rhs[(jr, jw)] = xscale * &rhs[(jr, jw)];
                        }
                    }
                }
                let (mut sums, stack) =
                    unsafe { colmat::linalg::temp_mat_uninit::<T, _, _>(na, nw, stack) };
                let mut sums = sums.as_mat_mut();
                let (mut sump, stack) =
                    unsafe { colmat::linalg::temp_mat_uninit::<T, _, _>(na, nw, stack) };
                let mut sump = sump.as_mat_mut();
                for jw in 0..nw {
                    for ja in 0..na {
                        sums[(ja, jw)] = math_utils::zero();
                        sump[(ja, jw)] = math_utils::zero();
                        for jr in je..j {
                            sums[(ja, jw)] = &sums[(ja, jw)] + &A[(jr, j + ja)] * &rhs[(jr, jw)];
                            sump[(ja, jw)] = &sump[(ja, jw)] + &B[(jr, j + ja)] * &rhs[(jr, jw)];
                        }
                    }
                }
                for ja in 0..na {
                    if cplx_eigval {
                        rhs[(j + ja, 0)] = -&acoef * &sums[(ja, 0)] + &bcoefr * &sump[(ja, 0)]
                            - &bcoefi * &sump[(ja, 1)];
                        rhs[(j + ja, 1)] = -&acoef * &sums[(ja, 1)]
                            + &bcoefr * &sump[(ja, 1)]
                            + &bcoefi * &sump[(ja, 0)];
                    } else {
                        rhs[(j + ja, 0)] = -&acoef * &sums[(ja, 0)] + &bcoefr * &sump[(ja, 0)];
                    }
                }
                let scale = if cplx {
                    solve_complex_shifted_2x2(
                        dmin.copy(),
                        acoef.copy(),
                        A.submatrix(j, j, na, na).transpose(),
                        b0,
                        b1,
                        rhs.rb_mut().subrows_mut(j, na),
                        bcoefr.copy(),
                        bcoefi.copy(),
                        stack,
                    )
                } else {
                    solve_complex_shifted_1x1(
                        dmin.copy(),
                        acoef.copy(),
                        A.submatrix(j, j, na, na).transpose(),
                        b0,
                        rhs.rb_mut().subrows_mut(j, na),
                        bcoefr.copy(),
                        bcoefi.copy(),
                    );
                    math_utils::one()
                };
                // the left vector is zero above `je`, so this window (from `je`,
                // offset `j - je`) and the right vectors' form (from row 0, offset
                // `j`) give the same result; the offset only matters when the 2x2
                // solve returned `scale != 1`, which `shifted_2x2_scale` explains
                // is out of reach today.
                // `left_vectors_of_schur_pencil_*` pins this path
                xmax = rescale_gevd_rhs(rhs.rb_mut().get_mut(je..j + na, ..), j - je, na, scale);
                j += na;
            }
            let (mut tmp, _) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
            let mut tmp = tmp.as_mat_mut();
            colmat::linalg::matmul::matmul(
                tmp.rb_mut(),
                Accum::Replace,
                u.rb().get(.., je..),
                rhs.rb().get(je.., ..),
                math_utils::one(),
                par.reborrow(),
            );
            let mut u = u.rb_mut().subcols_mut(je, nw);
            u.copy_from(&tmp);
            let scale = &u.norm_l2().recip();
            zip!(u).for_each(|unzip!(u)| {
                *u *= scale;
            });
            je += nw;
        }
    }
    if let Some(mut u) = Z {
        let mut je = n;
        while je > 0 {
            je -= 1;
            let cplx_eigval = je > 0 && A[(je, je - 1)] != math_utils::zero();
            let nw = if cplx_eigval { 2 } else { 1 };
            je -= nw - 1;
            if !cplx_eigval && A[(je, je)].abs().fmax(B[(je, je)].abs()) < *safmin {
                u.rb_mut().col_mut(je).fill(math_utils::zero::<T>());
                u[(je, je)] = math_utils::one();
                continue;
            }
            let mut acoef;
            let mut acoefa;
            let mut bcoefa;
            let mut bcoefr;
            let mut bcoefi;
            let (mut rhs, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
            let mut rhs = rhs.as_mat_mut();
            if !cplx_eigval {
                let temp =
                    &safmin.fmax((A[(je, je)].abs() * ascale).fmax(B[(je, je)].abs() * bscale));
                let salfar = &A[(je, je)].mul_real(temp).mul_real(ascale);
                let sbeta = &(B[(je, je)].real() * temp * bscale);
                acoef = sbeta * ascale;
                bcoefr = salfar * bscale;
                bcoefi = math_utils::zero();
                let mut scale = math_utils::one();
                let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
                let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.fmin(big);
                }
                if lsb {
                    scale = scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
                }
                if lsa || lsb {
                    scale = scale.fmin(
                        math_utils::one::<T>()
                            / (safmin
                                * math_utils::one::<T>().fmax(acoef.abs()).fmax(bcoefr.abs())),
                    );
                    if lsa {
                        acoef = ascale * (&scale * sbeta)
                    } else {
                        acoef = &scale * &acoef
                    }
                    if lsb {
                        bcoefr = bscale * (&scale * salfar)
                    } else {
                        bcoefr = &scale * &bcoefr
                    }
                }
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs();
                rhs[(je, 0)] = math_utils::one();
                for jr in 0..je {
                    rhs[(jr, 0)] = &bcoefr * &B[(jr, je)] - &acoef * &A[(jr, je)];
                }
            } else {
                let (scale, _, wr, _, wi) = generalized_eigval_2x2(
                    (
                        A[(je, je)].copy(),
                        A[(je, je + 1)].copy(),
                        A[(je + 1, je)].copy(),
                        A[(je + 1, je + 1)].copy(),
                    ),
                    (
                        B[(je, je)].copy(),
                        B[(je, je + 1)].copy(),
                        B[(je + 1, je)].copy(),
                        B[(je + 1, je + 1)].copy(),
                    ),
                );
                acoef = scale;
                bcoefr = wr;
                bcoefi = wi;
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs() + bcoefi.abs();
                let mut scale = math_utils::one();
                if &acoefa * ulp < *safmin && acoefa >= *safmin {
                    scale = (safmin / ulp) / &acoefa;
                }
                if &bcoefa * ulp < *safmin && bcoefa >= *safmin {
                    scale = scale.fmax((safmin / ulp) / &bcoefa);
                }
                if safmin * &acoefa > *ascale {
                    scale = ascale / (safmin * &acoefa);
                }
                if safmin * &bcoefa > *bscale {
                    scale = scale.fmin(bscale / (safmin * &bcoefa));
                }
                if scale != math_utils::one() {
                    acoef = &scale * &acoef;
                    acoefa = acoef.abs();
                    bcoefr = &scale * &bcoefr;
                    bcoefi = &scale * &bcoefi;
                    bcoefa = bcoefr.abs() + bcoefi.abs();
                }
                let temp = &(&acoef * &A[(je + 1, je)]);
                let temp2r = &(&acoef * &A[(je + 1, je + 1)] - &bcoefr * &B[(je + 1, je + 1)]);
                let temp2i = &(-&bcoefi * &B[(je + 1, je + 1)]);
                if temp.abs() > temp2r.abs() + temp2i.abs() {
                    rhs[(je + 1, 0)] = math_utils::one();
                    rhs[(je + 1, 1)] = math_utils::zero();
                    rhs[(je, 0)] = -temp2r / temp;
                    rhs[(je, 1)] = -temp2i / temp;
                } else {
                    rhs[(je, 0)] = math_utils::one();
                    rhs[(je, 1)] = math_utils::zero();
                    let temp = &(&acoef * &A[(je, je + 1)]);
                    rhs[(je + 1, 0)] = (&bcoefr * &B[(je, je)] - &acoef * &A[(je, je)]) / temp;
                    rhs[(je + 1, 1)] = &bcoefi * &B[(je, je)] / temp;
                }
                let creala = &(&acoef * &rhs[(je, 0)]);
                let cimaga = &(&acoef * &rhs[(je, 1)]);
                let crealb = &(&bcoefr * &rhs[(je, 0)] - &bcoefi * &rhs[(je, 1)]);
                let cimagb = &(&bcoefi * &rhs[(je, 0)] + &bcoefr * &rhs[(je, 1)]);
                let cre2a = &(&acoef * &rhs[(je + 1, 0)]);
                let cim2a = &(&acoef * &rhs[(je + 1, 1)]);
                let cre2b = &(&bcoefr * &rhs[(je + 1, 0)] - &bcoefi * &rhs[(je + 1, 1)]);
                let cim2b = &(&bcoefi * &rhs[(je + 1, 0)] + &bcoefr * &rhs[(je + 1, 1)]);
                for jr in 0..je {
                    rhs[(jr, 0)] = -creala * &A[(jr, je)] + crealb * &B[(jr, je)]
                        - cre2a * &A[(jr, je + 1)]
                        + cre2b * &B[(jr, je + 1)];
                    rhs[(jr, 1)] = -cimaga * &A[(jr, je)] + cimagb * &B[(jr, je)]
                        - cim2a * &A[(jr, je + 1)]
                        + cim2b * &B[(jr, je + 1)];
                }
            }
            let dmin = safmin
                .fmax(ulp * &acoefa * anorm)
                .fmax(ulp * &bcoefa * bnorm);
            let mut j = je;
            while j > 0 {
                j -= 1;
                let cplx = j > 0 && A[(j, j - 1)] != math_utils::zero();
                let na = if cplx { 2 } else { 1 };
                j -= na - 1;
                let b0 = B[(j, j)].copy();

                let b1 = if cplx {
                    B[(j + 1, j + 1)].copy()
                } else {
                    math_utils::zero()
                };
                let scale = if cplx {
                    solve_complex_shifted_2x2(
                        dmin.copy(),
                        acoef.copy(),
                        A.submatrix(j, j, na, na),
                        b0,
                        b1,
                        rhs.rb_mut().subrows_mut(j, na),
                        bcoefr.copy(),
                        bcoefi.copy(),
                        stack,
                    )
                } else {
                    solve_complex_shifted_1x1(
                        dmin.copy(),
                        acoef.copy(),
                        A.submatrix(j, j, na, na),
                        b0,
                        rhs.rb_mut().subrows_mut(j, na),
                        bcoefr.copy(),
                        bcoefi.copy(),
                    );
                    math_utils::one()
                };
                rescale_gevd_rhs(rhs.rb_mut().get_mut(..je + nw, ..), j, na, scale);
                for ja in 0..na {
                    if cplx_eigval {
                        let creala = &(&acoef * &rhs[(j + ja, 0)]);
                        let cimaga = &(&acoef * &rhs[(j + ja, 1)]);
                        let crealb = &(&bcoefr * &rhs[(j + ja, 0)] - &bcoefi * &rhs[(j + ja, 1)]);
                        let cimagb = &(&bcoefi * &rhs[(j + ja, 0)] + &bcoefr * &rhs[(j + ja, 1)]);
                        for jr in 0..j {
                            rhs[(jr, 0)] -= creala * &A[(jr, j + ja)] - crealb * &B[(jr, j + ja)];
                            rhs[(jr, 1)] -= cimaga * &A[(jr, j + ja)] - cimagb * &B[(jr, j + ja)];
                        }
                    } else {
                        let creala = &(&acoef * &rhs[(j + ja, 0)]);
                        let crealb = &(&bcoefr * &rhs[(j + ja, 0)]);
                        for jr in 0..j {
                            rhs[(jr, 0)] -= creala * &A[(jr, j + ja)] - crealb * &B[(jr, j + ja)];
                        }
                    }
                }
            }
            let (mut tmp, _) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, nw, stack);
            let mut tmp = tmp.as_mat_mut();
            colmat::linalg::matmul::matmul(
                tmp.rb_mut(),
                Accum::Replace,
                u.rb().get(.., ..je + nw),
                rhs.rb().get(..je + nw, ..),
                math_utils::one(),
                par.reborrow(),
            );
            let mut u = u.rb_mut().subcols_mut(je, nw);
            u.copy_from(&tmp);
            let scale = &u.norm_l2().recip();
            zip!(u).for_each(|unzip!(u)| {
                *u *= scale;
            });
        }
    }
}
fn qz_to_gevd_cplx<T: ComplexField>(
    A: MatRef<'_, T>,
    B: MatRef<'_, T>,
    Q: Option<MatMut<'_, T>>,
    Z: Option<MatMut<'_, T>>,
    mut par: Par,
    stack: &mut MemStack,
) {
    let n = A.nrows();
    if n == 0 {
        return;
    }
    let ulp = &math_utils::eps::<T::Real>();
    let safmin = &math_utils::min_positive::<T::Real>();
    let smallnum = &(safmin * math_utils::from_f64::<T::Real>(n as f64));
    let small = &(smallnum * ulp.recip());
    let bignum = &smallnum.recip();
    let big = &small.recip();
    let (mut acolnorm, stack) = colmat::linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
    let acolnorm = acolnorm.as_mat_mut();
    let (mut bcolnorm, stack) = colmat::linalg::temp_mat_zeroed::<T::Real, _, _>(n, 1, stack);
    let bcolnorm = bcolnorm.as_mat_mut();
    let mut acolnorm = acolnorm.col_mut(0);
    let mut bcolnorm = bcolnorm.col_mut(0);
    let mut anorm = math_utils::zero::<T::Real>();
    let mut bnorm = math_utils::zero::<T::Real>();
    let mut j = 0;
    while j < n {
        let a = A.rb().col(j).get(..j).norm_l1();
        acolnorm[j] = a.copy();
        anorm = anorm.fmax(a + A[(j, j)].abs());
        j += 1;
    }
    for j in 0..n {
        let b = &B.rb().col(j).get(..j).norm_l1();
        bcolnorm[j] = b.copy();
        bnorm = bnorm.fmax(b + B[(j, j)].abs());
    }
    let anorm = &anorm;
    let bnorm = &bnorm;
    let ascale = &anorm.fmax(safmin).recip();
    let bscale = &bnorm.fmax(safmin).recip();
    if let Some(mut u) = Q {
        let mut je = 0usize;
        while je < n {
            if A[(je, je)].abs().fmax(B[(je, je)].abs()) < *safmin {
                u.rb_mut().col_mut(je).fill(math_utils::zero::<T>());
                u[(je, je)] = math_utils::one::<T::Real>().to_cplx();
                je += 1;
                continue;
            }
            let mut acoef;
            let acoefa;
            let bcoefa;
            let mut bcoef;
            let mut xmax;
            let (mut rhs, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
            let mut rhs = rhs.as_mat_mut().col_mut(0);
            {
                let temp =
                    &safmin.fmax((A[(je, je)].abs() * ascale).fmax(B[(je, je)].abs() * bscale));
                let salfar = &A[(je, je)].mul_real(temp).mul_real(ascale);
                let sbeta = &(B[(je, je)].real() * temp * bscale);
                acoef = sbeta * ascale;
                bcoef = salfar.mul_real(bscale);
                let mut scale = math_utils::one::<T::Real>();
                let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
                let lsb = salfar.abs() >= *safmin && bcoef.abs() < *small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.fmin(big);
                }
                if lsb {
                    scale = scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
                }
                if lsa || lsb {
                    scale = scale.fmin(
                        math_utils::one::<T::Real>()
                            / (safmin
                                * math_utils::one::<T::Real>()
                                    .fmax(acoef.abs())
                                    .fmax(bcoef.abs())),
                    );
                    if lsa {
                        acoef = sbeta * &scale * ascale
                    } else {
                        acoef *= &scale
                    }
                    if lsb {
                        bcoef = salfar.mul_real(&scale).mul_real(bscale)
                    } else {
                        bcoef = bcoef.mul_real(&scale)
                    }
                }
                acoefa = acoef.abs();
                bcoefa = bcoef.abs();
                rhs[je] = math_utils::one::<T::Real>().to_cplx();
                xmax = math_utils::one::<T::Real>();
            }
            let dmin = safmin
                .fmax(ulp * &acoefa * anorm)
                .fmax(ulp * &bcoefa * bnorm);
            let mut j = je + 1;
            while j < n {
                let xscale = &xmax.recip();
                let temp = &acolnorm[j]
                    .fmax(&bcolnorm[j])
                    .fmax(&acoefa * &acolnorm[j] + &bcoefa * &bcolnorm[j]);
                let b0 = B[(j, j)].copy();
                if *temp > bignum * xscale {
                    for jr in je..j {
                        rhs[jr] = rhs[jr].mul_real(xscale);
                    }
                    xmax *= xscale;
                }
                let mut sums = math_utils::zero::<T>();
                let mut sump = math_utils::zero::<T>();
                for jr in je..j {
                    sums += A[(jr, j)].conj() * &rhs[jr];
                    sump += B[(jr, j)].conj() * &rhs[jr];
                }
                rhs[j] = bcoef.conj() * &sump - sums.mul_real(&acoef);
                solve_shifted_1x1(
                    dmin.copy(),
                    acoef.copy(),
                    A[(j, j)].conj(),
                    b0.conj(),
                    &mut rhs[j],
                    bcoef.conj(),
                );
                j += 1;
            }
            let (mut tmp, _) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
            let mut tmp = tmp.as_mat_mut().col_mut(0);
            colmat::linalg::matmul::matmul(
                tmp.rb_mut(),
                Accum::Replace,
                u.rb().get(.., je..),
                rhs.rb().get(je..),
                math_utils::one::<T::Real>().to_cplx(),
                par.reborrow(),
            );
            let mut u = u.rb_mut().col_mut(je);
            u.copy_from(&tmp);
            let scale = &u.norm_l2().recip();
            zip!(u).for_each(|unzip!(u)| {
                *u = u.mul_real(scale);
            });
            je += 1;
        }
    }
    if let Some(mut u) = Z {
        let mut je = n;
        while je > 0 {
            je -= 1;
            if A[(je, je)].abs().fmax(B[(je, je)].abs()) < *safmin {
                u.rb_mut().col_mut(je).fill(math_utils::zero::<T>());
                u[(je, je)] = math_utils::one::<T::Real>().to_cplx();
                continue;
            }
            let mut acoef;
            let acoefa;
            let bcoefa;
            let mut bcoefr;
            let (mut rhs, stack) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
            let mut rhs = rhs.as_mat_mut().col_mut(0);
            {
                let temp =
                    &safmin.fmax((A[(je, je)].abs() * ascale).fmax(B[(je, je)].abs() * bscale));
                let salfar = &A[(je, je)].mul_real(temp).mul_real(ascale);
                let sbeta = &(B[(je, je)].real() * temp * bscale);
                acoef = sbeta * ascale;
                bcoefr = salfar.mul_real(bscale);
                let mut scale = math_utils::one::<T::Real>();
                let lsa = sbeta.abs() >= *safmin && acoef.abs() < *small;
                let lsb = salfar.abs() >= *safmin && bcoefr.abs() < *small;
                if lsa {
                    scale = (small / sbeta.abs()) * anorm.fmin(big);
                }
                if lsb {
                    scale = scale.fmax((small / salfar.abs()) * bnorm.fmin(big));
                }
                if lsa || lsb {
                    scale = scale.fmin(
                        math_utils::one::<T::Real>()
                            / (safmin
                                * math_utils::one::<T::Real>()
                                    .fmax(acoef.abs())
                                    .fmax(bcoefr.abs())),
                    );
                    if lsa {
                        acoef = sbeta * &scale * ascale
                    } else {
                        acoef *= &scale
                    }
                    if lsb {
                        bcoefr = salfar.mul_real(&scale).mul_real(bscale)
                    } else {
                        bcoefr = bcoefr.mul_real(&scale)
                    }
                }
                acoefa = acoef.abs();
                bcoefa = bcoefr.abs();
                rhs[je] = math_utils::one::<T::Real>().to_cplx();
                for jr in 0..je {
                    rhs[jr] = &bcoefr * &B[(jr, je)] - A[(jr, je)].mul_real(&acoef);
                }
            }
            let dmin = safmin
                .fmax(ulp * &acoefa * anorm)
                .fmax(ulp * &bcoefa * bnorm);
            let mut j = je;
            while j > 0 {
                j -= 1;
                let b0 = B[(j, j)].copy();
                solve_shifted_1x1(
                    dmin.copy(),
                    acoef.copy(),
                    A[(j, j)].copy(),
                    b0,
                    &mut rhs[j],
                    bcoefr.copy(),
                );
                let creala = &rhs[j].mul_real(&acoef);
                let crealb = &(&bcoefr * &rhs[j]);
                for jr in 0..j {
                    rhs[jr] = &rhs[jr] - creala * &A[(jr, j)] + crealb * &B[(jr, j)];
                }
            }
            let (mut tmp, _) = colmat::linalg::temp_mat_zeroed::<T, _, _>(n, 1, stack);
            let mut tmp = tmp.as_mat_mut().col_mut(0);
            colmat::linalg::matmul::matmul(
                tmp.rb_mut(),
                Accum::Replace,
                u.rb().get(.., ..je + 1),
                rhs.rb().get(..je + 1),
                math_utils::one::<T::Real>().to_cplx(),
                par.reborrow(),
            );
            let mut u = u.rb_mut().col_mut(je);
            u.copy_from(&tmp);
            let scale = &u.norm_l2().recip();
            zip!(u).for_each(|unzip!(u)| {
                *u = u.mul_real(scale);
            });
        }
    }
}
fn qz_to_gevd_real_scratch<T: RealField>(n: usize, _: ParShape) -> StackReq {
    StackReq::all_of(&[
        temp_mat_scratch::<T>(n, 1),
        temp_mat_scratch::<T>(n, 1),
        temp_mat_scratch::<T>(n, 2),
        temp_mat_scratch::<T>(2, 2),
        temp_mat_scratch::<T>(2, 2),
        StackReq::all_of(&[
            temp_mat_scratch::<T>(2, 2),
            temp_mat_scratch::<T>(n, 1),
            colmat::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, Complex<T>>(
                2,
                2,
                ParShape::Seq,
                Default::default(),
            ),
            colmat::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, Complex<T>>(
                2,
                1,
                ParShape::Seq,
            ),
        ]),
        temp_mat_scratch::<T>(n, 2),
    ])
}

fn qz_to_gevd_cplx_scratch<T: ComplexField>(n: usize, _: ParShape) -> StackReq {
    StackReq::all_of(&[temp_mat_scratch::<T>(n, 1); 4])
}

fn copy_sign<T: RealField>(magnitude: T, sign: T) -> T {
    if sign < math_utils::zero::<T>() {
        -magnitude
    } else {
        magnitude
    }
}

fn generalized_eigval_2x2<T: RealField>(
    (ref a11, ref a12, ref a21, ref a22): (T, T, T, T),
    (b11, ref b12, ref b21, b22): (T, T, T, T),
) -> (T, T, T, T, T) {
    let safety = &math_utils::from_f64::<T>(128.0);
    let safmin = &(math_utils::min_positive::<T>() * safety);
    let safmax = &safmin.recip();
    let rtmin = &safmin.sqrt();
    let rtmax = &rtmin.recip();
    let half = &math_utils::from_f64::<T>(0.5);
    let zero = math_utils::zero::<T>;
    let one = math_utils::one::<T>;
    let anorm = &safmin
        .fmax(a11.abs() + a21.abs())
        .fmax(a12.abs() + a22.abs());
    let ascale = &(one() / anorm);
    let a11 = &(a11 * ascale);
    let a12 = &(a12 * ascale);
    let a21 = &(a21 * ascale);
    let a22 = &(a22 * ascale);
    let mut b11 = b11;
    let _ = b21;
    let mut b22 = b22;
    let bmin = &(rtmin * b11.abs().fmax(b12.abs()).fmax(b22.abs()).fmax(rtmin));
    if b11.abs() < *bmin {
        b11 = copy_sign(bmin.copy(), b11);
    }
    if b22.abs() < *bmin {
        b22 = copy_sign(bmin.copy(), b22);
    }
    let bnorm = safmin.fmax(b11.abs()).fmax(b12.abs() + b22.abs());
    let bsize = &b11.abs().fmax(b22.abs());
    let bscale = &(one() / bsize);
    let b11 = &(b11 * bscale);
    let b12 = &(b12 * bscale);
    let b22 = &(b22 * bscale);
    let binv11 = &(one() / b11);
    let binv22 = &(one() / b22);
    let s1 = &(a11 * binv11);
    let s2 = &(a22 * binv22);
    let as11;
    let as12;
    let as22;
    let ss;
    let abi22;
    let pp;
    let shift;
    if s1.abs() <= s2.abs() {
        as12 = a12 - s1 * b12;
        as22 = a22 - s1 * b22;
        ss = a21 * (binv11 * binv22);
        abi22 = as22 * binv22 - &ss * b12;
        pp = half * &abi22;
        shift = s1;
    } else {
        as12 = a12 - s2 * b12;
        as11 = a11 - s2 * b11;
        ss = a21 * (binv11 * binv22);
        abi22 = -&ss * b12;
        pp = half * (as11 * binv11 + &abi22);
        shift = s2;
    }
    let qq = &ss * as12;
    let discr;
    let r;
    let pp = &pp;
    let abi22 = &abi22;
    if (pp * rtmin).abs() >= one() {
        discr = (pp * rtmin).abs2() + qq * safmin;
        r = discr.abs().sqrt() * rtmax;
    } else {
        if pp.abs2() + qq.abs() <= *safmin {
            discr = (rtmax * pp).abs2() + qq * safmax;
            r = discr.abs().sqrt() * rtmin;
        } else {
            discr = pp.abs2() + qq;
            r = discr.abs().sqrt();
        }
    }
    let (mut wr1, mut wr2, mut wi);
    if discr >= zero() || r == zero() {
        let sum = pp + copy_sign(r.copy(), pp.copy());
        let diff = pp - copy_sign(r.copy(), pp.copy());
        let wbig = shift + sum;
        let mut wsmall = shift + diff;
        if wbig.abs() * half > safmin.fmax(wsmall.abs()) {
            let wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
            wsmall = wdet / &wbig;
        }
        if pp > abi22 {
            wr1 = wbig.fmin(&wsmall);
            wr2 = wbig.fmax(&wsmall);
        } else {
            wr1 = wbig.fmax(&wsmall);
            wr2 = wbig.fmin(&wsmall);
        }
        wi = zero();
    } else {
        wr1 = shift + pp;
        wr2 = wr1.copy();
        wi = r;
    }
    let c1 = &(bsize * (safmin * ascale.fmax(one())));
    let c2 = &(safmin * bnorm.fmax(one()));
    let c3 = &(bsize * safmin);

    let c4 = if *ascale <= one() && *bsize <= one() {
        one().fmin((ascale / safmin) * bsize)
    } else {
        one()
    };
    let c5 = if *ascale <= one() || *bsize <= one() {
        one().fmin(ascale * bsize)
    } else {
        one()
    };
    let fuzzy1 = &math_utils::from_f64::<T>(1.00001);
    let wabs = &(wr1.abs() + wi.abs());
    let wsize = &safmin
        .fmax(c1.fmax((fuzzy1 * (wabs * c2 + c3)).fmax(c4.fmin(c5.fmax(wabs).mul_pow2(half)))));
    let scale1: T;
    let mut scale2: T = one();
    if *wsize != one() {
        let wscale = &(one() / wsize);
        if *wsize > one() {
            scale1 = (ascale.fmax(bsize) * wscale) * ascale.fmin(bsize);
        } else {
            scale1 = (ascale.fmin(bsize) * wscale) * ascale.fmax(bsize);
        }
        wr1 *= wscale;
        if wi != zero() {
            wi *= wscale;
            wr2 = wr1.copy();
            scale2 = scale1.copy();
        }
    } else {
        scale1 = ascale * bsize;
        scale2 = scale1.copy();
    }
    if wi == zero() {
        let wsize = &safmin
            .fmax(c1)
            .fmax(fuzzy1 * (wr2.abs() * c2 + c3))
            .fmax(c4.fmin(wr2.abs().fmax(c5).mul_pow2(half)));
        if *wsize != one() {
            let wscale = &(one() / wsize);
            if *wsize > one() {
                scale2 = (ascale.fmax(bsize) * wscale) * ascale.fmin(bsize);
            } else {
                scale2 = (ascale.fmin(bsize) * wscale) * ascale.fmax(bsize);
            }
            wr2 *= wscale;
        } else {
            scale2 = ascale * bsize;
        }
    }
    (scale1, scale2, wr1, wr2, wi)
}

fn blocks<T: RealField>(s: MatRef<'_, T>) -> Result<Vec<(usize, usize)>, usize> {
    let n = s.nrows();
    let mut result = Vec::new();
    let mut j = 0;
    while j < n {
        let pair = j + 1 < n && s[(j + 1, j)] != math_utils::zero::<T>();
        if pair {
            if j + 2 < n && s[(j + 2, j + 1)] != math_utils::zero::<T>() {
                return Err(5);
            }
        }
        let len = if pair { 2 } else { 1 };
        result.push((j, len));
        j += len;
    }
    Ok(result)
}

fn pair_selected<I: LapackInt>(howmny: u8, select: *const I, index: usize, len: usize) -> bool {
    if howmny == b'A' || howmny == b'B' {
        return true;
    }
    if select.is_null() {
        return false;
    }
    unsafe { (0..len).any(|k| (*select.add(index + k)).val() != 0) }
}

fn normalize_real<T: RealField>(x: &mut Mat<T>, layout: &[(usize, usize)]) {
    let n = x.nrows();
    for &(j, len) in layout {
        let mut max = math_utils::zero::<T>();
        for i in 0..n {
            let value = if len == 2 {
                x[(i, j)].abs() + x[(i, j + 1)].abs()
            } else {
                x[(i, j)].abs()
            };
            max = max.fmax(value);
        }
        if max > math_utils::zero::<T>() && max.is_finite() {
            let scale = max.recip();
            for k in j..j + len {
                for i in 0..n {
                    x[(i, k)] = x[(i, k)].mul_real(scale.copy());
                }
            }
        }
    }
}

fn normalize_complex<T: ComplexField>(x: &mut Mat<T>) {
    for j in 0..x.ncols() {
        let mut max = math_utils::zero::<T::Real>();
        for i in 0..x.nrows() {
            max = max.fmax(x[(i, j)].abs());
        }
        if max > math_utils::zero::<T::Real>() && max.is_finite() {
            let scale = max.recip();
            for i in 0..x.nrows() {
                x[(i, j)] = x[(i, j)].mul_real(scale.copy());
            }
        }
    }
}

unsafe fn real_pencil<T: RealField + ReflectorScalar, I: LapackInt>(
    side: u8,
    howmny: u8,
    select: *const I,
    n: isize,
    s: *const T,
    lds: isize,
    p: *const T,
    ldp: isize,
    vl: *mut T,
    ldvl: isize,
    vr: *mut T,
    ldvr: isize,
    mm: isize,
    m: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let left = matches!(side, b'L' | b'B');
    let right = matches!(side, b'R' | b'B');
    let back = howmny == b'B';
    if !matches!(side, b'L' | b'R' | b'B') {
        *info = I::from_i64(-1);
        crate::invalid(1);
        return;
    }
    if !matches!(howmny, b'A' | b'S' | b'B') {
        *info = I::from_i64(-2);
        crate::invalid(2);
        return;
    }
    if n < 0 {
        *info = I::from_i64(-4);
        crate::invalid(4);
        return;
    }
    if lds < 1.max(n) {
        *info = I::from_i64(-6);
        crate::invalid(6);
        return;
    }
    if ldp < 1.max(n) {
        *info = I::from_i64(-8);
        crate::invalid(8);
        return;
    }
    let nn = n as usize;
    let s_ref = MatRef::from_raw_parts(s, nn, nn, 1, lds);
    let p_ref = MatRef::from_raw_parts(p, nn, nn, 1, ldp);
    let layout = match blocks(s_ref) {
        Ok(v) => v,
        Err(arg) => {
            *info = I::from_i64(-(arg as i64));
            crate::invalid(arg as i32);
            return;
        }
    };
    // LAPACK requires triangular P on the diagonal blocks of S.
    for &(j, len) in &layout {
        if len == 2
            && (p_ref[(j, j)] == math_utils::zero::<T>()
                || p_ref[(j + 1, j + 1)] == math_utils::zero::<T>()
                || p_ref[(j, j + 1)] != math_utils::zero::<T>())
        {
            *info = I::from_i64(-7);
            crate::invalid(7);
            return;
        }
    }
    if (left && ldvl < 1.max(n)) || ldvl < 1 {
        *info = I::from_i64(-10);
        crate::invalid(10);
        return;
    }
    if (right && ldvr < 1.max(n)) || ldvr < 1 {
        *info = I::from_i64(-12);
        crate::invalid(12);
        return;
    }
    let mut count = 0usize;
    if howmny == b'S' {
        for &(j, len) in &layout {
            if pair_selected(howmny, select, j, len) {
                count += len;
            }
        }
    } else {
        count = nn;
    }
    if mm < count as isize {
        *info = I::from_i64(-13);
        crate::invalid(13);
        return;
    }
    *m = I::from_i64(count as i64);
    if n == 0 {
        return;
    }

    let mut q = if left {
        Some(Mat::<T>::identity(nn, nn))
    } else {
        None
    };
    let mut z = if right {
        Some(Mat::<T>::identity(nn, nn))
    } else {
        None
    };
    if back {
        if let Some(q) = q.as_mut() {
            for j in 0..nn {
                for i in 0..nn {
                    q[(i, j)] = *vl.offset(i as isize + j as isize * ldvl);
                }
            }
        }
        if let Some(z) = z.as_mut() {
            for j in 0..nn {
                for i in 0..nn {
                    z[(i, j)] = *vr.offset(i as isize + j as isize * ldvr);
                }
            }
        }
    }

    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(qz_to_gevd_real_scratch::<T>(nn, par.shape()));
        qz_to_gevd_real(
            s_ref,
            p_ref,
            q.as_mut().map(|m| m.as_mut()),
            z.as_mut().map(|m| m.as_mut()),
            par,
            MemStack::new(&mut mem),
        );
    });

    let mut out_left = left.then(|| Mat::<T>::zeros(nn, nn));
    let mut out_right = right.then(|| Mat::<T>::zeros(nn, nn));
    if let Some(x) = q.as_mut() {
        normalize_real(x, &layout);
        if let Some(out) = out_left.as_mut() {
            out.copy_from(x.as_ref());
        }
    }
    if let Some(x) = z.as_mut() {
        normalize_real(x, &layout);
        if let Some(out) = out_right.as_mut() {
            out.copy_from(x.as_ref());
        }
    }
    let mut column = 0usize;
    for &(j, len) in &layout {
        if !pair_selected(howmny, select, j, len) {
            continue;
        }
        if let Some(out) = out_left.as_ref() {
            for k in 0..len {
                for i in 0..nn {
                    *vl.offset(i as isize + (column + k) as isize * ldvl) = out[(i, j + k)];
                }
            }
        }
        if let Some(out) = out_right.as_ref() {
            for k in 0..len {
                for i in 0..nn {
                    *vr.offset(i as isize + (column + k) as isize * ldvr) = out[(i, j + k)];
                }
            }
        }
        column += len;
    }
}

unsafe fn complex_pencil<T: ComplexField + ReflectorScalar, I: LapackInt>(
    side: u8,
    howmny: u8,
    select: *const I,
    n: isize,
    s: *const T,
    lds: isize,
    p: *const T,
    ldp: isize,
    vl: *mut T,
    ldvl: isize,
    vr: *mut T,
    ldvr: isize,
    mm: isize,
    m: *mut I,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let left = matches!(side, b'L' | b'B');
    let right = matches!(side, b'R' | b'B');
    let back = howmny == b'B';
    if !matches!(side, b'L' | b'R' | b'B') {
        *info = I::from_i64(-1);
        crate::invalid(1);
        return;
    }
    if !matches!(howmny, b'A' | b'S' | b'B') {
        *info = I::from_i64(-2);
        crate::invalid(2);
        return;
    }
    if n < 0 {
        *info = I::from_i64(-4);
        crate::invalid(4);
        return;
    }
    if lds < 1.max(n) {
        *info = I::from_i64(-6);
        crate::invalid(6);
        return;
    }
    if ldp < 1.max(n) {
        *info = I::from_i64(-8);
        crate::invalid(8);
        return;
    }
    let nn = n as usize;
    let s_ref = MatRef::from_raw_parts(s, nn, nn, 1, lds);
    let p_ref = MatRef::from_raw_parts(p, nn, nn, 1, ldp);
    if (left && ldvl < 1.max(n)) || ldvl < 1 {
        *info = I::from_i64(-10);
        crate::invalid(10);
        return;
    }
    if (right && ldvr < 1.max(n)) || ldvr < 1 {
        *info = I::from_i64(-12);
        crate::invalid(12);
        return;
    }
    let mut count = 0usize;
    if howmny == b'S' {
        for j in 0..nn {
            if pair_selected(howmny, select, j, 1) {
                count += 1;
            }
        }
    } else {
        count = nn;
    }
    if mm < count as isize {
        *info = I::from_i64(-13);
        crate::invalid(13);
        return;
    }
    *m = I::from_i64(count as i64);
    if n == 0 {
        return;
    }

    let mut q = if left {
        Some(Mat::<T>::identity(nn, nn))
    } else {
        None
    };
    let mut z = if right {
        Some(Mat::<T>::identity(nn, nn))
    } else {
        None
    };
    if back {
        if let Some(q) = q.as_mut() {
            for j in 0..nn {
                for i in 0..nn {
                    q[(i, j)] = *vl.offset(i as isize + j as isize * ldvl);
                }
            }
        }
        if let Some(z) = z.as_mut() {
            for j in 0..nn {
                for i in 0..nn {
                    z[(i, j)] = *vr.offset(i as isize + j as isize * ldvr);
                }
            }
        }
    }

    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(qz_to_gevd_cplx_scratch::<T>(nn, par.shape()));
        qz_to_gevd_cplx(
            s_ref,
            p_ref,
            q.as_mut().map(|m| m.as_mut()),
            z.as_mut().map(|m| m.as_mut()),
            par,
            MemStack::new(&mut mem),
        );
    });

    if let Some(x) = q.as_mut() {
        normalize_complex(x);
    }
    if let Some(x) = z.as_mut() {
        normalize_complex(x);
    }
    let mut column = 0usize;
    for j in 0..nn {
        if !pair_selected(howmny, select, j, 1) {
            continue;
        }
        if let Some(x) = q.as_ref() {
            for i in 0..nn {
                *vl.offset(i as isize + column as isize * ldvl) = x[(i, j)];
            }
        }
        if let Some(x) = z.as_ref() {
            for i in 0..nn {
                *vr.offset(i as isize + column as isize * ldvr) = x[(i, j)];
            }
        }
        column += 1;
    }
}

use paste::paste;
macro_rules! export {
    ($name:ident,$func:ident,$p:ident,$t:ty,$i:ty,[$($suffix:tt)*])=>{paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p tgevc $($suffix)*>](
            side:*const u8,howmny:*const u8,select:*const $i,n:*const $i,
            s:*const $t,lds:*const $i,p:*const $t,ldp:*const $i,
            vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,
            mm:*const $i,m:*mut $i,work:*mut $t,info:*mut $i,
        ) {
            let _=work;
            $func::<$t,$i>(
                crate::kernels::flag(side),crate::kernels::flag(howmny),select,
                (*n).val(),s,(*lds).val(),p,(*ldp).val(),vl,(*ldvl).val(),
                vr,(*ldvr).val(),(*mm).val(),m,info,
            )
        }
    }};
}
export!(real, real_pencil, s, f32, i32, [_]);
export!(real, real_pencil, d, f64, i32, [_]);
export!(complex, complex_pencil, c, colmat::c32, i32, [_]);
export!(complex, complex_pencil, z, colmat::c64, i32, [_]);
export!(real, real_pencil, s, f32, i64, [_64_]);
export!(real, real_pencil, d, f64, i64, [_64_]);
export!(complex, complex_pencil, c, colmat::c32, i64, [_64_]);
export!(complex, complex_pencil, z, colmat::c64, i64, [_64_]);
