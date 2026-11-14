//! LAPACK/BLAS plane rotations composed from native JacobiRotation.
use crate::generalized_selfadjoint::GeneralizedScalar;
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, Scalar};
use colmat::{linalg::jacobi::JacobiRotation, ColMut, RowMut};
use paste::paste;

#[derive(Clone, Copy)]
enum Convention {
    Blas,
    Lapack,
    Positive,
}
fn generate<T: ReflectorScalar>(f: T, g: T, convention: Convention) -> (T, T, T) {
    if T::COMPLEX {
        let (j, r) = JacobiRotation::rotg(f, g);
        return (j.c, -j.s.cj(), r);
    }
    let (a, b) = (f.re(), g.re());
    if g == T::z() {
        if matches!(convention, Convention::Positive) {
            return (
                T::real(if a.is_sign_negative() { -1.0 } else { 1.0 }),
                T::z(),
                T::real(a.abs()),
            );
        }
        return (T::o(), T::z(), f);
    }
    if f == T::z() {
        if matches!(convention, Convention::Blas) {
            return (T::z(), T::o(), g);
        }
        return (
            T::z(),
            T::real(if b.is_sign_negative() { -1.0 } else { 1.0 }),
            T::real(b.abs()),
        );
    }
    // Native real ROTG derives coefficients by dividing by r. Normalize first
    // so that a representable rotation survives an overflowing output r.
    let scale = a.abs().max(b.abs());
    let (j, mut r) = JacobiRotation::rotg(T::real(a / scale), T::real(b / scale));
    let (mut c, mut s) = (j.c, -j.s);
    let flip = match convention {
        Convention::Blas => false,
        Convention::Lapack => a.is_sign_negative() != r.re().is_sign_negative(),
        Convention::Positive => r.re().is_sign_negative(),
    };
    if flip {
        c = -c;
        s = -s;
        r = -r;
    }
    (c, s, T::real(r.re() * scale))
}
// Evaluate a*b/(c*d) without forming overflowing products or prematurely
// underflowing the small coefficient of a shifted rotation.
fn product_quotient(a: f64, b: f64, c: f64, d: f64) -> f64 {
    fn parts(value: f64) -> (f64, i32) {
        let exponent = (((value.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022);
        (value / 2.0_f64.powi(exponent), exponent)
    }
    let (a, ea) = parts(a);
    let (b, eb) = parts(b);
    let (c, ec) = parts(c);
    let (d, ed) = parts(d);
    let quotient = (a * b) / (c * d);
    if quotient == 0.0 {
        return quotient;
    }
    let (mantissa, eq) = parts(quotient);
    let exponent = ea + eb - ec - ed + eq;
    if exponent > 1023 {
        f64::INFINITY.copysign(mantissa)
    } else if exponent < -1075 {
        0.0_f64.copysign(mantissa)
    } else if exponent < -1022 {
        (mantissa * 2.0_f64.powi(exponent + 1022)) * f64::MIN_POSITIVE
    } else {
        mantissa * 2.0_f64.powi(exponent)
    }
}
fn shifted<T: ReflectorScalar>(x: T, y: T, sigma: T) -> (T, T) {
    let ax = x.re().abs();
    let shift = sigma.re();
    let threshold = T::EPS / 2.0;
    let sign = T::real(if x.re() < 0.0 { -1.0 } else { 1.0 });
    let (mut z, mut w) = if (shift == 0.0 && ax < threshold) || (ax == shift && y == T::z()) {
        (T::z(), T::z())
    } else if shift == 0.0 {
        (T::real(ax), sign * y)
    } else if ax < threshold {
        (-(sigma * sigma), T::z())
    } else {
        (
            (sign * (T::real(ax) - sigma)) * (sign + sigma / x),
            sign * y,
        )
    };
    if (!z.re().is_finite() || !w.re().is_finite())
        && x.re().is_finite()
        && y.re().is_finite()
        && shift.is_finite()
    {
        if ax < threshold {
            z = -T::o();
            w = T::z();
        } else {
            let difference = ax * 0.5 - shift * 0.5;
            let sum = ax * 0.5 + shift * 0.5;
            if difference == 0.0 || sum == 0.0 {
                z = T::z();
                w = sign * y;
            } else {
                let direction = if difference.is_sign_negative() != sum.is_sign_negative() {
                    -1.0
                } else {
                    1.0
                };
                let ratio = product_quotient(sign.re() * y.re(), ax * 0.25, difference, sum);
                z = T::real(direction);
                w = T::real(direction * ratio);
            }
        }
    }
    let (s, c, _) = generate(w, z, Convention::Positive);
    (c, s)
}
fn application<T: Scalar>(c: T, s: T) -> JacobiRotation<T> {
    JacobiRotation { c, s: -s.cj() }
}
unsafe fn apply_pair<T: Scalar>(x: *mut T, y: *mut T, c: T, s: T) {
    let (u, _, v, _) = application(c, s).apply_on_the_left_2x2(*x, T::z(), *y, T::z());
    *x = u;
    *y = v;
}
unsafe fn two_sided<T: GeneralizedScalar>(x: *mut T, y: *mut T, z: *mut T, c: T, s: T) {
    // LAR2V uses conj(s) in its upper-left rotation, unlike LARTV.
    let rotation = JacobiRotation { c, s: -s };
    let (a, b, d, e) =
        rotation.apply_on_the_left_2x2(T::diagonal(x), *z, (*z).cj(), T::diagonal(y));
    let (a, b, _, e) = rotation.adjoint().apply_on_the_right_2x2(a, b, d, e);
    *x = T::real(a.re());
    *y = T::real(e.re());
    *z = b;
}
unsafe fn sequence<T: Scalar, R: Scalar>(
    side: u8,
    pivot: u8,
    direct: u8,
    m: isize,
    n: isize,
    c: *const R,
    s: *const R,
    a: *mut T,
    lda: isize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    for step in 0..order.saturating_sub(1) {
        let k = if direct == b'F' {
            step
        } else {
            order - 2 - step
        };
        let (first, second) = match pivot {
            b'V' => (k, k + 1),
            b'T' => (0, k + 1),
            _ => (k, order - 1),
        };
        let (co, si) = (T::real((*c.offset(k)).re()), T::real((*s.offset(k)).re()));
        if co == T::o() && si == T::z() {
            continue;
        }
        let rotation = application(co, si);
        if side == b'L' {
            rotation.apply_on_the_left_in_place((
                RowMut::from_raw_parts_mut(a.offset(first), n as usize, lda),
                RowMut::from_raw_parts_mut(a.offset(second), n as usize, lda),
            ));
        } else {
            rotation.transpose().apply_on_the_right_in_place((
                ColMut::from_raw_parts_mut(a.offset(first * lda), m as usize, 1),
                ColMut::from_raw_parts_mut(a.offset(second * lda), m as usize, 1),
            ));
        }
    }
}
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! common {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lartg $suffix>](f:*const $t,g:*const $t,c:*mut $r,s:*mut $t,r:*mut $t) {
            let(co,si,rr)=generate(*f,*g,Convention::Lapack);*c=co.re() as $r;*s=si;*r=rr;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p largv $suffix>](n:*const $i,x:*mut $t,incx:*const $i,
            y:*mut $t,incy:*const $i,c:*mut $r,incc:*const $i) {
            let n=*n as isize;if n<=0{return;}
            let(ix,iy,ic)=(*incx as isize,*incy as isize,*incc as isize);
            check!(3=>ix>0,5=>iy>0,7=>ic>0);
            for j in 0..n {
                let(xp,yp)=(x.offset(j*ix),y.offset(j*iy));
                let(co,si,rr)=generate(*xp,*yp,Convention::Blas);
                *xp=rr;*yp=si;*c.offset(j*ic)=co.re() as $r;
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lartv $suffix>](n:*const $i,x:*mut $t,incx:*const $i,
            y:*mut $t,incy:*const $i,c:*const $r,s:*const $t,incc:*const $i) {
            let n=*n as isize;if n<=0{return;}
            let(ix,iy,ic)=(*incx as isize,*incy as isize,*incc as isize);
            check!(3=>ix>0,5=>iy>0,8=>ic>0);
            for j in 0..n {apply_pair(x.offset(j*ix),y.offset(j*iy),<$t as Scalar>::real(*c.offset(j*ic) as f64),*s.offset(j*ic));}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lar2v $suffix>](n:*const $i,x:*mut $t,y:*mut $t,z:*mut $t,
            incx:*const $i,c:*const $r,s:*const $t,incc:*const $i) {
            let n=*n as isize;if n<=0{return;}
            let(ix,ic)=(*incx as isize,*incc as isize);check!(5=>ix>0,8=>ic>0);
            for j in 0..n {two_sided(x.offset(j*ix),y.offset(j*ix),z.offset(j*ix),
                <$t as Scalar>::real(*c.offset(j*ic) as f64),*s.offset(j*ic));}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lasr $suffix>](side:*const u8,pivot:*const u8,direct:*const u8,
            m:*const $i,n:*const $i,c:*const $r,s:*const $r,a:*mut $t,lda:*const $i) {
            let(side,pivot,direct,mm,nn,ld)=(flag(side),flag(pivot),flag(direct),*m as isize,*n as isize,*lda as isize);
            check!(1=>matches!(side,b'L'|b'R'),2=>matches!(pivot,b'V'|b'T'|b'B'),
                3=>matches!(direct,b'F'|b'B'),4=>mm>=0,5=>nn>=0,9=>ld>=mm.max(1));
            sequence(side,pivot,direct,mm,nn,c,s,a,ld);
        }
    }};
}
macro_rules! real {
    ($p:ident,$t:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p rotg $suffix>](a:*mut $t,b:*mut $t,c:*mut $t,s:*mut $t) {
            let(f,g)=(*a,*b);let(co,si,r)=generate(f,g,Convention::Blas);
            let z=if g==0.0 {0.0} else if f.abs()>g.abs(){si}
                else if co!=0.0 {1.0/co} else {1.0};
            *a=r;*b=z;*c=co;*s=si;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lartgp $suffix>](f:*const $t,g:*const $t,c:*mut $t,s:*mut $t,r:*mut $t) {
            let(co,si,rr)=generate(*f,*g,Convention::Positive);*c=co;*s=si;*r=rr;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lartgs $suffix>](x:*const $t,y:*const $t,sigma:*const $t,c:*mut $t,s:*mut $t) {
            let(co,si)=shifted(*x,*y,*sigma);*c=co;*s=si;
        }
    }};
}
macro_rules! complex {
    ($p:ident,$t:ty,$r:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p rotg $suffix>](a:*mut $t,b:*const $t,c:*mut $r,s:*mut $t) {
            let(co,si,r)=generate(*a,*b,Convention::Blas);*a=r;*c=co.re() as $r;*s=si;
        }
    }};
}
common!(s, f32, f32, i32, _);
common!(d, f64, f64, i32, _);
common!(c, colmat::c32, f32, i32, _);
common!(z, colmat::c64, f64, i32, _);
common!(s, f32, f32, i64, _64_);
common!(d, f64, f64, i64, _64_);
common!(c, colmat::c32, f32, i64, _64_);
common!(z, colmat::c64, f64, i64, _64_);
real!(s, f32, i32, _);
real!(d, f64, i32, _);
real!(s, f32, i64, _64_);
real!(d, f64, i64, _64_);
complex!(c, colmat::c32, f32, _);
complex!(z, colmat::c64, f64, _);
complex!(c, colmat::c32, f32, _64_);
complex!(z, colmat::c64, f64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};
    #[test]
    fn empty_and_identity_operations_do_not_read_arrays() {
        unsafe {
            sequence::<f64, f64>(b'L', b'V', b'F', 0, 3, null(), null(), null_mut(), 1);
            sequence::<f64, f64>(b'L', b'V', b'F', 1, 3, null(), null(), null_mut(), 1);
            sequence::<f64, f64>(
                b'L',
                b'V',
                b'F',
                3,
                2,
                [1.0, 1.0].as_ptr(),
                [0.0, 0.0].as_ptr(),
                null_mut(),
                3,
            );
            dlargv_64_(
                &0,
                null_mut(),
                null(),
                null_mut(),
                null(),
                null_mut(),
                null(),
            );
            dlartv_64_(
                &0,
                null_mut(),
                null(),
                null_mut(),
                null(),
                null(),
                null(),
                null(),
            );
            dlar2v_64_(
                &0,
                null_mut(),
                null_mut(),
                null_mut(),
                null(),
                null(),
                null(),
                null(),
            );
        }
    }
    #[test]
    fn shifted_rotation_preserves_a_small_representable_sine() {
        let x = f64::EPSILON;
        let y = f64::MAX / 7.0;
        let sigma = f64::MAX.sqrt() / 2.0;
        let (c, s) = shifted(x, y, sigma);
        assert!((c + 1.0).abs() < 1e-14);
        assert!((s / (4.0 * x / 7.0) - 1.0).abs() < 1e-14);
        assert_eq!(
            product_quotient(f64::MAX, f64::MAX, f64::MAX, f64::MAX),
            1.0
        );
        assert_eq!(
            product_quotient(
                f64::MIN_POSITIVE,
                f64::MIN_POSITIVE,
                f64::MIN_POSITIVE,
                f64::MIN_POSITIVE
            ),
            1.0
        );
        assert_eq!(
            product_quotient(f64::MIN_POSITIVE, 1.0, 2.0_f64.powi(52), 1.0),
            f64::from_bits(1)
        );
    }
    #[test]
    fn overflowing_r_retains_a_unit_rotation() {
        let (c, s, r) = generate(f64::MAX, f64::MAX, Convention::Lapack);
        assert!((c * c + s * s - 1.0).abs() < 1e-14);
        assert!((c - s).abs() < 1e-14);
        assert!(r.is_infinite());
    }
}
