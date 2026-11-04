//! Source-precision rank bounds from borrowed R and caller-owned accumulation arrays.
use super::{Factors, Numeric};
use crate::operators::NativeValue;
use faer::dyn_stack::{MemStack, StackReq};
use faer::traits::ComplexField;
pub(super) fn scratch(n: usize) -> StackReq {
    if n == 0 {
        StackReq::EMPTY
    } else {
        StackReq::all_of(&[StackReq::new::<f64>(n); 3])
    }
}
fn size<T: ComplexField>(f: &Numeric<T>) -> usize {
    match f {
        Numeric::Qr(x) => x.dimensions().1,
        Numeric::SimplicialQr(x) => x.r.ncols(),
        _ => 0,
    }
}
pub(super) fn requirements(f: &Factors) -> StackReq {
    let n = match f {
        Factors::F32(x) => size(x),
        Factors::F64(x) => size(x),
        Factors::C32(x) => size(x),
        Factors::C64(x) => size(x),
    };
    scratch(n)
}
pub(super) fn needs<T: NativeValue>(
    f: &Numeric<T>,
    epsilon: f64,
    stack: &mut MemStack,
) -> Option<bool> {
    let n = size(f);
    if !matches!(
        f,
        Numeric::Qr(_) | Numeric::SimplicialQr(_) | Numeric::WideQr(_)
    ) {
        return None;
    }
    if n == 0 {
        return Some(false);
    }
    let (mut rows, stack) = stack.make_with::<f64>(n, |_| 0.0);
    let (mut cols, stack) = stack.make_with::<f64>(n, |_| 0.0);
    let (mut diagonal, _) = stack.make_with::<f64>(n, |_| 0.0);
    let mut diagonal_only = true;
    let mut visit = |i: usize, j: usize, v: &T| {
        let (re, im) = v.parts();
        let x = re.hypot(im);
        rows[i] += x;
        cols[j] += x;
        if i != j && x != 0.0 {
            diagonal_only = false;
        }
        if i == j {
            diagonal[i] += x;
        }
    };
    match f {
        Numeric::Qr(x) => x.visit_r(&mut visit),
        Numeric::SimplicialQr(x) => {
            for e in x.r.as_ref().triplet_iter() {
                visit(e.row, e.col, e.val);
            }
        }
        _ => unreachable!(),
    }
    if diagonal_only {
        let lo = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = diagonal.iter().copied().fold(0.0, f64::max);
        return Some(
            !lo.is_finite()
                || !hi.is_finite()
                || lo <= n as f64 * epsilon * hi * (1.0 + 8.0 * epsilon),
        );
    }
    let margin = (0..n)
        .map(|i| 2.0 * diagonal[i] - rows[i] - 8.0 * f64::EPSILON * rows[i])
        .fold(f64::INFINITY, f64::min);
    let upper = rows.iter().copied().fold(0.0, f64::max).sqrt()
        * cols.iter().copied().fold(0.0, f64::max).sqrt();
    let lower = margin / (n as f64).sqrt();
    Some(
        !lower.is_finite()
            || !upper.is_finite()
            || lower <= 0.0
            || lower <= n as f64 * epsilon * upper,
    )
}
pub(super) fn prepare(f: &Factors, stack: &mut MemStack) -> Option<bool> {
    match f {
        Factors::F32(x) => needs(x, f32::EPSILON as f64, stack),
        Factors::F64(x) => needs(x, f64::EPSILON, stack),
        Factors::C32(x) => needs(x, f32::EPSILON as f64, stack),
        Factors::C64(x) => needs(x, f64::EPSILON, stack),
    }
}
