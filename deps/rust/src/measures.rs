//! Determinant, trace and log-|det| of borrowed native views by structure.
//! Dense determinants use faer's `MatRef::determinant(par)`; triangular and
//! diagonal operands only read their stored diagonal. Nothing is retained.
use crate::guarded;
use crate::teams::{with_team, OwnedTeam};
use crate::views::{source, Borrowed, Spec};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatRef, Par};
use rustcall_julia_macros::julia;

/// Natural logarithm of the concrete real fields.
pub(crate) trait Ln {
    fn ln_(&self) -> Self;
}
impl Ln for f32 {
    fn ln_(&self) -> Self {
        self.ln()
    }
}
impl Ln for f64 {
    fn ln_(&self) -> Self {
        self.ln()
    }
}
/// Structures: 0 dense, 1 lower, 2 upper, 3 unit lower, 4 unit upper,
/// 5 a diagonal stored as an n×1 column. Codes: 0 det, 1 trace, 2 log|det|
/// written as `[log, phase]` (phase is the product of entry signs).
fn measure<T: ComplexField<Real: Ln>>(
    a: MatRef<'_, T>,
    conjugate: bool,
    structure: i32,
    code: i32,
    par: Par<'_>,
    out: *mut u8,
) -> Result<(), String> {
    let n = a.nrows();
    let square = if structure == 5 {
        a.ncols() == 1
    } else {
        a.ncols() == n
    };
    if !square || !(0..=5).contains(&structure) {
        return Err("measure requires a square operand".into());
    }
    let unit = structure == 3 || structure == 4;
    let d = |i: usize| -> T {
        let v = if structure == 5 {
            &a[(i, 0)]
        } else {
            &a[(i, i)]
        };
        if conjugate {
            conj(v)
        } else {
            copy(v)
        }
    };
    let out = out.cast::<T>();
    match code {
        0 => {
            let v = if structure == 0 {
                let x = a.determinant(par);
                if conjugate {
                    conj(&x)
                } else {
                    x
                }
            } else if unit {
                one::<T>()
            } else {
                (0..n).fold(one::<T>(), |s, i| mul(&s, &d(i)))
            };
            unsafe { out.write(v) };
        }
        1 => {
            let v = if unit {
                (0..n).fold(zero::<T>(), |s, _| add(&s, &one::<T>()))
            } else {
                (0..n).fold(zero::<T>(), |s, i| add(&s, &d(i)))
            };
            unsafe { out.write(v) };
        }
        2 => {
            if structure == 0 {
                return Err("dense log-determinants use a factorization".into());
            }
            let (mut log, mut phase) = (zero::<T::Real>(), one::<T>());
            if !unit {
                for i in 0..n {
                    let x = d(i);
                    let r = abs(&x);
                    log = add(&log, &r.ln_());
                    phase = if r == zero::<T::Real>() {
                        zero::<T>()
                    } else {
                        mul(&phase, &mul_real(&x, &recip(&r)))
                    };
                }
            }
            unsafe {
                out.write(from_real(&log));
                out.add(1).write(phase);
            }
        }
        _ => return Err("unknown measure".into()),
    }
    Ok(())
}
unsafe fn run(
    p: *const u8,
    desc: *const isize,
    structure: i32,
    code: i32,
    out: *mut u8,
    par: Par<'_>,
) -> Result<(), String> {
    let s = Spec::read(desc)?;
    match source(p, s)? {
        Borrowed::F32(a) => measure(a, s.conjugate, structure, code, par, out),
        Borrowed::F64(a) => measure(a, s.conjugate, structure, code, par, out),
        Borrowed::C32(a) => measure(a, s.conjugate, structure, code, par, out),
        Borrowed::C64(a) => measure(a, s.conjugate, structure, code, par, out),
    }
}
/// `team` is a FaerTeam handle or Julia `nothing` (RustCall 0.10 optional owner).
#[julia]
pub fn view_measure(
    p: *const u8,
    desc: *const isize,
    structure: i32,
    code: i32,
    out: *mut u8,
    team: Option<&OwnedTeam>,
) -> Result<(), String> {
    guarded(|| {
        with_team(team, |par| unsafe {
            run(p, desc, structure, code, out, par)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OwnedMatrix;
    use faer::Mat;
    #[test]
    fn structured_measures() {
        let mut a = OwnedMatrix::new();
        a.matrix = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                (i + 2) as f64
            } else {
                (i + 3 * j) as f64
            }
        });
        let p = (&a as *const OwnedMatrix).cast::<u8>();
        let d = [0isize, 0, 1, 3, 0, 1, 3, 0, 0];
        let mut out = [0.0f64; 2];
        let det = |s: i32, out: &mut [f64; 2]| {
            view_measure(p, d.as_ptr(), s, 0, out.as_mut_ptr().cast(), None).map(|_| out[0])
        };
        assert!((det(0, &mut out).unwrap() - a.matrix.determinant(Par::Seq)).abs() < 1e-12);
        assert_eq!(det(1, &mut out).unwrap(), 24.0);
        assert_eq!(det(3, &mut out).unwrap(), 1.0);
        view_measure(p, d.as_ptr(), 4, 1, out.as_mut_ptr().cast(), None).unwrap();
        assert_eq!(out[0], 3.0);
        view_measure(p, d.as_ptr(), 2, 2, out.as_mut_ptr().cast(), None).unwrap();
        assert!((out[0] - 24f64.ln()).abs() < 1e-12 && out[1] == 1.0);
        assert!(view_measure(p, d.as_ptr(), 0, 2, out.as_mut_ptr().cast(), None).is_err());
        let wide = [0isize, 0, 1, 2, 0, 1, 3, 0, 0];
        assert!(view_measure(p, wide.as_ptr(), 0, 0, out.as_mut_ptr().cast(), None).is_err());
    }
}
