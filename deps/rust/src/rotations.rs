//! Julia Givens conventions adapted to faer's Jacobi rotation kernels.
use crate::collections::{VectorData, VectorOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::traits::{math_utils::*, ComplexField};
use faer::{linalg::jacobi::JacobiRotation, MatMut};
use rustcall_julia_macros::julia;
fn parameters<T: ComplexField>(p: T, q: T) -> (T, T, T) {
    // Avoid forming real coefficients from an already-rounded subnormal r.
    let (mut j, mut r) = if T::IS_REAL && p != zero::<T>() && q != zero::<T>() {
        let mut scale = if abs(&p) > abs(&q) {
            copy(&p)
        } else {
            copy(&q)
        };
        if real(&scale) < zero::<T::Real>() {
            scale = -scale;
        }
        let (j, r) = JacobiRotation::rotg(&p / &scale, &q / &scale);
        (j, r * scale)
    } else {
        JacobiRotation::rotg(copy(&p), copy(&q))
    };
    // faer's complex zero-q branch returns one rather than the input p.
    if q == zero::<T>() {
        r = copy(&p);
    }
    // Match Julia's real lartg sign convention when q dominates p.
    if T::IS_REAL
        && p != zero::<T>()
        && q != zero::<T>()
        && abs(&p) <= abs(&q)
        && real(&r) < zero::<T::Real>()
    {
        j.c = -j.c;
        j.s = -j.s;
        r = -r;
    }
    (j.c, -conj(&j.s), r)
}
#[julia]
pub fn givens_parameters(input: *const u8, output: *mut u8, tag: i32) -> Result<(), String> {
    guarded(|| {
        unsafe fn run<T: ComplexField>(input: *const u8, output: *mut u8) {
            let p = copy(&*input.cast::<T>());
            let q = copy(&*input.cast::<T>().add(1));
            let (c, s, r) = parameters(p, q);
            output.cast::<T>().write(c);
            output.cast::<T>().add(1).write(s);
            output.cast::<T>().add(2).write(r);
        }
        unsafe {
            match tag {
                0 => run::<f32>(input, output),
                1 => run::<f64>(input, output),
                2 => run::<faer::c32>(input, output),
                3 => run::<faer::c64>(input, output),
                _ => return Err("unsupported Givens scalar".into()),
            }
        }
        Ok(())
    })
}
unsafe fn apply<T: ComplexField>(
    a: MatMut<'_, T>,
    coeff: *const u8,
    i: usize,
    j: usize,
    right: bool,
) -> Result<(), String> {
    let bound = if right { a.ncols() } else { a.nrows() };
    if i == j || i >= bound || j >= bound {
        return Err("Givens indices must be distinct and within the selected axis".into());
    }
    let c = copy(&*coeff.cast::<T>());
    let s = copy(&*coeff.cast::<T>().add(1));
    if imag(&c) != zero::<T::Real>() {
        return Err("Givens cosine must be real".into());
    }
    let rotation = JacobiRotation { c, s: -conj(&s) };
    if right {
        rotation.apply_on_the_right_in_place(a.two_cols_mut(i, j));
    } else {
        rotation.apply_on_the_left_in_place(a.two_rows_mut(i, j));
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn givens_apply(
        &mut self,
        coeff: *const u8,
        i: usize,
        j: usize,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe { apply(self.matrix.as_mut(), coeff, i, j, right) })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn givens_apply(
        &mut self,
        coeff: *const u8,
        i: usize,
        j: usize,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            match self.data.as_mut().ok_or("matrix is not initialized")? {
                MatrixData::F32(a) => apply(a.as_mut(), coeff, i, j, right),
                MatrixData::F64(a) => apply(a.as_mut(), coeff, i, j, right),
                MatrixData::C32(a) => apply(a.as_mut(), coeff, i, j, right),
                MatrixData::C64(a) => apply(a.as_mut(), coeff, i, j, right),
            }
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn givens_apply(
        &mut self,
        coeff: *const u8,
        i: usize,
        j: usize,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| unsafe {
            match self.data.as_mut().ok_or("vector is not initialized")? {
                VectorData::F32(a) => apply(a.as_mat_mut(), coeff, i, j, right),
                VectorData::F64(a) => apply(a.as_mat_mut(), coeff, i, j, right),
                VectorData::C32(a) => apply(a.as_mat_mut(), coeff, i, j, right),
                VectorData::C64(a) => apply(a.as_mat_mut(), coeff, i, j, right),
            }
        })
    }
}
