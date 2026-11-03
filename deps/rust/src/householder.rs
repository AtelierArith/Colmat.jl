//! Independently owned elementary Householder reflections.
use crate::collections::{VectorData, VectorOwner, VectorStore};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder as hh;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Col, Conj, Mat, MatMut, Par};
use rustcall_julia_macros::julia;
trait RealDivide: ComplexField {
    fn divided(&self, s: &Self::Real) -> Self;
}
macro_rules! divide {($($t:ty),*)=>{$(impl RealDivide for $t {fn divided(&self,s:&Self::Real)->Self {*self / *s}})*};}
divide!(f32, f64, c32, c64);
#[derive(Clone)]
struct Reflection<T: ComplexField> {
    basis: Mat<T>,
    tau: T::Real,
    beta: T,
    norm: T::Real,
    inverse: T,
}
fn make<T: RealDivide>(input: &[T]) -> Reflection<T> {
    let n = input.len();
    let mut scale = zero::<T::Real>();
    for value in input {
        for magnitude in [abs(&real(value)), abs(&imag(value))] {
            if magnitude > scale {
                scale = magnitude;
            }
        }
    }
    if scale == zero::<T::Real>() {
        scale = one::<T::Real>();
    }
    let mut basis = Mat::from_fn(n, 1, |i, _| input[i].divided(&scale));
    let mut beta = if n == 0 {
        zero::<T>()
    } else {
        copy(&basis[(0, 0)])
    };
    let (tau, norm, inverse) = if n == 0 {
        (infinity::<T::Real>(), zero::<T::Real>(), infinity::<T>())
    } else {
        let tail = basis.as_mut().col_mut(0).subrows_mut(1, n - 1);
        let info = hh::make_householder_in_place(&mut beta, tail);
        basis[(0, 0)] = one::<T>();
        (
            info.tau,
            info.norm * &scale,
            info.head_with_beta_inv.divided(&scale),
        )
    };
    Reflection {
        basis,
        tau,
        beta: mul_real(&beta, &scale),
        norm,
        inverse,
    }
}
#[derive(Clone)]
enum ReflectionData {
    F32(Reflection<f32>),
    F64(Reflection<f64>),
    C32(Reflection<c32>),
    C64(Reflection<c64>),
}
#[julia]
#[derive(Default)]
pub struct OwnedHouseholder {
    data: Option<ReflectionData>,
}
impl OwnedHouseholder {
    fn data(&self) -> Result<&ReflectionData, String> {
        self.data
            .as_ref()
            .ok_or("Householder is not initialized".into())
    }
}
#[julia]
impl OwnedHouseholder {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn householder_load(&mut self, input: *const u8, n: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            unsafe fn load<T: RealDivide>(p: *const u8, n: usize) -> Result<Reflection<T>, String> {
                let values = if n == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(p.cast::<T>(), n)
                };
                if values.iter().any(|v| !is_finite(v)) {
                    return Err("Householder input must be finite".into());
                }
                Ok(make(values))
            }
            self.data = Some(unsafe {
                match tag {
                    0 => ReflectionData::F32(load(input, n)?),
                    1 => ReflectionData::F64(load(input, n)?),
                    2 => ReflectionData::C32(load(input, n)?),
                    3 => ReflectionData::C64(load(input, n)?),
                    _ => return Err("unsupported Householder scalar".into()),
                }
            });
            Ok(())
        })
    }
    #[julia]
    pub fn householder_clone(&mut self, source: &Self, conjugate: bool) -> Result<(), String> {
        guarded(|| {
            fn transform<T: ComplexField>(f: &Reflection<T>, conjugate: bool) -> Reflection<T> {
                let mut f = f.clone();
                if conjugate {
                    for i in 0..f.basis.nrows() {
                        f.basis[(i, 0)] = conj(&f.basis[(i, 0)]);
                    }
                    f.beta = conj(&f.beta);
                    f.inverse = conj(&f.inverse);
                }
                f
            }
            self.data = Some(match source.data()? {
                ReflectionData::F32(f) => ReflectionData::F32(transform(f, conjugate)),
                ReflectionData::F64(f) => ReflectionData::F64(transform(f, conjugate)),
                ReflectionData::C32(f) => ReflectionData::C32(transform(f, conjugate)),
                ReflectionData::C64(f) => ReflectionData::C64(transform(f, conjugate)),
            });
            Ok(())
        })
    }
    #[julia]
    pub fn householder_info(&self, output: *mut u8) -> Result<(), String> {
        guarded(|| {
            unsafe fn write<T: ComplexField>(f: &Reflection<T>, p: *mut u8) {
                let p = p.cast::<T>();
                p.write(copy(&f.beta));
                p.add(1).write(from_real::<T>(&f.tau));
                p.add(2).write(from_real::<T>(&f.norm));
                p.add(3).write(copy(&f.inverse));
            }
            unsafe {
                match self.data()? {
                    ReflectionData::F32(f) => write(f, output),
                    ReflectionData::F64(f) => write(f, output),
                    ReflectionData::C32(f) => write(f, output),
                    ReflectionData::C64(f) => write(f, output),
                }
            };
            Ok(())
        })
    }
    #[julia]
    pub fn householder_element(&self, i: usize, j: usize, output: *mut u8) -> Result<(), String> {
        guarded(|| {
            unsafe fn write<T: ComplexField>(
                f: &Reflection<T>,
                i: usize,
                j: usize,
                p: *mut u8,
            ) -> Result<(), String> {
                if i >= f.basis.nrows() || j >= f.basis.nrows() {
                    return Err("Householder index out of bounds".into());
                }
                let delta = if i == j { one::<T>() } else { zero::<T>() };
                let value =
                    delta - mul_real(&(&f.basis[(i, 0)] * conj(&f.basis[(j, 0)])), &recip(&f.tau));
                p.cast::<T>().write(value);
                Ok(())
            }
            unsafe {
                match self.data()? {
                    ReflectionData::F32(f) => write(f, i, j, output),
                    ReflectionData::F64(f) => write(f, i, j, output),
                    ReflectionData::C32(f) => write(f, i, j, output),
                    ReflectionData::C64(f) => write(f, i, j, output),
                }
            }
        })
    }
}
fn apply<T: ComplexField>(f: &Reflection<T>, a: MatMut<'_, T>, right: bool) -> Result<(), String> {
    let n = f.basis.nrows();
    if if right {
        a.ncols() != n
    } else {
        a.nrows() != n
    } {
        return Err("Householder application dimensions must match".into());
    }
    if n == 0 || f.tau == infinity::<T::Real>() {
        return Ok(());
    }
    let factor = Mat::from_fn(1, 1, |_, _| from_real::<T>(&f.tau));
    let req = if right {
        hh::apply_block_householder_on_the_right_in_place_scratch::<T>(n, 1, a.nrows())
    } else {
        hh::apply_block_householder_on_the_left_in_place_scratch::<T>(n, 1, a.ncols())
    };
    let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
    if right {
        hh::apply_block_householder_on_the_right_in_place_with_conj(
            f.basis.as_ref(),
            factor.as_ref(),
            Conj::No,
            a,
            Par::Seq,
            MemStack::new(&mut mem),
        );
    } else {
        hh::apply_block_householder_on_the_left_in_place_with_conj(
            f.basis.as_ref(),
            factor.as_ref(),
            Conj::No,
            a,
            Par::Seq,
            MemStack::new(&mut mem),
        );
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn householder_apply(
        &mut self,
        reflection: &OwnedHouseholder,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| match reflection.data()? {
            ReflectionData::F64(f) => apply(f, self.matrix.as_mut(), right),
            _ => Err("Householder scalar types must match".into()),
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn householder_apply(
        &mut self,
        reflection: &OwnedHouseholder,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| {
            match (
                reflection.data()?,
                self.data.as_mut().ok_or("matrix is not initialized")?,
            ) {
                (ReflectionData::F32(f), MatrixData::F32(a)) => apply(f, a.as_mut(), right),
                (ReflectionData::F64(f), MatrixData::F64(a)) => apply(f, a.as_mut(), right),
                (ReflectionData::C32(f), MatrixData::C32(a)) => apply(f, a.as_mut(), right),
                (ReflectionData::C64(f), MatrixData::C64(a)) => apply(f, a.as_mut(), right),
                _ => Err("Householder scalar types must match".into()),
            }
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn householder_apply(
        &mut self,
        reflection: &OwnedHouseholder,
        right: bool,
    ) -> Result<(), String> {
        guarded(|| {
            match (
                reflection.data()?,
                self.data.as_mut().ok_or("vector is not initialized")?,
            ) {
                (ReflectionData::F32(f), VectorData::F32(a)) => apply(f, a.as_mat_mut(), right),
                (ReflectionData::F64(f), VectorData::F64(a)) => apply(f, a.as_mat_mut(), right),
                (ReflectionData::C32(f), VectorData::C32(a)) => apply(f, a.as_mat_mut(), right),
                (ReflectionData::C64(f), VectorData::C64(a)) => apply(f, a.as_mat_mut(), right),
                _ => Err("Householder scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn householder_vector_from(&mut self, reflection: &OwnedHouseholder) -> Result<(), String> {
        guarded(|| {
            fn vector<T: ComplexField>(f: &Reflection<T>) -> VectorStore<T> {
                VectorStore::Column(Col::from_fn(f.basis.nrows(), |i| copy(&f.basis[(i, 0)])))
            }
            self.data = Some(match reflection.data()? {
                ReflectionData::F32(f) => VectorData::F32(vector(f)),
                ReflectionData::F64(f) => VectorData::F64(vector(f)),
                ReflectionData::C32(f) => VectorData::C32(vector(f)),
                ReflectionData::C64(f) => VectorData::C64(vector(f)),
            });
            Ok(())
        })
    }
}
#[julia]
impl OwnedHouseholder {
    #[julia]
    pub fn householder_parts(
        &mut self,
        basis: *const u8,
        n: usize,
        info: *const u8,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            unsafe fn load<T: ComplexField>(
                p: *const u8,
                n: usize,
                info: *const u8,
            ) -> Result<Reflection<T>, String> {
                let info = std::slice::from_raw_parts(info.cast::<T>(), 4);
                let values = if n == 0 {
                    &[]
                } else {
                    std::slice::from_raw_parts(p.cast::<T>(), n)
                };
                if values.iter().any(|v| !is_finite(v))
                    || (n > 0 && values[0] != one::<T>())
                    || imag(&info[1]) != zero::<T::Real>()
                    || !(real(&info[1]) > zero::<T::Real>())
                {
                    return Err("invalid Householder basis/tau".into());
                }
                Ok(Reflection {
                    basis: Mat::from_fn(n, 1, |i, _| copy(&values[i])),
                    tau: real(&info[1]),
                    beta: copy(&info[0]),
                    norm: real(&info[2]),
                    inverse: copy(&info[3]),
                })
            }
            self.data = Some(unsafe {
                match tag {
                    0 => ReflectionData::F32(load(basis, n, info)?),
                    1 => ReflectionData::F64(load(basis, n, info)?),
                    2 => ReflectionData::C32(load(basis, n, info)?),
                    3 => ReflectionData::C64(load(basis, n, info)?),
                    _ => return Err("unsupported Householder scalar".into()),
                }
            });
            Ok(())
        })
    }
}
