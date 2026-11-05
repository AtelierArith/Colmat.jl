//! Compact block Householder products using faer's upgrade/sequence kernels.
use crate::collections::{VectorData, VectorOwner};
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder as hh;
use faer::prelude::ReborrowMut;
use faer::traits::{math_utils::*, ComplexField};
use faer::{c32, c64, Conj, Mat, MatMut, MatRef, Par};
use rustcall_julia_macros::julia;
#[derive(Clone)]
struct Sequence<T> {
    basis: Mat<T>,
    factor: Mat<T>,
}
unsafe fn load<T: ComplexField>(
    p: *const u8,
    m: usize,
    k: usize,
    tau: *const u8,
    b: usize,
    provided: *const u8,
    mut par: Par<'_>,
) -> Result<Sequence<T>, String> {
    if k > m || b == 0 {
        return Err("invalid Householder sequence dimensions/blocksize".into());
    }
    construction::span::<T>(p,m.checked_mul(k).ok_or("Householder basis size overflow")?)?;
    construction::span::<T>(tau,k)?;
    let count=b.checked_mul(k).ok_or("Householder factor size overflow")?;
    if !provided.is_null(){construction::span::<T>(provided,count)?;}
    let mut basis = Mat::from_fn(m, k, |i, j| {
        if i < j {
            zero::<T>()
        } else if i == j {
            one::<T>()
        } else {
            copy(&*p.cast::<T>().add(i + j * m))
        }
    });
    let mut factor = Mat::zeros(b, k);
    for j in 0..k {
        let t = copy(&*tau.cast::<T>().add(j));
        if imag(&t) != zero::<T::Real>() || !(real(&t) > zero::<T::Real>()) {
            return Err("Householder denominators must be positive real values or infinity".into());
        }
        factor[(j % b, j)] = t;
        for i in j + 1..m {
            if !is_finite(&basis[(i, j)]) {
                return Err("Householder essentials must be finite".into());
            }
            if real(&factor[(j % b, j)]) == infinity::<T::Real>() {
                basis[(i, j)] = zero::<T>();
            }
        }
    }
    let mut start = 0;
    while provided.is_null() && start < k {
        let width = b.min(k - start);
        hh::upgrade_householder_factor(
            factor.as_mut().submatrix_mut(0, start, width, width),
            basis.as_ref().submatrix(start, start, m - start, width),
            width,
            1,
            par.reborrow(),
        );
        start += width;
    }
    if !provided.is_null() {
        for j in 0..k {
            for i in 0..=j % b {
                factor[(i, j)] = copy(&*provided.cast::<T>().add(i + j * b));
            }
        }
    }
    Ok(Sequence { basis, factor })
}
#[derive(Clone)]
enum Data {
    F32(Sequence<f32>),
    F64(Sequence<f64>),
    C32(Sequence<c32>),
    C64(Sequence<c64>),
}
#[julia]
#[derive(Default)]
pub struct OwnedHouseholderSequence {
    data: Option<Data>,
}
impl OwnedHouseholderSequence {
    fn data(&self) -> Result<&Data, String> {
        self.data
            .as_ref()
            .ok_or("Householder sequence is not initialized".into())
    }
}
#[julia]
impl OwnedHouseholderSequence {
    #[julia]
    pub fn new() -> Self {
        Self::default()
    }
    #[julia]
    pub fn sequence_load(
        &mut self,
        basis: *const u8,
        m: usize,
        k: usize,
        tau: *const u8,
        b: usize,
        tag: i32,
        provided: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe {
                match tag {
                    0 => Data::F32(load(basis, m, k, tau, b, provided, Par::Seq)?),
                    1 => Data::F64(load(basis, m, k, tau, b, provided, Par::Seq)?),
                    2 => Data::C32(load(basis, m, k, tau, b, provided, Par::Seq)?),
                    3 => Data::C64(load(basis, m, k, tau, b, provided, Par::Seq)?),
                    _ => return Err("unsupported sequence scalar".into()),
                }
            });
            Ok(())
        })
    }
    #[julia]
    pub fn sequence_clone(&mut self, source: &Self) -> Result<(), String> {
        guarded(|| {
            self.data = Some(source.data()?.clone());
            Ok(())
        })
    }
    #[julia]
    pub fn sequence_taus(&self, output: *mut u8, k: usize) -> Result<(), String> {
        guarded(|| {
            unsafe fn write<T: ComplexField>(
                s: &Sequence<T>,
                out: *mut u8,
                k: usize,
            ) -> Result<(), String> {
                if k != s.basis.ncols() {
                    return Err("Householder tau length mismatch".into());
                }
                for j in 0..k {
                    out.cast::<T>()
                        .add(j)
                        .write(copy(&s.factor[(j % s.factor.nrows(), j)]));
                }
                Ok(())
            }
            unsafe {
                match self.data()? {
                    Data::F32(s) => write(s, output, k),
                    Data::F64(s) => write(s, output, k),
                    Data::C32(s) => write(s, output, k),
                    Data::C64(s) => write(s, output, k),
                }
            }
        })
    }
}
fn requirement<T: ComplexField>(
    s: &Sequence<T>,
    extent: usize,
    right: bool,
    transpose: bool,
) -> faer::dyn_stack::StackReq {
    let (m, k, b) = (s.basis.nrows(), s.basis.ncols(), s.factor.nrows());
    if k == 0 || (m == 1 && k == 1) {
        return faer::dyn_stack::StackReq::EMPTY;
    }
    match (right, transpose) {
        (false, false) => {
            hh::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(m, b, extent)
        }
        (false, true) => {
            hh::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T>(
                m, b, extent,
            )
        }
        (true, false) => {
            hh::apply_block_householder_sequence_on_the_right_in_place_scratch::<T>(m, b, extent)
        }
        (true, true) => {
            hh::apply_block_householder_sequence_transpose_on_the_right_in_place_scratch::<T>(
                m, b, extent,
            )
        }
    }
}
fn apply<T: ComplexField>(
    s: &Sequence<T>,
    a: MatMut<'_, T>,
    right: bool,
    transpose: bool,
    conjugate: bool,
) -> Result<(), String> {
    let req = requirement(
        s,
        if right { a.nrows() } else { a.ncols() },
        right,
        transpose,
    );
    let mut mem = MemBuffer::try_new(req).map_err(|e| format!("{e:?}"))?;
    apply_in(
        s,
        a,
        right,
        transpose,
        conjugate,
        Par::Seq,
        MemStack::new(&mut mem),
    )
}
fn apply_in<T: ComplexField>(
    s: &Sequence<T>,
    a: MatMut<'_, T>,
    right: bool,
    transpose: bool,
    conjugate: bool,
    par: Par,
    stack: &mut MemStack,
) -> Result<(), String> {
    apply_views_in(
        s.basis.as_ref(),
        s.factor.as_ref(),
        a,
        right,
        transpose,
        conjugate,
        par,
        stack,
    )
}
fn apply_views_in<T: ComplexField>(
    basis: MatRef<'_, T>,
    factor: MatRef<'_, T>,
    mut a: MatMut<'_, T>,
    right: bool,
    transpose: bool,
    conjugate: bool,
    par: Par,
    stack: &mut MemStack,
) -> Result<(), String> {
    let m = basis.nrows();
    let k = basis.ncols();
    let b = factor.nrows();
    if if right {
        a.ncols() != m
    } else {
        a.nrows() != m
    } {
        return Err("sequence application dimensions must match".into());
    }
    if k == 0 {
        return Ok(());
    }
    // faer's rank-one SIMD path skips a reflection with no tail, although
    // an explicitly supplied finite tau can make it a scalar sign flip.
    let scalar_tail = m == k && (b == 1 || k % b == 1);
    let active = if scalar_tail { k - 1 } else { k };
    let scalar = if scalar_tail {
        let tau = &factor[((k - 1) % b, k - 1)];
        if real(tau) == infinity::<T::Real>() {
            one::<T>()
        } else {
            one::<T>() - recip(tau)
        }
    } else {
        one::<T>()
    };
    fn scale_last<T: ComplexField>(a: &mut MatMut<'_, T>, right: bool, scale: &T) {
        if right {
            let col = a.ncols() - 1;
            for i in 0..a.nrows() {
                a[(i, col)] = &a[(i, col)] * scale;
            }
        } else {
            let row = a.nrows() - 1;
            for j in 0..a.ncols() {
                a[(row, j)] = scale * &a[(row, j)];
            }
        }
    }
    if active == 0 {
        if scalar_tail {
            scale_last(&mut a, right, &scalar);
        }
        return Ok(());
    }
    if scalar_tail && right == transpose {
        scale_last(&mut a, right, &scalar);
    }
    let conj = if conjugate { Conj::Yes } else { Conj::No };
    let v = basis.subcols(0, active);
    let f = factor.subcols(0, active);
    match (right, transpose) {
        (false, false) => hh::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            v,
            f,
            conj,
            a.rb_mut(),
            par,
            stack,
        ),
        (false, true) => {
            hh::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                v,
                f,
                conj,
                a.rb_mut(),
                par,
                stack,
            )
        }
        (true, false) => hh::apply_block_householder_sequence_on_the_right_in_place_with_conj(
            v,
            f,
            conj,
            a.rb_mut(),
            par,
            stack,
        ),
        (true, true) => {
            hh::apply_block_householder_sequence_transpose_on_the_right_in_place_with_conj(
                v,
                f,
                conj,
                a.rb_mut(),
                par,
                stack,
            )
        }
    }
    if scalar_tail && right != transpose {
        scale_last(&mut a, right, &scalar);
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn sequence_apply(
        &mut self,
        source: &OwnedHouseholderSequence,
        right: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| match source.data()? {
            Data::F64(s) => apply(s, self.matrix.as_mut(), right, transpose, conjugate),
            _ => Err("sequence scalar types must match".into()),
        })
    }
    #[julia]
    pub fn sequence_component(
        &mut self,
        source: &OwnedHouseholderSequence,
        factor: bool,
    ) -> Result<(), String> {
        guarded(|| match source.data()? {
            Data::F64(s) => {
                self.matrix = if factor {
                    s.factor.clone()
                } else {
                    s.basis.clone()
                };
                Ok(())
            }
            _ => Err("sequence scalar types must match".into()),
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn sequence_apply(
        &mut self,
        source: &OwnedHouseholderSequence,
        right: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            match (
                source.data()?,
                self.data.as_mut().ok_or("matrix is not initialized")?,
            ) {
                (Data::F32(s), MatrixData::F32(a)) => {
                    apply(s, a.as_mut(), right, transpose, conjugate)
                }
                (Data::F64(s), MatrixData::F64(a)) => {
                    apply(s, a.as_mut(), right, transpose, conjugate)
                }
                (Data::C32(s), MatrixData::C32(a)) => {
                    apply(s, a.as_mut(), right, transpose, conjugate)
                }
                (Data::C64(s), MatrixData::C64(a)) => {
                    apply(s, a.as_mut(), right, transpose, conjugate)
                }
                _ => Err("sequence scalar types must match".into()),
            }
        })
    }
    #[julia]
    pub fn sequence_component(
        &mut self,
        source: &OwnedHouseholderSequence,
        factor: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(match source.data()? {
                Data::F32(s) => MatrixData::F32(if factor {
                    s.factor.clone()
                } else {
                    s.basis.clone()
                }),
                Data::F64(s) => MatrixData::F64(if factor {
                    s.factor.clone()
                } else {
                    s.basis.clone()
                }),
                Data::C32(s) => MatrixData::C32(if factor {
                    s.factor.clone()
                } else {
                    s.basis.clone()
                }),
                Data::C64(s) => MatrixData::C64(if factor {
                    s.factor.clone()
                } else {
                    s.basis.clone()
                }),
            });
            Ok(())
        })
    }
}
#[julia]
impl VectorOwner {
    #[julia]
    pub fn sequence_apply(
        &mut self,
        source: &OwnedHouseholderSequence,
        right: bool,
        transpose: bool,
        conjugate: bool,
    ) -> Result<(), String> {
        guarded(|| {
            match (
                source.data()?,
                self.data.as_mut().ok_or("vector is not initialized")?,
            ) {
                (Data::F32(s), VectorData::F32(a)) => {
                    apply(s, a.as_mat_mut(), right, transpose, conjugate)
                }
                (Data::F64(s), VectorData::F64(a)) => {
                    apply(s, a.as_mat_mut(), right, transpose, conjugate)
                }
                (Data::C32(s), VectorData::C32(a)) => {
                    apply(s, a.as_mat_mut(), right, transpose, conjugate)
                }
                (Data::C64(s), VectorData::C64(a)) => {
                    apply(s, a.as_mat_mut(), right, transpose, conjugate)
                }
                _ => Err("sequence scalar types must match".into()),
            }
        })
    }
}

// Reuse scalar-tail correction when an external reduction supplies compact Q.
pub(crate) fn apply_compact<T: ComplexField>(
    basis: Mat<T>,
    factor: Mat<T>,
    out: MatMut<'_, T>,
) -> Result<(), String> {
    apply(&Sequence { basis, factor }, out, false, false, false)
}

pub(crate) fn apply_compact_conj<T: ComplexField>(
    basis: Mat<T>,
    factor: Mat<T>,
    out: MatMut<'_, T>,
    conjugate: bool,
) -> Result<(), String> {
    apply(&Sequence { basis, factor }, out, false, false, conjugate)
}

mod workspace;

mod construction;

pub(crate) fn apply_compact_in<T: ComplexField>(
    basis: MatRef<'_, T>,
    factor: MatRef<'_, T>,
    out: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) -> Result<(), String> {
    apply_views_in(basis, factor, out, false, false, false, par, stack)
}
