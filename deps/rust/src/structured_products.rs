//! Structured multiplication with exact primitive snapshots.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::linalg::matmul::triangular::{self, BlockStructure as B};
use faer::traits::{math_utils::*, ComplexField};
use faer::{Accum, Mat, Par};
use rustcall_julia_macros::julia;
fn structure(tag: i32) -> Result<B, String> {
    Ok(match tag {
        0 => B::Rectangular,
        1 => B::TriangularLower,
        2 => B::TriangularUpper,
        3 => B::UnitTriangularLower,
        4 => B::UnitTriangularUpper,
        _ => return Err("invalid structure".into()),
    })
}
unsafe fn product<T: ComplexField>(
    p: *const u8,
    q: *const u8,
    m: usize,
    k: usize,
    n: usize,
    sa: i32,
    sb: i32,
) -> Result<Mat<T>, String> {
    if (sa != 0 && m != k) || (sb != 0 && k != n) {
        return Err("structured operand must be square".into());
    }
    m.checked_mul(k)
        .and_then(|_| k.checked_mul(n))
        .ok_or("shape overflow")?;
    let mut a = Mat::<T>::from_fn(m, k, |i, j| copy(&*p.cast::<T>().add(i + j * m)));
    let mut b = Mat::<T>::from_fn(k, n, |i, j| copy(&*q.cast::<T>().add(i + j * k)));
    if sa == 5 || sb == 5 {
        for (x, tag) in [(&mut a, sa), (&mut b, sb)] {
            if (1..=4).contains(&tag) {
                for j in 0..x.ncols() {
                    for i in 0..x.nrows() {
                        if (tag == 1 || tag == 3) && i < j || (tag == 2 || tag == 4) && i > j {
                            x[(i, j)] = zero::<T>();
                        }
                        if (tag == 3 || tag == 4) && i == j {
                            x[(i, j)] = one::<T>();
                        }
                    }
                }
            }
        }
    }
    if sa == 5 {
        return Ok(a.as_ref().diagonal() * b.as_ref());
    }
    if sb == 5 {
        return Ok(a.as_ref() * b.as_ref().diagonal());
    }
    let mut c = Mat::<T>::zeros(m, n);
    triangular::matmul_with_conj(
        c.as_mut(),
        B::Rectangular,
        Accum::Replace,
        a.as_ref(),
        structure(sa)?,
        faer::Conj::No,
        b.as_ref(),
        structure(sb)?,
        faer::Conj::No,
        one::<T>(),
        Par::Seq,
    );
    Ok(c)
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn structured_product(
        &mut self,
        p: *const u8,
        q: *const u8,
        m: usize,
        k: usize,
        n: usize,
        sa: i32,
        sb: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = unsafe { product::<f64>(p, q, m, k, n, sa, sb)? };
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn structured_product(
        &mut self,
        p: *const u8,
        q: *const u8,
        m: usize,
        k: usize,
        n: usize,
        sa: i32,
        sb: i32,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe {
                match tag {
                    0 => MatrixData::F32(product::<f32>(p, q, m, k, n, sa, sb)?),
                    2 => MatrixData::C32(product::<faer::c32>(p, q, m, k, n, sa, sb)?),
                    3 => MatrixData::C64(product::<faer::c64>(p, q, m, k, n, sa, sb)?),
                    _ => return Err("invalid structured scalar".into()),
                }
            });
            Ok(())
        })
    }
}
