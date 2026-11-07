//! Native scalar conversion of a borrowed matrix or view into a fresh owner,
//! so mixed-precision promotion needs no Julia staging array. Real sources
//! widen into real or complex targets; complex sources convert between complex
//! precisions only (a complex-to-real conversion is rejected, as in Julia).
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{source, Borrowed, Spec};
use crate::{guarded, OwnedMatrix};
use faer::{c32, c64, Mat, MatRef};
use rustcall_julia_macros::julia;

fn map<S: Copy, T>(a: MatRef<'_, S>, conjugate: bool, f: impl Fn(S, bool) -> T) -> Mat<T> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| f(a[(i, j)], conjugate))
}
unsafe fn convert(p: *const u8, desc: *const isize, tag: i32) -> Result<MatrixData, String> {
    let s = Spec::read(desc)?;
    let c = s.conjugate;
    let cj32 = |x: c32, k: bool| if k { x.conj() } else { x };
    let cj64 = |x: c64, k: bool| if k { x.conj() } else { x };
    Ok(match (source(p, s)?, tag) {
        (Borrowed::F32(a), 0) => MatrixData::F32(map(a, c, |x, _| x)),
        (Borrowed::F32(a), 1) => MatrixData::F64(map(a, c, |x, _| x as f64)),
        (Borrowed::F32(a), 2) => MatrixData::C32(map(a, c, |x, _| c32::new(x, 0.0))),
        (Borrowed::F32(a), 3) => MatrixData::C64(map(a, c, |x, _| c64::new(x as f64, 0.0))),
        (Borrowed::F64(a), 0) => MatrixData::F32(map(a, c, |x, _| x as f32)),
        (Borrowed::F64(a), 1) => MatrixData::F64(map(a, c, |x, _| x)),
        (Borrowed::F64(a), 2) => MatrixData::C32(map(a, c, |x, _| c32::new(x as f32, 0.0))),
        (Borrowed::F64(a), 3) => MatrixData::C64(map(a, c, |x, _| c64::new(x, 0.0))),
        (Borrowed::C32(a), 2) => MatrixData::C32(map(a, c, cj32)),
        (Borrowed::C32(a), 3) => MatrixData::C64(map(a, c, |x, k| {
            let x = cj32(x, k);
            c64::new(x.re as f64, x.im as f64)
        })),
        (Borrowed::C64(a), 2) => MatrixData::C32(map(a, c, |x, k| {
            let x = cj64(x, k);
            c32::new(x.re as f32, x.im as f32)
        })),
        (Borrowed::C64(a), 3) => MatrixData::C64(map(a, c, cj64)),
        (Borrowed::C32(_) | Borrowed::C64(_), 0 | 1) => {
            return Err("complex matrices cannot convert to a real scalar type".into())
        }
        _ => return Err("unsupported scalar tag".into()),
    })
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn view_convert(&mut self, p: *const u8, desc: *const isize) -> Result<(), String> {
        guarded(|| match unsafe { convert(p, desc, 1)? } {
            MatrixData::F64(x) => {
                self.matrix = x;
                Ok(())
            }
            _ => Err("Float64 conversion produced another type".into()),
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn view_convert(
        &mut self,
        p: *const u8,
        desc: *const isize,
        tag: i32,
    ) -> Result<(), String> {
        guarded(|| {
            self.data = Some(unsafe { convert(p, desc, tag)? });
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn widen_and_reject_complex_to_real() {
        let mut a = TypedMatrix {
            data: Some(MatrixData::F32(Mat::from_fn(2, 2, |i, j| {
                (i + 2 * j) as f32 + 0.5
            }))),
        };
        let d = [1isize, 0, 1, 2, 0, 1, 2, 1, 0]; // transposed whole view
        let p = (&mut a as *mut TypedMatrix).cast::<u8>().cast_const();
        let mut out = OwnedMatrix::new();
        out.view_convert(p, d.as_ptr()).unwrap();
        assert_eq!(out.matrix[(0, 1)], 1.5);
        let mut z = TypedMatrix { data: None };
        z.view_convert(p, d.as_ptr(), 3).unwrap();
        match &z.data {
            Some(MatrixData::C64(x)) => assert_eq!(x[(1, 0)], c64::new(2.5, 0.0)),
            _ => panic!(),
        }
        let zp = (&z as *const TypedMatrix).cast::<u8>();
        let whole = [1isize, 0, 1, 2, 0, 1, 2, 0, 1]; // conjugated
        let mut r = TypedMatrix { data: None };
        assert!(r.view_convert(zp, whole.as_ptr(), 1).is_err());
        r.view_convert(zp, whole.as_ptr(), 2).unwrap();
        assert!(matches!(r.data, Some(MatrixData::C32(_))));
    }
}
