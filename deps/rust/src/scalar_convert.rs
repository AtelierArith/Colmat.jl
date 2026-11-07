//! Native scalar conversion of a borrowed matrix or view into a fresh owner,
//! so mixed-precision promotion needs no Julia staging array. Real sources
//! widen into real or complex targets; complex sources convert between complex
//! precisions only (a complex-to-real conversion is rejected, as in Julia).
use crate::typed::{MatrixData, TypedMatrix};
use crate::views::{converted, Spec};
use crate::{guarded, OwnedMatrix};
use rustcall_julia_macros::julia;

unsafe fn convert(p: *const u8, desc: *const isize, tag: i32) -> Result<MatrixData, String> {
    converted(p, Spec::read(desc)?, tag, true)
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
    use faer::{c64, Mat};
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
