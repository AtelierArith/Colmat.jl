//! Precision conversion with LAPACK's component range test before narrowing.
use crate::kernels::flag;
use colmat::{MatMut, MatRef};
use paste::paste;

trait Convert<U>: Copy {
    fn outside(self) -> bool;
    fn convert(self) -> U;
}
impl Convert<f64> for f32 {
    fn outside(self) -> bool {
        false
    }
    fn convert(self) -> f64 {
        self as f64
    }
}
impl Convert<f32> for f64 {
    fn outside(self) -> bool {
        self < -(f32::MAX as f64) || self > f32::MAX as f64
    }
    fn convert(self) -> f32 {
        self as f32
    }
}
impl Convert<colmat::c64> for colmat::c32 {
    fn outside(self) -> bool {
        false
    }
    fn convert(self) -> colmat::c64 {
        colmat::c64::new(self.re as f64, self.im as f64)
    }
}
impl Convert<colmat::c32> for colmat::c64 {
    fn outside(self) -> bool {
        self.re.outside() || self.im.outside()
    }
    fn convert(self) -> colmat::c32 {
        colmat::c32::new(self.re as f32, self.im as f32)
    }
}
unsafe fn convert<T: Convert<U>, U>(
    triangle: Option<bool>,
    m: usize,
    n: usize,
    a: *const T,
    la: usize,
    b: *mut U,
    lb: usize,
) -> bool {
    if m == 0 || n == 0 {
        return true;
    }
    for j in 0..n {
        let (start, end) = match triangle {
            Some(true) => (0, m.min(j + 1)),
            Some(false) => (j.min(m), m),
            None => (0, m),
        };
        let mut stop = end;
        for i in start..end {
            if (*a.add(i + j * la)).outside() {
                stop = i;
                break;
            }
        }
        if start < stop {
            let src = MatRef::from_raw_parts(a.add(start + j * la), stop - start, 1, 1, 0);
            let dst = MatMut::from_raw_parts_mut(b.add(start + j * lb), stop - start, 1, 1, 0);
            colmat::zip!(dst, src).for_each(|colmat::unzip!(out, input)| *out = (*input).convert());
        }
        if stop != end {
            return false;
        }
    }
    true
}
macro_rules! general {
    ($name:ident,$src:ty,$dst:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$name $suffix>](m:*const $i,n:*const $i,a:*const $src,la:*const $i,b:*mut $dst,lb:*const $i,info:*mut $i) {
            *info=0;
            let(m,n,la,lb)=(*m as isize,*n as isize,*la as isize,*lb as isize);
            for (arg,valid) in [(1,m>=0),(2,n>=0),(4,la>=m.max(1)),(6,lb>=m.max(1))] {
                if !valid {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if !convert(None,m as usize,n as usize,a,la as usize,b,lb as usize) {*info=1;}
        }
    }};
}
macro_rules! triangular {
    ($name:ident,$src:ty,$dst:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$name $suffix>](u:*const u8,n:*const $i,a:*const $src,la:*const $i,b:*mut $dst,lb:*const $i,info:*mut $i) {
            *info=0;
            let(n,la,lb)=(*n as isize,*la as isize,*lb as isize);
            for(arg,valid) in [(2,n>=0),(4,la>=n.max(1)),(6,lb>=n.max(1))] {
                if !valid {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if !convert(Some(flag(u)==b'U'),n as usize,n as usize,a,la as usize,b,lb as usize) {*info=1;}
        }
    }};
}
general!(slag2d, f32, f64, i32, _);
general!(dlag2s, f64, f32, i32, _);
general!(clag2z, colmat::c32, colmat::c64, i32, _);
general!(zlag2c, colmat::c64, colmat::c32, i32, _);
triangular!(dlat2s, f64, f32, i32, _);
triangular!(zlat2c, colmat::c64, colmat::c32, i32, _);
general!(slag2d, f32, f64, i64, _64_);
general!(dlag2s, f64, f32, i64, _64_);
general!(clag2z, colmat::c32, colmat::c64, i64, _64_);
general!(zlag2c, colmat::c64, colmat::c32, i64, _64_);
triangular!(dlat2s, f64, f32, i64, _64_);
triangular!(zlat2c, colmat::c64, colmat::c32, i64, _64_);
