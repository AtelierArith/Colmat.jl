//! Safe-range LASCL with native traversal of only the selected storage.
use crate::kernels::{flag, Scalar};
use colmat::MatMut;
use paste::paste;

unsafe fn multiply<T: Scalar>(
    kind: u8,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    a: *mut T,
    ld: usize,
    factor: f64,
) {
    for j in 0..n {
        let (start, end) = match kind {
            b'L' => (j.min(m), m),
            b'U' => (0, m.min(j + 1)),
            b'H' => (0, m.min(j + 2)),
            b'B' => (0, (kl + 1).min(n - j)),
            b'Q' => (ku.saturating_sub(j), ku + 1),
            b'Z' => (
                (kl + ku).saturating_sub(j).max(kl),
                (2 * kl + ku + 1).min((kl + ku + m).saturating_sub(j)),
            ),
            _ => (0, m),
        };
        if start >= end {
            continue;
        }
        // Match the reference Fortran complex expression A(I,J)*MUL,
        // including promotion of MUL to complex and its zero cross terms.
        // Componentwise real scaling differs for infinities and signed zero.
        MatMut::from_raw_parts_mut(a.add(start + j * ld), end - start, 1, 1, 0)
            .for_each_mut(|x| *x = *x * T::real(factor));
    }
}
macro_rules! check {($info:expr,$($arg:expr=>$valid:expr),+)=>{$(if !$valid{*$info=-$arg;crate::invalid($arg);return;})+};}
macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p lascl $suffix>](kind:*const u8,kl:*const $i,ku:*const $i,
            from:*const $r,to:*const $r,m:*const $i,n:*const $i,a:*mut $t,ld:*const $i,info:*mut $i) {
            *info=0;
            let(kind,kl,ku,m,n,ld)=(flag(kind),*kl as isize,*ku as isize,*m as isize,*n as isize,*ld as isize);
            let(mut from,mut to)=(*from,*to);
            let band=matches!(kind,b'B'|b'Q'|b'Z');
            let symmetric=matches!(kind,b'B'|b'Q');
            check!(info,1=>matches!(kind,b'G'|b'L'|b'U'|b'H'|b'B'|b'Q'|b'Z'),
                4=>from!=0.0&&!from.is_nan(),5=>!to.is_nan(),6=>m>=0,7=>n>=0&&(!symmetric||m==n));
            if band {
                check!(info,2=>kl>=0&&kl<=(m-1).max(0),3=>ku>=0&&ku<=(n-1).max(0)&&(!symmetric||kl==ku));
                let required=match kind {b'B'=>Some(kl+1),b'Q'=>Some(ku+1),
                    _=>kl.checked_mul(2).and_then(|v|v.checked_add(ku)).and_then(|v|v.checked_add(1))};
                check!(info,9=>required.is_some_and(|required|ld>=required));
            }else{check!(info,9=>ld>=m.max(1));}
            if m==0||n==0 {return;}
            let small=<$r>::MIN_POSITIVE;
            let big=1.0/small;
            loop {
                let from_small=from*small;
                let factor:$r;
                let done;
                if from_small==from {
                    factor=to/from;done=true;
                }else{
                    let to_small=to/big;
                    if to_small==to {
                        factor=to;done=true;
                    }else if from_small.abs()>to.abs()&&to!=0.0 {
                        factor=small;from=from_small;done=false;
                    }else if to_small.abs()>from.abs() {
                        factor=big;to=to_small;done=false;
                    }else{
                        factor=to/from;done=true;
                        if factor==1.0 {return;}
                    }
                }
                multiply(kind,m as usize,n as usize,kl.max(0) as usize,ku.max(0) as usize,a,ld as usize,factor as f64);
                if done {return;}
            }
        }
    }};
}
exports!(s, f32, f32, i32, _);
exports!(d, f64, f64, i32, _);
exports!(c, colmat::c32, f32, i32, _);
exports!(z, colmat::c64, f64, i32, _);
exports!(s, f32, f32, i64, _64_);
exports!(d, f64, f64, i64, _64_);
exports!(c, colmat::c32, f32, i64, _64_);
exports!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_and_identity_scaling_need_no_matrix_storage() {
        unsafe {
            let mut info = -777;
            dlascl_(
                b"G".as_ptr(),
                &-99,
                &-99,
                &3.0,
                &3.0,
                &3,
                &4,
                std::ptr::null_mut(),
                &3,
                &mut info,
            );
            assert_eq!(info, 0);
            let mut info64 = -777_i64;
            clascl_64_(
                b"Z".as_ptr(),
                &0,
                &0,
                &1.0,
                &2.0,
                &0,
                &4,
                std::ptr::null_mut(),
                &1,
                &mut info64,
            );
            assert_eq!(info64, 0);
        }
    }
}
