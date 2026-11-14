//! Diagonal positive-definite equilibration via native real diagonal views.
use crate::kernels::{flag, uplo, Scalar};
use colmat::{Mat, MatMut};
use paste::paste;
macro_rules! check {($info:expr,$($arg:expr=>$valid:expr),+)=>{$(if !$valid{*$info=-$arg;crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  unsafe fn [<$p diagonal_equil $suffix>](n:usize,a:*const $t,s:*mut $r,cond:*mut $r,amax:*mut $r,
      radix:bool,offset:impl Fn(usize)->usize)->$i {
   if n==0 {*cond=1.0;*amax=0.0;return 0;}
   let components=if <$t>::COMPLEX {2}else{1};
   // For Hermitian storage even the imaginary diagonal is not consulted.
   let diagonal=Mat::<$r>::from_fn(n,1,|i,_| *((a as *const $r).add(components*offset(i))));
   let mut output=MatMut::from_raw_parts_mut(s,n,1,1,0);
   output.copy_from(diagonal.as_ref());
   let(mut minimum,mut maximum)=(*s,*s);
   output.for_each_mut(|x| {minimum=minimum.min(*x);maximum=maximum.max(*x);});
   *amax=maximum;
   if minimum<=0.0 {
    for i in 0..n {if *s.add(i)<=0.0 {return i as $i+1;}}
   }else{
    let coefficient=-0.5/(2.0 as $r).ln();
    output.for_each_mut(|x| *x=if radix {(2.0 as $r).powi((coefficient*x.ln()) as i32)}else{1.0/x.sqrt()});
    *cond=minimum.sqrt()/maximum.sqrt();
   }
   0
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p poequ $suffix>](n:*const $i,a:*const $t,ld:*const $i,s:*mut $r,cond:*mut $r,amax:*mut $r,info:*mut $i) {
   *info=0;let(n,ld)=(*n as isize,*ld as isize);
   check!(info,1=>n>=0,3=>ld>=n.max(1));
   *info=[<$p diagonal_equil $suffix>](n as usize,a,s,cond,amax,false,|i|i*(ld as usize+1));
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p poequb $suffix>](n:*const $i,a:*const $t,ld:*const $i,s:*mut $r,cond:*mut $r,amax:*mut $r,info:*mut $i) {
   *info=0;let(n,ld)=(*n as isize,*ld as isize);
   check!(info,1=>n>=0,3=>ld>=n.max(1));
   *info=[<$p diagonal_equil $suffix>](n as usize,a,s,cond,amax,true,|i|i*(ld as usize+1));
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p pbequ $suffix>](u:*const u8,n:*const $i,kd:*const $i,a:*const $t,ld:*const $i,s:*mut $r,cond:*mut $r,amax:*mut $r,info:*mut $i) {
   *info=0;let(u,n,kd,ld)=(flag(u),*n as isize,*kd as isize,*ld as isize);
   check!(info,1=>uplo(u),2=>n>=0,3=>kd>=0,5=>kd.checked_add(1).is_some_and(|required|ld>=required));
   let row=if u==b'U' {kd as usize}else{0};
   *info=[<$p diagonal_equil $suffix>](n as usize,a,s,cond,amax,false,|i|row+i*ld as usize);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p ppequ $suffix>](u:*const u8,n:*const $i,a:*const $t,s:*mut $r,cond:*mut $r,amax:*mut $r,info:*mut $i) {
   *info=0;let(u,n)=(flag(u),*n as isize);
   check!(info,1=>uplo(u),2=>n>=0);
   *info=[<$p diagonal_equil $suffix>](n as usize,a,s,cond,amax,false,|i|if u==b'U' {i*(i+3)/2}else{i*n as usize-i*i.saturating_sub(1)/2});
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
    fn packed_lower_diagonal_and_empty_storage() {
        unsafe {
            let a = [1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 9.0];
            let mut scales = [0.0; 3];
            let (mut cond, mut max, mut info) = (0.0, 0.0, -777);
            dppequ_(
                b"L".as_ptr(),
                &3,
                a.as_ptr(),
                scales.as_mut_ptr(),
                &mut cond,
                &mut max,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(scales, [1.0, 0.5, 1.0 / 3.0]);
            assert_eq!(cond, 1.0 / 3.0);
            assert_eq!(max, 9.0);
            let (mut cond, mut max, mut info) = (777.0, 777.0, -777_i64);
            zpoequb_64_(
                &0,
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
                &mut cond,
                &mut max,
                &mut info,
            );
            assert_eq!((info, cond, max), (0, 1.0, 0.0));
        }
    }
}
