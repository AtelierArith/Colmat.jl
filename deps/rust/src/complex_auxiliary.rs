//! Complex auxiliary transforms and mixed real/complex products.
use crate::kernels::{flag, Scalar};
use colmat::{Accum, Conj, MatMut, MatRef};
use paste::paste;
macro_rules! check {($($arg:expr=>$valid:expr),+)=>{$(if !$valid{crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p lacgv $suffix>](n:*const $i,x:*mut $t,inc:*const $i) {
   let(n,inc)=(*n as isize,*inc as isize);
   if n<=0 || (inc==0&&n%2==0) {return;}
   let Some(offset)=(n-1).checked_mul(inc) else {crate::invalid(3);return;};
   let begin=if inc<0 {let Some(v)=offset.checked_neg() else{crate::invalid(3);return;};v}else{0};
   MatMut::from_raw_parts_mut(x.offset(begin),if inc==0 {1}else{n as usize},1,inc,0)
    .for_each_mut(|v| *v=v.cj());
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p lacp2 $suffix>](u:*const u8,m:*const $i,n:*const $i,a:*const $r,la:*const $i,b:*mut $t,lb:*const $i) {
   let(u,m,n,la,lb)=(flag(u),*m as isize,*n as isize,*la as isize,*lb as isize);
   check!(2=>m>=0,3=>n>=0,5=>la>=m.max(1),7=>lb>=m.max(1));
   if m==0||n==0 {return;}
   for j in 0..n {
    let(start,end)=match u {b'U'=>(0,m.min(j+1)),b'L'=>(j.min(m),m),_=>(0,m)};
    if start==end {continue;}
    let src=MatRef::from_raw_parts(a.offset(start+j*la),(end-start) as usize,1,1,0);
    let dst=MatMut::from_raw_parts_mut(b.offset(start+j*lb),(end-start) as usize,1,1,0);
    colmat::zip!(dst,src).for_each(|colmat::unzip!(out,input)| *out=<$t>::new(*input,0.0));
   }
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p lacrm $suffix>](m:*const $i,n:*const $i,a:*const $t,la:*const $i,b:*const $r,lb:*const $i,c:*mut $t,lc:*const $i,_work:*mut $r) {
   let(m,n,la,lb,lc)=(*m as isize,*n as isize,*la as isize,*lb as isize,*lc as isize);
   check!(1=>m>=0,2=>n>=0,4=>la>=m.max(1),6=>lb>=n.max(1),8=>lc>=m.max(1));
   if m==0||n==0 {return;}
   for component in 0..2 {
    let lhs=MatRef::from_raw_parts((a as *const $r).add(component),m as usize,n as usize,2,2*la);
    let rhs=MatRef::from_raw_parts(b,n as usize,n as usize,1,lb);
    let dst=MatMut::from_raw_parts_mut((c as *mut $r).add(component),m as usize,n as usize,2,2*lc);
    crate::threading::with_par(|par| colmat::linalg::matmul::matmul_with_conj(dst,Accum::Replace,lhs,Conj::No,rhs,Conj::No,1.0,par));
   }
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p larcm $suffix>](m:*const $i,n:*const $i,a:*const $r,la:*const $i,b:*const $t,lb:*const $i,c:*mut $t,lc:*const $i,_work:*mut $r) {
   let(m,n,la,lb,lc)=(*m as isize,*n as isize,*la as isize,*lb as isize,*lc as isize);
   check!(1=>m>=0,2=>n>=0,4=>la>=m.max(1),6=>lb>=m.max(1),8=>lc>=m.max(1));
   if m==0||n==0 {return;}
   for component in 0..2 {
    let lhs=MatRef::from_raw_parts(a,m as usize,m as usize,1,la);
    let rhs=MatRef::from_raw_parts((b as *const $r).add(component),m as usize,n as usize,2,2*lb);
    let dst=MatMut::from_raw_parts_mut((c as *mut $r).add(component),m as usize,n as usize,2,2*lc);
    crate::threading::with_par(|par| colmat::linalg::matmul::matmul_with_conj(dst,Accum::Replace,lhs,Conj::No,rhs,Conj::No,1.0,par));
   }
  }
 }};
}
exports!(c, colmat::c32, f32, i32, _);
exports!(z, colmat::c64, f64, i32, _);
exports!(c, colmat::c32, f32, i64, _64_);
exports!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_transforms_and_products_need_no_storage() {
        unsafe {
            clacgv_(&0, std::ptr::null_mut(), &-2);
            zlacp2_64_(
                b"U".as_ptr(),
                &0,
                &4,
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
                &1,
            );
            clacrm_(
                &0,
                &4,
                std::ptr::null(),
                &1,
                std::ptr::null(),
                &4,
                std::ptr::null_mut(),
                &1,
                std::ptr::null_mut(),
            );
            zlarcm_64_(
                &4,
                &0,
                std::ptr::null(),
                &4,
                std::ptr::null(),
                &4,
                std::ptr::null_mut(),
                &4,
                std::ptr::null_mut(),
            );
        }
    }
}
