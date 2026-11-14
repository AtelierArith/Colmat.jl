//! General/band row and column equilibration through native column maps.
use crate::householder::ReflectorScalar;
use crate::kernels::Scalar;
use colmat::{Mat, MatMut, MatRef};
use paste::paste;
macro_rules! check {($info:expr,$($arg:expr=>$valid:expr),+)=>{$(if !$valid{*$info=-$arg;crate::invalid($arg);return;})+};}
macro_rules! exports {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  unsafe fn [<$p abs_column $suffix>](a:*const $t,ld:usize,m:usize,j:usize,band:Option<(usize,usize)>)->(usize,Mat<$r>) {
   let(start,end)=match band {
    Some((kl,ku))=>(j.saturating_sub(ku).min(m),(j+kl+1).min(m)),
    None=>(0,m),
   };
   if start>=end {return(start,Mat::zeros(0,1));}
   let row=match band {Some((_,ku))=>ku+start-j,None=>0};
   let column=MatRef::from_raw_parts(a.add(row+j*ld),end-start,1,1,0);
   (start,column.map(|x| (x.re() as $r).abs()+(x.im() as $r).abs()))
  }
  unsafe fn [<$p invert_equil $suffix>](values:*mut $r,count:usize,radix:bool,condition:*mut $r)->usize {
   let(mut minimum,mut maximum)=(1.0/<$r>::MIN_POSITIVE,0.0 as $r);
   let mut view=MatMut::from_raw_parts_mut(values,count,1,1,0);
   view.for_each_mut(|x| {
    if radix&&*x>0.0 {*x=(x.ln()/(2.0 as $r).ln()).trunc().exp2();}
    minimum=minimum.min(*x);maximum=maximum.max(*x);
   });
   if minimum==0.0 {
    for i in 0..count {if *values.add(i)==0.0{return i+1;}}
   }
   let small=<$r>::MIN_POSITIVE;
   let big=1.0/small;
   view.for_each_mut(|x| *x=1.0/x.max(small).min(big));
   *condition=minimum.max(small)/maximum.min(big);
   0
  }
  unsafe fn [<$p general_equil $suffix>](m:usize,n:usize,a:*const $t,ld:usize,r:*mut $r,c:*mut $r,
      rowcond:*mut $r,colcond:*mut $r,amax:*mut $r,band:Option<(usize,usize)>,radix:bool)->$i {
   if m==0||n==0 {*rowcond=1.0;*colcond=1.0;*amax=0.0;return 0;}
   MatMut::from_raw_parts_mut(r,m,1,1,0).fill(0.0);
   for j in 0..n {
    let(start,column)=[<$p abs_column $suffix>](a,ld,m,j,band);
    if column.nrows()==0 {continue;}
    let dst=MatMut::from_raw_parts_mut(r.add(start),column.nrows(),1,1,0);
    colmat::zip!(dst,column.as_ref()).for_each(|colmat::unzip!(out,value)| *out=out.max(*value));
   }
   // AMAX for the radix variants is the maximum quantized row maximum,
   // as in GEEQUB/GBEQUB, rather than the original matrix maximum.
   let mut maximum=0.0 as $r;
   MatMut::from_raw_parts_mut(r,m,1,1,0).for_each_mut(|x| {
    if radix&&*x>0.0 {*x=(x.ln()/(2.0 as $r).ln()).trunc().exp2();}
    maximum=maximum.max(*x);
   });
   *amax=maximum;
   let bad=[<$p invert_equil $suffix>](r,m,false,rowcond);
   if bad!=0 {return bad as $i;}
   MatMut::from_raw_parts_mut(c,n,1,1,0).fill(0.0);
   for j in 0..n {
    let(start,mut column)=[<$p abs_column $suffix>](a,ld,m,j,band);
    if column.nrows()==0 {continue;}
    let weights=MatRef::from_raw_parts(r.add(start),column.nrows(),1,1,0);
    let mut maximum=0.0 as $r;
    colmat::zip!(column.as_mut(),weights).for_each(|colmat::unzip!(value,weight)| {maximum=maximum.max(*value* *weight);});
    *c.add(j)=maximum;
   }
   let bad=[<$p invert_equil $suffix>](c,n,radix,colcond);
   if bad!=0 {(m+bad) as $i}else{0}
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p geequ $suffix>](m:*const $i,n:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i) {
   [<$p geequ_impl $suffix>](m,n,a,ld,r,c,rc,cc,amax,info,false);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p geequb $suffix>](m:*const $i,n:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i) {
   [<$p geequ_impl $suffix>](m,n,a,ld,r,c,rc,cc,amax,info,true);
  }
  unsafe fn [<$p geequ_impl $suffix>](m:*const $i,n:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i,radix:bool) {
   *info=0;let(m,n,ld)=(*m as isize,*n as isize,*ld as isize);
   check!(info,1=>m>=0,2=>n>=0,4=>ld>=m.max(1));
   *info=[<$p general_equil $suffix>](m as usize,n as usize,a,ld as usize,r,c,rc,cc,amax,None,radix);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p gbequ $suffix>](m:*const $i,n:*const $i,kl:*const $i,ku:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i) {
   [<$p gbequ_impl $suffix>](m,n,kl,ku,a,ld,r,c,rc,cc,amax,info,false);
  }
  #[no_mangle]
  pub unsafe extern "C" fn [<$p gbequb $suffix>](m:*const $i,n:*const $i,kl:*const $i,ku:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i) {
   [<$p gbequ_impl $suffix>](m,n,kl,ku,a,ld,r,c,rc,cc,amax,info,true);
  }
  unsafe fn [<$p gbequ_impl $suffix>](m:*const $i,n:*const $i,kl:*const $i,ku:*const $i,a:*const $t,ld:*const $i,r:*mut $r,c:*mut $r,rc:*mut $r,cc:*mut $r,amax:*mut $r,info:*mut $i,radix:bool) {
   *info=0;let(m,n,kl,ku,ld)=(*m as isize,*n as isize,*kl as isize,*ku as isize,*ld as isize);
   check!(info,1=>m>=0,2=>n>=0,3=>kl>=0,4=>ku>=0,6=>ld>=kl+ku+1);
   *info=[<$p general_equil $suffix>](m as usize,n as usize,a,ld as usize,r,c,rc,cc,amax,Some((kl as usize,ku as usize)),radix);
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
    fn zero_row_and_empty_band_do_not_access_undefined_outputs() {
        unsafe {
            let a = [0.0; 4];
            let mut rows = [777.0; 2];
            let (mut max, mut info) = (777.0, -777);
            dgeequ_(
                &2,
                &2,
                a.as_ptr(),
                &2,
                rows.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut max,
                &mut info,
            );
            assert_eq!((info, max), (1, 0.0));
            assert_eq!(rows, [0.0; 2]);
            let (mut rc, mut cc, mut max, mut info) = (777.0, 777.0, 777.0, -777_i64);
            zgbequb_64_(
                &7,
                &0,
                &2,
                &3,
                std::ptr::null(),
                &6,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut rc,
                &mut cc,
                &mut max,
                &mut info,
            );
            assert_eq!((info, rc, cc, max), (0, 1.0, 1.0, 0.0));
        }
    }
}
