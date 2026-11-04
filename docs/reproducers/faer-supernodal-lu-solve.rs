use faer::{Mat, Par, ParShape};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::linalg::{lu::*, SupernodalThreshold};
fn main() {
 for n in [0,1,3,12,40] {
  let a = Mat::<f64>::from_fn(n,n,|i,j| if i==j {0.1} else if (i+j)%3==0 || i.abs_diff(j)==1 {((i*17+j*13)%19) as f64 - 9.0} else {0.0});
  let entries: Vec<_> = (0..n).flat_map(|j| (0..n).map(move |i| (i,j))).filter(|&(i,j)| a[(i,j)]!=0.0).map(|(i,j)| Triplet {row:i,col:j,val:a[(i,j)]}).collect();
  let sp = SparseColMat::<usize,f64>::try_new_from_triplets(n,n,&entries).unwrap();
  for threshold in [SupernodalThreshold::AUTO,SupernodalThreshold::FORCE_SUPERNODAL,SupernodalThreshold::FORCE_SIMPLICIAL] {
   let sym = factorize_symbolic_lu(sp.symbolic(),LuSymbolicParams {supernodal_flop_ratio_threshold:threshold,..Default::default()}).unwrap();
   let mut num = NumericLu::new();
   let mut mem = MemBuffer::new(sym.factorize_numeric_lu_scratch::<f64>(ParShape::Seq,Default::default()));
   let f = sym.factorize_numeric_lu(&mut num,sp.as_ref(),Par::Seq,MemStack::new(&mut mem),Default::default()).unwrap();
   println!("n={n} threshold={threshold:?}");
   let mut x = Mat::<f64>::from_fn(n,1,|i,_| (i+1) as f64);
   let b = x.clone();
   let mut work = MemBuffer::new(sym.solve_in_place_scratch::<f64>(1,ParShape::Seq));
   f.solve_in_place_with_conj(faer::Conj::No,x.as_mut(),Par::Seq,MemStack::new(&mut work));
   let residual = (&a * &x - &b).norm_max();
   println!("solve residual={residual}");
   
  }
  println!("n={n} solve comparison complete");
 }
}
