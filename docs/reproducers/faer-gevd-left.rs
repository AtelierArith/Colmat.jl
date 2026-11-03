use faer::{Mat, Par};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::gevd::{gevd_real, gevd_scratch, ComputeEigenvectors as V};
fn main() {
    let n=64;
    let x=Mat::<f64>::from_fn(n,n,|i,j| (1+i+j*n) as f64/(n*n) as f64);
    let b=&x*x.transpose()+Mat::<f64>::identity(n,n);
    let c=Mat::<f64>::from_fn(n,n,|i,j| if i==j {2.0} else if i.abs_diff(j)==1 {-1.0} else {0.0});
    let a=&b*&c;
    for mode in [0,2,1] {
        let left=mode>0;
        let mut extra=Mat::<f64>::zeros(n,n);
        let mut ac=a.clone();let mut bc=b.clone();
        let mut re=Mat::<f64>::zeros(n,1);let mut im=Mat::<f64>::zeros(n,1);let mut beta=Mat::<f64>::zeros(n,1);
        let mut u=Mat::<f64>::zeros(n,n);
        let mut mem=MemBuffer::new(gevd_scratch::<f64>(n,if left {V::Yes}else{V::No},if mode==1{V::No}else{V::Yes},Par::Seq.shape(),Default::default()));
        let (ul,ur)=if mode==2 {(Some(u.as_mut()),Some(extra.as_mut()))} else if left {(Some(u.as_mut()),None)}else{(None,Some(u.as_mut()))};
        gevd_real(ac.as_mut(),bc.as_mut(),re.as_mut().col_mut(0).as_diagonal_mut(),im.as_mut().col_mut(0).as_diagonal_mut(),beta.as_mut().col_mut(0).as_diagonal_mut(),ul,ur,Par::Seq,MemStack::new(&mut mem),Default::default()).unwrap();
        let finite=(0..n).filter(|&i| (re[(i,0)]/beta[(i,0)]).is_finite()).count();
        println!("mode={mode}: finite eigenvalues={finite}/{n}; finite vectors={}",u.as_ref().is_all_finite());
        if mode==2 {
            let lhs=u.transpose()*&a;
            let rhs=u.transpose()*&b;
            let mut residual=0.0_f64;
            for j in 0..n { for i in 0..n {
                residual=residual.max((beta[(i,0)]*lhs[(i,j)]-re[(i,0)]*rhs[(i,j)]).abs());
            }}
            println!("both-output left homogeneous max residual={residual:e}");
            assert!(residual<1e-8);
        }
        assert_eq!(finite,n,"requesting left vectors must preserve the finite spectrum");
    }
}
