//! BIN symmetric equilibration, composed with native GEMV and vector maps.
//! Algorithm: Livne and Golub, Scaling by Binormalization (2004), as used
//! by LAPACK SYEQUB/HEEQUB. Magnitudes use CABS1 for complex inputs.
use crate::householder::ReflectorScalar;
use crate::kernels::{flag, uplo, Scalar};
use colmat::{Accum, Conj, Mat, MatMut};
use paste::paste;

fn bin(matrix: &Mat<f64>) -> Option<(Mat<f64>, f64)> {
    let n = matrix.nrows();
    let mut scales = Mat::from_fn(n, 1, |i, _| {
        let maximum = (0..n).map(|j| matrix[(i, j)]).fold(0.0, f64::max);
        1.0 / maximum
    });
    if (0..n).any(|i| !scales[(i, 0)].is_finite()) {
        return None;
    }
    let mut beta = Mat::<f64>::zeros(n, 1);
    let tolerance = 1.0 / (2.0 * n as f64).sqrt();
    let mut average = 0.0;
    for _ in 0..100 {
        crate::threading::with_par(|par| {
            colmat::linalg::matmul::matmul_with_conj(
                beta.as_mut(),
                Accum::Replace,
                matrix.as_ref(),
                Conj::No,
                scales.as_ref(),
                Conj::No,
                1.0,
                par,
            );
        });
        average = (0..n).map(|i| scales[(i, 0)] * beta[(i, 0)]).sum::<f64>() / n as f64;
        if !average.is_finite() || average <= 0.0 {
            return None;
        }
        let deviations = Mat::from_fn(n, 1, |i, _| scales[(i, 0)] * beta[(i, 0)] - average);
        let maximum = deviations.norm_max();
        // Normalize before native norm_l2: the pinned native implementation
        // loses subnormals or overflows without this range reduction.
        let deviation = if maximum == 0.0 {
            0.0
        } else {
            deviations.as_ref().map(|x| *x / maximum).norm_l2() / (n as f64).sqrt() * maximum
        };
        if deviation < tolerance * average {
            break;
        }
        for i in 0..n {
            let diagonal = matrix[(i, i)];
            let old = scales[(i, 0)];
            let a = (n as f64 - 1.0) * diagonal;
            let b = (n as f64 - 2.0) * (beta[(i, 0)] - diagonal * old);
            let c = -(diagonal * old) * old + 2.0 * beta[(i, 0)] * old - n as f64 * average;
            // All matrix entries are nonnegative, so c is nonpositive.
            // Avoid squaring b or forming a*c before taking its square root.
            if c > 0.0 || !c.is_finite() {
                return None;
            }
            let discriminant = b.hypot(2.0 * (-c).sqrt() * a.sqrt());
            if discriminant <= 0.0 || !discriminant.is_finite() {
                return None;
            }
            let next = -2.0 * c / (b + discriminant);
            if next <= 0.0 || !next.is_finite() {
                return None;
            }
            let delta = next - old;
            let product = (0..n).map(|j| scales[(j, 0)] * matrix[(j, i)]).sum::<f64>();
            colmat::zip!(beta.as_mut().col_mut(0), matrix.col(i))
                .for_each(|colmat::unzip!(v, t)| *v += delta * *t);
            average += (product + beta[(i, 0)]) * delta / n as f64;
            scales[(i, 0)] = next;
        }
    }
    if average > 0.0 && average.is_finite() {
        Some((scales, average))
    } else {
        None
    }
}

// A second normalization is used when the global normalization cannot
// complete a finite iteration. Every nonzero row retains a nonzero maximum after
// symmetric row normalization, including the full f64 exponent range.
fn equilibrate(logs: &Mat<f64>) -> Result<Vec<f64>, usize> {
    let n = logs.nrows();
    let rows: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| logs[(i, j)])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    if let Some(i) = rows.iter().position(|x| *x == f64::NEG_INFINITY) {
        return Err(i + 1);
    }
    if rows.iter().any(|x| !x.is_finite()) {
        return Err(0);
    }
    let maximum = rows.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for symmetric in [false, true] {
        let shifts: Vec<f64> = rows
            .iter()
            .map(|x| if symmetric { -0.5 * *x } else { -0.5 * maximum })
            .collect();
        let matrix = Mat::from_fn(n, n, |i, j| (logs[(i, j)] + shifts[i] + shifts[j]).exp());
        if let Some((scales, average)) = bin(&matrix) {
            let normalization = -0.5 * average.ln();
            let result: Vec<f64> = (0..n)
                .map(|i| {
                    let exponent = ((scales[(i, 0)].ln() + normalization + shifts[i])
                        / std::f64::consts::LN_2)
                        .trunc();
                    exponent.exp2()
                })
                .collect();
            if result.iter().all(|x| x.is_finite() && *x > 0.0) {
                return Ok(result);
            }
        }
    }
    Err(0)
}
macro_rules! exports {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p syequb $suffix>](u:*const u8,n:*const $i,a:*const $t,ld:*const $i,s:*mut $r,cond:*mut $r,amax:*mut $r,_work:*mut $t,info:*mut $i) {
   *info=0;let(u,n,ld)=(flag(u),*n as isize,*ld as isize);
   for(arg,valid) in [(1,uplo(u)),(2,n>=0),(4,ld>=n.max(1))] {
    if !valid {*info=-arg;crate::invalid(arg as i32);return;}
   }
   *amax=0.0;
   if n==0 {*cond=1.0;return;}
   let n=n as usize;
   let mut maximum=0.0 as $r;
   let logs=Mat::from_fn(n,n,|i,j| {
    let(row,col)=if (u==b'U'&&i<=j)||(u==b'L'&&i>=j) {(i,j)}else{(j,i)};
    let value=*a.add(row+col*ld as usize);
    let(re,im)=((value.re() as $r).abs(),(value.im() as $r).abs());
    let magnitude=re+im;
    maximum=maximum.max(magnitude);
    if magnitude.is_finite() {(magnitude as f64).ln()}else{
     let hi=(re as f64).max(im as f64);let lo=(re as f64).min(im as f64);
     hi.ln()+(lo/hi).ln_1p()
    }
   });
   *amax=maximum;
   match equilibrate(&logs) {
    Ok(values)=>{
     let(mut minimum,mut maximum)=(1.0/<$r>::MIN_POSITIVE,0.0 as $r);
     let source=Mat::from_fn(n,1,|i,_| values[i] as $r);
     if (0..n).any(|i| !source[(i,0)].is_finite()||source[(i,0)]<=0.0) {*info=-1;return;}
     let mut dst=MatMut::from_raw_parts_mut(s,n,1,1,0);
     dst.copy_from(source.as_ref());
     dst.for_each_mut(|x| {minimum=minimum.min(*x);maximum=maximum.max(*x);});
     *cond=minimum.max(<$r>::MIN_POSITIVE)/maximum.min(1.0/<$r>::MIN_POSITIVE);
    }
    Err(row)=>{*info=if row==0 {-1}else{row as $i};}
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
macro_rules! hermitian {
 ($p:ident,$t:ty,$r:ty,$i:ty,$suffix:tt)=>{paste!{
  #[no_mangle]
  pub unsafe extern "C" fn [<$p heequb $suffix>](u:*const u8,n:*const $i,a:*const $t,ld:*const $i,s:*mut $r,cond:*mut $r,amax:*mut $r,work:*mut $t,info:*mut $i) {
   [<$p syequb $suffix>](u,n,a,ld,s,cond,amax,work,info);
  }
 }};
}
hermitian!(c, colmat::c32, f32, i32, _);
hermitian!(z, colmat::c64, f64, i32, _);
hermitian!(c, colmat::c32, f32, i64, _64_);
hermitian!(z, colmat::c64, f64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_exponent_range_uses_a_finite_symmetric_scaling() {
        let diagonal = [f64::from_bits(1), 1.0, f64::MAX];
        let logs = Mat::from_fn(3, 3, |i, j| {
            if i == j {
                diagonal[i].ln()
            } else {
                f64::NEG_INFINITY
            }
        });
        let result = equilibrate(&logs).unwrap();
        assert!(result.iter().all(|x| x.is_finite() && *x > 0.0));
        let energies: Vec<f64> = (0..3)
            .map(|i| diagonal[i].ln() + 2.0 * result[i].ln())
            .collect();
        let range = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - energies.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(range < 16.0_f64.ln());
    }
    #[test]
    fn empty_equilibration_needs_no_input_or_workspace() {
        unsafe {
            let (mut cond, mut max, mut info) = (777.0, 777.0, -777_i64);
            zheequb_64_(
                b"L".as_ptr(),
                &0,
                std::ptr::null(),
                &1,
                std::ptr::null_mut(),
                &mut cond,
                &mut max,
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!((info, cond, max), (0, 1.0, 0.0));
        }
    }
}
