//! LAPACK-controlled pivot and tolerance semantics over native pivoted LLT/GEMM.
use crate::householder::ReflectorScalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{c32, c64, Mat};

trait PivotScalar: ReflectorScalar {
    unsafe fn diagonal(pointer: *const Self) -> f64;
}
macro_rules! real_diagonal {
    ($t:ty) => {
        impl PivotScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> f64 {
                *pointer as f64
            }
        }
    };
}
macro_rules! complex_diagonal {
    ($t:ty) => {
        impl PivotScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> f64 {
                core::ptr::addr_of!((*pointer).re).read() as f64
            }
        }
    };
}
real_diagonal!(f32);
real_diagonal!(f64);
complex_diagonal!(c32);
complex_diagonal!(c64);

fn pivot_root<T: ReflectorScalar>(value: T) -> T {
    use colmat::linalg::cholesky::llt_pivoting::factor::{
        cholesky_in_place, cholesky_in_place_scratch,
    };
    let mut diagonal = Mat::from_fn(1, 1, |_, _| value);
    let (mut permutation, mut inverse) = ([0_usize], [0_usize]);
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, T>(
            1,
            par.shape(),
            Default::default(),
        ));
        // The caller has already selected a strictly positive finite/extended pivot.
        let _ = cholesky_in_place(
            diagonal.as_mut(),
            &mut permutation,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
    });
    diagonal[(0, 0)]
}
unsafe fn factor<T: PivotScalar>(
    uplo: u8,
    n: usize,
    a: *mut T,
    lda: isize,
    tol: f64,
) -> (Vec<usize>, usize) {
    let mut residual = Mat::from_fn(n, n, |i, j| {
        if i == j {
            T::real(T::diagonal(a.offset(i as isize + j as isize * lda)))
        } else if (uplo == b'L' && i > j) || (uplo == b'U' && i < j) {
            *a.offset(i as isize + j as isize * lda)
        } else {
            (*a.offset(j as isize + i as isize * lda)).cj()
        }
    });
    let mut l = Mat::<T>::zeros(n, n);
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut rank = n;
    let mut stop_value = None;
    let mut cutoff = tol;
    for j in 0..n {
        let mut pivot = j;
        let mut value = residual[(j, j)].re();
        for i in j + 1..n {
            let candidate = residual[(i, i)].re();
            if candidate > value {
                pivot = i;
                value = candidate;
            }
        }
        if j == 0 {
            if value <= 0.0 || value.is_nan() {
                return (permutation, 0);
            }
            if tol < 0.0 {
                cutoff = (T::real(n as f64 * (T::EPS / 2.0)) * T::real(value)).re();
            }
        } else if value <= cutoff || value <= 0.0 || value.is_nan() {
            rank = j;
            stop_value = Some(value);
            break;
        }
        if pivot != j {
            permutation.swap(j, pivot);
            for i in 0..n {
                let x = residual[(j, i)];
                residual[(j, i)] = residual[(pivot, i)];
                residual[(pivot, i)] = x;
            }
            for i in 0..n {
                let x = residual[(i, j)];
                residual[(i, j)] = residual[(i, pivot)];
                residual[(i, pivot)] = x;
            }
            for k in 0..j {
                let x = l[(j, k)];
                l[(j, k)] = l[(pivot, k)];
                l[(pivot, k)] = x;
            }
        }
        let diagonal = pivot_root(T::real(value));
        l[(j, j)] = diagonal;
        let inverse = T::real(1.0 / diagonal.re());
        for i in j + 1..n {
            l[(i, j)] = residual[(i, j)] * inverse;
        }
        let rest = n - j - 1;
        if rest > 0 {
            let ls = residual.as_ref().col_stride();
            let ll = l.as_ref().col_stride();
            let column = l.as_ref().as_ptr().offset(j as isize * ll + j as isize + 1);
            crate::kernels::gemm(
                b'N',
                b'C',
                rest as isize,
                rest as isize,
                1,
                -T::o(),
                column,
                ll,
                column,
                ll,
                T::o(),
                residual
                    .as_mut()
                    .as_ptr_mut()
                    .offset((j + 1) as isize * (ls + 1)),
                ls,
            );
            for i in j + 1..n {
                residual[(i, i)] = T::real(residual[(i, i)].re());
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                let (row, col) = if uplo == b'L' { (i, j) } else { (j, i) };
                // Columns beyond the numerical rank keep the original entries.
                // Read them before the first write to the caller's buffer.
                let mut value = if col < rank {
                    l[(row, col)]
                } else {
                    *a.offset(permutation[row] as isize + permutation[col] as isize * lda)
                };
                if row == rank && col == rank {
                    if let Some(stop) = stop_value {
                        value = T::real(stop);
                    }
                }
                if uplo == b'U' {
                    value = value.cj();
                }
                l[(row, col)] = value;
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            if (uplo == b'L' && i >= j) || (uplo == b'U' && i <= j) {
                let (row, col) = if uplo == b'L' { (i, j) } else { (j, i) };
                *a.offset(i as isize + j as isize * lda) = l[(row, col)];
            }
        }
    }
    (permutation, rank)
}
use paste::paste;
macro_rules! exports {
    ($p:ident,$family:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $family $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,piv:*mut $i,rank:*mut $i,tol:*const $r,_work:*mut $r,info:*mut $i){
            *info=0;let up=crate::kernels::flag(uplo);let(nn,la)=(*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            let(permutation,computed)=factor(up,nn as usize,a,la,*tol as f64);
            for j in 0..nn as usize{*piv.add(j)=(permutation[j]+1) as $i;}
            *rank=computed as $i;if computed<nn as usize{*info=1;}
        }
    }};
}
exports!(s, pstrf, f32, f32, i32, _);
exports!(s, pstf2, f32, f32, i32, _);
exports!(s, pstrf, f32, f32, i64, _64_);
exports!(s, pstf2, f32, f32, i64, _64_);
exports!(d, pstrf, f64, f64, i32, _);
exports!(d, pstf2, f64, f64, i32, _);
exports!(d, pstrf, f64, f64, i64, _64_);
exports!(d, pstf2, f64, f64, i64, _64_);
exports!(c, pstrf, c32, f32, i32, _);
exports!(c, pstf2, c32, f32, i32, _);
exports!(c, pstrf, c32, f32, i64, _64_);
exports!(c, pstf2, c32, f32, i64, _64_);
exports!(z, pstrf, c64, f64, i32, _);
exports!(z, pstf2, c64, f64, i32, _);
exports!(z, pstrf, c64, f64, i64, _64_);
exports!(z, pstf2, c64, f64, i64, _64_);
