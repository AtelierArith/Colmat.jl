//! Protected complete-pivot LU and scaled solve for LAPACK GETC2/GESC2.
use crate::kernels::Scalar;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{MatMut, MatRef};

trait CompleteScalar: Scalar {
    const EPS: f64;
    const SMALL: f64;
    const LARGE: f64;
    fn component(self) -> f64;
    fn rounded(value: f64) -> f64 {
        Self::real(value).re()
    }
}
macro_rules! real_scalar {
    ($t:ty) => {
        impl CompleteScalar for $t {
            const EPS: f64 = <$t>::EPSILON as f64;
            const SMALL: f64 = (<$t>::MIN_POSITIVE / <$t>::EPSILON) as f64;
            const LARGE: f64 = <$t>::MAX as f64;
            fn component(self) -> f64 {
                self.abs() as f64
            }
        }
    };
}
macro_rules! complex_scalar {
    ($t:ty,$r:ty) => {
        impl CompleteScalar for $t {
            const EPS: f64 = <$r>::EPSILON as f64;
            const SMALL: f64 = (<$r>::MIN_POSITIVE / <$r>::EPSILON) as f64;
            const LARGE: f64 = <$r>::MAX as f64;
            fn component(self) -> f64 {
                (self.re.abs() as f64).max(self.im.abs() as f64)
            }
        }
    };
}
real_scalar!(f32);
real_scalar!(f64);
complex_scalar!(colmat::c32, f32);
complex_scalar!(colmat::c64, f64);

// Truncate BEFORE the pivot that would need perturbation. The public API leaves
// the corresponding Schur complement available for the protected continuation.
unsafe fn prefix<T: CompleteScalar>(
    n: usize,
    a: *mut T,
    ld: isize,
    minimum: f64,
) -> (usize, Vec<usize>, Vec<usize>) {
    use colmat::linalg::lu::full_pivoting::factor::*;
    let (mut rows, mut inverse_rows, mut columns, mut inverse_columns) = (
        vec![0usize; n],
        vec![0usize; n],
        vec![0usize; n],
        vec![0usize; n],
    );
    let mut params: colmat::Spec<FullPivLuParams, T> = Default::default();
    params.config.abs_tol = Some(minimum);
    let rank = crate::threading::with_par(|par| {
        let mut memory = MemBuffer::new(lu_in_place_scratch::<usize, T>(n, n, par.shape(), params));
        let (info, _, _) = lu_in_place(
            MatMut::from_raw_parts_mut(a, n, n, 1, ld),
            &mut rows,
            &mut inverse_rows,
            &mut columns,
            &mut inverse_columns,
            par,
            MemStack::new(&mut memory),
            params,
        );
        info.rank
    });
    (rank, rows, columns)
}
fn swaps(permutation: &[usize], rank: usize) -> Vec<usize> {
    let n = permutation.len();
    let mut current: Vec<usize> = (0..n).collect();
    let mut result: Vec<usize> = (0..n).collect();
    for i in 0..rank {
        let p = (i..n).find(|&j| current[j] == permutation[i]).unwrap();
        result[i] = p;
        current.swap(i, p);
    }
    result
}
unsafe fn factor<T: CompleteScalar>(
    n: usize,
    a: *mut T,
    ld: isize,
) -> (Vec<usize>, Vec<usize>, usize) {
    if n == 0 {
        return (vec![], vec![], 0);
    }
    if n == 1 {
        let info = if (*a).magnitude() < T::SMALL {
            *a = T::real(T::SMALL);
            1
        } else {
            0
        };
        return (vec![0], vec![0], info);
    }
    let mut largest = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            largest = largest.max((*a.offset(i as isize + j as isize * ld)).magnitude());
        }
    }
    let minimum = T::rounded(T::EPS * largest).max(T::SMALL);
    let (rank, rows, columns) = prefix(n, a, ld, minimum);
    let (mut row_swaps, mut column_swaps) = (swaps(&rows, rank), swaps(&columns, rank));
    let mut info = 0;
    for k in rank..n {
        let (mut p, mut q, mut largest) = (k, k, 0.0f64);
        for j in k..n {
            for i in k..n {
                let value = (*a.offset(i as isize + j as isize * ld)).magnitude();
                if value >= largest {
                    largest = value;
                    p = i;
                    q = j;
                }
            }
        }
        for j in 0..n {
            std::ptr::swap(
                a.offset(k as isize + j as isize * ld),
                a.offset(p as isize + j as isize * ld),
            );
        }
        for i in 0..n {
            std::ptr::swap(
                a.offset(i as isize + k as isize * ld),
                a.offset(i as isize + q as isize * ld),
            );
        }
        row_swaps[k] = p;
        column_swaps[k] = q;
        let pivot = a.offset(k as isize + k as isize * ld);
        if (*pivot).magnitude() < minimum {
            *pivot = T::real(minimum);
            info = k + 1;
        }
        let tail = n - k - 1;
        if tail == 0 {
            continue;
        }
        crate::threading::with_par(|mut par| {
            colmat::linalg::triangular_solve::solve_upper_triangular_in_place(
                MatRef::from_raw_parts(pivot, 1, 1, 1, ld),
                MatMut::from_raw_parts_mut(pivot.add(1), 1, tail, 1, 1),
                par.reborrow(),
            );
            colmat::linalg::matmul::matmul(
                MatMut::from_raw_parts_mut(pivot.offset(ld + 1), tail, tail, 1, ld),
                colmat::Accum::Add,
                MatRef::from_raw_parts(pivot.add(1), tail, 1, 1, ld),
                MatRef::from_raw_parts(pivot.offset(ld), 1, tail, 1, ld),
                -T::o(),
                par,
            );
        });
    }
    (row_swaps, column_swaps, info)
}
unsafe fn maximum<T: CompleteScalar>(rhs: *const T, n: usize) -> f64 {
    (0..n)
        .map(|i| (*rhs.add(i)).component())
        .fold(0.0f64, f64::max)
}
// Store the scale in the caller's real precision after every step. If its
// cumulative product underflows, zero the solution as well as the reported scale.
unsafe fn rescale<T: CompleteScalar>(rhs: *mut T, n: usize, total: &mut f64, factor: f64) -> bool {
    let factor = T::rounded(factor.min(1.0));
    let next = T::rounded(*total * factor);
    if factor == 0.0 || next == 0.0 {
        for i in 0..n {
            *rhs.add(i) = T::z();
        }
        *total = 0.0;
        return false;
    }
    let multiplier = T::real(factor);
    for i in 0..n {
        *rhs.add(i) = *rhs.add(i) * multiplier;
    }
    *total = next;
    true
}
unsafe fn solve<T: CompleteScalar, I: crate::lapack::LapackInt>(
    n: usize,
    a: *const T,
    ld: isize,
    rhs: *mut T,
    rows: *const I,
    columns: *const I,
) -> f64 {
    if n == 0 {
        return 1.0;
    }
    for i in 0..n - 1 {
        std::ptr::swap(rhs.add(i), rhs.add((*rows.add(i)).val() as usize - 1));
    }
    let mut scale = 1.0;
    let safe = (1.0 / T::SMALL) / 16.0;
    // Guard the forward rank-one updates, not only the eventual upper solve.
    for k in 0..n - 1 {
        let tail = n - k - 1;
        let coefficient = (k + 1..n)
            .map(|i| (*a.offset(i as isize + k as isize * ld)).component())
            .fold(1.0f64, f64::max);
        let target = safe / coefficient;
        let largest = maximum(rhs, n);
        if largest > target && !rescale(rhs, n, &mut scale, target / largest) {
            return 0.0;
        }
        crate::threading::with_par(|par| {
            colmat::linalg::matmul::matmul(
                MatMut::from_raw_parts_mut(rhs.add(k + 1), tail, 1, 1, tail as isize),
                colmat::Accum::Add,
                MatRef::from_raw_parts(a.offset(k as isize * ld + k as isize + 1), tail, 1, 1, ld),
                MatRef::from_raw_parts(rhs.add(k), 1, 1, 1, 1),
                -T::o(),
                par,
            )
        });
    }
    let mut normalized = vec![T::z(); n];
    for k in (0..n).rev() {
        let tail = n - k - 1;
        let pivot = a.offset(k as isize + k as isize * ld);
        let denominator = (*pivot).component();
        if denominator == 0.0 {
            for i in 0..n {
                *rhs.add(i) = T::z();
            }
            return 0.0;
        }
        // Normalize the U row BEFORE multiplying solved components, as in
        // Reference LAPACK. Updating unnormalized columns can require a zero
        // scale even when a nonzero scaled solution is representable.
        for j in 0..tail {
            normalized[j] = *pivot.offset((j + 1) as isize * ld);
        }
        if tail > 0 {
            crate::threading::with_par(|par| {
                colmat::linalg::triangular_solve::solve_upper_triangular_in_place(
                    MatRef::from_raw_parts(pivot, 1, 1, 1, ld),
                    MatMut::from_raw_parts_mut(normalized.as_mut_ptr(), 1, tail, 1, 1),
                    par,
                )
            });
        }
        let coefficient = normalized[..tail]
            .iter()
            .map(|x| x.component())
            .fold(1.0f64, f64::max);
        let bound = ((T::LARGE / 16.0) / coefficient) / (2.0 * (tail.max(1) as f64));
        let largest = maximum(rhs, n);
        if largest > bound && !rescale(rhs, n, &mut scale, bound / largest) {
            return 0.0;
        }
        let numerator = (*rhs.add(k)).component();
        let bound = bound / 4.0;
        if numerator / denominator > bound {
            let factor = (bound / numerator) * denominator;
            if !rescale(rhs, n, &mut scale, factor) {
                return 0.0;
            }
        }
        crate::threading::with_par(|mut par| {
            colmat::linalg::triangular_solve::solve_upper_triangular_in_place(
                MatRef::from_raw_parts(pivot, 1, 1, 1, ld),
                MatMut::from_raw_parts_mut(rhs.add(k), 1, 1, 1, 1),
                par.reborrow(),
            );
            if tail > 0 {
                colmat::linalg::matmul::matmul(
                    MatMut::from_raw_parts_mut(rhs.add(k), 1, 1, 1, 1),
                    colmat::Accum::Add,
                    MatRef::from_raw_parts(normalized.as_ptr(), 1, tail, 1, 1),
                    MatRef::from_raw_parts(rhs.add(k + 1), tail, 1, 1, tail as isize),
                    -T::o(),
                    par,
                );
            }
        });
    }
    for i in (0..n - 1).rev() {
        std::ptr::swap(rhs.add(i), rhs.add((*columns.add(i)).val() as usize - 1));
    }
    scale
}
use paste::paste;
macro_rules! exports {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p getc2 $s>](n:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,jpiv:*mut $i,info:*mut $i) {
            *info=0;let(nn,ld)=(*n as isize,*lda as isize);
            for(arg,valid)in[(1,nn>=0),(3,ld>=nn.max(1))]{if !valid{*info=-arg;crate::invalid(arg as i32);return;}}
            if nn==0{return;}
            let(rows,columns,status)=factor(nn as usize,a,ld);
            for i in 0..nn as usize {*ipiv.add(i)=(rows[i]+1)as $i;*jpiv.add(i)=(columns[i]+1)as $i;}
            *info=status as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gesc2 $s>](n:*const $i,a:*const $t,lda:*const $i,rhs:*mut $t,ipiv:*const $i,jpiv:*const $i,scale:*mut $r) {
            *scale=1.0;let(nn,ld)=(*n as isize,*lda as isize);
            for(arg,valid)in[(1,nn>=0),(3,ld>=nn.max(1))]{if !valid{crate::invalid(arg);return;}}
            *scale=solve(nn as usize,a,ld,rhs,ipiv,jpiv)as $r;
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
    fn native_prefix_stops_before_a_protected_pivot() {
        let mut a =
            colmat::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { [2.0, 1.0, 0.0][i] } else { 0.0 });
        let (rank, _, _) =
            unsafe { prefix(3, a.as_mut().as_ptr_mut(), a.as_ref().col_stride(), 1e-15) };
        assert_eq!(rank, 2);
        let mut a =
            colmat::Mat::<f64>::from_fn(3, 3, |i, j| if i == j { [2.0, 1.0, 0.5][i] } else { 0.0 });
        let (rank, _, _) =
            unsafe { prefix(3, a.as_mut().as_ptr_mut(), a.as_ref().col_stride(), 1e-15) };
        assert_eq!(rank, 3);
    }
}
