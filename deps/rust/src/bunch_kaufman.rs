//! LAPACK representation adapters for native real-symmetric/Hermitian LDL.
use crate::householder::ReflectorScalar;
use colmat::diag::Diag;
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::Mat;

struct Factors<T: ReflectorScalar> {
    matrix: Mat<T>,
    subdiagonal: Diag<T>,
    permutation: Vec<usize>,
    inverse: Vec<usize>,
}
fn native_factor<T: ReflectorScalar>(mut matrix: Mat<T>, unblocked: bool) -> Factors<T> {
    use colmat::linalg::cholesky::lblt::factor::{
        cholesky_in_place, cholesky_in_place_scratch, LbltParams, PivotingStrategy,
    };
    let n = matrix.nrows();
    let mut subdiagonal = Diag::<T>::zeros(n);
    let mut permutation = vec![0_usize; n];
    let mut inverse = vec![0_usize; n];
    let mut params: colmat::Spec<LbltParams, T> = Default::default();
    params.config.pivoting = PivotingStrategy::Partial;
    if unblocked {
        params.config.block_size = 1;
    }
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(cholesky_in_place_scratch::<usize, T>(
            n,
            par.shape(),
            params,
        ));
        cholesky_in_place(
            matrix.as_mut(),
            subdiagonal.as_mut(),
            &mut permutation,
            &mut inverse,
            par,
            MemStack::new(&mut mem),
            params,
        );
    });
    Factors {
        matrix,
        subdiagonal,
        permutation,
        inverse,
    }
}
fn native_solve<T: ReflectorScalar>(factors: &Factors<T>, rhs: &mut Mat<T>) {
    use colmat::linalg::cholesky::lblt::solve::{solve_in_place, solve_in_place_scratch};
    let n = factors.matrix.nrows();
    let columns = rhs.ncols();
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(solve_in_place_scratch::<usize, T>(n, columns, par.shape()));
        let permutation = unsafe {
            colmat::perm::PermRef::new_unchecked(&factors.permutation, &factors.inverse, n)
        };
        solve_in_place(
            factors.matrix.as_ref(),
            factors.matrix.as_ref().diagonal(),
            factors.subdiagonal.as_ref(),
            permutation,
            rhs.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schur::multiply;
    use colmat::c64;
    fn check<T: ReflectorScalar>() {
        let n = 5;
        let diagonal = [0.0, 0.0, 3.0, 0.0, 2.0];
        let original = Mat::<T>::from_fn(n, n, |i, j| {
            if i == j {
                T::real(diagonal[i])
            } else if i > j {
                T::parts(
                    if i == 1 && j == 0 {
                        2.0
                    } else {
                        0.03 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        0.01 * (i - j) as f64
                    } else {
                        0.0
                    },
                )
            } else {
                T::parts(
                    if j == 1 && i == 0 {
                        2.0
                    } else {
                        0.03 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        -0.01 * (j - i) as f64
                    } else {
                        0.0
                    },
                )
            }
        });
        for unblocked in [false, true] {
            let factors = native_factor(original.clone(), unblocked);
            let (packed, pivots) = pack_lower(&factors);
            let converted = unpack_lower(&packed, &pivots);
            assert_eq!(factors.permutation, converted.permutation);
            for j in 0..n {
                for i in j..n {
                    assert!(
                        (factors.matrix[(i, j)] - converted.matrix[(i, j)]).magnitude() < 1e-13
                    );
                }
            }
            for j in 0..n {
                assert!((factors.subdiagonal[j] - converted.subdiagonal[j]).magnitude() < 1e-13);
            }
            let l = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    T::o()
                } else if i > j {
                    factors.matrix[(i, j)]
                } else {
                    T::z()
                }
            });
            let d = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    factors.matrix[(i, i)]
                } else if i == j + 1 {
                    factors.subdiagonal[j]
                } else if j == i + 1 {
                    factors.subdiagonal[i].cj()
                } else {
                    T::z()
                }
            });
            let reconstructed = multiply(&multiply(&l, b'N', &d, b'N'), b'N', &l, b'C');
            for j in 0..n {
                for i in 0..n {
                    assert!(
                        (reconstructed[(i, j)]
                            - original[(factors.permutation[i], factors.permutation[j])])
                            .magnitude()
                            < 1e-10
                    );
                }
            }
            let rhs = Mat::from_fn(n, 2, |i, j| T::real((i + j + 1) as f64));
            let mut solution = rhs.clone();
            native_solve(&factors, &mut solution);
            let reconstructed = multiply(&original, b'N', &solution, b'N');
            for j in 0..2 {
                for i in 0..n {
                    assert!((reconstructed[(i, j)] - rhs[(i, j)]).magnitude() < 1e-10);
                }
            }
        }
    }
    #[test]
    fn real_and_complex_factor_and_solve() {
        check::<f64>();
        check::<c64>();
    }
}

fn pack_lower<T: ReflectorScalar>(factors: &Factors<T>) -> (Mat<T>, Vec<isize>) {
    let n = factors.matrix.nrows();
    let mut packed = Mat::<T>::zeros(n, n);
    let mut pivots = vec![0_isize; n];
    let mut order: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = j + 1 < n && factors.subdiagonal[j] != T::z();
        let size = if pair { 2 } else { 1 };
        let moved = j + size - 1;
        // Partial pivoting leaves the first row of a two-column pivot in place.
        assert!(!pair || order[j] == factors.permutation[j]);
        let pivot = (moved..n)
            .find(|&i| order[i] == factors.permutation[moved])
            .unwrap();
        order.swap(moved, pivot);
        let encoded = (pivot + 1) as isize * if pair { -1 } else { 1 };
        for k in j..j + size {
            pivots[k] = encoded;
            packed[(k, k)] = factors.matrix[(k, k)];
            for i in j + size..n {
                // Undo later row swaps of the global native L for this LAPACK column.
                packed[(i, k)] = factors.matrix[(factors.inverse[order[i]], k)];
            }
        }
        if pair {
            packed[(j + 1, j)] = factors.subdiagonal[j];
        }
        j += size;
    }
    (packed, pivots)
}
fn unpack_lower<T: ReflectorScalar>(packed: &Mat<T>, pivots: &[isize]) -> Factors<T> {
    let n = packed.nrows();
    let mut matrix = Mat::<T>::zeros(n, n);
    let mut subdiagonal = Diag::<T>::zeros(n);
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = pivots[j] < 0;
        let size = if pair { 2 } else { 1 };
        let moved = j + size - 1;
        let pivot = pivots[j].unsigned_abs() - 1;
        permutation.swap(moved, pivot);
        for k in 0..j {
            let value = matrix[(moved, k)];
            matrix[(moved, k)] = matrix[(pivot, k)];
            matrix[(pivot, k)] = value;
        }
        for k in j..j + size {
            matrix[(k, k)] = packed[(k, k)];
            for i in j + size..n {
                matrix[(i, k)] = packed[(i, k)];
            }
        }
        if pair {
            subdiagonal[j] = packed[(j + 1, j)];
        }
        j += size;
    }
    let mut inverse = vec![0_usize; n];
    for i in 0..n {
        inverse[permutation[i]] = i;
    }
    Factors {
        matrix,
        subdiagonal,
        permutation,
        inverse,
    }
}

trait BkScalar: ReflectorScalar {
    unsafe fn diagonal(pointer: *const Self) -> Self;
}
macro_rules! real_diagonal {
    ($t:ty) => {
        impl BkScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> Self {
                *pointer
            }
        }
    };
}
macro_rules! complex_diagonal {
    ($t:ty) => {
        impl BkScalar for $t {
            unsafe fn diagonal(pointer: *const Self) -> Self {
                Self::new(core::ptr::addr_of!((*pointer).re).read(), 0.0)
            }
        }
    };
}
real_diagonal!(f32);
real_diagonal!(f64);
complex_diagonal!(colmat::c32);
complex_diagonal!(colmat::c64);
unsafe fn read_lower<T: BkScalar>(
    uplo: u8,
    n: usize,
    a: *const T,
    lda: isize,
    hermitian: bool,
) -> Mat<T> {
    let mut matrix = Mat::<T>::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            let (row, col) = if uplo == b'L' {
                (i, j)
            } else {
                (n - 1 - i, n - 1 - j)
            };
            let pointer = a.offset(row as isize + col as isize * lda);
            let value = if i == j {
                T::diagonal(pointer)
            } else {
                *pointer
            };
            matrix[(i, j)] = value;
            if hermitian && i != j {
                matrix[(j, i)] = value.cj();
            }
        }
    }
    matrix
}
unsafe fn write_lower<T: ReflectorScalar>(uplo: u8, matrix: &Mat<T>, a: *mut T, lda: isize) {
    let n = matrix.nrows();
    for j in 0..n {
        for i in j..n {
            let (row, col) = if uplo == b'L' {
                (i, j)
            } else {
                (n - 1 - i, n - 1 - j)
            };
            *a.offset(row as isize + col as isize * lda) = if i == j {
                T::real(matrix[(i, j)].re())
            } else {
                matrix[(i, j)]
            };
        }
    }
}
fn singular<T: ReflectorScalar>(factors: &Factors<T>) -> usize {
    let n = factors.matrix.nrows();
    let mut j = 0;
    while j < n {
        if factors.subdiagonal[j] != T::z() {
            j += 2;
        } else {
            if factors.matrix[(j, j)] == T::z() {
                return j + 1;
            }
            j += 1;
        }
    }
    0
}
unsafe fn factor_lapack<T: BkScalar>(
    uplo: u8,
    n: usize,
    a: *mut T,
    lda: isize,
    unblocked: bool,
) -> (Vec<isize>, usize) {
    let factors = native_factor(read_lower(uplo, n, a, lda, true), unblocked);
    let info = singular(&factors);
    let (packed, pivots) = pack_lower(&factors);
    write_lower(uplo, &packed, a, lda);
    if uplo == b'L' {
        (pivots, info)
    } else {
        let pivots = (0..n)
            .map(|j| {
                let p = pivots[n - 1 - j];
                p.signum() * (n as isize + 1 - p.abs())
            })
            .collect();
        (pivots, if info == 0 { 0 } else { n + 1 - info })
    }
}
unsafe fn external_factors<T: BkScalar>(
    uplo: u8,
    n: usize,
    a: *const T,
    lda: isize,
    pivots: &[isize],
) -> Factors<T> {
    let packed = read_lower(uplo, n, a, lda, false);
    if uplo == b'L' {
        unpack_lower(&packed, pivots)
    } else {
        let lower: Vec<isize> = (0..n)
            .map(|j| {
                let p = pivots[n - 1 - j];
                p.signum() * (n as isize + 1 - p.abs())
            })
            .collect();
        unpack_lower(&packed, &lower)
    }
}
unsafe fn solve_lapack<T: BkScalar>(
    uplo: u8,
    n: usize,
    nrhs: usize,
    a: *const T,
    lda: isize,
    pivots: &[isize],
    b: *mut T,
    ldb: isize,
) {
    let factors = external_factors(uplo, n, a, lda, pivots);
    let mut rhs = Mat::from_fn(n, nrhs, |i, j| {
        *b.offset((if uplo == b'L' { i } else { n - 1 - i }) as isize + j as isize * ldb)
    });
    native_solve(&factors, &mut rhs);
    for j in 0..nrhs {
        for i in 0..n {
            *b.offset((if uplo == b'L' { i } else { n - 1 - i }) as isize + j as isize * ldb) =
                rhs[(i, j)];
        }
    }
}
use crate::kernels::flag;
use paste::paste;
macro_rules! factor_exports{
    ($p:ident,$kind:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind trf $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(up,nn,la,lw)=(flag(uplo),*n as isize,*lda as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1)),(7,lw== -1||lw>=1)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(1);if lw== -1{return;}
            let(pivots,err)=factor_lapack(up,nn as usize,a,la,false);
            for j in 0..nn as usize{*ipiv.add(j)=pivots[j] as $i;}*info=err as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tf2 $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,info:*mut $i){
            *info=0;let(up,nn,la)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            let(pivots,err)=factor_lapack(up,nn as usize,a,la,true);
            for j in 0..nn as usize{*ipiv.add(j)=pivots[j] as $i;}*info=err as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind trs $s>](uplo:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,ipiv:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
            *info=0;let(up,nn,nr,la,lb)=(flag(uplo),*n as isize,*nrhs as isize,*lda as isize,*ldb as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(3,nr>=0),(5,la>=nn.max(1)),(8,lb>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            if nn==0||nr==0{return;}
            let pivots:Vec<isize>=(0..nn as usize).map(|j|*ipiv.add(j) as isize).collect();
            solve_lapack(up,nn as usize,nr as usize,a,la,&pivots,b,lb);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind trs2 $s>](uplo:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,ipiv:*const $i,b:*mut $t,ldb:*const $i,_work:*mut $t,info:*mut $i){
            [<$p $kind trs $s>](uplo,n,nrhs,a,lda,ipiv,b,ldb,info);
        }
    }};
}
factor_exports!(s, sy, f32, i32, _);
factor_exports!(d, sy, f64, i32, _);
factor_exports!(c, he, colmat::c32, i32, _);
factor_exports!(z, he, colmat::c64, i32, _);
factor_exports!(s, sy, f32, i64, _64_);
factor_exports!(d, sy, f64, i64, _64_);
factor_exports!(c, he, colmat::c32, i64, _64_);
factor_exports!(z, he, colmat::c64, i64, _64_);

fn native_inverse<T: ReflectorScalar>(factors: &Factors<T>) -> Mat<T> {
    use colmat::linalg::cholesky::lblt::inverse::{inverse, inverse_scratch};
    let n = factors.matrix.nrows();
    let mut result = Mat::<T>::zeros(n, n);
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(inverse_scratch::<usize, T>(n, par.shape()));
        let permutation = unsafe {
            colmat::perm::PermRef::new_unchecked(&factors.permutation, &factors.inverse, n)
        };
        inverse(
            result.as_mut(),
            factors.matrix.as_ref(),
            factors.matrix.as_ref().diagonal(),
            factors.subdiagonal.as_ref(),
            permutation,
            par,
            MemStack::new(&mut mem),
        );
    });
    result
}
unsafe fn inverse_lapack<T: BkScalar>(
    uplo: u8,
    n: usize,
    a: *mut T,
    lda: isize,
    pivots: &[isize],
) -> usize {
    if n == 0 {
        return 0;
    }
    let factors = external_factors(uplo, n, a, lda, pivots);
    let info = singular(&factors);
    if info != 0 {
        return if uplo == b'L' { info } else { n + 1 - info };
    }
    let inverted = native_inverse(&factors);
    write_lower(uplo, &inverted, a, lda);
    0
}
macro_rules! inverse_driver_exports{
    ($p:ident,$kind:ident,$t:ty,$i:ty,$s:tt)=>{paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tri $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*const $i,_work:*mut $t,info:*mut $i){
            *info=0;let(up,nn,la)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            let pivots:Vec<isize>=(0..nn as usize).map(|j|*ipiv.add(j) as isize).collect();
            *info=inverse_lapack(up,nn as usize,a,la,&pivots) as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tri2 $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(up,nn,la,lw)=(flag(uplo),*n as isize,*lda as isize,*lwork as isize);
            // Common inverse path: its unblocked caller workspace is N elements.
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1)),(7,lw== -1||lw>=nn.max(1))]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(nn.max(1));if lw== -1{return;}
            [<$p $kind tri $s>](uplo,n,a,lda,ipiv,work,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tri2x $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*const $i,work:*mut $t,_nb:*const $i,info:*mut $i){
            // NB is accepted as a blocking hint; the common native inverse owns scratch.
            [<$p $kind tri $s>](uplo,n,a,lda,ipiv,work,info);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind sv $s>](uplo:*const u8,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,b:*mut $t,ldb:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
            *info=0;let(up,nn,nr,la,lb,lw)=(flag(uplo),*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(3,nr>=0),(5,la>=nn.max(1)),(8,lb>=nn.max(1)),(10,lw== -1||lw>=1)]{if !ok{*info=-arg;crate::invalid(arg as i32);return;}}
            *work=crate::lapack::workspace::<$t>(1);if lw== -1{return;}
            [<$p $kind trf $s>](uplo,n,a,lda,ipiv,work,lwork,info);
            if *info==0{[<$p $kind trs $s>](uplo,n,nrhs,a,lda,ipiv,b,ldb,info);}
        }
    }};
}
inverse_driver_exports!(s, sy, f32, i32, _);
inverse_driver_exports!(d, sy, f64, i32, _);
inverse_driver_exports!(c, he, colmat::c32, i32, _);
inverse_driver_exports!(z, he, colmat::c64, i32, _);
inverse_driver_exports!(s, sy, f32, i64, _64_);
inverse_driver_exports!(d, sy, f64, i64, _64_);
inverse_driver_exports!(c, he, colmat::c32, i64, _64_);
inverse_driver_exports!(z, he, colmat::c64, i64, _64_);
