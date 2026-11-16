//! ROOK uses two independently encoded swaps for each two-column pivot.
//! Storage reference: LAPACK 3.12.1 DSYTRF_ROOK, IPIV contract:
//! https://netlib.org/lapack/explore-html/db/df4/dsytrf__rook_8f_source.html
//! Numerical operations compose the pinned colmat public LBLT APIs.
use super::*;

fn pack_lower_rook<T: ReflectorScalar>(factors: &Factors<T>) -> (Mat<T>, Vec<isize>) {
    let n = factors.matrix.nrows();
    let mut packed = Mat::<T>::zeros(n, n);
    let mut pivots = vec![0; n];
    let mut order: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = j + 1 < n && factors.subdiagonal[j] != T::z();
        let size = if pair { 2 } else { 1 };
        for k in j..j + size {
            let pivot = (k..n)
                .find(|&i| order[i] == factors.permutation[k])
                .unwrap();
            order.swap(k, pivot);
            pivots[k] = (pivot + 1) as isize * if pair { -1 } else { 1 };
        }
        for k in j..j + size {
            packed[(k, k)] = factors.matrix[(k, k)];
            for i in j + size..n {
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

fn unpack_lower_rook<T: ReflectorScalar>(packed: &Mat<T>, pivots: &[isize]) -> Factors<T> {
    let n = packed.nrows();
    let mut matrix = Mat::<T>::zeros(n, n);
    let mut subdiagonal = Diag::<T>::zeros(n);
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut j = 0;
    while j < n {
        let pair = pivots[j] < 0;
        let size = if pair { 2 } else { 1 };
        for k in j..j + size {
            let pivot = pivots[k].unsigned_abs() - 1;
            permutation.swap(k, pivot);
            for column in 0..j {
                let value = matrix[(k, column)];
                matrix[(k, column)] = matrix[(pivot, column)];
                matrix[(pivot, column)] = value;
            }
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
    let mut inverse = vec![0; n];
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

#[cfg(test)]
mod tests {
    use super::*;
    use colmat::linalg::cholesky::lblt::factor::PivotingStrategy;

    #[test]
    fn rook_two_swap_indices_and_later_column_permutations() {
        let matrix = Mat::from_fn(4, 4, |i, j| {
            if i == j {
                (i + 1) as f64
            } else if i > j && !(i == 1 && j == 0) {
                (10 * i + j) as f64
            } else {
                0.0
            }
        });
        let mut subdiagonal = Diag::<f64>::zeros(4);
        subdiagonal[0] = 2.0;
        let factors = Factors {
            matrix,
            subdiagonal,
            permutation: vec![2, 3, 1, 0],
            inverse: vec![3, 2, 0, 1],
        };
        let (packed, pivots) = pack_lower_rook(&factors);
        assert_eq!(pivots, [-3, -4, 4, 4]);
        assert_eq!(packed[(2, 0)], factors.matrix[(3, 0)]);
        assert_eq!(packed[(3, 0)], factors.matrix[(2, 0)]);
        let restored = unpack_lower_rook(&packed, &pivots);
        assert_eq!(restored.permutation, factors.permutation);
        assert_eq!(restored.inverse, factors.inverse);
        for j in 0..4 {
            for i in j..4 {
                assert_eq!(restored.matrix[(i, j)], factors.matrix[(i, j)]);
            }
        }
        assert_eq!(restored.subdiagonal[0], 2.0);
    }

    fn native_roundtrip<T: ReflectorScalar>() {
        let n = 6;
        let original = Mat::from_fn(n, n, |i, j| {
            if i == j {
                T::real(if i < 2 { 0.0 } else { (i + 1) as f64 })
            } else if i > j {
                T::parts(
                    if i == 1 && j == 0 {
                        3.0
                    } else {
                        0.1 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        0.03 * (i - j) as f64
                    } else {
                        0.0
                    },
                )
            } else {
                T::parts(
                    if j == 1 && i == 0 {
                        3.0
                    } else {
                        0.1 * (i + j) as f64
                    },
                    if T::COMPLEX {
                        -0.03 * (j - i) as f64
                    } else {
                        0.0
                    },
                )
            }
        });
        for strategy in [PivotingStrategy::Rook, PivotingStrategy::RookDiag] {
            for unblocked in [false, true] {
                let factors = native_factor_with_pivoting(original.clone(), unblocked, strategy);
                let (packed, pivots) = pack_lower_rook(&factors);
                assert!(pivots.iter().any(|&p| p < 0));
                let restored = unpack_lower_rook(&packed, &pivots);
                assert_eq!(restored.permutation, factors.permutation);
                let mut rhs = original.clone();
                native_solve(&restored, &mut rhs);
                for j in 0..n {
                    for i in 0..n {
                        assert!(
                            (rhs[(i, j)] - if i == j { T::o() } else { T::z() }).magnitude()
                                < 200.0 * T::EPS
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn rook_native_four_scalar_types() {
        native_roundtrip::<f32>();
        native_roundtrip::<f64>();
        native_roundtrip::<colmat::c32>();
        native_roundtrip::<colmat::c64>();
    }
}

unsafe fn factor_lapack<T: BkScalar>(
    uplo: u8,
    n: usize,
    a: *mut T,
    lda: isize,
    unblocked: bool,
) -> (Vec<isize>, usize) {
    let factors = native_factor_with_pivoting(
        read_lower(uplo, n, a, lda, true),
        unblocked,
        colmat::linalg::cholesky::lblt::factor::PivotingStrategy::Rook,
    );
    let info = singular(&factors);
    let (packed, pivots) = pack_lower_rook(&factors);
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
        unpack_lower_rook(&packed, pivots)
    } else {
        let lower: Vec<isize> = (0..n)
            .map(|j| {
                let p = pivots[n - 1 - j];
                p.signum() * (n as isize + 1 - p.abs())
            })
            .collect();
        unpack_lower_rook(&packed, &lower)
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

macro_rules! rook_exports {
    ($p:ident,$kind:ident,$t:ty,$i:ty,$s:tt) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind trf _rook $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let(up,nn,la,lw)=(flag(uplo),*n as isize,*lda as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1)),(7,lw== -1||lw>=1)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(1);
            if lw== -1||nn==0 {return;}
            if !crate::kernels::matrix_addressable::<$t>(nn,nn,la) {*info=-4;crate::invalid(4);return;}
            let(pivots,err)=factor_lapack(up,nn as usize,a,la,false);
            for j in 0..nn as usize {*ipiv.add(j)=pivots[j] as $i;}
            *info=err as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tf2 _rook $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,info:*mut $i) {
            *info=0;
            let(up,nn,la)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if nn==0 {return;}
            if !crate::kernels::matrix_addressable::<$t>(nn,nn,la) {*info=-4;crate::invalid(4);return;}
            let(pivots,err)=factor_lapack(up,nn as usize,a,la,true);
            for j in 0..nn as usize {*ipiv.add(j)=pivots[j] as $i;}
            *info=err as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind trs _rook $s>](uplo:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,ipiv:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {
            *info=0;
            let(up,nn,nr,la,lb)=(flag(uplo),*n as isize,*nrhs as isize,*lda as isize,*ldb as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(3,nr>=0),(5,la>=nn.max(1)),(8,lb>=nn.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if nn==0||nr==0 {return;}
            for(arg,ok)in[(5,crate::kernels::matrix_addressable::<$t>(nn,nn,la)),(8,crate::kernels::matrix_addressable::<$t>(nn,nr,lb))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let pivots:Vec<isize>=(0..nn as usize).map(|j|*ipiv.add(j) as isize).collect();
            solve_lapack(up,nn as usize,nr as usize,a,la,&pivots,b,lb);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind tri _rook $s>](uplo:*const u8,n:*const $i,a:*mut $t,lda:*const $i,ipiv:*const $i,_work:*mut $t,info:*mut $i) {
            *info=0;
            let(up,nn,la)=(flag(uplo),*n as isize,*lda as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(4,la>=nn.max(1))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            if nn==0 {return;}
            if !crate::kernels::matrix_addressable::<$t>(nn,nn,la) {*info=-4;crate::invalid(4);return;}
            let pivots:Vec<isize>=(0..nn as usize).map(|j|*ipiv.add(j) as isize).collect();
            *info=inverse_lapack(up,nn as usize,a,la,&pivots) as $i;
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $kind sv _rook $s>](uplo:*const u8,n:*const $i,nrhs:*const $i,a:*mut $t,lda:*const $i,ipiv:*mut $i,b:*mut $t,ldb:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i) {
            *info=0;
            let(up,nn,nr,la,lb,lw)=(flag(uplo),*n as isize,*nrhs as isize,*lda as isize,*ldb as isize,*lwork as isize);
            for(arg,ok)in[(1,matches!(up,b'U'|b'L')),(2,nn>=0),(3,nr>=0),(5,la>=nn.max(1)),(8,lb>=nn.max(1)),(10,lw== -1||lw>=1)] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            *work=crate::lapack::workspace::<$t>(1);
            if lw== -1||nn==0 {return;}
            for(arg,ok)in[(5,crate::kernels::matrix_addressable::<$t>(nn,nn,la)),(8,nr==0||crate::kernels::matrix_addressable::<$t>(nn,nr,lb))] {
                if !ok {*info=-arg;crate::invalid(arg as i32);return;}
            }
            let(pivots,err)=factor_lapack(up,nn as usize,a,la,false);
            for j in 0..nn as usize {*ipiv.add(j)=pivots[j] as $i;}
            *info=err as $i;
            if err==0 && nr>0 {solve_lapack(up,nn as usize,nr as usize,a,la,&pivots,b,lb);}
        }
    }};
}
rook_exports!(s, sy, f32, i32, _);
rook_exports!(d, sy, f64, i32, _);
rook_exports!(c, he, colmat::c32, i32, _);
rook_exports!(z, he, colmat::c64, i32, _);
rook_exports!(s, sy, f32, i64, _64_);
rook_exports!(d, sy, f64, i64, _64_);
rook_exports!(c, he, colmat::c32, i64, _64_);
rook_exports!(z, he, colmat::c64, i64, _64_);
