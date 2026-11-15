//! General tridiagonal LU adapted to the same native band elimination.
//! Four stored rows retain adjacent pivots and the second superdiagonal.
use crate::kernels::{flag, Scalar};
use colmat::{Accum, Conj, MatMut, MatRef};

unsafe fn band<T: Scalar>(
    n: usize,
    dl: *const T,
    d: *const T,
    du: *const T,
    du2: Option<*const T>,
) -> Vec<T> {
    let mut ab = vec![T::z(); 4 * n];
    for j in 0..n {
        ab[4 * j + 2] = *d.add(j);
        if j + 1 < n {
            ab[4 * j + 3] = *dl.add(j);
            ab[4 * (j + 1) + 1] = *du.add(j);
        }
        if j + 2 < n {
            if let Some(du2) = du2 {
                ab[4 * (j + 2)] = *du2.add(j);
            }
        }
    }
    ab
}
unsafe fn copy_upper<T: Scalar>(n: usize, ab: &[T], d: *mut T, du: *mut T, du2: Option<*mut T>) {
    for j in 0..n {
        *d.add(j) = ab[4 * j + 2];
        if j + 1 < n {
            *du.add(j) = ab[4 * (j + 1) + 1];
        }
        if j + 2 < n {
            if let Some(du2) = du2 {
                *du2.add(j) = ab[4 * (j + 2)];
            }
        }
    }
}
unsafe fn factor<T: Scalar>(
    n: usize,
    dl: *mut T,
    d: *mut T,
    du: *mut T,
    du2: *mut T,
) -> (Vec<usize>, usize) {
    let mut ab = band(n, dl, d, du, None);
    let result = crate::band_lu::factor(n, n, 1, 1, ab.as_mut_ptr(), 4);
    copy_upper(n, &ab, d, du, Some(du2));
    for j in 0..n.saturating_sub(1) {
        *dl.add(j) = ab[4 * j + 3];
    }
    result
}
unsafe fn solve<T: Scalar>(
    n: usize,
    nrhs: usize,
    dl: *const T,
    d: *const T,
    du: *const T,
    du2: *const T,
    piv: &[usize],
    b: *mut T,
    ldb: usize,
    tr: u8,
) {
    if n == 0 || nrhs == 0 {
        return;
    }
    let ab = band(n, dl, d, du, Some(du2));
    crate::band_lu::solve(n, 1, 1, ab.as_ptr(), 4, piv, b, ldb, nrhs, tr);
}
unsafe fn driver<T: Scalar>(
    n: usize,
    nrhs: usize,
    dl: *mut T,
    d: *mut T,
    du: *mut T,
    b: *mut T,
    ldb: usize,
) -> usize {
    let mut ab = band(n, dl, d, du, None);
    let (_, info) =
        crate::band_lu::factor_steps(n, n, 1, 1, ab.as_mut_ptr(), 4, true, |j, p, km, l| {
            if nrhs == 0 || km == 0 {
                return;
            }
            for c in 0..nrhs {
                std::ptr::swap(b.add(j + c * ldb), b.add(p + c * ldb));
            }
            let left = MatRef::from_raw_parts(l, 1, 1, 1, 0);
            let right = MatRef::from_raw_parts(b.add(j), 1, nrhs, 0, ldb as isize);
            let target = MatMut::from_raw_parts_mut(b.add(j + 1), 1, nrhs, 0, ldb as isize);
            crate::threading::with_par(|par| {
                colmat::linalg::matmul::matmul_with_conj(
                    target,
                    Accum::Add,
                    left,
                    Conj::No,
                    right,
                    Conj::No,
                    -T::o(),
                    par,
                );
            });
        });
    copy_upper(n, &ab, d, du, None);
    // Unlike GTTRF, GTSV overwrites DL with U's second superdiagonal and
    // leaves its final element untouched. Singular calls stop immediately.
    let completed = if info == 0 {
        n.saturating_sub(1)
    } else {
        info - 1
    };
    for j in 0..completed.min(n.saturating_sub(2)) {
        *dl.add(j) = ab[4 * (j + 2)];
    }
    if info == 0 {
        for c in 0..nrhs {
            crate::structured_triangular::solve_upper_band(
                n,
                2,
                ab.as_ptr(),
                4,
                b.add(c * ldb),
                b'N',
            );
        }
    }
    info
}
macro_rules! valid {($info:expr,$($arg:expr=>$test:expr),+)=>{$(if !$test{crate::lapack::bad(&mut *$info,$arg);return;})+};}
macro_rules! exports {($p:ident,$t:ty,$i:ty,$s:tt)=>{paste::paste!{
    #[no_mangle]
    pub unsafe extern "C" fn [<$p gttrf $s>](n:*const $i,dl:*mut $t,d:*mut $t,du:*mut $t,du2:*mut $t,p:*mut $i,info:*mut $i) {
        *info=0;valid!(info,1=>*n>=0);let n=*n as usize;if n==0{return;}
        let(piv,err)=factor(n,dl,d,du,du2);for(j,v)in piv.iter().enumerate(){*p.add(j)=(*v+1)as $i;}*info=err as $i;
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<$p gttrs $s>](tr:*const u8,n:*const $i,nrhs:*const $i,dl:*const $t,d:*const $t,du:*const $t,du2:*const $t,p:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i) {
        *info=0;let tr=flag(tr);valid!(info,1=>crate::kernels::trans(tr),2=>*n>=0,3=>*nrhs>=0,10=>*ldb>=(*n).max(1));
        let(n,nrhs)=(*n as usize,*nrhs as usize);if n==0||nrhs==0{return;}
        let piv:Vec<usize>=(0..n).map(|j|*p.add(j)as usize-1).collect();solve(n,nrhs,dl,d,du,du2,&piv,b,*ldb as usize,tr);
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<$p gtts2 $s>](itr:*const $i,n:*const $i,nrhs:*const $i,dl:*const $t,d:*const $t,du:*const $t,du2:*const $t,p:*const $i,b:*mut $t,ldb:*const $i) {
        for(arg,valid)in[(1,matches!(*itr,0..=2)),(2,*n>=0),(3,*nrhs>=0),(10,*ldb>=(*n).max(1))]{if !valid{crate::invalid(arg);return;}}
        let(n,nrhs)=(*n as usize,*nrhs as usize);if n==0||nrhs==0{return;}
        let piv:Vec<usize>=(0..n).map(|j|*p.add(j)as usize-1).collect();let tr=match *itr{0=>b'N',1=>b'T',_=>b'C'};
        solve(n,nrhs,dl,d,du,du2,&piv,b,*ldb as usize,tr);
    }
    #[no_mangle]
    pub unsafe extern "C" fn [<$p gtsv $s>](n:*const $i,nrhs:*const $i,dl:*mut $t,d:*mut $t,du:*mut $t,b:*mut $t,ldb:*const $i,info:*mut $i) {
        *info=0;valid!(info,1=>*n>=0,2=>*nrhs>=0,7=>*ldb>=(*n).max(1));if *n==0{return;}
        *info=driver(*n as usize,*nrhs as usize,dl,d,du,b,*ldb as usize)as $i;
    }
}};}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, colmat::c32, i32, _);
exports!(z, colmat::c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, colmat::c32, i64, _64_);
exports!(z, colmat::c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn driver_stops_at_singular_pivot_and_preserves_final_dl() {
        unsafe {
            let mut dl = [2.0, 0.0];
            let mut d = [2.0, 1.0, 9.0];
            let mut du = [1.0, 3.0];
            let mut b = [4.0, 6.0, 8.0];
            assert_eq!(
                driver(
                    3,
                    1,
                    dl.as_mut_ptr(),
                    d.as_mut_ptr(),
                    du.as_mut_ptr(),
                    b.as_mut_ptr(),
                    3
                ),
                2
            );
            assert_eq!(d, [2.0, 0.0, 9.0]);
            assert_eq!(du, [1.0, 3.0]);
            assert_eq!(dl, [0.0, 0.0]);
            assert_eq!(b, [4.0, 2.0, 8.0]);
            let mut dl = [2.0];
            let mut d = [2.0, 1.0];
            let mut du = [1.0];
            let mut b = [4.0, 6.0];
            assert_eq!(
                driver(
                    2,
                    1,
                    dl.as_mut_ptr(),
                    d.as_mut_ptr(),
                    du.as_mut_ptr(),
                    b.as_mut_ptr(),
                    2
                ),
                2
            );
            assert_eq!(dl, [2.0]);
            assert_eq!(b, [4.0, 2.0]);
        }
    }
    #[test]
    fn no_rhs_still_factors_and_reports_singularity() {
        unsafe {
            let mut dl = [2.0];
            let mut d = [2.0, 1.0];
            let mut du = [1.0];
            assert_eq!(
                driver(
                    2,
                    0,
                    dl.as_mut_ptr(),
                    d.as_mut_ptr(),
                    du.as_mut_ptr(),
                    std::ptr::null_mut(),
                    2
                ),
                2
            );
            assert_eq!(d, [2.0, 0.0]);
            assert_eq!(dl, [2.0]);
        }
    }
    #[test]
    fn empty_and_zero_rhs_calls_do_not_read_factor_arrays() {
        unsafe {
            let mut info = 777i64;
            dgttrf_64_(
                &0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!(info, 0);
            dgttrs_64_(
                b"C".as_ptr(),
                &((1i64 << 32) + 1),
                &0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                &((1i64 << 32) + 1),
                &mut info,
            );
            assert_eq!(info, 0);
            dgttrf_64_(
                &(-(1i64 << 32) + 1),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut info,
            );
            assert_eq!(info, -1);
            assert_eq!(crate::colmat_blas_last_error(), 1);
        }
    }
}
