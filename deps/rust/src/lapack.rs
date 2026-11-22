//! LAPACK drivers over colmat's public decomposition and solve APIs.
use crate::kernels::{flag, Scalar};
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::prelude::Reborrow;
use colmat::{Conj, Mat, MatMut, MatRef};

/// Borrow a column-major Fortran buffer without copying it.
pub unsafe fn input_ref<'a, T: Scalar>(
    a: *const T,
    m: usize,
    n: usize,
    ld: isize,
) -> MatRef<'a, T> {
    MatRef::from_raw_parts(a, m, n, 1, ld)
}
/// Borrow a column-major Fortran buffer for in-place output without copying it.
pub unsafe fn input_mut<'a, T: Scalar>(a: *mut T, m: usize, n: usize, ld: isize) -> MatMut<'a, T> {
    MatMut::from_raw_parts_mut(a, m, n, 1, ld)
}
/// Borrow the transposed view of a column-major buffer, so the lower triangle of
/// the view is the upper triangle of the caller's matrix.
pub unsafe fn input_mut_transposed<'a, T: Scalar>(a: *mut T, n: usize, ld: isize) -> MatMut<'a, T> {
    MatMut::from_raw_parts_mut(a, n, n, ld, 1)
}
/// Borrow the transpose of a column-major buffer, so the lower triangle of the
/// view is the caller's upper triangle.
pub unsafe fn input_ref_transposed<'a, T: Scalar>(
    a: *const T,
    n: usize,
    ld: isize,
) -> MatRef<'a, T> {
    MatRef::from_raw_parts(a, n, n, ld, 1)
}
pub unsafe fn output_ref<T: Scalar>(a: MatRef<'_, T>, p: *mut T, ld: isize) {
    // The destination is column-major, so a contiguous source is a single copy.
    if a.row_stride() == 1 && a.col_stride() == a.nrows() as isize && ld == a.nrows() as isize {
        std::ptr::copy_nonoverlapping(a.as_ptr(), p, a.nrows() * a.ncols());
        return;
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            *p.offset(i as isize + j as isize * ld) = a[(i, j)];
        }
    }
}
pub unsafe fn output<T: Scalar>(a: &Mat<T>, p: *mut T, ld: isize) {
    output_ref(a.as_ref(), p, ld);
}
/// Factor the caller's buffer in place. `a` is overwritten with the L/U factors,
/// matching LAPACK's xGETRF contract.
pub fn lu<T: Scalar>(mut a: MatMut<'_, T>) -> (Vec<usize>, usize) {
    use colmat::linalg::lu::partial_pivoting::factor::*;
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    if k == 0 {
        return (vec![], 0);
    }
    let mut row_perm = vec![0usize; m];
    let mut row_perm_inv = vec![0usize; m];
    let work = crate::threading::real_work(m, n, k, T::COMPLEX);
    crate::threading::with_par_if(work, crate::threading::LU_WORK, |par| {
        let mut mem = MemBuffer::new(lu_in_place_scratch::<usize, T>(
            m,
            n,
            par.shape(),
            Default::default(),
        ));
        lu_in_place(
            a.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
    });
    let p = row_perm;
    let mut current: Vec<usize> = (0..m).collect();
    let mut piv = vec![0; k];
    let mut info = 0;
    for j in 0..k {
        let pos = (j..m).find(|&i| current[i] == p[j]).unwrap();
        piv[j] = pos;
        current.swap(j, pos);
    }
    for i in 0..k {
        if a[(i, i)] == T::z() {
            info = i + 1;
            break;
        }
    }
    (piv, info)
}
pub fn solve_lu<T: Scalar>(a: MatRef<'_, T>, p: &[usize], mut b: MatMut<'_, T>, t: u8) {
    use colmat::linalg::triangular_solve::*;
    let n = a.nrows();
    let swap = |b: &mut MatMut<'_, T>, i: usize, j: usize| {
        for c in 0..b.ncols() {
            let x = b[(i, c)];
            b[(i, c)] = b[(j, c)];
            b[(j, c)] = x;
        }
    };
    crate::threading::with_par(|mut par| {
        if t == b'N' {
            for (i, &j) in p.iter().enumerate() {
                swap(&mut b, i, j);
            }
            solve_unit_lower_triangular_in_place(a, b.as_mut(), par.reborrow());
            solve_upper_triangular_in_place(a, b.as_mut(), par);
        } else {
            let conj = if t == b'C' { Conj::Yes } else { Conj::No };
            solve_lower_triangular_in_place_with_conj(
                a.transpose(),
                conj,
                b.as_mut(),
                par.reborrow(),
            );
            solve_unit_upper_triangular_in_place_with_conj(a.transpose(), conj, b.as_mut(), par);
            for i in (0..n).rev() {
                swap(&mut b, i, p[i]);
            }
        }
    });
}
/// Factor the caller's buffer in place. The lower triangle is used regardless of
/// the caller's UPLO: an upper-triangular view is transposed before the call.
/// colmat-rs #192 tunes the f64/c64 LLT block to 64 for Zen 5. On AVX2-only
/// x86-64 (Skylake and similar) 32 is faster; see colmat-rs issue #257. Fall
/// back to the untuned LDLT default (128) only when neither applies.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn prefer_narrow_llt_block() -> bool {
    std::is_x86_feature_detected!("avx2") && !std::is_x86_feature_detected!("avx512f")
}
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn prefer_narrow_llt_block() -> bool {
    false
}
pub fn cholesky<T: Scalar>(mut a: MatMut<'_, T>) -> usize {
    use colmat::linalg::cholesky::llt::factor::*;
    let mut params: colmat::Spec<LltParams, T> = Default::default();
    if params.config.block_size >= 128 || prefer_narrow_llt_block() {
        params.config.block_size = 32;
        params.config.recursion_threshold = 32;
    }
    let n = a.nrows();
    let work = crate::threading::real_work(n, n, n, T::COMPLEX);
    crate::threading::with_par_if(work, crate::threading::CHOLESKY_WORK, |par| {
        let mut mem = MemBuffer::new(cholesky_in_place_scratch::<T>(
            a.nrows(),
            par.shape(),
            params,
        ));
        match cholesky_in_place(
            a.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            params,
        ) {
            Ok(_) => 0,
            Err(LltError::NonPositivePivot { index }) => index + 1,
        }
    })
}
thread_local! {
    /// `factor_cholesky` saves the elements of the triangle LAPACK does not own
    /// that a column-major factorization overwrites. The BLAS entry point is
    /// called once per factorization and allocating a fresh buffer at n=1024
    /// dominated the call with page-fault jitter, so each calling thread keeps
    /// its buffer and grows it to the largest shape it has seen. `u64` gives
    /// the 8-byte alignment every `Scalar` needs.
    static CHOLESKY_SCRATCH: std::cell::RefCell<Vec<u64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Save or restore the `n * (n - 1) / 2` elements of the strict upper triangle.
/// Both directions visit the entries in the same order, so `saved` can be a
/// plain stack of column segments.
unsafe fn copy_strict_upper<T: Scalar>(
    n: usize,
    a: *mut T,
    ld: isize,
    saved: *mut T,
    into_saved: bool,
) {
    let mut len = 0;
    for j in 1..n {
        let opposite = a.offset(j as isize * ld);
        if into_saved {
            std::ptr::copy_nonoverlapping(opposite, saved.add(len), j);
        } else {
            std::ptr::copy_nonoverlapping(saved.add(len), opposite, j);
        }
        len += j;
    }
}

unsafe fn factor_cholesky_in<T: Scalar>(
    saved: *mut T,
    u: u8,
    n: usize,
    a: *mut T,
    ld: isize,
) -> usize {
    // LAPACK owns only the selected triangle. The native LLT writes some of the
    // opposite triangle for a column-major (UPLO='L') input, so that part is
    // saved and put back. For UPLO='U' the transposed view makes every leaf
    // row-major, which the native leaf stages through a column-major copy of its
    // lower triangle: it never touches the caller's lower triangle, so there is
    // nothing to save.
    if u == b'L' {
        copy_strict_upper(n, a, ld, saved, true);
    }
    // LAPACK ignores the imaginary part of a Hermitian diagonal. The native leaf
    // divides the diagonal by its real square root, so a non-zero imaginary part
    // would survive in the factor.
    for j in 0..n {
        let diag = a.offset(j as isize + j as isize * ld);
        *diag = T::real((*diag).re());
    }
    let matrix = MatMut::from_raw_parts_mut(a, n, n, 1, ld);
    let info = if u == b'L' {
        cholesky(matrix)
    } else {
        cholesky(matrix.transpose_mut())
    };
    if u == b'L' {
        copy_strict_upper(n, a, ld, saved, false);
    }
    info
}
/// Factor the caller's buffer in place while preserving the triangle LAPACK
/// does not own. The saved part is reused per calling thread; the UPLO='U' path
/// factors the lower triangle of the transposed view, which is the caller's
/// upper triangle and is left untouched.
pub unsafe fn factor_cholesky<T: Scalar>(u: u8, n: usize, a: *mut T, ld: isize) -> usize {
    if n == 0 {
        return 0;
    }
    let words = if u == b'L' {
        (n * (n - 1) / 2 * core::mem::size_of::<T>()).div_ceil(8)
    } else {
        0
    };
    if words == 0 {
        return factor_cholesky_in(core::ptr::NonNull::<T>::dangling().as_ptr(), u, n, a, ld);
    }
    CHOLESKY_SCRATCH.with(|scratch| match scratch.try_borrow_mut() {
        Ok(mut buffer) => {
            if buffer.len() < words {
                buffer.resize(words, 0);
            }
            factor_cholesky_in(buffer.as_mut_ptr() as *mut T, u, n, a, ld)
        }
        Err(_) => {
            let mut buffer = vec![0u64; words];
            factor_cholesky_in(buffer.as_mut_ptr() as *mut T, u, n, a, ld)
        }
    })
}

pub fn solve_cholesky<T: Scalar>(a: MatRef<'_, T>, mut b: MatMut<'_, T>, u: u8) {
    use colmat::linalg::triangular_solve::*;
    crate::threading::with_par(|mut par| {
        if u == b'L' {
            // A = L L^H with L in the lower triangle of `a`.
            solve_lower_triangular_in_place(a, b.as_mut(), par.reborrow());
            solve_upper_triangular_in_place_with_conj(a.transpose(), Conj::Yes, b.as_mut(), par);
        } else {
            // A = U^H U with U in the upper triangle of `a`. The transpose view
            // exposes the stored upper triangle as a lower triangle, and the
            // conjugate flag makes it U^H.
            solve_lower_triangular_in_place_with_conj(
                a.transpose(),
                Conj::Yes,
                b.as_mut(),
                par.reborrow(),
            );
            solve_upper_triangular_in_place(a, b.as_mut(), par);
        }
    });
}
pub unsafe fn cholesky_output<T: Scalar>(a: &Mat<T>, p: *mut T, ld: isize, u: u8) {
    for j in 0..a.ncols() {
        for i in j..a.nrows() {
            if u == b'L' {
                *p.offset(i as isize + j as isize * ld) = a[(i, j)];
            } else {
                *p.offset(j as isize + i as isize * ld) = a[(i, j)].cj();
            }
        }
    }
}
type SvdResult<T> = (Mat<T>, Vec<f64>, Mat<T>);
type GeneralOutput<T> = (Vec<T>, Vec<f64>, usize, usize, Vec<f64>, f64);

/// Compute the SVD into caller-provided vectors. The native driver overwrites
/// every element of each requested singular-vector matrix before reading it, so
/// LAPACK's buffers can be borrowed directly when their layout matches the
/// column-major result (see `svd_driver`).
fn svd_native<T: Scalar>(
    a: MatRef<'_, T>,
    u: usize,
    v: usize,
    mut s: colmat::diag::DiagMut<'_, T>,
    left: Option<MatMut<'_, T>>,
    right: Option<MatMut<'_, T>>,
) -> Result<(), ()> {
    use colmat::linalg::svd::*;
    let (m, n) = (a.nrows(), a.ncols());
    let cv = |x: usize, dim: usize| {
        if x == 0 {
            ComputeSvdVectors::No
        } else if x == dim {
            ComputeSvdVectors::Full
        } else {
            ComputeSvdVectors::Thin
        }
    };
    let work = crate::threading::real_work(m, n, m.min(n), T::COMPLEX);
    let limit = if T::COMPLEX {
        crate::threading::SVD_WORK_COMPLEX
    } else {
        crate::threading::SVD_WORK
    };
    crate::threading::with_par_if(work, limit, |par| {
        let mut mem = MemBuffer::new(svd_scratch::<T>(
            m,
            n,
            cv(u, m),
            cv(v, n),
            par.shape(),
            Default::default(),
        ));
        colmat::linalg::svd::svd(
            a.as_ref(),
            s.as_mut(),
            left,
            right,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| ())
    })
}

pub fn svd<T: Scalar>(a: MatRef<'_, T>, u: usize, v: usize) -> Result<SvdResult<T>, ()> {
    use colmat::diag::Diag;
    let (m, n) = (a.nrows(), a.ncols());
    let mut s = Diag::<T>::zeros(m.min(n));
    let mut left = Mat::<T>::zeros(m, u);
    let mut right = Mat::<T>::zeros(n, v);
    svd_native(
        a,
        u,
        v,
        s.as_mut(),
        if u == 0 { None } else { Some(left.as_mut()) },
        if v == 0 { None } else { Some(right.as_mut()) },
    )?;
    Ok((left, (0..m.min(n)).map(|i| s[i].re()).collect(), right))
}
pub fn selfadj<T: Scalar>(a: MatRef<'_, T>, vectors: bool) -> Result<(Vec<f64>, Mat<T>), ()> {
    use colmat::{diag::Diag, linalg::evd::*};
    let n = a.nrows();
    // LAPACK treats a Hermitian diagonal as real regardless of the stored
    // imaginary bits. Larger native EVD paths can inspect those bits, so
    // normalize them at this boundary. The ordinary real-diagonal path keeps
    // borrowing the caller's selected lower triangle without another copy.
    let normalized =
        (T::COMPLEX && (0..n).any(|i| a[(i, i)] != T::real(a[(i, i)].re()))).then(|| {
            Mat::from_fn(n, n, |i, j| {
                if i == j {
                    T::real(a[(i, i)].re())
                } else if i > j {
                    a[(i, j)]
                } else {
                    T::z()
                }
            })
        });
    let a = normalized.as_ref().map_or(a, |matrix| matrix.as_ref());
    let mut s = Diag::<T>::zeros(n);
    let mut u = Mat::<T>::zeros(n, if vectors { n } else { 0 });
    let work = crate::threading::real_work(n, n, n, T::COMPLEX);
    let limit = if T::COMPLEX {
        crate::threading::SELFADJOINT_WORK_COMPLEX
    } else {
        crate::threading::SELFADJOINT_WORK
    };
    crate::threading::with_par_if(work, limit, |par| {
        let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<T>(
            n,
            if vectors {
                ComputeEigenvectors::Yes
            } else {
                ComputeEigenvectors::No
            },
            par.shape(),
            Default::default(),
        ));
        self_adjoint_evd(
            a.as_ref(),
            s.as_mut(),
            if vectors { Some(u.as_mut()) } else { None },
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| ())?;
        Ok(((0..n).map(|i| s[i].re()).collect(), u))
    })
}

// LAPACK queries stored in Float32 must round upward, otherwise a large
// integer minimum can round down and reject the very workspace we recommend.
pub fn workspace<T: Scalar>(n: isize) -> T {
    let value = T::real(n as f64);
    if value.re() < n as f64 {
        T::real((n as f64) * (1.0 + f32::EPSILON as f64))
    } else {
        value
    }
}

pub fn bad(info: &mut impl LapackInt, arg: i32) {
    *info = LapackInt::from_i64(-(arg as i64));
    crate::invalid(arg);
}
pub trait LapackInt: Copy {
    fn val(self) -> isize;
    fn from_i64(v: i64) -> Self;
}
impl LapackInt for i32 {
    fn val(self) -> isize {
        self as isize
    }
    fn from_i64(v: i64) -> Self {
        v as Self
    }
}
impl LapackInt for i64 {
    fn val(self) -> isize {
        self as isize
    }
    fn from_i64(v: i64) -> Self {
        v
    }
}
// Central driver checks retain LAPACK's negative INFO convention.
macro_rules! valid {($info:expr,$($arg:expr=>$condition:expr),+ $(,)?)=>{$(if !$condition {bad(&mut *$info,$arg);return;})+};}

macro_rules! basic {
($p:ident,$t:ty,$r:ty,$i:ty,$s:tt)=>{paste::paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p getrf $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,piv:*mut $i,info:*mut $i){
 *info=0;let(m,n,ld)=((*m).val(),(*n).val(),(*lda).val());valid!(info,1=>m>=0,2=>n>=0,4=>ld>=1.max(m));if m==0||n==0{return;}
 let mat=input_mut(a,m as usize,n as usize,ld);let(p,err)=lu(mat);for(i,v)in p.iter().enumerate(){*piv.add(i)=<$i>::from_i64(*v as i64+1);}*info=err as $i;
}
#[no_mangle]
pub unsafe extern "C" fn [<$p getrs $s>](t:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,piv:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;let(t,n,k,la,lb)=(flag(t),(*n).val(),(*nrhs).val(),(*lda).val(),(*ldb).val());valid!(info,1=>crate::kernels::trans(t),2=>n>=0,3=>k>=0,5=>la>=1.max(n),8=>lb>=1.max(n));if n==0||k==0{return;}
 let mat=input_ref(a,n as usize,n as usize,la);let rhs=input_mut(b,n as usize,k as usize,lb);let p:Vec<usize>=(0..n as usize).map(|i|(*piv.add(i)).val() as usize-1).collect();solve_lu(mat,&p,rhs,t);
}
#[no_mangle]
pub unsafe extern "C" fn [<$p gesv $s>](n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,p:*mut $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;valid!(info,1=>*n>=0,2=>*k>=0,4=>*lda>=1.max(*n),7=>*ldb>=1.max(*n));[<$p getrf $s>](n,n,a,lda,p,info);if *info==0{[<$p getrs $s>](b"N".as_ptr(),n,k,a,lda,p,b,ldb,info);}
}
#[no_mangle]
pub unsafe extern "C" fn [<$p getri $s>](n:*const $i,a:*mut $t,lda:*const $i,p:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
 *info=0;valid!(info,1=>*n>=0,3=>*lda>=1.max(*n),6=>*lwork== -1||*lwork>=1.max(*n));*work=workspace::<$t>((1.max(*n)).val());if *lwork== -1||*n==0{return;}
 let n=(*n).val() as usize;let inv={let mat=input_ref(a,n,n,(*lda).val());for i in 0..n{if mat[(i,i)]==<$t>::z(){*info=(i+1)as $i;return;}}
 let mut inv=Mat::<$t>::identity(n,n);let piv:Vec<usize>=(0..n).map(|i|(*p.add(i)).val()as usize-1).collect();solve_lu(mat,&piv,inv.as_mut(),b'N');inv};output(&inv,a,(*lda).val());
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potrf $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){
 *info=0;let(u,n,ld)=(flag(u),(*n).val(),(*lda).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,4=>ld>=1.max(n));if n==0{return;}*info=factor_cholesky(u,n as usize,a,ld)as $i;
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potrs $s>](u:*const u8,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;let(u,n,k,la,lb)=(flag(u),(*n).val(),(*k).val(),(*lda).val(),(*ldb).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,3=>k>=0,5=>la>=1.max(n),7=>lb>=1.max(n));if n==0||k==0{return;}let mat=input_ref(a,n as usize,n as usize,la);let rhs=input_mut(b,n as usize,k as usize,lb);solve_cholesky(mat,rhs,u);
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potri $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){
 *info=0;let(u,n,ld)=(flag(u),(*n).val(),(*lda).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,4=>ld>=1.max(n));if n==0{return;}let inv={let mat=input_ref(a,n as usize,n as usize,ld);for i in 0..n as usize{if mat[(i,i)]==<$t>::z(){*info=(i+1)as $i;return;}}let mut inv=Mat::<$t>::identity(n as usize,n as usize);solve_cholesky(mat,inv.as_mut(),u);inv};cholesky_output(&inv,a,ld,u);
}
}};}
macro_rules! factor_aliases {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {paste::paste!{
        #[no_mangle]
        pub unsafe extern "C" fn [<$p getf2 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,p:*mut $i,info:*mut $i){[<$p getrf $s>](m,n,a,lda,p,info);}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p getrf2 $s>](m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,p:*mut $i,info:*mut $i){[<$p getrf $s>](m,n,a,lda,p,info);}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p potf2 $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){[<$p potrf $s>](u,n,a,lda,info);}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p potrf2 $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){[<$p potrf $s>](u,n,a,lda,info);}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p posv $s>](u:*const u8,n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
            *info=0;valid!(info,1=>crate::kernels::uplo(flag(u)),2=>*n>=0,3=>*k>=0,5=>*lda>=1.max(*n),7=>*ldb>=1.max(*n));
            [<$p potrf $s>](u,n,a,lda,info);if *info==0{[<$p potrs $s>](u,n,k,a,lda,b,ldb,info);}
        }
    }};
}
factor_aliases!(s, f32, i32, _);
factor_aliases!(d, f64, i32, _);
factor_aliases!(c, colmat::c32, i32, _);
factor_aliases!(z, colmat::c64, i32, _);
factor_aliases!(s, f32, i64, _64_);
factor_aliases!(d, f64, i64, _64_);
factor_aliases!(c, colmat::c32, i64, _64_);
factor_aliases!(z, colmat::c64, i64, _64_);

basic!(s, f32, f32, i32, _);
basic!(d, f64, f64, i32, _);
basic!(c, colmat::c32, f32, i32, _);
basic!(z, colmat::c64, f64, i32, _);
basic!(s, f32, f32, i64, _64_);
basic!(d, f64, f64, i64, _64_);
basic!(c, colmat::c32, f32, i64, _64_);
basic!(z, colmat::c64, f64, i64, _64_);

unsafe fn svd_driver<T: Scalar, R: Scalar, I: LapackInt>(
    dd: bool,
    ju: u8,
    jv: u8,
    m: isize,
    n: isize,
    a: *mut T,
    la: isize,
    s: *mut R,
    u: *mut T,
    lu: isize,
    vt: *mut T,
    lv: isize,
    work: *mut T,
    lwork: isize,
    info: *mut I,
) {
    *info = I::from_i64(0);
    let r = m.min(n).max(0);
    let big = m.max(n).max(0);
    let minwork = if dd {
        if T::COMPLEX {
            match ju {
                b'N' => 2 * r + big,
                b'O' => 2 * r * r + 2 * r + big,
                b'S' => r * r + 3 * r,
                _ => r * r + 2 * r + big,
            }
        } else {
            match ju {
                b'N' => 3 * r + big.max(7 * r),
                b'O' => 3 * r + big.max(5 * r * r + 4 * r),
                b'S' => 4 * r * r + 7 * r,
                _ => 4 * r * r + 6 * r + big,
            }
        }
    } else if T::COMPLEX {
        2 * r + big
    } else {
        (3 * r + big).max(5 * r)
    }
    .max(1);
    if dd {
        valid!(info,1=>matches!(ju,b'N'|b'S'|b'A'|b'O'),2=>m>=0,3=>n>=0,5=>la>=1.max(m),8=>lu>=1&&(ju!=b'A'&&ju!=b'S'&&!(ju==b'O'&&m<n)||lu>=m),10=>lv>=1&&(ju!=b'A'&&!(ju==b'O'&&m>=n)||lv>=n)&&(ju!=b'S'||lv>=r),12=>lwork== -1||lwork>=minwork);
    } else {
        valid!(info,1=>matches!(ju,b'N'|b'S'|b'A'|b'O'),2=>matches!(jv,b'N'|b'S'|b'A'|b'O')&&!(ju==b'O'&&jv==b'O'),3=>m>=0,4=>n>=0,6=>la>=1.max(m),9=>lu>=1&&(!matches!(ju,b'A'|b'S')||lu>=m),11=>lv>=1&&(jv!=b'A'||lv>=n)&&(jv!=b'S'||lv>=r),13=>lwork== -1||lwork>=minwork);
    }
    *work = workspace::<T>(minwork);
    if lwork == -1 {
        return;
    }
    let ucols = if ju == b'N' {
        0
    } else if ju == b'A' || (dd && ju == b'O' && m < n) {
        m
    } else {
        r
    } as usize;
    let vcols = if (dd && ju == b'N') || (!dd && jv == b'N') {
        0
    } else if (dd && (ju == b'A' || (ju == b'O' && m >= n))) || (!dd && jv == b'A') {
        n
    } else {
        r
    } as usize;
    if r == 0 {
        let left = Mat::<T>::identity(m as usize, ucols);
        let right = Mat::<T>::identity(n as usize, vcols);
        if ucols > 0 {
            let (dest, ld) = if ju == b'O' && (!dd || m >= n) {
                (a, la)
            } else {
                (u, lu)
            };
            output(&left, dest, ld);
        }
        if vcols > 0 {
            let (dest, ld) = if (!dd && jv == b'O') || (dd && ju == b'O' && m < n) {
                (a, la)
            } else {
                (vt, lv)
            };
            for j in 0..n as usize {
                for i in 0..vcols {
                    *dest.offset(i as isize + j as isize * ld) = right[(j, i)].cj();
                }
            }
        }
        return;
    }
    let mat = input_ref(a, m as usize, n as usize, la);
    // The native driver checks the input with a full `norm_max` scan before it
    // writes anything, so an explicit finiteness pass here would only repeat it.
    let left_direct = ucols > 0 && lu >= m && !(ju == b'O' && (!dd || m >= n));
    let right_direct =
        vcols > 0 && lv >= vcols as isize && !((!dd && jv == b'O') || (dd && ju == b'O' && m < n));
    // Real singular values share the caller's scalar type: let the native driver
    // write them straight into LAPACK's S array. Complex values need one scratch
    // diagonal, whose real parts are copied out below.
    let mut complex_values = if T::COMPLEX {
        vec![T::z(); r as usize]
    } else {
        Vec::new()
    };
    let values: &mut [T] = if T::COMPLEX {
        &mut complex_values
    } else {
        unsafe { std::slice::from_raw_parts_mut(s.cast::<T>(), r as usize) }
    };
    let mut left_store: Option<Mat<T>> = None;
    let mut left_view = if ucols == 0 {
        None
    } else if left_direct {
        Some(unsafe { MatMut::from_raw_parts_mut(u, m as usize, ucols, 1, lu) })
    } else {
        left_store = Some(Mat::<T>::zeros(m as usize, ucols));
        left_store.as_mut().map(|left| left.as_mut())
    };
    let mut right_store: Option<Mat<T>> = None;
    let mut right_view = if vcols == 0 {
        None
    } else if right_direct {
        // LAPACK stores V^H, and the native driver produces V column-major. A
        // view of the destination with V's shape and transposed strides maps
        // entry (row, col) of V to `vt[col, row]`; for real scalars that is
        // already V^H, and complex entries are conjugated in place below.
        Some(unsafe { MatMut::from_raw_parts_mut(vt, n as usize, vcols, lv, 1) })
    } else {
        right_store = Some(Mat::<T>::zeros(n as usize, vcols));
        right_store.as_mut().map(|right| right.as_mut())
    };
    let status = svd_native(
        mat,
        ucols,
        vcols,
        colmat::diag::DiagMut::from_slice_mut(values),
        left_view.as_mut().map(|left| left.as_mut()),
        right_view.as_mut().map(|right| right.as_mut()),
    );
    drop(left_view);
    drop(right_view);
    if status.is_err() {
        *info = I::from_i64(1);
        return;
    }
    if T::COMPLEX {
        for (i, value) in complex_values.iter().enumerate() {
            *s.add(i) = R::real(value.re());
        }
    }
    if let Some(left) = left_store.as_ref() {
        let (dest, ld) = if ju == b'O' && (!dd || m >= n) {
            (a, la)
        } else {
            (u, lu)
        };
        output(left, dest, ld);
    }
    if right_direct {
        if T::COMPLEX {
            // The native driver wrote V through the transposed view, so the
            // destination holds V^T unconjugated; conjugate it in place.
            for j in 0..n as usize {
                for i in 0..vcols {
                    let place = vt.offset(i as isize + j as isize * lv);
                    *place = (*place).cj();
                }
            }
        }
    } else if let Some(right) = right_store.as_ref() {
        let (dest, ld) = if (!dd && jv == b'O') || (dd && ju == b'O' && m < n) {
            (a, la)
        } else {
            (vt, lv)
        };
        for j in 0..n as usize {
            for i in 0..vcols {
                *dest.offset(i as isize + j as isize * ld) = right[(j, i)].cj();
            }
        }
    }
}
macro_rules! svd_exports {
($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$($extra:ident:$ext:ty),*)=>{paste::paste!{
 #[no_mangle]
 pub unsafe extern "C" fn [<$p gesdd $s>](j:*const u8,m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,values:*mut $r,u:*mut $t,ldu:*const $i,vt:*mut $t,ldvt:*const $i,work:*mut $t,lwork:*const $i,$($extra:$ext,)*_iwork:*mut $i,info:*mut $i){
  $(let _=$extra;)*svd_driver(true,flag(j),flag(j),(*m).val(),(*n).val(),a,(*lda).val(),values,u,(*ldu).val(),vt,(*ldvt).val(),work,(*lwork).val(),info);
 }
 #[no_mangle]
 pub unsafe extern "C" fn [<$p gesvd $s>](ju:*const u8,jv:*const u8,m:*const $i,n:*const $i,a:*mut $t,lda:*const $i,values:*mut $r,u:*mut $t,ldu:*const $i,vt:*mut $t,ldvt:*const $i,work:*mut $t,lwork:*const $i,$($extra:$ext,)*info:*mut $i){
  $(let _=$extra;)*svd_driver(false,flag(ju),flag(jv),(*m).val(),(*n).val(),a,(*lda).val(),values,u,(*ldu).val(),vt,(*ldvt).val(),work,(*lwork).val(),info);
 }
}};}
svd_exports!(s, f32, f32, i32, _,);
svd_exports!(d, f64, f64, i32, _,);
svd_exports!(c,colmat::c32,f32,i32,_,rwork:*mut f32);
svd_exports!(z,colmat::c64,f64,i32,_,rwork:*mut f64);
svd_exports!(s, f32, f32, i64, _64_,);
svd_exports!(d, f64, f64, i64, _64_,);
svd_exports!(c,colmat::c32,f32,i64,_64_,rwork:*mut f32);
svd_exports!(z,colmat::c64,f64,i64,_64_,rwork:*mut f64);

pub struct GeneralResult<T: Scalar> {
    pub(crate) w: Vec<T>,
    pub(crate) wi: Vec<f64>,
    pub(crate) left: Mat<T>,
    pub(crate) right: Mat<T>,
}
pub trait General: crate::general_evd::SharedEvd {
    fn balance(a: MatMut<'_, Self>, job: u8) -> Result<(Vec<usize>, Vec<f64>, usize, usize), ()>;
}
macro_rules! balance_impl {
    ($t:ty,$r:ty) => {
        fn balance(
            a: MatMut<'_, Self>,
            job: u8,
        ) -> Result<(Vec<usize>, Vec<f64>, usize, usize), ()> {
            use colmat::linalg::evd::*;
            let n = a.nrows();
            let mut p = vec![0; n];
            let mut s = vec![0.0 as $r; n];
            let mut mem = MemBuffer::new(balance_scratch());
            let range = balance_in_place_with(
                a,
                &mut p,
                &mut s,
                match job {
                    b'P' => BalanceJob::Permute,
                    b'S' => BalanceJob::Scale,
                    b'B' => BalanceJob::Both,
                    _ => BalanceJob::None,
                },
                colmat::Par::Seq,
                MemStack::new(&mut mem),
            )
            .map_err(|_| ())?;
            Ok((
                p,
                s.into_iter().map(|x| x as f64).collect(),
                range.start,
                if n > 0 && range.start == range.end {
                    range.start + 1
                } else {
                    range.end
                },
            ))
        }
    };
}
impl General for f32 {
    balance_impl!(f32, f32);
}
impl General for f64 {
    balance_impl!(f64, f64);
}
impl General for colmat::c32 {
    balance_impl!(colmat::c32, f32);
}
impl General for colmat::c64 {
    balance_impl!(colmat::c64, f64);
}

/// Undo balancing on `v` in place and restore LAPACK's eigenvector
/// normalization: every column has unit 2-norm and the largest component of a
/// conjugate pair is real.
///
/// The Schur eigenvector routine already normalizes, so the second normalization
/// is only needed when a scale factor actually changed the rows.
fn unbalance<T: crate::householder::ReflectorScalar>(
    v: &mut Mat<T>,
    p: &[usize],
    s: &[f64],
    left: bool,
    wi: &[f64],
) {
    let n = v.nrows();
    let m = v.ncols();
    let identity = p.iter().enumerate().all(|(i, &x)| x == i);
    let scaled = s.iter().any(|&x| x != 1.0);
    let factor = |i: usize| T::real(if left { 1.0 / s[i] } else { s[i] });
    if identity {
        if scaled {
            for i in 0..n {
                let f = factor(i);
                for j in 0..m {
                    v[(i, j)] = v[(i, j)] * f;
                }
            }
        }
    } else {
        let source = v.clone();
        for j in 0..m {
            for i in 0..n {
                v[(p[i], j)] = source[(i, j)] * factor(i);
            }
        }
    }
    if !identity || scaled {
        use colmat::traits::ext::ComplexFieldExt;
        use colmat::traits::math_utils::{hypot, recip, zero};
        let mut j = 0;
        while j < m {
            let paired = !T::COMPLEX && wi[j] > 0.0 && j + 1 < m;
            let norm = if paired {
                hypot(&v.col(j).norm_l2(), &v.col(j + 1).norm_l2())
            } else {
                v.col(j).norm_l2()
            };
            if norm != zero() {
                let inv = recip(&norm);
                for c in j..j + if paired { 2 } else { 1 } {
                    for i in 0..n {
                        v[(i, c)] = v[(i, c)].mul_real(&inv);
                    }
                }
            }
            j += if paired { 2 } else { 1 };
        }
    }
    let mut j = 0;
    while j < m {
        let paired = !T::COMPLEX && wi[j] > 0.0 && j + 1 < m;
        if T::COMPLEX && n > 0 {
            let mut best = 0;
            for i in 1..n {
                if v[(i, j)].magnitude() > v[(best, j)].magnitude() {
                    best = i;
                }
            }
            let x = v[(best, j)];
            if x.magnitude() > 0.0 {
                let phase = x.cj() / T::real(x.magnitude());
                for i in 0..n {
                    v[(i, j)] *= phase;
                }
            }
        }
        if paired && n > 0 {
            let mut best = 0;
            for i in 1..n {
                if v[(i, j)].re().hypot(v[(i, j + 1)].re())
                    > v[(best, j)].re().hypot(v[(best, j + 1)].re())
                {
                    best = i;
                }
            }
            let (a, b) = (v[(best, j)].re(), v[(best, j + 1)].re());
            let r = a.hypot(b);
            if r > 0.0 {
                let (c, s) = (T::real(a / r), T::real(b / r));
                for i in 0..n {
                    let (x, y) = (v[(i, j)], v[(i, j + 1)]);
                    v[(i, j)] = x * c + y * s;
                    v[(i, j + 1)] = y * c - x * s;
                }
            }
        }
        j += if paired { 2 } else { 1 };
    }
}
unsafe fn general_driver<T: General>(
    job: u8,
    jl: u8,
    jr: u8,
    n: usize,
    a: *mut T,
    la: isize,
    vl: *mut T,
    ll: isize,
    vr: *mut T,
    lr: isize,
) -> Result<GeneralOutput<T>, ()> {
    if n == 0 {
        return Ok((vec![], vec![], 0, 0, vec![], 0.0));
    }
    let (p, s, lo, hi) = {
        let _measure = crate::evd_profile::Scope::new(0);
        T::balance(input_mut(a, n, n, la), job)?
    };
    let abnrm;
    let mut e = {
        let mat = input_ref(a, n, n, la);
        abnrm = {
            let _measure = crate::evd_profile::Scope::new(1);
            (0..n)
                .map(|j| (0..n).map(|i| mat[(i, j)].magnitude()).sum::<f64>())
                .fold(0.0f64, f64::max)
        };
        let _measure = crate::evd_profile::Scope::new(2);
        crate::general_evd::compute(input_mut(a, n, n, la), jl == b'V', jr == b'V')?
    };
    if jl == b'V' {
        {
            let _measure = crate::evd_profile::Scope::new(5);
            unbalance(&mut e.left, &p, &s, true, &e.wi);
        }
        let _measure = crate::evd_profile::Scope::new(6);
        output(&e.left, vl, ll);
    }
    if jr == b'V' {
        {
            let _measure = crate::evd_profile::Scope::new(7);
            unbalance(&mut e.right, &p, &s, false, &e.wi);
        }
        let _measure = crate::evd_profile::Scope::new(8);
        output(&e.right, vr, lr);
    }
    let encoded = {
        let _measure = crate::evd_profile::Scope::new(9);
        crate::balance::encode_scale(&p, s, lo, hi)
    };
    Ok((e.w, e.wi, lo, hi, encoded, abnrm))
}

macro_rules! eigen_exports{
($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr,($($wi:ident:$wity:ty,)?),($($extra:ident:$ext:ty,)*),($($gextra:ident:$gext:ty,)*))=>{paste::paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p geevx $s>](bal:*const u8,jl:*const u8,jr:*const u8,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,w:*mut $t,$($wi:$wity,)?vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,ilo:*mut $i,ihi:*mut $i,scale:*mut $r,abnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,work:*mut $t,lwork:*const $i,$($extra:$ext,)*info:*mut $i){
 let _measure_total=crate::evd_profile::Scope::new(14);
 let (b0,l0,r0,se,n0,la0,ll0,lr0,lw0)=(flag(bal),flag(jl),flag(jr),flag(sense),(*n).val(),(*lda).val(),(*ldvl).val(),(*ldvr).val(),(*lwork).val());
 let offset=if <$t>::COMPLEX{0}else{1};
 valid!(info,1=>matches!(b0,b'N'|b'P'|b'S'|b'B'),2=>matches!(l0,b'N'|b'V'),3=>matches!(r0,b'N'|b'V'),4=>matches!(se,b'N'|b'E'|b'V'|b'B') && (!matches!(se,b'E'|b'B') || (l0==b'V' && r0==b'V')),5=>n0>=0,7=>la0>=n0.max(1),10+offset=>ll0>=1 && (l0!=b'V'||ll0>=n0),12+offset=>lr0>=1 && (r0!=b'V'||lr0>=n0));
 if se!=b'N' {
  if lw0 != -1 {
   let Some(minimum)=crate::geevx_expert::minimum(n0,<$t>::COMPLEX,l0==b'V',r0==b'V',se) else {bad(&mut *info,5);return;};
   if lw0<minimum {*work=workspace::<$t>(minimum);bad(&mut *info,20+offset);return;}
   valid!(info,7=>crate::kernels::matrix_addressable::<$t>(n0,n0,la0),10+offset=>l0!=b'V'||crate::kernels::matrix_addressable::<$t>(n0,n0,ll0),12+offset=>r0!=b'V'||crate::kernels::matrix_addressable::<$t>(n0,n0,lr0));
  }
  let ptr=crate::geevx_expert::pointer($id,if core::mem::size_of::<$i>()==8 {64}else{32});
  if !ptr.is_null() {
   let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const $i,*mut $t,*const $i,*mut $t,$($wity,)?*mut $t,*const $i,*mut $t,*const $i,*mut $i,*mut $i,*mut $r,*mut $r,*mut $r,*mut $r,*mut $t,*const $i,$($ext,)* *mut $i,usize,usize,usize,usize)=core::mem::transmute(ptr);
   f(bal,jl,jr,sense,n,a,lda,w,$($wi,)?vl,ldvl,vr,ldvr,ilo,ihi,scale,abnrm,rconde,rcondv,work,lwork,$($extra,)*info,1,1,1,1);
   if (*info).val()<0 {crate::invalid(-(*info).val() as i32);}return;
  }
  let ptr=crate::geevx_expert::pointer($id,64);
  if ptr.is_null() {
   // Expert calls require initialization through load_backend. Preserve INFO
   // rather than fabricating a LAPACK value for a missing implementation.
   crate::invalid(-1);return;
  }
  let (n64,la64,ll64,lr64,lw64)=(n0 as i64,la0 as i64,ll0 as i64,lr0 as i64,lw0 as i64);
  let (mut lo64,mut hi64,mut info64)=(i64::MIN,i64::MIN,i64::MIN);
  $(let (extra64,extra_state)=<$ext as crate::geevx_expert::Auxiliary>::widen($extra,n0 as usize,lw0== -1,matches!(se,b'V'|b'B'));)*
  let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const i64,*mut $t,*const i64,*mut $t,$($wity,)?*mut $t,*const i64,*mut $t,*const i64,*mut i64,*mut i64,*mut $r,*mut $r,*mut $r,*mut $r,*mut $t,*const i64,$(<$ext as crate::geevx_expert::Auxiliary>::Wide,)* *mut i64,usize,usize,usize,usize)=core::mem::transmute(ptr);
  f(bal,jl,jr,sense,&n64,a,&la64,w,$($wi,)?vl,&ll64,vr,&lr64,&mut lo64,&mut hi64,scale,abnrm,rconde,rcondv,work,&lw64,$( { let _ = $extra; extra64 },)*&mut info64,1,1,1,1);
  if [lo64,hi64,info64].iter().any(|&v|v!=i64::MIN && <$i>::try_from(v).is_err()){crate::invalid(-2);return;}
  $(if !<$ext as crate::geevx_expert::Auxiliary>::commit($extra,extra_state){crate::invalid(-2);return;})*
  if lo64!=i64::MIN {*ilo=lo64 as $i;}if hi64!=i64::MIN {*ihi=hi64 as $i;}
  if info64!=i64::MIN {*info=info64 as $i;if info64<0{crate::invalid(-info64 as i32);}}
  return;
 }
 *info=0;let(bal,jl,jr,nn,la,ll,lr,lw)=(flag(bal),flag(jl),flag(jr),(*n).val(),(*lda).val(),(*ldvl).val(),(*ldvr).val(),(*lwork).val());let off=if <$t>::COMPLEX{0}else{1};
 valid!(info,1=>matches!(bal,b'N'|b'P'|b'S'|b'B'),2=>matches!(jl,b'N'|b'V'),3=>matches!(jr,b'N'|b'V'),4=>flag(sense)==b'N',5=>nn>=0,7=>la>=1.max(nn),10+off=>ll>=1&&(jl!=b'V'||ll>=nn),12+off=>lr>=1&&(jr!=b'V'||lr>=nn),20+off=>lw== -1||lw>=1.max(if <$t>::COMPLEX{2*nn}else if jl==b'V'||jr==b'V'{3*nn}else{2*nn}));
 *work=workspace::<$t>(1.max(4*nn));if lw== -1{return;}
 if nn==0{return;}
 match general_driver(bal,jl,jr,nn as usize,a,la,vl,ll,vr,lr){Err(_)=>*info=1,Ok((values,imag,lo,hi,sc,an))=>{let _=&imag;for i in 0..nn as usize{*w.add(i)=values[i];$(*$wi.add(i)=imag[i]as $r;)?*scale.add(i)=sc[i]as $r;}*ilo=(lo+1)as $i;*ihi=hi as $i;*abnrm=an as $r;}}
}
#[no_mangle]
pub unsafe extern "C" fn [<$p geev $s>](jl:*const u8,jr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,w:*mut $t,$($wi:$wity,)?vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,work:*mut $t,lwork:*const $i,$($gextra:$gext,)*info:*mut $i){
 let _measure_total=crate::evd_profile::Scope::new(14);
 *info=0;let(jl,jr,nn,la,ll,lr,lw)=(flag(jl),flag(jr),(*n).val(),(*lda).val(),(*ldvl).val(),(*ldvr).val(),(*lwork).val());let off=if <$t>::COMPLEX{0}else{1};
 valid!(info,1=>matches!(jl,b'N'|b'V'),2=>matches!(jr,b'N'|b'V'),3=>nn>=0,5=>la>=1.max(nn),8+off=>ll>=1&&(jl!=b'V'||ll>=nn),10+off=>lr>=1&&(jr!=b'V'||lr>=nn),12+off=>lw== -1||lw>=1.max(if <$t>::COMPLEX{2*nn}else if jl==b'V'||jr==b'V'{4*nn}else{3*nn}));
 *work=workspace::<$t>(1.max(4*nn));if lw== -1{return;}
 match general_driver(b'B',jl,jr,nn as usize,a,la,vl,ll,vr,lr){Err(_)=>*info=1,Ok((values,imag,_,_,_,_))=>{let _=&imag;for i in 0..nn as usize{*w.add(i)=values[i];$(*$wi.add(i)=imag[i]as $r;)?}}}
 $(let _=$gextra;)*
}
}};}
eigen_exports!(s,f32,f32,i32,_,0,(wi:*mut f32,),(iwork:*mut i32,),());
eigen_exports!(d,f64,f64,i32,_,1,(wi:*mut f64,),(iwork:*mut i32,),());
eigen_exports!(c,colmat::c32,f32,i32,_,2,(),(rwork:*mut f32,),(rwork:*mut f32,));
eigen_exports!(z,colmat::c64,f64,i32,_,3,(),(rwork:*mut f64,),(rwork:*mut f64,));
eigen_exports!(s,f32,f32,i64,_64_,0,(wi:*mut f32,),(iwork:*mut i64,),());
eigen_exports!(d,f64,f64,i64,_64_,1,(wi:*mut f64,),(iwork:*mut i64,),());
eigen_exports!(c,colmat::c32,f32,i64,_64_,2,(),(rwork:*mut f32,),(rwork:*mut f32,));
eigen_exports!(z,colmat::c64,f64,i64,_64_,3,(),(rwork:*mut f64,),(rwork:*mut f64,));

pub unsafe fn symmetric_driver<T: Scalar, R: Scalar, I: LapackInt>(
    job: u8,
    range: u8,
    u: u8,
    n: usize,
    a: *mut T,
    la: isize,
    vl: f64,
    vu: f64,
    il: usize,
    iu: usize,
    m: *mut I,
    w: *mut R,
    z: *mut T,
    lz: isize,
    support: *mut I,
) -> Result<(), ()> {
    let mat = if u == b'L' {
        input_ref(a, n, n, la)
    } else {
        // The transpose view exposes the stored upper triangle as a lower
        // triangle. Its eigenvectors are the conjugates of the caller's, so
        // they are conjugated again before being written out.
        input_ref_transposed(a, n, la)
    };
    let (values, mut vectors) = selfadj(mat, job == b'V')?;
    if range == b'V'
        && values.len() > 1
        && crate::selfadjoint_range::uncertain_endpoint::<T>(&values, vl, vu)
    {
        // No caller outputs have been written. The range driver can still
        // recover with its saved implementation using the original matrix.
        return Err(());
    }
    if u != b'L' {
        for j in 0..vectors.ncols() {
            for i in 0..vectors.nrows() {
                vectors[(i, j)] = vectors[(i, j)].cj();
            }
        }
    }
    let selected: Vec<usize> = (0..n)
        .filter(|&i| {
            range == b'A'
                || (range == b'I' && i + 1 >= il && i < iu)
                || (range == b'V' && values[i] > vl && values[i] <= vu)
        })
        .collect();
    *m = I::from_i64(selected.len() as i64);
    for (j, &i) in selected.iter().enumerate() {
        *w.add(j) = R::real(values[i]);
        if job == b'V' {
            for r in 0..n {
                *z.offset(r as isize + j as isize * lz) = vectors[(r, i)];
            }
            if !support.is_null() {
                *support.add(2 * j) = I::from_i64(1);
                *support.add(2 * j + 1) = I::from_i64(n as i64);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod balanced_normalization_regression {
    use super::*;
    use crate::householder::ReflectorScalar;

    fn check<T: ReflectorScalar>(exponent: i32) {
        let vectors = Mat::from_fn(2, 2, |i, j| {
            if i == 0 {
                T::parts(if j == 0 { 0.6 } else { -0.6 }, 0.8)
            } else {
                T::parts(0.3, -0.5)
            }
        });
        for left in [false, true] {
            let factor = 2.0f64.powi(if left { -exponent } else { exponent });
            let mut output = vectors.clone();
            unbalance(&mut output, &[0, 1], &[factor, 1.0], left, &[0.0, 0.0]);
            for j in 0..2 {
                let mut norm = 0.0f64;
                for i in 0..2 {
                    let z = output[(i, j)];
                    assert!(
                        z.re().is_finite() && z.im().is_finite(),
                        "non-finite normalized component"
                    );
                    norm = norm.hypot(z.magnitude());
                }
                assert!((norm - 1.0).abs() <= 32.0 * T::EPS);
                assert!((output[(0, j)].re() - 1.0).abs() <= 32.0 * T::EPS);
                assert!(output[(0, j)].im().abs() <= 32.0 * T::EPS);
            }
        }
    }
    #[test]
    fn complex_balanced_vectors_keep_finite_unit_norm_at_large_scale() {
        check::<colmat::c32>(80);
        check::<colmat::c64>(664);
    }
}

#[cfg(test)]
mod shared_evd_regression {
    use super::*;
    use colmat::c64;

    fn z<T: General>(x: T) -> c64 {
        c64::new(x.re(), x.im())
    }
    fn check<T: General>(exponents: &[i32]) {
        let n = 6;
        for &exponent in exponents {
            let scale = 2.0f64.powi(exponent);
            let original = Mat::from_fn(n, n, |i, j| {
                let re = if i == j {
                    (i + 1) as f64 / 3.0
                } else if i / 2 == j / 2 {
                    if i < j {
                        -2.0
                    } else {
                        2.0
                    }
                } else {
                    ((i * 3 + j * 7 + 2) % 11) as f64 / 17.0 - 0.3
                };
                T::parts(
                    re * scale,
                    if T::COMPLEX {
                        ((i * 5 + j * 3) % 7) as f64 / 13.0 * scale
                    } else {
                        0.0
                    },
                )
            });
            for job in [b'N', b'P', b'S', b'B'] {
                for (left, right) in [(false, false), (true, false), (false, true), (true, true)] {
                    let mut a = original.clone();
                    let mut vl = Mat::<T>::zeros(n, n);
                    let mut vr = Mat::<T>::zeros(n, n);
                    let (w, wi, _, _, _, _) = unsafe {
                        general_driver(
                            job,
                            if left { b'V' } else { b'N' },
                            if right { b'V' } else { b'N' },
                            n,
                            a.as_mut().as_ptr_mut(),
                            a.as_ref().col_stride(),
                            vl.as_mut().as_ptr_mut(),
                            vl.as_ref().col_stride(),
                            vr.as_mut().as_ptr_mut(),
                            vr.as_ref().col_stride(),
                        )
                    }
                    .expect("shared EVD converges");
                    for j in 0..n {
                        for i in j + if T::COMPLEX { 1 } else { 2 }..n {
                            assert_eq!(a[(i, j)], T::z());
                        }
                    }
                    for (wanted, is_left, vectors) in [(left, true, &vl), (right, false, &vr)] {
                        if !wanted {
                            continue;
                        }
                        let mut j = 0;
                        while j < n {
                            let pair = !T::COMPLEX && wi[j] > 0.0 && j + 1 < n;
                            let lambda = if T::COMPLEX {
                                z(w[j])
                            } else {
                                c64::new(w[j].re(), wi[j])
                            };
                            let lambda = if is_left { lambda.conj() } else { lambda };
                            let v: Vec<c64> = (0..n)
                                .map(|i| {
                                    if pair {
                                        c64::new(vectors[(i, j)].re(), vectors[(i, j + 1)].re())
                                    } else {
                                        z(vectors[(i, j)])
                                    }
                                })
                                .collect();
                            let mut residual = 0.0f64;
                            let mut norm = 0.0f64;
                            for i in 0..n {
                                norm = norm.hypot(v[i].norm());
                                let mut av = c64::new(0.0, 0.0);
                                for k in 0..n {
                                    let x = if is_left {
                                        z(original[(k, i)]).conj()
                                    } else {
                                        z(original[(i, k)])
                                    };
                                    av += c64::new(x.re / scale, x.im / scale) * v[k];
                                }
                                let lam = c64::new(lambda.re / scale, lambda.im / scale);
                                residual = residual.hypot((av - lam * v[i]).norm());
                            }
                            assert!(
                                (norm - 1.0).abs() < 100.0 * T::EPS,
                                "norm={norm}, exponent={exponent}"
                            );
                            // Subnormal eigenvalue restoration adds absolute component rounding.
                            let quantum = T::SAFE * (T::EPS / 2.0) * T::EPS;
                            let tolerance = n as f64 * (1000.0 * T::EPS + quantum / scale);
                            assert!(residual < tolerance, "residual={residual}, tolerance={tolerance}, left={is_left}, pair={pair}, exponent={exponent}, job={job}");
                            j += if pair { 2 } else { 1 };
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn shared_general_real32() {
        check::<f32>(&[0, 100, -140]);
    }
    #[test]
    fn shared_general_real64() {
        check::<f64>(&[0, 900, -1000]);
    }
    #[test]
    fn shared_general_complex32() {
        check::<colmat::c32>(&[0, 100, -140]);
    }
    #[test]
    fn shared_general_complex64() {
        check::<colmat::c64>(&[0, 900, -1000]);
    }
}

#[cfg(test)]
mod cholesky_triangle_regression {
    use super::*;
    use colmat::{c64, Mat};

    fn spd_f64(n: usize) -> Mat<f64> {
        let l = Mat::<f64>::from_fn(n, n, |i, j| {
            if i == j {
                2.0 + i as f64 * 0.01
            } else if i > j {
                ((i * 7 + j * 3) % 11) as f64 / 23.0 - 0.2
            } else {
                0.0
            }
        });
        Mat::from_fn(n, n, |i, j| (0..n).map(|k| l[(i, k)] * l[(j, k)]).sum())
    }

    fn spd_c64(n: usize) -> Mat<c64> {
        let l = Mat::<c64>::from_fn(n, n, |i, j| {
            let re = if i == j {
                2.0 + i as f64 * 0.01
            } else if i > j {
                ((i * 7 + j * 3) % 11) as f64 / 23.0 - 0.2
            } else {
                0.0
            };
            let im = if i > j {
                ((i * 5 + j * 2) % 7) as f64 / 31.0 - 0.1
            } else {
                0.0
            };
            c64::new(re, im)
        });
        Mat::from_fn(n, n, |i, j| {
            (0..n).map(|k| l[(i, k)] * l[(j, k)].conj()).sum()
        })
    }

    fn check<T: Scalar>(original: &Mat<T>) {
        let n = original.nrows();
        for u in [b'L', b'U'] {
            let selected = |i: usize, j: usize| {
                if u == b'L' {
                    i >= j
                } else {
                    i <= j
                }
            };
            let mut a = Mat::<T>::from_fn(n, n, |i, j| {
                if selected(i, j) {
                    original[(i, j)]
                } else {
                    T::real(12345.0)
                }
            });
            let opposite_before: Vec<T> = (0..n)
                .flat_map(|j| (0..n).map(move |i| (i, j)))
                .filter(|&(i, j)| !selected(i, j))
                .map(|(i, j)| a[(i, j)])
                .collect();
            let info =
                unsafe { factor_cholesky(u, n, a.as_mut().as_ptr_mut(), a.as_ref().col_stride()) };
            assert_eq!(info, 0);
            let opposite_after: Vec<T> = (0..n)
                .flat_map(|j| (0..n).map(move |i| (i, j)))
                .filter(|&(i, j)| !selected(i, j))
                .map(|(i, j)| a[(i, j)])
                .collect();
            assert_eq!(opposite_before, opposite_after, "u={}", u as char);
            // Reconstruct A from the selected factor: L L^H or U^H U.
            let mut scale = 0.0f64;
            let mut error = 0.0f64;
            for j in 0..n {
                for i in 0..n {
                    let mut sum = T::z();
                    for k in 0..n.min(i.min(j) + 1) {
                        sum += if u == b'L' {
                            a[(i, k)] * a[(j, k)].cj()
                        } else {
                            a[(k, i)].cj() * a[(k, j)]
                        };
                    }
                    error = error.max((sum - original[(i, j)]).abs1());
                    scale = scale.max(original[(i, j)].abs1());
                }
            }
            assert!(
                error <= 1e-10 * scale.max(1.0),
                "u={} error={error}",
                u as char
            );
        }
    }

    #[test]
    fn both_triangles_match_the_factor_contract() {
        check(&spd_f64(37));
        check(&spd_c64(37));
    }

    /// The 37-square case stays inside one leaf of the blocked recursion; at
    /// 257 the factorization splits into leaves and runs its trailing updates,
    /// so the preserved-opposite-triangle contract has to hold there too.
    #[test]
    fn blocked_triangle_matches_the_factor_contract() {
        check(&spd_f64(257));
        check(&spd_c64(257));
    }
}
