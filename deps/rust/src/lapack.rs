//! LAPACK drivers over colmat's public decomposition and solve APIs.
use crate::kernels::{flag, Scalar};
use colmat::dyn_stack::{MemBuffer, MemStack};
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
pub unsafe fn output<T: Scalar>(a: &Mat<T>, p: *mut T, ld: isize) {
    // The destination is column-major, so a contiguous source is a single copy.
    if a.row_stride() == 1 && a.col_stride() == a.nrows() as isize && ld == a.nrows() as isize
    {
        std::ptr::copy_nonoverlapping(a.as_ptr(), p, a.nrows() * a.ncols());
        return;
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            *p.offset(i as isize + j as isize * ld) = a[(i, j)];
        }
    }
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
    crate::threading::with_par(|par| {
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
pub fn cholesky<T: Scalar>(mut a: MatMut<'_, T>) -> usize {
    use colmat::linalg::cholesky::llt::factor::*;
    crate::threading::with_par(|par| {
        let mut mem = MemBuffer::new(cholesky_in_place_scratch::<T>(
            a.nrows(),
            par.shape(),
            Default::default(),
        ));
        match cholesky_in_place(
            a.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        ) {
            Ok(_) => 0,
            Err(LltError::NonPositivePivot { index }) => index + 1,
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

pub fn svd<T: Scalar>(a: MatRef<'_, T>, u: usize, v: usize) -> Result<SvdResult<T>, ()> {
    use colmat::{diag::Diag, linalg::svd::*};
    let (m, n) = (a.nrows(), a.ncols());
    let mut s = Diag::<T>::zeros(m.min(n));
    let mut left = Mat::<T>::zeros(m, u);
    let mut right = Mat::<T>::zeros(n, v);
    crate::threading::with_par(|par| {
        let cv = |x: usize, dim: usize| {
            if x == 0 {
                ComputeSvdVectors::No
            } else if x == dim {
                ComputeSvdVectors::Full
            } else {
                ComputeSvdVectors::Thin
            }
        };
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
            if u == 0 { None } else { Some(left.as_mut()) },
            if v == 0 { None } else { Some(right.as_mut()) },
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| ())?;
        Ok((left, (0..m.min(n)).map(|i| s[i].re()).collect(), right))
    })
}
pub fn selfadj<T: Scalar>(a: MatRef<'_, T>, vectors: bool) -> Result<(Vec<f64>, Mat<T>), ()> {
    use colmat::{diag::Diag, linalg::evd::*};
    let n = a.nrows();
    let mut s = Diag::<T>::zeros(n);
    let mut u = Mat::<T>::zeros(n, if vectors { n } else { 0 });
    crate::threading::with_par(|par| {
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
 *info=0;let(u,n,ld)=(flag(u),(*n).val(),(*lda).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,4=>ld>=1.max(n));if n==0{return;}let mat=if u==b'L'{input_mut(a,n as usize,n as usize,ld)}else{input_mut_transposed(a,n as usize,ld)};*info=cholesky(mat)as $i;
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
    let result = if r == 0 {
        Ok((
            Mat::<T>::identity(m as usize, ucols),
            vec![],
            Mat::<T>::identity(n as usize, vcols),
        ))
    } else {
        let mat = input_ref(a, m as usize, n as usize, la);
        if !mat.is_all_finite() {
            Err(())
        } else {
            svd(mat, ucols, vcols)
        }
    };
    match result {
        Err(_) => *info = I::from_i64(1),
        Ok((left, values, right)) => {
            for (i, v) in values.iter().enumerate() {
                *s.add(i) = R::real(*v);
            }
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
    w: Vec<T>,
    wi: Vec<f64>,
    left: Mat<T>,
    right: Mat<T>,
}
pub trait General: Scalar {
    fn eigen(a: MatRef<'_, Self>, left: bool, right: bool) -> Result<GeneralResult<Self>, ()>;
    fn balance(a: MatMut<'_, Self>, job: u8) -> Result<(Vec<usize>, Vec<f64>, usize, usize), ()>;
    fn schur(a: MatMut<'_, Self>) -> Result<(), ()>;
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
macro_rules! general_real {
    ($t:ty) => {
        impl General for $t {
            balance_impl!($t, $t);
            fn eigen(
                a: MatRef<'_, Self>,
                left: bool,
                right: bool,
            ) -> Result<GeneralResult<Self>, ()> {
                use colmat::{diag::Diag, linalg::evd::*};
                let n = a.nrows();
                let mut wr = Diag::<$t>::zeros(n);
                let mut wi = Diag::<$t>::zeros(n);
                let mut vl = Mat::<$t>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<$t>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let yes = |v| {
                        if v {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(evd_scratch::<$t>(
                        n,
                        yes(left),
                        yes(right),
                        par.shape(),
                        Default::default(),
                    ));
                    evd_real(
                        a.as_ref(),
                        wr.as_mut(),
                        wi.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(GeneralResult {
                    w: (0..n).map(|i| wr[i]).collect(),
                    wi: (0..n).map(|i| wi[i] as f64).collect(),
                    left: vl,
                    right: vr,
                })
            }
            fn schur(mut a: MatMut<'_, Self>) -> Result<(), ()> {
                use colmat::linalg::{
                    evd::{hessenberg::*, *},
                    qr::no_pivoting::factor::recommended_block_size,
                };
                use colmat::Col;
                let n = a.nrows();
                if n < 2 {
                    return Ok(());
                }
                crate::threading::with_par(|par| {
                    let bs = recommended_block_size::<Self>(n - 1, n - 1);
                    let mut coeff = Mat::<Self>::zeros(bs, n - 1);
                    let mut mem = MemBuffer::new(hessenberg_in_place_scratch::<Self>(
                        n,
                        bs,
                        par.shape(),
                        Default::default(),
                    ));
                    hessenberg_in_place(
                        a.as_mut(),
                        coeff.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    );
                });
                for j in 0..n {
                    for i in j + 2..n {
                        a[(i, j)] = Self::z();
                    }
                }
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<Self>(
                        n,
                        false,
                        par.shape(),
                        Default::default(),
                    ));
                    let mut rw = Col::<Self>::zeros(n);
                    let mut iw = Col::<Self>::zeros(n);
                    hessenberg_schur_real(
                        a.as_mut(),
                        None,
                        rw.as_mut(),
                        iw.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })
            }
        }
    };
}
macro_rules! general_complex {
    ($t:ty,$r:ty) => {
        impl General for $t {
            balance_impl!($t, $r);
            fn eigen(
                a: MatRef<'_, Self>,
                left: bool,
                right: bool,
            ) -> Result<GeneralResult<Self>, ()> {
                use colmat::{diag::Diag, linalg::evd::*};
                let n = a.nrows();
                let mut w = Diag::<$t>::zeros(n);
                let mut vl = Mat::<$t>::zeros(n, if left { n } else { 0 });
                let mut vr = Mat::<$t>::zeros(n, if right { n } else { 0 });
                crate::threading::with_par(|par| {
                    let yes = |v| {
                        if v {
                            ComputeEigenvectors::Yes
                        } else {
                            ComputeEigenvectors::No
                        }
                    };
                    let mut mem = MemBuffer::new(evd_scratch::<$t>(
                        n,
                        yes(left),
                        yes(right),
                        par.shape(),
                        Default::default(),
                    ));
                    evd_cplx(
                        a.as_ref(),
                        w.as_mut(),
                        if left { Some(vl.as_mut()) } else { None },
                        if right { Some(vr.as_mut()) } else { None },
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })?;
                Ok(GeneralResult {
                    w: (0..n).map(|i| w[i]).collect(),
                    wi: vec![0.0; n],
                    left: vl,
                    right: vr,
                })
            }
            fn schur(mut a: MatMut<'_, Self>) -> Result<(), ()> {
                use colmat::linalg::{
                    evd::{hessenberg::*, *},
                    qr::no_pivoting::factor::recommended_block_size,
                };
                use colmat::Col;
                let n = a.nrows();
                if n < 2 {
                    return Ok(());
                }
                crate::threading::with_par(|par| {
                    let bs = recommended_block_size::<Self>(n - 1, n - 1);
                    let mut coeff = Mat::<Self>::zeros(bs, n - 1);
                    let mut mem = MemBuffer::new(hessenberg_in_place_scratch::<Self>(
                        n,
                        bs,
                        par.shape(),
                        Default::default(),
                    ));
                    hessenberg_in_place(
                        a.as_mut(),
                        coeff.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    );
                });
                for j in 0..n {
                    for i in j + 2..n {
                        a[(i, j)] = Self::z();
                    }
                }
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<Self>(
                        n,
                        false,
                        par.shape(),
                        Default::default(),
                    ));
                    let mut sw = Col::<Self>::zeros(n);
                    hessenberg_schur_cplx(
                        a.as_mut(),
                        None,
                        sw.as_mut(),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| ())
                })
            }
        }
    };
}
general_real!(f32);
general_real!(f64);
general_complex!(colmat::c32, f32);
general_complex!(colmat::c64, f64);

fn unbalance<T: Scalar>(v: &Mat<T>, p: &[usize], s: &[f64], left: bool, wi: &[f64]) -> Mat<T> {
    let n = v.nrows();
    let mut out = Mat::zeros(n, v.ncols());
    for j in 0..v.ncols() {
        for i in 0..n {
            out[(p[i], j)] = v[(i, j)] * T::real(if left { 1.0 / s[i] } else { s[i] });
        }
    }
    let mut j = 0;
    while j < out.ncols() {
        let paired = !T::COMPLEX && wi[j] > 0.0 && j + 1 < out.ncols();
        let cols = if paired { 2 } else { 1 };
        let mut norm = 0.0f64;
        for c in j..j + cols {
            for i in 0..n {
                norm = norm.hypot(out[(i, c)].magnitude());
            }
        }
        if norm != 0.0 {
            for c in j..j + cols {
                for i in 0..n {
                    out[(i, c)] /= T::real(norm);
                }
            }
        }
        if n > 0 && T::COMPLEX {
            let mut best = 0;
            for i in 1..n {
                if out[(i, j)].magnitude() > out[(best, j)].magnitude() {
                    best = i;
                }
            }
            let x = out[(best, j)];
            if x.magnitude() > 0.0 {
                let phase = x.cj() / T::real(x.magnitude());
                for i in 0..n {
                    out[(i, j)] *= phase;
                }
            }
        }
        if paired && n > 0 {
            let mut best = 0;
            for i in 1..n {
                if out[(i, j)].re().hypot(out[(i, j + 1)].re())
                    > out[(best, j)].re().hypot(out[(best, j + 1)].re())
                {
                    best = i;
                }
            }
            let (a, b) = (out[(best, j)].re(), out[(best, j + 1)].re());
            let r = a.hypot(b);
            if r > 0.0 {
                let (c, s) = (T::real(a / r), T::real(b / r));
                for i in 0..n {
                    let (x, y) = (out[(i, j)], out[(i, j + 1)]);
                    out[(i, j)] = x * c + y * s;
                    out[(i, j + 1)] = y * c - x * s;
                }
            }
        }
        j += cols;
    }
    out
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
    let (p, s, lo, hi) = T::balance(input_mut(a, n, n, la), job)?;
    let abnrm;
    let e = {
        let mat = input_ref(a, n, n, la);
        abnrm = (0..n)
            .map(|j| (0..n).map(|i| mat[(i, j)].magnitude()).sum::<f64>())
            .fold(0.0f64, f64::max);
        T::eigen(mat, jl == b'V', jr == b'V')?
    };
    T::schur(input_mut(a, n, n, la))?;
    if jl == b'V' {
        output(&unbalance(&e.left, &p, &s, true, &e.wi), vl, ll);
    }
    if jr == b'V' {
        output(&unbalance(&e.right, &p, &s, false, &e.wi), vr, lr);
    }
    let encoded = crate::balance::encode_scale(&p, s, lo, hi);
    Ok((e.w, e.wi, lo, hi, encoded, abnrm))
}

use core::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
static EXPERT_FALLBACKS: [AtomicPtr<c_void>; 8] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; 8];
#[no_mangle]
pub extern "C" fn colmat_lapack_set_expert_fallback(id: i32, bits: i32, p: *mut c_void) {
    if (0..4).contains(&id) {
        EXPERT_FALLBACKS[id as usize * 2 + usize::from(bits == 64)].store(p, Ordering::Relaxed);
    }
}

macro_rules! eigen_exports{
($p:ident,$t:ty,$r:ty,$i:ty,$s:tt,$id:expr,($($wi:ident:$wity:ty,)?),($($extra:ident:$ext:ty,)*),($($gextra:ident:$gext:ty,)*))=>{paste::paste!{
#[no_mangle]
pub unsafe extern "C" fn [<$p geevx $s>](bal:*const u8,jl:*const u8,jr:*const u8,sense:*const u8,n:*const $i,a:*mut $t,lda:*const $i,w:*mut $t,$($wi:$wity,)?vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,ilo:*mut $i,ihi:*mut $i,scale:*mut $r,abnrm:*mut $r,rconde:*mut $r,rcondv:*mut $r,work:*mut $t,lwork:*const $i,$($extra:$ext,)*info:*mut $i){
 if flag(sense)!=b'N'{let ptr=EXPERT_FALLBACKS[$id*2+usize::from(core::mem::size_of::<$i>()==8)].load(Ordering::Relaxed);if !ptr.is_null(){let f:unsafe extern "C" fn(*const u8,*const u8,*const u8,*const u8,*const $i,*mut $t,*const $i,*mut $t,$($wity,)?*mut $t,*const $i,*mut $t,*const $i,*mut $i,*mut $i,*mut $r,*mut $r,*mut $r,*mut $r,*mut $t,*const $i,$($ext,)* *mut $i,usize,usize,usize,usize)=core::mem::transmute(ptr);f(bal,jl,jr,sense,n,a,lda,w,$($wi,)?vl,ldvl,vr,ldvr,ilo,ihi,scale,abnrm,rconde,rcondv,work,lwork,$($extra,)*info,1,1,1,1);return;}}
 *info=0;let(bal,jl,jr,nn,la,ll,lr,lw)=(flag(bal),flag(jl),flag(jr),(*n).val(),(*lda).val(),(*ldvl).val(),(*ldvr).val(),(*lwork).val());let off=if <$t>::COMPLEX{0}else{1};
 valid!(info,1=>matches!(bal,b'N'|b'P'|b'S'|b'B'),2=>matches!(jl,b'N'|b'V'),3=>matches!(jr,b'N'|b'V'),4=>flag(sense)==b'N',5=>nn>=0,7=>la>=1.max(nn),10+off=>ll>=1&&(jl!=b'V'||ll>=nn),12+off=>lr>=1&&(jr!=b'V'||lr>=nn),20+off=>lw== -1||lw>=1.max(if <$t>::COMPLEX{2*nn}else if jl==b'V'||jr==b'V'{3*nn}else{2*nn}));
 *work=workspace::<$t>(1.max(4*nn));if lw== -1{return;}
 match general_driver(bal,jl,jr,nn as usize,a,la,vl,ll,vr,lr){Err(_)=>*info=1,Ok((values,imag,lo,hi,sc,an))=>{let _=&imag;for i in 0..nn as usize{*w.add(i)=values[i];$(*$wi.add(i)=imag[i]as $r;)?*scale.add(i)=sc[i]as $r;}*ilo=(lo+1)as $i;*ihi=hi as $i;*abnrm=an as $r;}}
}
#[no_mangle]
pub unsafe extern "C" fn [<$p geev $s>](jl:*const u8,jr:*const u8,n:*const $i,a:*mut $t,lda:*const $i,w:*mut $t,$($wi:$wity,)?vl:*mut $t,ldvl:*const $i,vr:*mut $t,ldvr:*const $i,work:*mut $t,lwork:*const $i,$($gextra:$gext,)*info:*mut $i){
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
