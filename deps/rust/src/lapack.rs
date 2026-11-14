//! LAPACK drivers over colmat's public decomposition and solve APIs.
use crate::kernels::{flag, Scalar};
use colmat::dyn_stack::{MemBuffer, MemStack};
use colmat::{Conj, Mat, MatRef};

pub unsafe fn input<T: Scalar>(a: *const T, m: usize, n: usize, ld: isize) -> Mat<T> {
    MatRef::from_raw_parts(a, m, n, 1, ld).to_owned()
}
pub unsafe fn output<T: Scalar>(a: &Mat<T>, p: *mut T, ld: isize) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            *p.offset(i as isize + j as isize * ld) = a[(i, j)];
        }
    }
}
pub fn lu<T: Scalar>(a: &mut Mat<T>) -> (Vec<usize>, usize) {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);
    if k == 0 {
        return (vec![], 0);
    }
    let f = crate::threading::with_par(|par| a.as_ref().partial_piv_lu(par));
    let (p, _) = f.P().arrays();
    let mut current: Vec<usize> = (0..m).collect();
    let mut piv = vec![0; k];
    let mut info = 0;
    for j in 0..k {
        let pos = (j..m).find(|&i| current[i] == p[j]).unwrap();
        piv[j] = pos;
        current.swap(j, pos);
    }
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = if i > j { f.L()[(i, j)] } else { f.U()[(i, j)] };
        }
    }
    for i in 0..k {
        if a[(i, i)] == T::z() {
            info = i + 1;
            break;
        }
    }
    (piv, info)
}
pub fn solve_lu<T: Scalar>(a: &Mat<T>, p: &[usize], b: &mut Mat<T>, t: u8) {
    use colmat::linalg::triangular_solve::*;
    let n = a.nrows();
    let swap = |b: &mut Mat<T>, i: usize, j: usize| {
        for c in 0..b.ncols() {
            let x = b[(i, c)];
            b[(i, c)] = b[(j, c)];
            b[(j, c)] = x;
        }
    };
    crate::threading::with_par(|mut par| {
        if t == b'N' {
            for (i, &j) in p.iter().enumerate() {
                swap(b, i, j);
            }
            solve_unit_lower_triangular_in_place(a.as_ref(), b.as_mut(), par.reborrow());
            solve_upper_triangular_in_place(a.as_ref(), b.as_mut(), par);
        } else {
            let conj = if t == b'C' { Conj::Yes } else { Conj::No };
            solve_lower_triangular_in_place_with_conj(
                a.as_ref().transpose(),
                conj,
                b.as_mut(),
                par.reborrow(),
            );
            solve_unit_upper_triangular_in_place_with_conj(
                a.as_ref().transpose(),
                conj,
                b.as_mut(),
                par,
            );
            for i in (0..n).rev() {
                swap(b, i, p[i]);
            }
        }
    });
}
pub fn cholesky<T: Scalar>(a: &mut Mat<T>) -> usize {
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
pub fn solve_cholesky<T: Scalar>(a: &Mat<T>, b: &mut Mat<T>) {
    use colmat::linalg::triangular_solve::*;
    crate::threading::with_par(|mut par| {
        solve_lower_triangular_in_place(a.as_ref(), b.as_mut(), par.reborrow());
        solve_upper_triangular_in_place_with_conj(
            a.as_ref().transpose(),
            Conj::Yes,
            b.as_mut(),
            par,
        );
    });
}
pub unsafe fn hermitian_input<T: Scalar>(a: *const T, n: usize, ld: isize, u: u8) -> Mat<T> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            T::real((*a.offset(i as isize + j as isize * ld)).re())
        } else if u == b'L' {
            if i > j {
                *a.offset(i as isize + j as isize * ld)
            } else {
                (*a.offset(j as isize + i as isize * ld)).cj()
            }
        } else if i < j {
            *a.offset(i as isize + j as isize * ld)
        } else {
            (*a.offset(j as isize + i as isize * ld)).cj()
        }
    })
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

pub fn svd<T: Scalar>(a: &Mat<T>, u: usize, v: usize) -> Result<SvdResult<T>, ()> {
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
pub fn selfadj<T: Scalar>(a: &Mat<T>, vectors: bool) -> Result<(Vec<f64>, Mat<T>), ()> {
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
 let mut mat=input(a,m as usize,n as usize,ld);let(p,err)=lu(&mut mat);output(&mat,a,ld);for(i,v)in p.iter().enumerate(){*piv.add(i)=<$i>::from_i64(*v as i64+1);}*info=err as $i;
}
#[no_mangle]
pub unsafe extern "C" fn [<$p getrs $s>](t:*const u8,n:*const $i,nrhs:*const $i,a:*const $t,lda:*const $i,piv:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;let(t,n,k,la,lb)=(flag(t),(*n).val(),(*nrhs).val(),(*lda).val(),(*ldb).val());valid!(info,1=>crate::kernels::trans(t),2=>n>=0,3=>k>=0,5=>la>=1.max(n),8=>lb>=1.max(n));if n==0||k==0{return;}
 let mat=input(a,n as usize,n as usize,la);let mut rhs=input(b,n as usize,k as usize,lb);let p:Vec<usize>=(0..n as usize).map(|i|(*piv.add(i)).val() as usize-1).collect();solve_lu(&mat,&p,&mut rhs,t);output(&rhs,b,lb);
}
#[no_mangle]
pub unsafe extern "C" fn [<$p gesv $s>](n:*const $i,k:*const $i,a:*mut $t,lda:*const $i,p:*mut $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;valid!(info,1=>*n>=0,2=>*k>=0,4=>*lda>=1.max(*n),7=>*ldb>=1.max(*n));[<$p getrf $s>](n,n,a,lda,p,info);if *info==0{[<$p getrs $s>](b"N".as_ptr(),n,k,a,lda,p,b,ldb,info);}
}
#[no_mangle]
pub unsafe extern "C" fn [<$p getri $s>](n:*const $i,a:*mut $t,lda:*const $i,p:*const $i,work:*mut $t,lwork:*const $i,info:*mut $i){
 *info=0;valid!(info,1=>*n>=0,3=>*lda>=1.max(*n),6=>*lwork== -1||*lwork>=1.max(*n));*work=workspace::<$t>((1.max(*n)).val());if *lwork== -1||*n==0{return;}
 let n=(*n).val() as usize;let mat=input(a,n,n,(*lda).val());for i in 0..n{if mat[(i,i)]==<$t>::z(){*info=(i+1)as $i;return;}}
 let mut inv=Mat::<$t>::identity(n,n);let piv:Vec<usize>=(0..n).map(|i|(*p.add(i)).val()as usize-1).collect();solve_lu(&mat,&piv,&mut inv,b'N');output(&inv,a,(*lda).val());
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potrf $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){
 *info=0;let(u,n,ld)=(flag(u),(*n).val(),(*lda).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,4=>ld>=1.max(n));if n==0{return;}let mut mat=hermitian_input(a,n as usize,ld,u);*info=cholesky(&mut mat)as $i;cholesky_output(&mat,a,ld,u);
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potrs $s>](u:*const u8,n:*const $i,k:*const $i,a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i,info:*mut $i){
 *info=0;let(u,n,k,la,lb)=(flag(u),(*n).val(),(*k).val(),(*lda).val(),(*ldb).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,3=>k>=0,5=>la>=1.max(n),7=>lb>=1.max(n));if n==0||k==0{return;}let mat=hermitian_input(a,n as usize,la,u);let mut rhs=input(b,n as usize,k as usize,lb);solve_cholesky(&mat,&mut rhs);output(&rhs,b,lb);
}
#[no_mangle]
pub unsafe extern "C" fn [<$p potri $s>](u:*const u8,n:*const $i,a:*mut $t,lda:*const $i,info:*mut $i){
 *info=0;let(u,n,ld)=(flag(u),(*n).val(),(*lda).val());valid!(info,1=>crate::kernels::uplo(u),2=>n>=0,4=>ld>=1.max(n));if n==0{return;}let mat=hermitian_input(a,n as usize,ld,u);for i in 0..n as usize{if mat[(i,i)]==<$t>::z(){*info=(i+1)as $i;return;}}let mut inv=Mat::<$t>::identity(n as usize,n as usize);solve_cholesky(&mat,&mut inv);cholesky_output(&inv,a,ld,u);
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
        let mat = input(a, m as usize, n as usize, la);
        if !mat.is_all_finite() {
            Err(())
        } else {
            svd(&mat, ucols, vcols)
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
    schur: Mat<T>,
}
pub trait General: Scalar {
    fn eigen(a: &Mat<Self>, left: bool, right: bool) -> Result<GeneralResult<Self>, ()>;
    fn balance(a: &mut Mat<Self>, job: u8) -> Result<(Vec<usize>, Vec<f64>, usize, usize), ()>;
}
fn hessenberg<T: Scalar>(a: &Mat<T>) -> Mat<T> {
    let n = a.nrows();
    let mut h = a.clone();
    if n < 2 {
        return h;
    }
    crate::threading::with_par(|par| {
        use colmat::linalg::{evd::hessenberg::*, qr::no_pivoting::factor::recommended_block_size};
        let bs = recommended_block_size::<T>(n - 1, n - 1);
        let mut coeff = Mat::<T>::zeros(bs, n - 1);
        let mut mem = MemBuffer::new(hessenberg_in_place_scratch::<T>(
            n,
            bs,
            par.shape(),
            Default::default(),
        ));
        hessenberg_in_place(
            h.as_mut(),
            coeff.as_mut(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
    });
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = T::z();
        }
    }
    h
}
macro_rules! balance_impl {
    ($t:ty,$r:ty) => {
        fn balance(a: &mut Mat<Self>, job: u8) -> Result<(Vec<usize>, Vec<f64>, usize, usize), ()> {
            use colmat::linalg::evd::*;
            let n = a.nrows();
            let mut p = vec![0; n];
            let mut s = vec![0.0 as $r; n];
            let mut mem = MemBuffer::new(balance_scratch());
            let range = balance_in_place_with(
                a.as_mut(),
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
                range.end,
            ))
        }
    };
}
macro_rules! general_real {
    ($t:ty) => {
        impl General for $t {
            balance_impl!($t, $t);
            fn eigen(a: &Mat<Self>, left: bool, right: bool) -> Result<GeneralResult<Self>, ()> {
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
                let mut h = hessenberg(a);
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<$t>(
                        n,
                        false,
                        par.shape(),
                        Default::default(),
                    ));
                    let mut rw = colmat::Col::<$t>::zeros(n);
                    let mut iw = colmat::Col::<$t>::zeros(n);
                    hessenberg_schur_real(
                        h.as_mut(),
                        None,
                        rw.as_mut(),
                        iw.as_mut(),
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
                    schur: h,
                })
            }
        }
    };
}
macro_rules! general_complex {
    ($t:ty,$r:ty) => {
        impl General for $t {
            balance_impl!($t, $r);
            fn eigen(a: &Mat<Self>, left: bool, right: bool) -> Result<GeneralResult<Self>, ()> {
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
                let mut h = hessenberg(a);
                crate::threading::with_par(|par| {
                    let mut mem = MemBuffer::new(hessenberg_schur_scratch::<$t>(
                        n,
                        false,
                        par.shape(),
                        Default::default(),
                    ));
                    let mut sw = colmat::Col::<$t>::zeros(n);
                    hessenberg_schur_cplx(
                        h.as_mut(),
                        None,
                        sw.as_mut(),
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
                    schur: h,
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
    let mut mat = input(a, n, n, la);
    let (p, s, lo, hi) = T::balance(&mut mat, job)?;
    let abnrm = (0..n)
        .map(|j| (0..n).map(|i| mat[(i, j)].magnitude()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let e = T::eigen(&mat, jl == b'V', jr == b'V')?;
    if jl == b'V' {
        output(&unbalance(&e.left, &p, &s, true, &e.wi), vl, ll);
    }
    if jr == b'V' {
        output(&unbalance(&e.right, &p, &s, false, &e.wi), vr, lr);
    }
    output(&e.schur, a, la);
    // Encode the final permutation as the outside-active-range swaps xGEBAK expects.
    let mut encoded = s;
    let mut current: Vec<usize> = (0..n).collect();
    for i in (hi..n).rev().chain(0..lo) {
        let pos = (0..n).find(|&k| current[k] == p[i]).unwrap();
        encoded[i] = (pos + 1) as f64;
        current.swap(i, pos);
    }
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
    let mat = hermitian_input(a, n, la, u);
    let (values, vectors) = selfadj(&mat, job == b'V')?;
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
