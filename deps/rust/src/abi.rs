//! Fortran BLAS symbols, generated for four scalars and two integer ABIs.
use crate::kernels::*;
use colmat::{c32, c64};
use paste::paste;

macro_rules! check { ($($arg:expr => $valid:expr),+ $(,)?) => { $(if !$valid {crate::invalid($arg); return;})+ }; }

macro_rules! common {
    ($p:ident,$t:ty,$r:ty,$i:ty,$s:tt) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gemm $s>](ta:*const u8,tb:*const u8,m:*const $i,n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
            let (ta,tb,m,n,k,la,lb,lc)=(flag(ta),flag(tb),*m as isize,*n as isize,*k as isize,*lda as isize,*ldb as isize,*ldc as isize);
            check!(1=>trans(ta),2=>trans(tb),3=>m>=0,4=>n>=0,5=>k>=0,8=>la>=1.max(if ta==b'N' {m} else {k}),10=>lb>=1.max(if tb==b'N' {k} else {n}),13=>lc>=1.max(m));
            gemm(ta,tb,m,n,k,*alpha,a,la,b,lb,*beta,c,lc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p gemv $s>](t:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,x:*const $t,ix:*const $i,beta:*const $t,y:*mut $t,iy:*const $i) {
            let (t,m,n,la,ix,iy)=(flag(t),*m as isize,*n as isize,*lda as isize,*ix as isize,*iy as isize);
            check!(1=>trans(t),2=>m>=0,3=>n>=0,6=>la>=1.max(m),8=>ix!=0,11=>iy!=0);
            gemv(t,m,n,*alpha,a,la,x,ix,*beta,y,iy);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p copy $s>](n:*const $i,x:*const $t,ix:*const $i,y:*mut $t,iy:*const $i) {
            let (n,ix,iy)=(*n as isize,*ix as isize,*iy as isize); if n<=0 {return;}
            let x=x.offset(start(n,ix)); let y=y.offset(start(n,iy));
            for j in 0..n {*y.offset(j*iy)=*x.offset(j*ix);}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p swap $s>](n:*const $i,x:*mut $t,ix:*const $i,y:*mut $t,iy:*const $i) {
            let (n,ix,iy)=(*n as isize,*ix as isize,*iy as isize); if n<=0 {return;}
            let x=x.offset(start(n,ix)); let y=y.offset(start(n,iy));
            for j in 0..n {std::ptr::swap(x.offset(j*ix),y.offset(j*iy));}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p scal $s>](n:*const $i,alpha:*const $t,x:*mut $t,inc:*const $i) {
            let (n,inc)=(*n as isize,*inc as isize); if n<=0||inc<=0 {return;}
            for j in 0..n {let p=x.offset(j*inc); *p *= *alpha;}
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p axpy $s>](n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,y:*mut $t,iy:*const $i) {
            axpby(*n as isize,*alpha,x,*ix as isize,<$t>::o(),y,*iy as isize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p axpby $s>](n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,beta:*const $t,y:*mut $t,iy:*const $i) {
            axpby(*n as isize,*alpha,x,*ix as isize,*beta,y,*iy as isize);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<i $p amax $s>](n:*const $i,x:*const $t,inc:*const $i) -> $i {
            iamax(*n as isize,x,*inc as isize) as $i
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p symv $s>](u:*const u8,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,x:*const $t,ix:*const $i,beta:*const $t,y:*mut $t,iy:*const $i) {
            let (u,n,la,ix,iy)=(flag(u),*n as isize,*lda as isize,*ix as isize,*iy as isize);
            check!(1=>uplo(u),2=>n>=0,5=>la>=1.max(n),7=>ix!=0,10=>iy!=0);
            symv(u,n,*alpha,a,la,x,ix,*beta,y,iy,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p syr $s>](u:*const u8,n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,a:*mut $t,lda:*const $i) {
            let (u,n,ix,la)=(flag(u),*n as isize,*ix as isize,*lda as isize);
            check!(1=>uplo(u),2=>n>=0,5=>ix!=0,7=>la>=1.max(n));
            rank1(u,n,*alpha,x,ix,a,la,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p symm $s>](side:*const u8,u:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
            let (s,u,m,n,la,lb,lc)=(flag(side),flag(u),*m as isize,*n as isize,*lda as isize,*ldb as isize,*ldc as isize);
            check!(1=>matches!(s,b'L'|b'R'),2=>uplo(u),3=>m>=0,4=>n>=0,7=>la>=1.max(if s==b'L'{m}else{n}),9=>lb>=1.max(m),12=>lc>=1.max(m));
            symm(s,u,m,n,*alpha,a,la,b,lb,*beta,c,lc,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p syrk $s>](u:*const u8,t:*const u8,n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
            let (u,t,n,k,la,lc)=(flag(u),flag(t),*n as isize,*k as isize,*lda as isize,*ldc as isize);
            check!(1=>uplo(u),2=>(matches!(t,b'N'|b'T') || (t==b'C' && !<$t>::COMPLEX)),3=>n>=0,4=>k>=0,7=>la>=1.max(if t==b'N'{n}else{k}),10=>lc>=1.max(n));
            rankk(u,t,n,k,*alpha,a,la,a,la,*beta,c,lc,false,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p syr2k $s>](u:*const u8,t:*const u8,n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
            let (u,t,n,k,la,lb,lc)=(flag(u),flag(t),*n as isize,*k as isize,*lda as isize,*ldb as isize,*ldc as isize);
            check!(1=>uplo(u),2=>(matches!(t,b'N'|b'T') || (t==b'C' && !<$t>::COMPLEX)),3=>n>=0,4=>k>=0,7=>la>=1.max(if t==b'N'{n}else{k}),9=>lb>=1.max(if t==b'N'{n}else{k}),12=>lc>=1.max(n));
            rankk(u,t,n,k,*alpha,a,la,b,lb,*beta,c,lc,false,true);
        }
        triangular_exports!($p,$t,$i,$s,trmv,false);
        triangular_exports!($p,$t,$i,$s,trsv,true);
        trimatrix_exports!($p,$t,$i,$s,trmm,false);
        trimatrix_exports!($p,$t,$i,$s,trsm,true);
    }};
}
macro_rules! triangular_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident,$solve:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](u:*const u8,t:*const u8,d:*const u8,n:*const $i,a:*const $t,lda:*const $i,x:*mut $t,inc:*const $i) {
            let (u,t,d,n,la,inc)=(flag(u),flag(t),flag(d),*n as isize,*lda as isize,*inc as isize);
            check!(1=>uplo(u),2=>trans(t),3=>diag(d),4=>n>=0,6=>la>=1.max(n),8=>inc!=0);
            trmv(u,t,d,n,a,la,x,inc,$solve);
        }
    }};
}
macro_rules! trimatrix_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident,$solve:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](side:*const u8,u:*const u8,t:*const u8,d:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*mut $t,ldb:*const $i) {
            let (s,u,t,d,m,n,la,lb)=(flag(side),flag(u),flag(t),flag(d),*m as isize,*n as isize,*lda as isize,*ldb as isize);
            check!(1=>matches!(s,b'L'|b'R'),2=>uplo(u),3=>trans(t),4=>diag(d),5=>m>=0,6=>n>=0,9=>la>=1.max(if s==b'L'{m}else{n}),11=>lb>=1.max(m));
            trmatrix(s,u,t,d,m,n,*alpha,a,la,b,lb,$solve);
        }
    }};
}
macro_rules! real_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p dot $s>](n:*const $i,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i) -> $t {dot(*n as isize,x,*ix as isize,y,*iy as isize,false)}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p nrm2 $s>](n:*const $i,x:*const $t,inc:*const $i) -> $t {nrm2(*n as isize,x,*inc as isize) as $t}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p asum $s>](n:*const $i,x:*const $t,inc:*const $i) -> $t {asum(*n as isize,x,*inc as isize) as $t}
        ger_exports!($p,$t,$i,$s,ger,false);
        rot_exports!($p,$t,$t,$t,$i,$s);
    }};
}
macro_rules! ger_exports {
    ($p:ident,$t:ty,$i:ty,$s:tt,$name:ident,$cj:expr) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $name $s>](m:*const $i,n:*const $i,alpha:*const $t,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i,a:*mut $t,lda:*const $i) {
            let (m,n,ix,iy,la)=(*m as isize,*n as isize,*ix as isize,*iy as isize,*lda as isize);
            check!(1=>m>=0,2=>n>=0,5=>ix!=0,7=>iy!=0,9=>la>=1.max(m));
            ger(m,n,*alpha,x,ix,y,iy,a,la,$cj);
        }
    }};
}
macro_rules! rot_exports {
    ($p:ident,$t:ty,$r:ty,$st:ty,$i:ty,$s:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p rot $s>](n:*const $i,x:*mut $t,ix:*const $i,y:*mut $t,iy:*const $i,c:*const $r,s:*const $st) {
            let (n,ix,iy)=(*n as isize,*ix as isize,*iy as isize);if n<=0 {return;}
            let x=x.offset(start(n,ix));let y=y.offset(start(n,iy));
            let c=<$t>::real(*c as f64);let s=<$t>::from(*s);
            for i in 0..n {let xp=x.offset(i*ix);let yp=y.offset(i*iy);let (a,b)=(*xp,*yp);*xp=c*a+s*b;*yp=c*b-s.cj()*a;}
        }
    }};
}
macro_rules! complex_exports {
    ($p:ident,$rp:ident,$t:ty,$r:ty,$i:ty,$s:tt) => {paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p dotc $s>](n:*const $i,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i) -> $t {dot(*n as isize,x,*ix as isize,y,*iy as isize,true)}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p dotu $s>](n:*const $i,x:*const $t,ix:*const $i,y:*const $t,iy:*const $i) -> $t {dot(*n as isize,x,*ix as isize,y,*iy as isize,false)}
        #[no_mangle]
        pub unsafe extern "C" fn [<$rp $p nrm2 $s>](n:*const $i,x:*const $t,inc:*const $i) -> $r {nrm2(*n as isize,x,*inc as isize) as $r}
        #[no_mangle]
        pub unsafe extern "C" fn [<$rp $p asum $s>](n:*const $i,x:*const $t,inc:*const $i) -> $r {asum(*n as isize,x,*inc as isize) as $r}
        #[no_mangle]
        pub unsafe extern "C" fn [<$p $rp scal $s>](n:*const $i,alpha:*const $r,x:*mut $t,inc:*const $i) {
            let (n,inc)=(*n as isize,*inc as isize);if n<=0||inc<=0 {return;}
            let a=<$t>::real(*alpha as f64);for i in 0..n {let p=x.offset(i*inc);*p=a * *p;}
        }
        ger_exports!($p,$t,$i,$s,geru,false);
        ger_exports!($p,$t,$i,$s,gerc,true);
        rot_exports!($p,$t,$r,$t,$i,$s);
        rot_exports!([<$p $rp>],$t,$r,$r,$i,$s);
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hemv $s>](u:*const u8,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,x:*const $t,ix:*const $i,beta:*const $t,y:*mut $t,iy:*const $i) {
            let (u,n,la,ix,iy)=(flag(u),*n as isize,*lda as isize,*ix as isize,*iy as isize);
            check!(1=>uplo(u),2=>n>=0,5=>la>=1.max(n),7=>ix!=0,10=>iy!=0);
            symv(u,n,*alpha,a,la,x,ix,*beta,y,iy,true);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p her $s>](u:*const u8,n:*const $i,alpha:*const $r,x:*const $t,ix:*const $i,a:*mut $t,lda:*const $i) {
            let (u,n,ix,la)=(flag(u),*n as isize,*ix as isize,*lda as isize);
            check!(1=>uplo(u),2=>n>=0,5=>ix!=0,7=>la>=1.max(n));
            rank1(u,n,<$t>::real(*alpha as f64),x,ix,a,la,true);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p hemm $s>](side:*const u8,u:*const u8,m:*const $i,n:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*const $t,ldb:*const $i,beta:*const $t,c:*mut $t,ldc:*const $i) {
            let (s,u,m,n,la,lb,lc)=(flag(side),flag(u),*m as isize,*n as isize,*lda as isize,*ldb as isize,*ldc as isize);
            check!(1=>matches!(s,b'L'|b'R'),2=>uplo(u),3=>m>=0,4=>n>=0,7=>la>=1.max(if s==b'L'{m}else{n}),9=>lb>=1.max(m),12=>lc>=1.max(m));
            symm(s,u,m,n,*alpha,a,la,b,lb,*beta,c,lc,true);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p herk $s>](u:*const u8,t:*const u8,n:*const $i,k:*const $i,alpha:*const $r,a:*const $t,lda:*const $i,beta:*const $r,c:*mut $t,ldc:*const $i) {
            let (u,t,n,k,la,lc)=(flag(u),flag(t),*n as isize,*k as isize,*lda as isize,*ldc as isize);
            check!(1=>uplo(u),2=>matches!(t,b'N'|b'C'),3=>n>=0,4=>k>=0,7=>la>=1.max(if t==b'N'{n}else{k}),10=>lc>=1.max(n));
            rankk(u,t,n,k,<$t>::real(*alpha as f64),a,la,a,la,<$t>::real(*beta as f64),c,lc,true,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p her2k $s>](u:*const u8,t:*const u8,n:*const $i,k:*const $i,alpha:*const $t,a:*const $t,lda:*const $i,b:*const $t,ldb:*const $i,beta:*const $r,c:*mut $t,ldc:*const $i) {
            let (u,t,n,k,la,lb,lc)=(flag(u),flag(t),*n as isize,*k as isize,*lda as isize,*ldb as isize,*ldc as isize);
            check!(1=>uplo(u),2=>matches!(t,b'N'|b'C'),3=>n>=0,4=>k>=0,7=>la>=1.max(if t==b'N'{n}else{k}),9=>lb>=1.max(if t==b'N'{n}else{k}),12=>lc>=1.max(n));
            rankk(u,t,n,k,*alpha,a,la,b,lb,<$t>::real(*beta as f64),c,lc,true,true);
        }
    }};
}
macro_rules! interface {
    ($i:ty,$s:tt) => {
        common!(s, f32, f32, $i, $s);
        common!(d, f64, f64, $i, $s);
        common!(c, c32, f32, $i, $s);
        common!(z, c64, f64, $i, $s);
        real_exports!(s, f32, $i, $s);
        real_exports!(d, f64, $i, $s);
        complex_exports!(c, s, c32, f32, $i, $s);
        complex_exports!(z, d, c64, f64, $i, $s);
    };
}
interface!(i32, _);
interface!(i64, _64_);

// CBLAS dot entry points are used directly by Julia. Their suffix is appended
// to a name without the Fortran underscore, as required by LBT.
macro_rules! cblas_real {
    ($name:ident,$t:ty,$i:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(n: $i, x: *const $t, ix: $i, y: *const $t, iy: $i) -> $t {
            dot(n as isize, x, ix as isize, y, iy as isize, false)
        }
    };
}
macro_rules! cblas_complex {
    ($name:ident,$t:ty,$i:ty,$cj:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            n: $i,
            x: *const $t,
            ix: $i,
            y: *const $t,
            iy: $i,
            out: *mut $t,
        ) {
            *out = dot(n as isize, x, ix as isize, y, iy as isize, $cj);
        }
    };
}
cblas_real!(cblas_sdot, f32, i32);
cblas_real!(cblas_ddot, f64, i32);
cblas_real!(cblas_sdot64_, f32, i64);
cblas_real!(cblas_ddot64_, f64, i64);
cblas_complex!(cblas_cdotc_sub, c32, i32, true);
cblas_complex!(cblas_cdotu_sub, c32, i32, false);
cblas_complex!(cblas_zdotc_sub, c64, i32, true);
cblas_complex!(cblas_zdotu_sub, c64, i32, false);
cblas_complex!(cblas_cdotc_sub64_, c32, i64, true);
cblas_complex!(cblas_cdotu_sub64_, c32, i64, false);
cblas_complex!(cblas_zdotc_sub64_, c64, i64, true);
cblas_complex!(cblas_zdotu_sub64_, c64, i64, false);
