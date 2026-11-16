use super::*;

macro_rules! matrix_cases {
    ($name:ident,$p:ident,$t:ty,$coefficient:expr,$make:expr,$op:ident,$h:expr,$tol:expr) => {paste! {
        #[test]
        fn $name() {
            let value=|i:usize,j:usize| ($make)((1+i+3*j) as f64/8.0,(i as f64-j as f64)/16.0);
            let alpha=($make)(0.75,0.25);
            let beta=($make)(-0.5,0.125);
            for order in [101,102] {for side in [141,142] {for u in [121,122] {
                let (m,n)=(3usize,2usize);
                let dim=if side==141 {m}else{n};
                let lda=dim+2;
                let ldb=if order==101 {n+1}else{m+1};
                let ldc=ldb+1;
                let index=|i,j,ld| if order==101 {i*ld+j}else{i+j*ld};
                let sentinel=<$t>::real(97.0);
                let mut a=vec![<$t>::real(f64::NAN);lda*dim];
                let entry=|i,j| {
                    if i==j {if $h {<$t>::real(2.0+i as f64)}else{value(i,j)}}
                    else if (u==121&&i<j)||(u==122&&i>j) {value(i,j)}
                    else {let v=value(j,i);if $h {v.cj()}else{v}}
                };
                for i in 0..dim {for j in 0..dim {
                    if (u==121&&i<=j)||(u==122&&i>=j) {
                        a[index(i,j,lda)]=if $h&&i==j {($make)(2.0+i as f64,f64::NAN)}else{entry(i,j)};
                    }
                }}
                let mut b=vec![sentinel;ldb*m.max(n)];
                let mut initial=vec![sentinel;ldc*m.max(n)];
                for i in 0..m {for j in 0..n {b[index(i,j,ldb)]=value(i+1,j);initial[index(i,j,ldc)]=value(i,j+2);}}
                let mut expected=initial.clone();
                for i in 0..m {for j in 0..n {
                    let mut sum=<$t>::z();
                    for l in 0..dim {sum+=if side==141 {entry(i,l)*value(l+1,j)}else{value(i+1,l)*entry(l,j)};}
                    expected[index(i,j,ldc)]=alpha*sum+beta*initial[index(i,j,ldc)];
                }}
                for wide in [false,true] {
                    let mut c=initial.clone();
                    unsafe {
                        if wide {[<cblas_ $p $op 64_>](order,side,u,m as i64,n as i64,($coefficient)(&alpha),a.as_ptr(),lda as i64,b.as_ptr(),ldb as i64,($coefficient)(&beta),c.as_mut_ptr(),ldc as i64);}
                        else {[<cblas_ $p $op>](order,side,u,m as i32,n as i32,($coefficient)(&alpha),a.as_ptr(),lda as i32,b.as_ptr(),ldb as i32,($coefficient)(&beta),c.as_mut_ptr(),ldc as i32);}
                    }
                    for (actual,expected) in c.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tol,"wide={wide} order={order} side={side} u={u}");}
                }
            }}}
        }
    }};
}
matrix_cases!(
    symm_real32,
    s,
    f32,
    |x: &f32| *x,
    |r: f64, _i: f64| r as f32,
    symm,
    false,
    1e-5
);
matrix_cases!(
    symm_real64,
    d,
    f64,
    |x: &f64| *x,
    |r: f64, _i: f64| r,
    symm,
    false,
    1e-12
);
matrix_cases!(
    symm_complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    symm,
    false,
    1e-5
);
matrix_cases!(
    symm_complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    |r: f64, i: f64| c64::new(r, i),
    symm,
    false,
    1e-12
);
matrix_cases!(
    hemm_complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    hemm,
    true,
    1e-5
);
matrix_cases!(
    hemm_complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    |r: f64, i: f64| c64::new(r, i),
    hemm,
    true,
    1e-12
);

macro_rules! ger_cases {
    ($name:ident,$p:ident,$t:ty,$coefficient:expr,$make:expr,$op:ident,$h:expr,$tol:expr) => {paste! {
        #[test]
        fn $name() {
            let value=|i:usize,j:usize| ($make)((1+i+3*j) as f64/8.0,(i as f64-j as f64)/16.0);
            let alpha=($make)(0.75,0.25);
            let (m,n)=(3usize,2usize);
            for order in [101,102] {for ix in [-2i32,1,3] {for iy in [-3i32,1,2] {
                let lda=if order==101 {n+2}else{m+2};
                let index=|i,j| if order==101 {i*lda+j}else{i+j*lda};
                let vi=|i:usize,len:usize,inc:i32| if inc<0 {(len-1-i)*(-inc) as usize}else{i*inc as usize};
                let sentinel=<$t>::real(97.0);
                let mut x=vec![sentinel;1+(m-1)*ix.unsigned_abs() as usize];
                let mut y=vec![sentinel;1+(n-1)*iy.unsigned_abs() as usize];
                for i in 0..m {x[vi(i,m,ix)]=value(i,1);}
                for j in 0..n {y[vi(j,n,iy)]=value(j,2);}
                let mut initial=vec![sentinel;lda*m.max(n)];
                let mut expected=initial.clone();
                for i in 0..m {for j in 0..n {
                    initial[index(i,j)]=value(i,j);
                    let y=value(j,2);
                    expected[index(i,j)]=value(i,j)+alpha*value(i,1)*if $h {y.cj()}else{y};
                }}
                for wide in [false,true] {
                    let mut a=initial.clone();
                    unsafe {
                        if wide {[<cblas_ $p $op 64_>](order,m as i64,n as i64,($coefficient)(&alpha),x.as_ptr(),ix as i64,y.as_ptr(),iy as i64,a.as_mut_ptr(),lda as i64);}
                        else {[<cblas_ $p $op>](order,m as i32,n as i32,($coefficient)(&alpha),x.as_ptr(),ix,y.as_ptr(),iy,a.as_mut_ptr(),lda as i32);}
                    }
                    for (actual,expected) in a.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tol,"wide={wide} order={order} ix={ix} iy={iy}");}
                }
            }}}
        }
    }};
}
ger_cases!(
    ger_real32,
    s,
    f32,
    |x: &f32| *x,
    |r: f64, _i: f64| r as f32,
    ger,
    false,
    1e-5
);
ger_cases!(
    ger_real64,
    d,
    f64,
    |x: &f64| *x,
    |r: f64, _i: f64| r,
    ger,
    false,
    1e-12
);
ger_cases!(
    geru_complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    geru,
    false,
    1e-5
);
ger_cases!(
    geru_complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    |r: f64, i: f64| c64::new(r, i),
    geru,
    false,
    1e-12
);
ger_cases!(
    gerc_complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    gerc,
    true,
    1e-5
);
ger_cases!(
    gerc_complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    |r: f64, i: f64| c64::new(r, i),
    gerc,
    true,
    1e-12
);

macro_rules! vector_cases {
    ($name:ident,$p:ident,$t:ty,$coefficient:expr,$make:expr,$op:ident,$h:expr,$tol:expr) => {paste! {
        #[test]
        fn $name() {
            let value=|i:usize,j:usize| ($make)((1+i+3*j) as f64/8.0,(i as f64-j as f64)/16.0);
            let alpha=($make)(0.75,0.25);
            let n=3usize;
            for order in [101,102] {for u in [121,122] {for ix in [-2i32,1,3] {for iy in [-3i32,1,2] {for zero_beta in [false,true] {
                let beta=if zero_beta {<$t>::z()}else{($make)(-0.5,0.125)};
                let lda=n+2;
                let index=|i,j| if order==101 {i*lda+j}else{i+j*lda};
                let vi=|i:usize,inc:i32| if inc<0 {(n-1-i)*(-inc) as usize}else{i*inc as usize};
                let sentinel=<$t>::real(97.0);
                let mut a=vec![<$t>::real(f64::NAN);lda*n];
                let entry=|i,j| {
                    if i==j {<$t>::real(2.0+i as f64)}
                    else if (u==121&&i<j)||(u==122&&i>j) {value(i,j)}
                    else {let v=value(j,i);if $h {v.cj()}else{v}}
                };
                for i in 0..n {for j in 0..n {
                    if (u==121&&i<=j)||(u==122&&i>=j) {
                        a[index(i,j)]=if $h&&i==j {($make)(2.0+i as f64,f64::NAN)}else{entry(i,j)};
                    }
                }}
                let mut x=vec![sentinel;1+(n-1)*ix.unsigned_abs() as usize];
                let mut initial=vec![sentinel;1+(n-1)*iy.unsigned_abs() as usize];
                for i in 0..n {x[vi(i,ix)]=value(i,1);initial[vi(i,iy)]=if zero_beta {<$t>::real(f64::NAN)}else{value(i,2)};}
                let mut expected=initial.clone();
                for i in 0..n {
                    let mut sum=<$t>::z();
                    for j in 0..n {sum+=entry(i,j)*value(j,1);}
                    expected[vi(i,iy)]=alpha*sum+if zero_beta {<$t>::z()}else{beta*value(i,2)};
                }
                for wide in [false,true] {
                    let mut y=initial.clone();
                    unsafe {
                        if wide {[<cblas_ $p $op 64_>](order,u,n as i64,($coefficient)(&alpha),a.as_ptr(),lda as i64,x.as_ptr(),ix as i64,($coefficient)(&beta),y.as_mut_ptr(),iy as i64);}
                        else {[<cblas_ $p $op>](order,u,n as i32,($coefficient)(&alpha),a.as_ptr(),lda as i32,x.as_ptr(),ix,($coefficient)(&beta),y.as_mut_ptr(),iy);}
                    }
                    for (actual,expected) in y.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tol,"wide={wide} order={order} u={u} ix={ix} iy={iy} beta0={zero_beta}");}
                }
            }}}}}
        }
    }};
}
vector_cases!(
    symv_real32,
    s,
    f32,
    |x: &f32| *x,
    |r: f64, _i: f64| r as f32,
    symv,
    false,
    1e-5
);
vector_cases!(
    symv_real64,
    d,
    f64,
    |x: &f64| *x,
    |r: f64, _i: f64| r,
    symv,
    false,
    1e-12
);
vector_cases!(
    hemv_complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    hemv,
    true,
    1e-5
);
vector_cases!(
    hemv_complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    |r: f64, i: f64| c64::new(r, i),
    hemv,
    true,
    1e-12
);

#[test]
fn symmetric_vector_quick_returns_and_c_argument_errors() {
    unsafe {
        cblas_zhemv64_(
            101,
            121,
            0,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
        cblas_zhemv64_(
            101,
            121,
            2,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            1,
        );
        assert_eq!(crate::colmat_blas_last_error(), 6);
        cblas_zhemv64_(
            101,
            121,
            2,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null(),
            1,
            core::ptr::null(),
            core::ptr::null_mut(),
            0,
        );
        assert_eq!(crate::colmat_blas_last_error(), 11);
        let zero = c64::new(0.0, 0.0);
        let two = c64::new(2.0, 0.0);
        let mut y = [c64::new(1.0, 2.0), c64::new(97.0, 0.0), c64::new(3.0, -1.0)];
        cblas_zhemv64_(
            101,
            121,
            2,
            &zero,
            core::ptr::null(),
            2,
            core::ptr::null(),
            -1,
            &two,
            y.as_mut_ptr(),
            -2,
        );
        assert_eq!(
            y,
            [c64::new(2.0, 4.0), c64::new(97.0, 0.0), c64::new(6.0, -2.0)]
        );
    }
}

macro_rules! rank_one_cases {
    ($name:ident,$p:ident,$t:ty,$r:ty,$make:expr,$op:ident,$h:expr,$tol:expr) => {paste! {
        #[test]
        fn $name() {
            let value=|i:usize,j:usize| ($make)((1+i+3*j) as f64/8.0,(i as f64-j as f64)/16.0);
            let alpha: $r=0.75;
            let n=3usize;
            for order in [101,102] {for u in [121,122] {for inc in [-2i32,1,3] {
                let lda=n+2;
                let index=|i,j| if order==101 {i*lda+j}else{i+j*lda};
                let vi=|i:usize| if inc<0 {(n-1-i)*(-inc) as usize}else{i*inc as usize};
                let sentinel=<$t>::real(97.0);
                let mut x=vec![sentinel;1+(n-1)*inc.unsigned_abs() as usize];
                for i in 0..n {x[vi(i)]=value(i,1);}
                let mut initial=vec![sentinel;lda*n];
                let mut expected=initial.clone();
                for i in 0..n {for j in 0..n {
                    if (u==121&&i<=j)||(u==122&&i>=j) {
                        initial[index(i,j)]=if $h&&i==j {($make)(2.0+i as f64,f64::NAN)}else{value(i,j)};
                        let old=if $h&&i==j {<$t>::real(2.0+i as f64)}else{value(i,j)};
                        let y=value(j,1);
                        let v=old+<$t>::real(alpha as f64)*value(i,1)*if $h {y.cj()}else{y};
                        expected[index(i,j)]=if $h&&i==j {<$t>::real(v.re())}else{v};
                    }
                }}
                for wide in [false,true] {
                    let mut a=initial.clone();
                    unsafe {
                        if wide {[<cblas_ $p $op 64_>](order,u,n as i64,alpha,x.as_ptr(),inc as i64,a.as_mut_ptr(),lda as i64);}
                        else {[<cblas_ $p $op>](order,u,n as i32,alpha,x.as_ptr(),inc,a.as_mut_ptr(),lda as i32);}
                    }
                    for (actual,expected) in a.iter().zip(&expected) {assert!((*actual-*expected).magnitude()<$tol,"wide={wide} order={order} u={u} inc={inc}");}
                }
            }}}
        }
    }};
}
rank_one_cases!(
    syr_real32,
    s,
    f32,
    f32,
    |r: f64, _i: f64| r as f32,
    syr,
    false,
    1e-5
);
rank_one_cases!(
    syr_real64,
    d,
    f64,
    f64,
    |r: f64, _i: f64| r,
    syr,
    false,
    1e-12
);
rank_one_cases!(
    her_complex32,
    c,
    c32,
    f32,
    |r: f64, i: f64| c32::new(r as f32, i as f32),
    her,
    true,
    1e-5
);
rank_one_cases!(
    her_complex64,
    z,
    c64,
    f64,
    |r: f64, i: f64| c64::new(r, i),
    her,
    true,
    1e-12
);
