use super::*;

#[test]
fn ignored_storage_and_original_c_argument_numbers() {
    unsafe {
        // Unit diagonal is implicit: no matrix element exists to read.
        let mut x = [3.0f64];
        cblas_dtrmv64_(
            101,
            121,
            113,
            132,
            1,
            core::ptr::null(),
            1,
            x.as_mut_ptr(),
            -1,
        );
        assert_eq!(x, [3.0]);
        cblas_dtrsv(
            102,
            122,
            114,
            132,
            1,
            core::ptr::null(),
            1,
            x.as_mut_ptr(),
            1,
        );
        assert_eq!(x, [3.0]);
        // Alpha zero clears B without reading A or existing B values.
        let mut b = [f64::NAN; 6];
        cblas_dtrmm(
            101,
            141,
            121,
            111,
            131,
            2,
            3,
            0.0,
            core::ptr::null(),
            2,
            b.as_mut_ptr(),
            3,
        );
        assert_eq!(b, [0.0; 6]);
        b.fill(f64::NAN);
        cblas_dtrsm64_(
            102,
            142,
            122,
            113,
            131,
            2,
            3,
            0.0,
            core::ptr::null(),
            3,
            b.as_mut_ptr(),
            2,
        );
        assert_eq!(b, [0.0; 6]);
        // Invalid headers return before coefficients or payload pointers.
        cblas_ztrsm64_(
            101,
            141,
            121,
            111,
            131,
            2,
            3,
            core::ptr::null(),
            core::ptr::null(),
            2,
            core::ptr::null_mut(),
            2,
        );
        assert_eq!(crate::colmat_blas_last_error(), 12);
        cblas_ztrmv64_(
            101,
            121,
            111,
            131,
            2,
            core::ptr::null(),
            2,
            core::ptr::null_mut(),
            0,
        );
        assert_eq!(crate::colmat_blas_last_error(), 9);
        cblas_ztrmm(
            101,
            141,
            121,
            111,
            131,
            0,
            3,
            core::ptr::null(),
            core::ptr::null(),
            1,
            core::ptr::null_mut(),
            3,
        );
        assert_eq!(crate::colmat_blas_last_error(), 0);
    }
}

macro_rules! cases {
    ($name:ident,$p:ident,$t:ty,$coefficient:expr,$imag:expr,$tol:expr) => {paste! {
        #[test]
        fn $name() {
            let value=|i:usize,j:usize| <$t>::real((1+i+2*j) as f64/16.0)+($imag)*<$t>::real((i as f64-j as f64)/32.0);
            let alpha=<$t>::real(0.75)+($imag)*<$t>::real(0.125);
            let sentinel=<$t>::real(97.0);
            for order in [101,102] {for side in [141,142] {for u in [121,122] {for t in 111..=114 {for d in [131,132] {
                let (m,n)=(3usize,2usize);
                let dim=if side==141 {m}else{n};
                let lda=dim+2;
                let ldb=if order==101 {n+2}else{m+2};
                let offset=|i,j,ld|if order==101 {i*ld+j}else{i+j*ld};
                // Unused triangle and unit diagonals contain NaNs, not zeros.
                let poison=<$t>::real(f64::NAN);
                let mut a=vec![poison;lda*dim];
                let entry=|i,j| {
                    if i==j {if d==132 {<$t>::o()}else{<$t>::real(2.0+i as f64/4.0)+($imag)*<$t>::real(0.125)}}
                    else if (u==121&&i<j)||(u==122&&i>j) {value(i,j)}else{<$t>::z()}
                };
                for i in 0..dim {for j in 0..dim {
                    if ((u==121&&i<=j)||(u==122&&i>=j))&&!(i==j&&d==132) {a[offset(i,j,lda)]=entry(i,j);}
                }}
                let op=|i,j| {
                    let v=if t==112||t==113 {entry(j,i)}else{entry(i,j)};
                    if t==113||t==114 {v.cj()}else{v}
                };
                let mut original=vec![sentinel;ldb*m.max(n)];
                let mut product=original.clone();
                let mut rhs=original.clone();
                for i in 0..m {for j in 0..n {
                    original[offset(i,j,ldb)]=value(i+1,j+1);
                    let mut sum=<$t>::z();
                    for l in 0..dim {
                        sum+=if side==141 {op(i,l)*value(l+1,j+1)}else{value(i+1,l+1)*op(l,j)};
                    }
                    product[offset(i,j,ldb)]=alpha*sum;
                    rhs[offset(i,j,ldb)]=sum/alpha;
                }}
                for wide in [false,true] {
                    let mut b=original.clone();
                    unsafe {
                        if wide {[<cblas_ $p trmm64_>](order,side,u,t,d,m as i64,n as i64,($coefficient)(&alpha),a.as_ptr(),lda as i64,b.as_mut_ptr(),ldb as i64);}
                        else {[<cblas_ $p trmm>](order,side,u,t,d,m as i32,n as i32,($coefficient)(&alpha),a.as_ptr(),lda as i32,b.as_mut_ptr(),ldb as i32);}
                    }
                    for (actual,expected) in b.iter().zip(&product) {assert!((*actual-*expected).magnitude()<$tol,"TRMM wide={wide} order={order} side={side} u={u} t={t} d={d}");}
                    let mut b=rhs.clone();
                    unsafe {
                        if wide {[<cblas_ $p trsm64_>](order,side,u,t,d,m as i64,n as i64,($coefficient)(&alpha),a.as_ptr(),lda as i64,b.as_mut_ptr(),ldb as i64);}
                        else {[<cblas_ $p trsm>](order,side,u,t,d,m as i32,n as i32,($coefficient)(&alpha),a.as_ptr(),lda as i32,b.as_mut_ptr(),ldb as i32);}
                    }
                    for (actual,expected) in b.iter().zip(&original) {assert!((*actual-*expected).magnitude()<$tol,"TRSM wide={wide} order={order} side={side} u={u} t={t} d={d}");}
                }
                // Vector multiply and solve use independently constructed RHS.
                for inc in [-2i32,1,3] {
                    let index=|i:usize| if inc<0 {(dim-1-i)*(-inc) as usize}else{i*inc as usize};
                    let mut original=vec![sentinel;1+(dim-1)*inc.unsigned_abs() as usize];
                    let mut product=original.clone();
                    for i in 0..dim {
                        original[index(i)]=value(i,2);
                        let mut sum=<$t>::z();
                        for j in 0..dim {sum+=op(i,j)*value(j,2);}
                        product[index(i)]=sum;
                    }
                    for wide in [false,true] {
                        let mut x=original.clone();
                        unsafe {
                            if wide {[<cblas_ $p trmv64_>](order,u,t,d,dim as i64,a.as_ptr(),lda as i64,x.as_mut_ptr(),inc as i64);}
                            else {[<cblas_ $p trmv>](order,u,t,d,dim as i32,a.as_ptr(),lda as i32,x.as_mut_ptr(),inc);}
                        }
                        for (actual,expected) in x.iter().zip(&product) {assert!((*actual-*expected).magnitude()<$tol,"TRMV wide={wide} order={order} u={u} t={t} d={d} inc={inc}");}
                        let mut x=product.clone();
                        unsafe {
                            if wide {[<cblas_ $p trsv64_>](order,u,t,d,dim as i64,a.as_ptr(),lda as i64,x.as_mut_ptr(),inc as i64);}
                            else {[<cblas_ $p trsv>](order,u,t,d,dim as i32,a.as_ptr(),lda as i32,x.as_mut_ptr(),inc);}
                        }
                        for (actual,expected) in x.iter().zip(&original) {assert!((*actual-*expected).magnitude()<$tol,"TRSV wide={wide} order={order} u={u} t={t} d={d} inc={inc}");}
                    }
                }
            }}}}}
        }
    }};
}
cases!(real32, s, f32, |x: &f32| *x, 0.0f32, 1e-5);
cases!(real64, d, f64, |x: &f64| *x, 0.0f64, 1e-12);
cases!(
    complex32,
    c,
    c32,
    |x: &c32| x as *const c32,
    c32::new(0.0, 1.0),
    1e-5
);
cases!(
    complex64,
    z,
    c64,
    |x: &c64| x as *const c64,
    c64::new(0.0, 1.0),
    1e-12
);
