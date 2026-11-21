//! LAPACK Householder adapters. All storage is borrowed only during the call.
use crate::kernels::*;
use colmat::{c32, c64};
use paste::paste;

pub(crate) trait ReflectorScalar: Scalar {
    const EPS: f64;
    const SAFE: f64;
    fn im(self) -> f64;
    fn parts(re: f64, im: f64) -> Self;
}
macro_rules! reflector_real {
    ($t:ty) => {
        impl ReflectorScalar for $t {
            const EPS: f64 = <$t>::EPSILON as f64;
            const SAFE: f64 = <$t>::MIN_POSITIVE as f64 / (<$t>::EPSILON as f64 / 2.0);
            fn im(self) -> f64 {
                0.0
            }
            fn parts(re: f64, _im: f64) -> Self {
                re as Self
            }
        }
    };
}
macro_rules! reflector_complex {
    ($t:ty,$r:ty) => {
        impl ReflectorScalar for $t {
            const EPS: f64 = <$r>::EPSILON as f64;
            const SAFE: f64 = <$r>::MIN_POSITIVE as f64 / (<$r>::EPSILON as f64 / 2.0);
            fn im(self) -> f64 {
                self.im as f64
            }
            fn parts(re: f64, im: f64) -> Self {
                Self::new(re as $r, im as $r)
            }
        }
    };
}
reflector_real!(f32);
reflector_real!(f64);
reflector_complex!(c32, f32);
reflector_complex!(c64, f64);

// Normalize before forming differences or squares. All scalar arithmetic is
// performed on components, avoiding overflow in complex division at extremes.
pub(crate) unsafe fn generate<T: ReflectorScalar>(
    n: isize,
    a: *mut T,
    x: *mut T,
    inc: isize,
    tau: *mut T,
    positive: bool,
) {
    *tau = T::z();
    if n <= 0 {
        return;
    }
    let (ar, ai) = ((*a).re(), (*a).im());
    let mut scale = ar.abs().max(ai.abs());
    for i in 0..n - 1 {
        let z = *x.offset(i * inc);
        scale = scale.max(z.re().abs()).max(z.im().abs());
    }
    if scale == 0.0 {
        return;
    }
    let (r, im) = (ar / scale, ai / scale);
    // Scaled sum of squares for the tail. `scale` is already the largest
    // component, so every normalized term is at most one: a plain accumulation
    // is overflow-free and avoids the two divisions per element that a
    // dynamic-scale update would take. Subnormal columns keep the division
    // form because their reciprocal overflows.
    let inv_scale = 1.0 / scale;
    let mut tail_ssq = 0.0f64;
    if inv_scale.is_finite() {
        for i in 0..n - 1 {
            let z = *x.offset(i * inc);
            let (xr, xi) = (z.re() * inv_scale, z.im() * inv_scale);
            tail_ssq += xr * xr + xi * xi;
        }
    } else {
        for i in 0..n - 1 {
            let z = *x.offset(i * inc);
            let (xr, xi) = (z.re() / scale, z.im() / scale);
            tail_ssq += xr * xr + xi * xi;
        }
    }
    let tail = tail_ssq.sqrt();
    if im == 0.0 && (tail == 0.0 || (positive && tail <= T::EPS * r.abs())) {
        if positive && r < 0.0 {
            *tau = T::real(2.0);
            *a = T::real(-ar);
            for i in 0..n - 1 {
                *x.offset(i * inc) = T::z();
            }
        }
        return;
    }
    let norm = r.hypot(im).hypot(tail);
    let beta = if positive { norm } else { -norm.copysign(r) };
    let dr = if positive && r >= 0.0 {
        -(im * (im / (r + norm)) + tail * (tail / (r + norm)))
    } else {
        r - beta
    };
    let tr = -dr / beta;
    let ti = -im / beta;
    if positive && tr.hypot(ti) <= T::SAFE {
        if ai == 0.0 && ar >= 0.0 {
            return;
        }
        let norm = r.hypot(im);
        *tau = T::parts(1.0 - r / norm, -im / norm);
        *a = T::real(norm * scale);
        for i in 0..n - 1 {
            *x.offset(i * inc) = T::z();
        }
        return;
    }
    *tau = T::parts(tr, ti);
    let divisor = dr.abs().max(im.abs());
    let (rr, ii) = (dr / divisor, im / divisor);
    let denom = rr * rr + ii * ii;
    // Two reciprocals replace two divisions per element in the loop.
    let inv_divisor = 1.0 / divisor;
    let inv_denom = 1.0 / denom;
    if inv_scale.is_finite() && inv_divisor.is_finite() {
        for i in 0..n - 1 {
            let p = x.offset(i * inc);
            let z = *p;
            let (xr, xi) = (
                z.re() * inv_scale * inv_divisor,
                z.im() * inv_scale * inv_divisor,
            );
            *p = T::parts(
                (xr * rr + xi * ii) * inv_denom,
                (xi * rr - xr * ii) * inv_denom,
            );
        }
    } else {
        for i in 0..n - 1 {
            let p = x.offset(i * inc);
            let z = *p;
            let (xr, xi) = ((z.re() / scale) / divisor, (z.im() / scale) / divisor);
            *p = T::parts((xr * rr + xi * ii) / denom, (xi * rr - xr * ii) / denom);
        }
    }
    *a = T::real(beta * scale);
}

// WORK is the LAPACK caller's length-N (left) or length-M (right) buffer.
// GEMV uses the native colmat multiplication path, including conjugation.
pub(crate) unsafe fn apply<T: Scalar>(
    side: u8,
    m: isize,
    n: isize,
    v: *const T,
    inc: isize,
    tau: T,
    c: *mut T,
    ldc: isize,
    work: *mut T,
) {
    if m == 0 || n == 0 || tau == T::z() {
        return;
    }
    let left = side == b'L';
    let len = if left { m } else { n };
    let v = v.offset(start(len, inc));
    // One reflector update is tiny even when the matrix is not; a thread team
    // per call costs more than the update for the sizes used by QR panels.
    let threaded = (m.max(0) as usize).saturating_mul(n.max(0) as usize) >= (1 << 18);
    if inc == 1 && !threaded {
        // Contiguous Householder fast path: one BLAS-2 product, then a rank-1
        // update whose conjugated coefficient is hoisted out of the inner loop.
        if left {
            crate::kernels::gemv_conj_contiguous(m, n, c, ldc, v, work);
            for j in 0..n {
                let alpha = -tau * (*work.offset(j)).cj();
                crate::kernels::axpy(m, alpha, v, c.offset(j * ldc));
            }
        } else {
            crate::kernels::gemv_n_contiguous(m, n, c, ldc, v, work);
            for j in 0..n {
                let alpha = -tau * (*v.offset(j)).cj();
                crate::kernels::axpy(m, alpha, work, c.offset(j * ldc));
            }
        }
        return;
    }
    if left {
        // w = C^H v; C -= tau v w^H.
        if threaded {
            gemv(
                b'C',
                m,
                n,
                T::o(),
                c,
                ldc,
                v.offset(-start(len, inc)),
                inc,
                T::z(),
                work,
                1,
            );
        } else {
            gemv_seq(
                b'C',
                m,
                n,
                T::o(),
                c,
                ldc,
                v.offset(-start(len, inc)),
                inc,
                T::z(),
                work,
                1,
            );
        }
        for j in 0..n {
            for i in 0..m {
                *c.offset(i + j * ldc) =
                    *c.offset(i + j * ldc) - tau * *v.offset(i * inc) * (*work.offset(j)).cj();
            }
        }
    } else {
        // w = C v; C -= tau w v^H.
        if threaded {
            gemv(
                b'N',
                m,
                n,
                T::o(),
                c,
                ldc,
                v.offset(-start(len, inc)),
                inc,
                T::z(),
                work,
                1,
            );
        } else {
            gemv_seq(
                b'N',
                m,
                n,
                T::o(),
                c,
                ldc,
                v.offset(-start(len, inc)),
                inc,
                T::z(),
                work,
                1,
            );
        }
        for j in 0..n {
            for i in 0..m {
                *c.offset(i + j * ldc) =
                    *c.offset(i + j * ldc) - tau * *work.offset(i) * (*v.offset(j * inc)).cj();
            }
        }
    }
}
// Hermitian similarity H C H^H, using only the selected triangle.
unsafe fn apply_symmetric<T: Scalar>(
    u: u8,
    n: isize,
    v: *const T,
    inc: isize,
    tau: T,
    c: *mut T,
    ldc: isize,
    work: *mut T,
) {
    if n == 0 || tau == T::z() {
        return;
    }
    symv(u, n, T::o(), c, ldc, v, inc, T::z(), work, 1, T::COMPLEX);
    let v = v.offset(start(n, inc));
    let mut inner = T::z();
    for i in 0..n {
        inner = inner + (*work.offset(i)).cj() * *v.offset(i * inc);
    }
    let alpha = -T::real(0.5) * tau * inner;
    for i in 0..n {
        *work.offset(i) = *work.offset(i) + alpha * *v.offset(i * inc);
    }
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
                let p = c.offset(i + j * ldc);
                let old = if T::COMPLEX && i == j {
                    T::real((*p).re())
                } else {
                    *p
                };
                let value = old
                    - tau * *v.offset(i * inc) * (*work.offset(j)).cj()
                    - tau.cj() * *work.offset(i) * (*v.offset(j * inc)).cj();
                *p = if T::COMPLEX && i == j {
                    T::real(value.re())
                } else {
                    value
                };
            }
        }
    }
}
// Expand only the meaningful part of V. Its unit diagonal and structural
// zeros must not be read from the caller (they commonly contain R or NaNs).
unsafe fn basis<T: Scalar>(
    direct: u8,
    store: u8,
    n: usize,
    k: usize,
    v: *const T,
    ld: isize,
) -> Vec<T> {
    let mut out = vec![T::z(); n * k];
    basis_into(direct, store, n, k, v, ld, &mut out);
    out
}

/// Fill a caller-owned `n x k` buffer with the explicit reflector basis.
pub(crate) unsafe fn basis_into<T: Scalar>(
    direct: u8,
    store: u8,
    n: usize,
    k: usize,
    v: *const T,
    ld: isize,
    out: &mut [T],
) {
    debug_assert_eq!(out.len(), n * k);
    for j in 0..k {
        let pivot = if direct == b'F' { j } else { n - k + j };
        for i in 0..n {
            out[i + j * n] = if i == pivot {
                T::o()
            } else if (direct == b'F' && i < pivot) || (direct == b'B' && i > pivot) {
                T::z()
            } else if store == b'C' {
                *v.offset(i as isize + j as isize * ld)
            } else {
                (*v.offset(j as isize + i as isize * ld)).cj()
            };
        }
    }
}
pub(crate) unsafe fn factor<T: Scalar>(
    direct: u8,
    store: u8,
    n: usize,
    k: usize,
    v: *const T,
    ldv: isize,
    tau: *const T,
    t: *mut T,
    ldt: isize,
) {
    if n == 0 || k == 0 {
        return;
    }
    let b = basis(direct, store, n, k, v, ldv);
    factor_with_basis(direct, n, k, &b, tau, t, ldt);
}

/// `factor` with a precomputed basis, so a QR panel builds the basis once and
/// shares it with the trailing update.
pub(crate) unsafe fn factor_with_basis<T: Scalar>(
    direct: u8,
    n: usize,
    k: usize,
    b: &[T],
    tau: *const T,
    t: *mut T,
    ldt: isize,
) {
    if k == 0 {
        return;
    }
    let mut value = vec![T::z(); k];
    if direct == b'F' {
        // Column-wise T construction, as in LARFT: the coefficient vector
        // `w = V[:, 0..j]^H v_j` costs O(n j) for a column instead of forming
        // the full k x k Gram matrix, and the triangular recurrence below is
        // one contiguous axpy per previous column.
        let mut w = vec![T::z(); k];
        for j in 0..k {
            let tj = *tau.add(j);
            let column = t.offset(j as isize * ldt);
            if tj != T::z() {
                if j > 0 {
                    gemv_conj_contiguous(
                        (n - j) as isize,
                        j as isize,
                        b.as_ptr().add(j),
                        n as isize,
                        b.as_ptr().add(j + j * n),
                        w.as_mut_ptr(),
                    );
                    for v in &mut value[..j] {
                        *v = T::z();
                    }
                    for l in 0..j {
                        let g = w[l];
                        let source = std::slice::from_raw_parts(t.offset(l as isize * ldt), l + 1);
                        for (acc, src) in value[..=l].iter_mut().zip(source) {
                            *acc = *acc + *src * g;
                        }
                    }
                }
                for (i, v) in value[..j].iter().enumerate() {
                    *column.add(i) = -tj * *v;
                }
            } else {
                for i in 0..j {
                    *column.add(i) = T::z();
                }
            }
            *column.add(j) = tj;
        }
        return;
    }
    // Backward direction: the coefficient vector of each column depends on all
    // later columns, so one Gram matrix is still the cheapest source.
    let mut gram = vec![T::z(); k * k];
    gemm(
        b'C',
        b'N',
        k as isize,
        k as isize,
        n as isize,
        T::o(),
        b.as_ptr(),
        n as isize,
        b.as_ptr(),
        n as isize,
        T::z(),
        gram.as_mut_ptr(),
        k as isize,
    );
    for step in 0..k {
        let j = k - 1 - step;
        let tj = *tau.add(j);
        let column = t.offset(j as isize * ldt);
        if tj != T::z() {
            for v in &mut value[j + 1..k] {
                *v = T::z();
            }
            for l in j + 1..k {
                let g = gram[l + j * k];
                let source = std::slice::from_raw_parts(t.offset(l as isize * ldt).add(l), k - l);
                for (acc, src) in value[l..k].iter_mut().zip(source) {
                    *acc = *acc + *src * g;
                }
            }
            for (i, v) in value[j + 1..k].iter().enumerate() {
                *column.add(j + 1 + i) = -tj * *v;
            }
        } else {
            for i in j + 1..k {
                *column.add(i) = T::z();
            }
        }
        *column.add(j) = tj;
    }
}
pub(crate) unsafe fn apply_forward_column_block_adjoint<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
) {
    let b = basis(b'F', b'C', m, k, v, ldv);
    let mut projection = vec![T::z(); k * n];
    let mut transformed = vec![T::z(); k * n];
    apply_forward_column_block_adjoint_with_basis(
        m,
        n,
        k,
        &b,
        t,
        ldt,
        c,
        ldc,
        &mut projection,
        &mut transformed,
    );
}

/// `apply_forward_column_block_adjoint` with caller-owned basis and work
/// buffers, and one thread-team acquisition for all three GEMMs.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_forward_column_block_adjoint_with_basis<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    b: &[T],
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
    projection: &mut [T],
    transformed: &mut [T],
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert_eq!(b.len(), m * k);
    debug_assert_eq!(projection.len(), k * n);
    debug_assert_eq!(transformed.len(), k * n);
    let mut triangular = vec![T::z(); k * k];
    for j in 0..k {
        for i in 0..=j {
            triangular[i + j * k] = *t.offset(i as isize + j as isize * ldt);
        }
    }
    let (m, n, k) = (m as isize, n as isize, k as isize);
    // W = V^H C, Y = T^H W, C -= V Y.
    let work = crate::threading::real_work(
        m.max(0) as usize,
        n.max(0) as usize,
        k.max(0) as usize,
        T::COMPLEX,
    );
    crate::threading::with_par_if(work, crate::threading::QR_PANEL_WORK, |mut par| {
        gemm_with_par(
            b'C',
            b'N',
            k,
            n,
            m,
            T::o(),
            b.as_ptr(),
            m,
            c,
            ldc,
            T::z(),
            projection.as_mut_ptr(),
            k,
            par.reborrow(),
        );
        gemm_with_par(
            b'C',
            b'N',
            k,
            n,
            k,
            T::o(),
            triangular.as_ptr(),
            k,
            projection.as_ptr(),
            k,
            T::z(),
            transformed.as_mut_ptr(),
            k,
            par.reborrow(),
        );
        gemm_with_par(
            b'N',
            b'N',
            m,
            n,
            k,
            -T::o(),
            b.as_ptr(),
            m,
            transformed.as_ptr(),
            k,
            T::o(),
            c,
            ldc,
            par,
        );
    });
}
pub(crate) unsafe fn apply_block<T: Scalar>(
    side: u8,
    tr: u8,
    direct: u8,
    store: u8,
    m: usize,
    n: usize,
    k: usize,
    v: *const T,
    ldv: isize,
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    let b = basis(direct, store, order, k, v, ldv);
    let width = k * m.max(n);
    let mut triangular = vec![T::z(); k * k];
    let mut work = vec![T::z(); width];
    let mut transformed = vec![T::z(); width];
    apply_block_with_basis(
        side,
        tr,
        direct,
        m,
        n,
        k,
        &b,
        t,
        ldt,
        c,
        ldc,
        &mut triangular,
        &mut work,
        &mut transformed,
    );
}

/// `apply_block` with caller-owned scratch: the basis, the dense copy of `T`,
/// and the two `k x max(m,n)` work buffers. A caller that applies several
/// blocks (GEMQRT) sizes them once instead of allocating and zeroing per block.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_block_with_basis<T: Scalar>(
    side: u8,
    tr: u8,
    direct: u8,
    m: usize,
    n: usize,
    k: usize,
    b: &[T],
    t: *const T,
    ldt: isize,
    c: *mut T,
    ldc: isize,
    triangular: &mut [T],
    work: &mut [T],
    transformed: &mut [T],
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let order = if side == b'L' { m } else { n };
    debug_assert_eq!(b.len(), order * k);
    debug_assert_eq!(triangular.len(), k * k);
    debug_assert!(work.len() >= k * m.max(n));
    debug_assert!(transformed.len() >= k * m.max(n));
    for v in triangular.iter_mut() {
        *v = T::z();
    }
    for j in 0..k {
        for i in 0..k {
            if (direct == b'F' && i <= j) || (direct == b'B' && i >= j) {
                triangular[i + j * k] = *t.offset(i as isize + j as isize * ldt);
            }
        }
    }
    // Apply H = I - V T V^H from its factored form. Materializing the full
    // order x order matrix costs O(order^3) per panel; the three-GEMM form
    // below is the O(order^2 k) LARFB route.
    // T^H (or T^T for real) when transpose='C'/'T', T for transpose='N'.
    let t_op = if tr == b'N' {
        b'N'
    } else if <T as Scalar>::COMPLEX {
        b'C'
    } else {
        b'T'
    };
    let panel_work = crate::threading::real_work(order, m.max(n), k, T::COMPLEX);
    // One team acquisition for the three GEMMs of a block, as in
    // `apply_forward_column_block_adjoint_with_basis`; below the panel limit
    // the whole block stays sequential.
    crate::threading::with_par_if(panel_work, crate::threading::QR_PANEL_WORK, |mut par| {
        if side == b'L' {
            // W = V^H C
            gemm_with_par(
                b'C',
                b'N',
                k as isize,
                n as isize,
                order as isize,
                T::o(),
                b.as_ptr(),
                order as isize,
                c as *const T,
                ldc,
                T::z(),
                work.as_mut_ptr(),
                k as isize,
                par.reborrow(),
            );
            // W <- T^H W, then C -= V W.
            gemm_with_par(
                t_op,
                b'N',
                k as isize,
                n as isize,
                k as isize,
                T::o(),
                triangular.as_ptr(),
                k as isize,
                work.as_ptr(),
                k as isize,
                T::z(),
                transformed.as_mut_ptr(),
                k as isize,
                par.reborrow(),
            );
            gemm_with_par(
                b'N',
                b'N',
                m as isize,
                n as isize,
                k as isize,
                -T::o(),
                b.as_ptr(),
                order as isize,
                transformed.as_ptr(),
                k as isize,
                T::o(),
                c,
                ldc,
                par,
            );
        } else {
            // W = C V
            gemm_with_par(
                b'N',
                b'N',
                m as isize,
                k as isize,
                order as isize,
                T::o(),
                c as *const T,
                ldc,
                b.as_ptr(),
                order as isize,
                T::z(),
                work.as_mut_ptr(),
                m as isize,
                par.reborrow(),
            );
            // W <- W T^H, then C -= W V^H.
            gemm_with_par(
                b'N',
                t_op,
                m as isize,
                k as isize,
                k as isize,
                T::o(),
                work.as_ptr(),
                m as isize,
                triangular.as_ptr(),
                k as isize,
                T::z(),
                transformed.as_mut_ptr(),
                m as isize,
                par.reborrow(),
            );
            gemm_with_par(
                b'N',
                b'C',
                m as isize,
                n as isize,
                k as isize,
                -T::o(),
                transformed.as_ptr(),
                m as isize,
                b.as_ptr(),
                order as isize,
                T::o(),
                c,
                ldc,
                par,
            );
        }
    });
}
macro_rules! exports {
    ($p:ident,$t:ty,$i:ty,$s:tt) => { paste! {
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larfg $s>](n:*const $i,a:*mut $t,x:*mut $t,inc:*const $i,tau:*mut $t) {
            if *inc<=0 {crate::invalid(4);return;}
            generate(*n as isize,a,x,*inc as isize,tau,false);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larfgp $s>](n:*const $i,a:*mut $t,x:*mut $t,inc:*const $i,tau:*mut $t) {
            if *inc<=0 {crate::invalid(4);return;}
            generate(*n as isize,a,x,*inc as isize,tau,true);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larft $s>](d:*const u8,s:*const u8,n:*const $i,k:*const $i,
            v:*const $t,ldv:*const $i,tau:*const $t,t:*mut $t,ldt:*const $i) {
            let (d,s,n,k,lv,lt)=(flag(d),flag(s),*n as isize,*k as isize,*ldv as isize,*ldt as isize);
            for (arg,valid) in [(1,matches!(d,b'F'|b'B')),(2,matches!(s,b'C'|b'R')),
                (3,n>=0),(4,k>=0 && k<=n),(6,lv>=1.max(if s==b'C' {n} else {k})),(9,lt>=k.max(1))] {
                if !valid {crate::invalid(arg);return;}
            }
            factor(d,s,n as usize,k as usize,v,lv,tau,t,lt);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larfb $s>](side:*const u8,tr:*const u8,d:*const u8,s:*const u8,
            m:*const $i,n:*const $i,k:*const $i,v:*const $t,ldv:*const $i,t:*const $t,ldt:*const $i,
            c:*mut $t,ldc:*const $i,_work:*mut $t,ldwork:*const $i) {
            let (side,tr,d,s)=(flag(side),flag(tr),flag(d),flag(s));
            let (m,n,k,lv,lt,lc,lw)=(*m as isize,*n as isize,*k as isize,*ldv as isize,*ldt as isize,*ldc as isize,*ldwork as isize);
            let order=if side==b'L' {m} else {n};
            for (arg,valid) in [(1,matches!(side,b'L'|b'R')),(2,tr==b'N'||tr==if <$t>::COMPLEX {b'C'} else {b'T'}),
                (3,matches!(d,b'F'|b'B')),(4,matches!(s,b'C'|b'R')),(5,m>=0),(6,n>=0),(7,k>=0 && k<=order),
                (9,lv>=1.max(if s==b'C' {order} else {k})),(11,lt>=k.max(1)),(13,lc>=m.max(1)),
                (15,lw>=1.max(if side==b'L' {n} else {m}))] {
                if !valid {crate::invalid(arg);return;}
            }
            apply_block(side,tr,d,s,m as usize,n as usize,k as usize,v,lv,t,lt,c,lc);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larfy $s>](u:*const u8,n:*const $i,v:*const $t,
            inc:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t) {
            let (u,n,inc,ldc)=(flag(u),*n as isize,*inc as isize,*ldc as isize);
            for (arg,valid) in [(1,uplo(u)),(2,n>=0),(4,inc!=0),(7,ldc>=n.max(1))] {
                if !valid {crate::invalid(arg);return;}
            }
            apply_symmetric(u,n,v,inc,*tau,c,ldc,work);
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larfx $s>](side:*const u8,m:*const $i,n:*const $i,
            v:*const $t,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t) {
            let one: $i = 1;
            // LARFX permits WORK to be unused for small reflectors.
            let order = if flag(side)==b'L' {*m} else {*n};
            if order <= 10 {
                let len = if flag(side)==b'L' {*n} else {*m};
                let mut scratch = vec![<$t>::z(); len.max(0) as usize];
                [<$p larf $s>](side,m,n,v,&one,tau,c,ldc,scratch.as_mut_ptr());
            } else {
                [<$p larf $s>](side,m,n,v,&one,tau,c,ldc,work);
            }
        }
        #[no_mangle]
        pub unsafe extern "C" fn [<$p larf $s>](side:*const u8,m:*const $i,n:*const $i,
            v:*const $t,inc:*const $i,tau:*const $t,c:*mut $t,ldc:*const $i,work:*mut $t) {
            let (side,m,n,inc,ldc)=(flag(side),*m as isize,*n as isize,*inc as isize,*ldc as isize);
            for (arg,valid) in [(1,matches!(side,b'L'|b'R')),(2,m>=0),(3,n>=0),(5,inc!=0),(8,ldc>=m.max(1))] {
                if !valid {crate::invalid(arg);return;}
            }
            apply(side,m,n,v,inc,*tau,c,ldc,work);
        }
    }};
}
exports!(s, f32, i32, _);
exports!(d, f64, i32, _);
exports!(c, c32, i32, _);
exports!(z, c64, i32, _);
exports!(s, f32, i64, _64_);
exports!(d, f64, i64, _64_);
exports!(c, c32, i64, _64_);
exports!(z, c64, i64, _64_);

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};
    #[test]
    fn identity_reflector_does_not_read_storage() {
        unsafe {
            apply::<f64>(b'L', 3, 4, null(), 1, 0.0, null_mut(), 3, null_mut());
            apply_symmetric::<c64>(b'U', 3, null(), 1, c64::z(), null_mut(), 3, null_mut());
            apply_block::<f64>(
                b'L',
                b'N',
                b'F',
                b'C',
                3,
                4,
                0,
                null(),
                3,
                null(),
                1,
                null_mut(),
                3,
            );
            let mut tau = 1.0;
            generate::<f64>(0, null_mut(), null_mut(), 1, &mut tau, false);
            assert_eq!(tau, 0.0);
        }
    }
    #[test]
    fn padded_output_and_invalid_arguments() {
        unsafe {
            let (m, n, ld, inc) = (2_i32, 2_i32, 4_i32, 1_i32);
            let v = [1.0, 2.0];
            let mut c = [1.0, 2.0, 91.0, 92.0, 3.0, 4.0, 93.0, 94.0];
            let mut work = [0.0; 2];
            dlarf_(
                &b'L',
                &m,
                &n,
                v.as_ptr(),
                &inc,
                &0.2,
                c.as_mut_ptr(),
                &ld,
                work.as_mut_ptr(),
            );
            assert_eq!([c[2], c[3], c[6], c[7]], [91.0, 92.0, 93.0, 94.0]);
            dlarf_(
                &b'?',
                &m,
                &n,
                null(),
                &inc,
                &0.2,
                null_mut(),
                &ld,
                null_mut(),
            );
            assert_eq!(crate::colmat_blas_last_error(), 1);
        }
    }
}
