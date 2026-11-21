use colmat::{c32, c64, Accum, Conj, MatMut, MatRef, Par};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Accumulate `x^2` into a scaled sum of squares (`scale^2 * ssq`) without
/// overflow or underflow. This is LAPACK's `?nrm2` update step; `hypot` gives
/// the same result but is far slower in the inner loops.
#[inline]
pub(crate) fn add_sq(scale: &mut f64, ssq: &mut f64, x: f64) {
    let x = x.abs();
    if x != 0.0 {
        if *scale < x {
            let ratio = *scale / x;
            *ssq = 1.0 + *ssq * ratio * ratio;
            *scale = x;
        } else {
            let ratio = x / *scale;
            *ssq += ratio * ratio;
        }
    }
}

pub trait Scalar:
    colmat::traits::ComplexField
    + Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const COMPLEX: bool;
    /// True for `f64`: the only scalar with a dispatched AVX2 reflector path.
    const REAL64: bool;
    fn z() -> Self;
    fn o() -> Self;
    fn cj(self) -> Self;
    fn abs1(self) -> f64;
    fn magnitude(self) -> f64;
    fn re(self) -> f64;
    fn reim(self) -> (f64, f64);
    fn real(x: f64) -> Self;
    fn from_parts(re: f64, im: f64) -> Self;
}
macro_rules! real_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            const COMPLEX: bool = false;
            const REAL64: bool = core::mem::size_of::<$t>() == 8;
            fn z() -> Self {
                0.0
            }
            fn o() -> Self {
                1.0
            }
            fn cj(self) -> Self {
                self
            }
            fn abs1(self) -> f64 {
                self.abs() as f64
            }
            fn magnitude(self) -> f64 {
                self.abs() as f64
            }
            fn re(self) -> f64 {
                self as f64
            }
            fn reim(self) -> (f64, f64) {
                (self as f64, 0.0)
            }
            fn real(x: f64) -> Self {
                x as Self
            }
            fn from_parts(re: f64, _im: f64) -> Self {
                re as Self
            }
        }
    };
}
real_scalar!(f32);
real_scalar!(f64);
macro_rules! complex_scalar {
    ($t:ty, $r:ty) => {
        impl Scalar for $t {
            const COMPLEX: bool = true;
            const REAL64: bool = false;
            fn z() -> Self {
                Self::new(0.0, 0.0)
            }
            fn o() -> Self {
                Self::new(1.0, 0.0)
            }
            fn cj(self) -> Self {
                Self::new(self.re, -self.im)
            }
            fn abs1(self) -> f64 {
                self.re.abs() as f64 + self.im.abs() as f64
            }
            fn magnitude(self) -> f64 {
                (self.re as f64).hypot(self.im as f64)
            }
            fn re(self) -> f64 {
                self.re as f64
            }
            fn reim(self) -> (f64, f64) {
                (self.re as f64, self.im as f64)
            }
            fn real(x: f64) -> Self {
                Self::new(x as $r, 0.0)
            }
            fn from_parts(re: f64, im: f64) -> Self {
                Self::new(re as $r, im as $r)
            }
        }
    };
}
complex_scalar!(c32, f32);
complex_scalar!(c64, f64);

pub unsafe fn flag(p: *const u8) -> u8 {
    (*p).to_ascii_uppercase()
}
pub fn trans(t: u8) -> bool {
    matches!(t, b'N' | b'T' | b'C')
}
pub fn uplo(u: u8) -> bool {
    matches!(u, b'U' | b'L')
}
pub fn diag(d: u8) -> bool {
    matches!(d, b'U' | b'N')
}
pub fn start(n: isize, inc: isize) -> isize {
    if inc < 0 {
        (1 - n) * inc
    } else {
        0
    }
}
pub unsafe fn get<T: Scalar>(a: *const T, ld: isize, i: isize, j: isize, t: u8) -> T {
    let v = if matches!(t, b'N' | b'R') {
        *a.offset(i + j * ld)
    } else {
        *a.offset(j + i * ld)
    };
    if matches!(t, b'C' | b'R') {
        v.cj()
    } else {
        v
    }
}
// Extent through the last referenced element, excluding the final column's
// unused padding. In particular, a one-column matrix never uses its LD.
pub(crate) fn matrix_addressable<T>(rows: isize, cols: isize, ld: isize) -> bool {
    if rows < 0 || cols < 0 || ld < 0 {
        return false;
    }
    if rows == 0 || cols == 0 {
        return true;
    }
    (cols - 1)
        .checked_mul(ld)
        .and_then(|x| x.checked_add(rows))
        .and_then(|x| x.checked_mul(std::mem::size_of::<T>() as isize))
        .is_some()
}
pub unsafe fn scale<T: Scalar>(m: isize, n: isize, beta: T, c: *mut T, ldc: isize) {
    if beta == T::o() {
        return;
    }
    for j in 0..n {
        for i in 0..m {
            let p = c.offset(i + j * ldc);
            // BLAS beta=0 must not read C (which may contain uninitialized values).
            *p = if beta == T::z() { T::z() } else { beta * *p };
        }
    }
}

pub unsafe fn gemm<T: Scalar>(
    ta: u8,
    tb: u8,
    m: isize,
    n: isize,
    k: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    b: *const T,
    ldb: isize,
    beta: T,
    c: *mut T,
    ldc: isize,
) {
    let work = crate::threading::real_work(
        m.max(0) as usize,
        n.max(0) as usize,
        k.max(0) as usize,
        T::COMPLEX,
    );
    crate::threading::with_par_if(work, crate::threading::GEMM_WORK, |par| {
        gemm_with_par(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, par)
    });
}

/// GEMM on an existing parallelism token. Callers that issue several GEMMs with
/// the same shape reuse one team acquisition instead of paying it per call.
pub(crate) unsafe fn gemm_with_par<T: Scalar>(
    ta: u8,
    tb: u8,
    m: isize,
    n: isize,
    k: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    b: *const T,
    ldb: isize,
    beta: T,
    c: *mut T,
    ldc: isize,
    par: Par<'_>,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == T::z() || k == 0 {
        scale(m, n, beta, c, ldc);
        return;
    }
    let lhs = MatRef::from_raw_parts(
        a,
        m as usize,
        k as usize,
        if matches!(ta, b'N' | b'R') { 1 } else { lda },
        if matches!(ta, b'N' | b'R') { lda } else { 1 },
    );
    let rhs = MatRef::from_raw_parts(
        b,
        k as usize,
        n as usize,
        if matches!(tb, b'N' | b'R') { 1 } else { ldb },
        if matches!(tb, b'N' | b'R') { ldb } else { 1 },
    );
    let acc = if beta == T::z() {
        Accum::Replace
    } else {
        scale(m, n, beta, c, ldc);
        Accum::Add
    };
    let dst = MatMut::from_raw_parts_mut(c, m as usize, n as usize, 1, ldc);
    colmat::linalg::matmul::matmul_with_conj(
        dst,
        acc,
        lhs,
        if matches!(ta, b'C' | b'R') {
            Conj::Yes
        } else {
            Conj::No
        },
        rhs,
        if matches!(tb, b'C' | b'R') {
            Conj::Yes
        } else {
            Conj::No
        },
        alpha,
        par,
    );
}

pub unsafe fn gemv<T: Scalar>(
    t: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
) {
    crate::threading::with_par(|par| gemv_impl(t, m, n, alpha, a, lda, x, ix, beta, y, iy, par));
}

/// Sequential GEMV for small updates such as one Householder reflector
/// application. Acquiring a thread team per reflector dominates the work.
pub(crate) unsafe fn gemv_seq<T: Scalar>(
    t: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
) {
    gemv_impl(t, m, n, alpha, a, lda, x, ix, beta, y, iy, Par::Seq);
}

/// GEMV on an existing parallelism token, so a blocked routine can issue
/// several GEMVs per team acquisition instead of one.
pub(crate) unsafe fn gemv_with_par<T: Scalar>(
    t: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
    par: Par<'_>,
) {
    gemv_impl(t, m, n, alpha, a, lda, x, ix, beta, y, iy, par);
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemv_impl<T: Scalar>(
    t: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
    par: Par<'_>,
) {
    // Reference BLAS quick return also applies to a zero inner dimension.
    if m == 0 || n == 0 || (alpha == T::z() && beta == T::o()) {
        return;
    }
    let (rows, cols) = if matches!(t, b'N' | b'R') {
        (m, n)
    } else {
        (n, m)
    };
    let y = y.offset(start(rows, iy));
    if alpha == T::z() {
        for i in 0..rows {
            scale(1, 1, beta, y.offset(i * iy), 1);
        }
        return;
    }
    let x = x.offset(start(cols, ix));
    let lhs = MatRef::from_raw_parts(
        a,
        rows as usize,
        cols as usize,
        if matches!(t, b'N' | b'R') { 1 } else { lda },
        if matches!(t, b'N' | b'R') { lda } else { 1 },
    );
    let rhs = MatRef::from_raw_parts(x, cols as usize, 1, ix, 0);
    let trans = if matches!(t, b'C' | b'R') {
        Conj::Yes
    } else {
        Conj::No
    };
    // The common Householder path writes a contiguous vector. Borrow it directly
    // instead of staging through an owned buffer and copying back.
    if iy == 1 {
        if beta != T::z() {
            scale(rows, 1, beta, y, 1);
        }
        let dst = MatMut::from_raw_parts_mut(y, rows as usize, 1, 1, 1);
        let compute = |par: Par<'_>| {
            colmat::linalg::matmul::matmul_with_conj(
                dst,
                if beta == T::z() {
                    Accum::Replace
                } else {
                    Accum::Add
                },
                lhs,
                trans,
                rhs,
                Conj::No,
                alpha,
                par,
            )
        };
        compute(par);
        return;
    }
    // Strided output (including negative strides) is staged.
    let mut result = vec![T::z(); rows as usize];
    if beta != T::z() {
        for i in 0..rows {
            result[i as usize] = beta * *y.offset(i * iy);
        }
    }
    let dst = MatMut::from_raw_parts_mut(result.as_mut_ptr(), rows as usize, 1, 1, rows);
    let compute = |par: Par<'_>| {
        colmat::linalg::matmul::matmul_with_conj(
            dst,
            if beta == T::z() {
                Accum::Replace
            } else {
                Accum::Add
            },
            lhs,
            trans,
            rhs,
            Conj::No,
            alpha,
            par,
        )
    };
    compute(par);
    for i in 0..rows {
        *y.offset(i * iy) = result[i as usize];
    }
}

pub unsafe fn dot<T: Scalar>(
    n: isize,
    x: *const T,
    ix: isize,
    y: *const T,
    iy: isize,
    conjugate: bool,
) -> T {
    if n <= 0 {
        return T::z();
    }
    let x = x.offset(start(n, ix));
    let y = y.offset(start(n, iy));
    let mut sum = T::z();
    for i in 0..n {
        let v = *x.offset(i * ix);
        sum += if conjugate { v.cj() } else { v } * *y.offset(i * iy);
    }
    sum
}
/// `?laqps` F row: `y[c] = alpha * sum_i conj(A[i, c]) x[i]` for a column-major
/// block of `rows` by `cols` elements with leading dimension `lda`.
///
/// LAPACK computes this with one transposed GEMV per reflector, and that GEMV
/// reads the whole trailing matrix: it is the dominant cost of a complex
/// column-pivoted QR. The generic path walks one output column at a time, so the
/// reflector vector is re-fetched from cache for every column and the kernel
/// reaches about two thirds of the streaming read bandwidth on a trailing block
/// larger than the last-level cache. The specialized kernel keeps eight output
/// columns in flight (one pass over the reflector feeds eight columns) and
/// deinterleaves `(re, im)` with `vpermt2pd`, which matches the streaming read
/// rate. Real scalars keep the generic path, where the same shape already runs
/// at the read rate.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn conj_gemv<T: Scalar>(
    y: *mut T,
    a: *const T,
    lda: isize,
    x: *const T,
    rows: usize,
    cols: usize,
    alpha: T,
    par: Par<'_>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // The crate implements `Scalar` for f32, f64, c32 and c64, so a complex
        // scalar of 16 bytes is `Complex<f64>`: the layout `frow::run` reads.
        if T::COMPLEX
            && core::mem::size_of::<T>() == 16
            && rows >= 32
            && cols >= 1
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("fma")
        {
            let (ar, ai) = alpha.reim();
            frow::run(
                y as *mut c64,
                a as *const c64,
                lda,
                x as *const c64,
                rows,
                cols,
                ar,
                ai,
                par,
            );
            return;
        }
    }
    gemv_with_par(
        if T::COMPLEX { b'C' } else { b'T' },
        rows as isize,
        cols as isize,
        alpha,
        a,
        lda,
        x,
        1,
        T::z(),
        y,
        1,
        par,
    );
}

/// `y[i] += alpha * sum_c op(A[i, c]) x[c]` for a column-major block of `rows`
/// by `cols` elements with leading dimension `lda`, `op` conjugating when
/// `CONJ`. The panel's column update, `F` accumulation and row update are all
/// this shape, and it is a level-2 operation whose generic path leaves most of
/// the load bandwidth unused. Real scalars keep the generic GEMV.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn col_axpy<T: Scalar, const CONJ: bool>(
    y: *mut T,
    a: *const T,
    lda: isize,
    x: *const T,
    rows: usize,
    cols: usize,
    alpha: T,
    par: Par<'_>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if T::COMPLEX
            && core::mem::size_of::<T>() == 16
            && rows >= 64
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("fma")
        {
            frow::run_axpy::<CONJ>(
                y as *mut c64,
                a as *const c64,
                lda,
                x as *const c64,
                rows,
                cols,
                alpha.reim(),
                par,
            );
            return;
        }
    }
    if !T::COMPLEX || !CONJ {
        gemv_with_par(
            b'N',
            rows as isize,
            cols as isize,
            alpha,
            a,
            lda,
            x,
            1,
            T::o(),
            y,
            1,
            par,
        );
        return;
    }
    // Conjugated columns on a machine without the SIMD path.
    for c in 0..cols {
        let t = alpha * *x.add(c);
        for i in 0..rows {
            *y.add(i) += t * (*a.add(i + c * lda as usize)).cj();
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod frow {
    use super::*;
    use std::arch::x86_64::*;

    /// Deinterleave two 512-bit vectors of `(re, im)` pairs into their real and
    /// imaginary parts: 8 complex in, 8 reals out each.
    #[inline]
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn split8(x: __m512d, y: __m512d) -> (__m512d, __m512d) {
        let even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        (
            _mm512_permutex2var_pd(x, even, y),
            _mm512_permutex2var_pd(x, odd, y),
        )
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn hsum(v: __m512d) -> f64 {
        let lo = _mm512_castpd512_pd256(v);
        let hi = _mm512_extractf64x4_pd(v, 1);
        let s = _mm256_add_pd(lo, hi);
        let s = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
        _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
    }

    /// Columns `[c0, c0 + n * CN)` of the F row, `CN` columns in flight.
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn dot_cols<const CN: usize>(
        y: *mut c64,
        a: *const f64,
        lda2: usize,
        x: *const f64,
        rows: usize,
        c0: usize,
        nblocks: usize,
    ) {
        for b in 0..nblocks {
            let c = c0 + b * CN;
            let mut accre = [_mm512_setzero_pd(); CN];
            let mut accim = [_mm512_setzero_pd(); CN];
            let mut i = 0usize;
            while i + 8 <= rows {
                let (xrv, xiv) = split8(
                    _mm512_loadu_pd(x.add(2 * i)),
                    _mm512_loadu_pd(x.add(2 * i + 8)),
                );
                for k in 0..CN {
                    let p = a.add((c + k) * lda2 + 2 * i);
                    let (arv, aiv) = split8(_mm512_loadu_pd(p), _mm512_loadu_pd(p.add(8)));
                    accre[k] = _mm512_fmadd_pd(arv, xrv, accre[k]);
                    accre[k] = _mm512_fmadd_pd(aiv, xiv, accre[k]);
                    accim[k] = _mm512_fmadd_pd(arv, xiv, accim[k]);
                    accim[k] = _mm512_fnmadd_pd(aiv, xrv, accim[k]);
                }
                i += 8;
            }
            for k in 0..CN {
                let mut re = hsum(accre[k]);
                let mut im = hsum(accim[k]);
                let p = a.add((c + k) * lda2);
                let mut t = i;
                while t < rows {
                    let ar = *p.add(2 * t);
                    let ai = *p.add(2 * t + 1);
                    re += ar * *x.add(2 * t) + ai * *x.add(2 * t + 1);
                    im += ar * *x.add(2 * t + 1) - ai * *x.add(2 * t);
                    t += 1;
                }
                *y.add(c + k) = c64::new(re, im);
            }
        }
    }

    /// One column at a time for the leftover columns (at most seven).
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn col1(y: *mut f64, a: *const f64, lda2: usize, x: *const f64, rows: usize, c: usize) {
        let mut accre = _mm512_setzero_pd();
        let mut accim = _mm512_setzero_pd();
        let p = a.add(c * lda2);
        let mut i = 0usize;
        while i + 8 <= rows {
            let (xrv, xiv) = split8(
                _mm512_loadu_pd(x.add(2 * i)),
                _mm512_loadu_pd(x.add(2 * i + 8)),
            );
            let (arv, aiv) = split8(
                _mm512_loadu_pd(p.add(2 * i)),
                _mm512_loadu_pd(p.add(2 * i + 8)),
            );
            accre = _mm512_fmadd_pd(arv, xrv, accre);
            accre = _mm512_fmadd_pd(aiv, xiv, accre);
            accim = _mm512_fmadd_pd(arv, xiv, accim);
            accim = _mm512_fnmadd_pd(aiv, xrv, accim);
            i += 8;
        }
        let mut re = hsum(accre);
        let mut im = hsum(accim);
        while i < rows {
            let ar = *p.add(2 * i);
            let ai = *p.add(2 * i + 1);
            re += ar * *x.add(2 * i) + ai * *x.add(2 * i + 1);
            im += ar * *x.add(2 * i + 1) - ai * *x.add(2 * i);
            i += 1;
        }
        *y.add(2 * c) = re;
        *y.add(2 * c + 1) = im;
    }

    type Ptr3 = (
        colmat::utils::thread::Ptr<c64>,
        colmat::utils::thread::Ptr<f64>,
        colmat::utils::thread::Ptr<f64>,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_range(
        ptrs: &Ptr3,
        eight: bool,
        lda2: usize,
        rows: usize,
        c0: usize,
        nblocks: usize,
    ) {
        if eight {
            dot_cols::<8>(ptrs.0 .0, ptrs.1 .0, lda2, ptrs.2 .0, rows, c0, nblocks)
        } else {
            dot_cols::<4>(ptrs.0 .0, ptrs.1 .0, lda2, ptrs.2 .0, rows, c0, nblocks)
        }
    }

    /// Interleave `(re, im)` halves back into `(re, im)` pairs.
    #[inline]
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn join8(yr: __m512d, yi: __m512d) -> (__m512d, __m512d) {
        let lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        (
            _mm512_permutex2var_pd(yr, lo, yi),
            _mm512_permutex2var_pd(yr, hi, yi),
        )
    }

    /// `y[r] += alpha * sum_c op(A[r, c]) x[c]` over rows `[r0, r1)` of a
    /// column-major block; `op` conjugates when `CONJ`.
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn axpy_rows<const CONJ: bool>(
        y: *mut f64,
        a: *const f64,
        lda2: usize,
        x: *const f64,
        r0: usize,
        r1: usize,
        cols: usize,
        alpha: (f64, f64),
    ) {
        let (alr, ali) = alpha;
        for c in 0..cols {
            let (xr, xi) = (*x.add(2 * c), *x.add(2 * c + 1));
            let tr = alr * xr - ali * xi;
            let ti = alr * xi + ali * xr;
            let trv = _mm512_set1_pd(tr);
            let tiv = _mm512_set1_pd(ti);
            let src = a.add(c * lda2);
            let mut i = r0;
            while i + 8 <= r1 {
                let (ar, ai) = split8(
                    _mm512_loadu_pd(src.add(2 * i)),
                    _mm512_loadu_pd(src.add(2 * i + 8)),
                );
                let (yr, yi) = split8(
                    _mm512_loadu_pd(y.add(2 * i)),
                    _mm512_loadu_pd(y.add(2 * i + 8)),
                );
                let (yr, yi) = if CONJ {
                    (
                        _mm512_fmadd_pd(trv, ar, _mm512_fmadd_pd(tiv, ai, yr)),
                        _mm512_fmadd_pd(tiv, ar, _mm512_fnmadd_pd(trv, ai, yi)),
                    )
                } else {
                    (
                        _mm512_fmadd_pd(trv, ar, _mm512_fnmadd_pd(tiv, ai, yr)),
                        _mm512_fmadd_pd(tiv, ar, _mm512_fmadd_pd(trv, ai, yi)),
                    )
                };
                let (lo, hi) = join8(yr, yi);
                _mm512_storeu_pd(y.add(2 * i), lo);
                _mm512_storeu_pd(y.add(2 * i + 8), hi);
                i += 8;
            }
            while i < r1 {
                let ar = *src.add(2 * i);
                let ai = *src.add(2 * i + 1);
                let yr = *y.add(2 * i);
                let yi = *y.add(2 * i + 1);
                if CONJ {
                    *y.add(2 * i) = yr + tr * ar + ti * ai;
                    *y.add(2 * i + 1) = yi + ti * ar - tr * ai;
                } else {
                    *y.add(2 * i) = yr + tr * ar - ti * ai;
                    *y.add(2 * i + 1) = yi + ti * ar + tr * ai;
                }
                i += 1;
            }
        }
    }

    type AxpyPtrs = (
        colmat::utils::thread::Ptr<c64>,
        colmat::utils::thread::Ptr<c64>,
        colmat::utils::thread::Ptr<c64>,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn axpy_range<const CONJ: bool>(
        ptrs: &AxpyPtrs,
        lda2: usize,
        r0: usize,
        r1: usize,
        cols: usize,
        alpha: (f64, f64),
    ) {
        axpy_rows::<CONJ>(
            ptrs.0 .0 as *mut f64,
            ptrs.1 .0 as *const f64,
            lda2,
            ptrs.2 .0 as *const f64,
            r0,
            r1,
            cols,
            alpha,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_axpy<const CONJ: bool>(
        y: *mut c64,
        a: *const c64,
        lda: isize,
        x: *const c64,
        rows: usize,
        cols: usize,
        alpha: (f64, f64),
        par: Par<'_>,
    ) {
        let lda2 = 2 * lda as usize;
        let degree = par.degree().max(1);
        let workers = degree.min(rows / 256).max(1);
        let per = rows.div_ceil(workers);
        let ptrs = (
            colmat::utils::thread::Ptr(y),
            colmat::utils::thread::Ptr(a as *mut c64),
            colmat::utils::thread::Ptr(x as *mut c64),
        );
        if workers > 1 {
            colmat::utils::thread::for_each(par, workers, 0..workers, |t| {
                let r0 = t * per;
                let r1 = (r0 + per).min(rows);
                if r0 < r1 {
                    unsafe { axpy_range::<CONJ>(&ptrs, lda2, r0, r1, cols, alpha) }
                }
            });
        } else {
            unsafe { axpy_range::<CONJ>(&ptrs, lda2, 0, rows, cols, alpha) }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run(
        y: *mut c64,
        a: *const c64,
        lda: isize,
        x: *const c64,
        rows: usize,
        cols: usize,
        alpha_re: f64,
        alpha_im: f64,
        par: Par<'_>,
    ) {
        let a = a as *const f64;
        let x = x as *const f64;
        let lda2 = 2 * lda as usize;
        // Eight columns in flight measure best once the block leaves the
        // last-level cache; inside it, four columns leave more registers for the
        // load stream.
        let eight = rows.saturating_mul(cols) * 16 > 6_000_000;
        let cn = if eight { 8 } else { 4 };
        let nblocks = cols / cn;
        let degree = par.degree().max(1);
        let workers = degree.min(nblocks).max(1);
        let per = nblocks.div_ceil(workers);
        let block = if eight { 8 } else { 4 };
        if workers > 1 {
            let ptrs = (
                colmat::utils::thread::Ptr(y),
                colmat::utils::thread::Ptr(a as *mut f64),
                colmat::utils::thread::Ptr(x as *mut f64),
            );
            colmat::utils::thread::for_each(par, workers, 0..workers, |t| {
                let b0 = t * per;
                let n = nblocks.saturating_sub(b0).min(per);
                if n > 0 {
                    unsafe { run_range(&ptrs, eight, lda2, rows, b0 * block, n) }
                }
            });
        } else {
            let ptrs = (
                colmat::utils::thread::Ptr(y),
                colmat::utils::thread::Ptr(a as *mut f64),
                colmat::utils::thread::Ptr(x as *mut f64),
            );
            unsafe { run_range(&ptrs, eight, lda2, rows, 0, nblocks) }
        }
        // Leftover columns, and the alpha scaling of everything computed above.
        let yf = y as *mut f64;
        for c in (nblocks * cn)..cols {
            col1(yf, a, lda2, x, rows, c);
        }
        if alpha_re != 1.0 || alpha_im != 0.0 {
            for c in 0..cols {
                let re = *yf.add(2 * c);
                let im = *yf.add(2 * c + 1);
                *yf.add(2 * c) = alpha_re * re - alpha_im * im;
                *yf.add(2 * c + 1) = alpha_re * im + alpha_im * re;
            }
        }
    }
}

pub unsafe fn nrm2<T: Scalar>(n: isize, x: *const T, inc: isize) -> f64 {
    if n <= 0 || inc <= 0 {
        return 0.0;
    }
    // Scaled sum of squares, as in LAPACK's ?nrm2. Feeding `magnitude` would
    // take one `hypot` per element, which dominated the initial norm pass of
    // complex GEQP3; splitting the components is the same norm without it.
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for i in 0..n {
        let (re, im) = (*x.offset(i * inc)).reim();
        add_sq(&mut scale, &mut ssq, re);
        add_sq(&mut scale, &mut ssq, im);
    }
    scale * ssq.sqrt()
}
pub unsafe fn asum<T: Scalar>(n: isize, x: *const T, inc: isize) -> f64 {
    if n <= 0 || inc <= 0 {
        return 0.0;
    }
    (0..n).map(|i| (*x.offset(i * inc)).abs1()).sum()
}
pub unsafe fn iamax<T: Scalar>(n: isize, x: *const T, inc: isize) -> isize {
    if n <= 0 || inc <= 0 {
        return 0;
    }
    let mut best = 0;
    // BLAS compares ABS(real) + ABS(imag) in the scalar's real precision.
    // Widening a complex-f32 sum before comparison breaks first-index ties.
    let mut max = T::real((*x).abs1()).re();
    for i in 1..n {
        let v = T::real((*x.offset(i * inc)).abs1()).re();
        if v > max {
            max = v;
            best = i;
        }
    }
    best + 1
}
pub unsafe fn axpby<T: Scalar>(
    n: isize,
    alpha: T,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
) {
    if n <= 0 {
        return;
    }
    let y = y.offset(start(n, iy));
    if alpha == T::z() {
        for i in 0..n {
            scale(1, 1, beta, y.offset(i * iy), 1);
        }
        return;
    }
    let x = x.offset(start(n, ix));
    for i in 0..n {
        let p = y.offset(i * iy);
        *p = alpha * *x.offset(i * ix) + if beta == T::z() { T::z() } else { beta * *p };
    }
}

// Only the selected triangle is read. Hermitian diagonal imaginary parts are ignored.
pub unsafe fn symmetric<T: Scalar>(
    a: *const T,
    ld: isize,
    i: isize,
    j: isize,
    u: u8,
    hermitian: bool,
) -> T {
    if i == j && hermitian {
        return T::real((*a.offset(i + j * ld)).re());
    }
    if (u == b'U' && i <= j) || (u == b'L' && i >= j) {
        *a.offset(i + j * ld)
    } else {
        let v = *a.offset(j + i * ld);
        if hermitian {
            v.cj()
        } else {
            v
        }
    }
}
pub unsafe fn symv<T: Scalar>(
    u: u8,
    n: isize,
    alpha: T,
    a: *const T,
    ld: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
    h: bool,
) {
    symv_with_conjugation(u, n, alpha, a, ld, x, ix, beta, y, iy, h, false);
}
pub(crate) unsafe fn symv_with_conjugation<T: Scalar>(
    u: u8,
    n: isize,
    alpha: T,
    a: *const T,
    ld: isize,
    x: *const T,
    ix: isize,
    beta: T,
    y: *mut T,
    iy: isize,
    h: bool,
    conjugate: bool,
) {
    if n == 0 || (alpha == T::z() && beta == T::o()) {
        return;
    }
    let y = y.offset(start(n, iy));
    if alpha == T::z() {
        for i in 0..n {
            scale(1, 1, beta, y.offset(i * iy), 1);
        }
        return;
    }
    let x = x.offset(start(n, ix));
    for i in 0..n {
        let mut sum = T::z();
        for j in 0..n {
            let value = symmetric(a, ld, i, j, u, h);
            sum += if conjugate { value.cj() } else { value } * *x.offset(j * ix);
        }
        let p = y.offset(i * iy);
        *p = alpha * sum + if beta == T::z() { T::z() } else { beta * *p };
    }
}
pub unsafe fn ger<T: Scalar>(
    m: isize,
    n: isize,
    alpha: T,
    x: *const T,
    ix: isize,
    y: *const T,
    iy: isize,
    a: *mut T,
    lda: isize,
    c: bool,
) {
    if m == 0 || n == 0 || alpha == T::z() {
        return;
    }
    let x = x.offset(start(m, ix));
    let y = y.offset(start(n, iy));
    for j in 0..n {
        let v = *y.offset(j * iy);
        let v = alpha * if c { v.cj() } else { v };
        for i in 0..m {
            let p = a.offset(i + j * lda);
            *p += *x.offset(i * ix) * v;
        }
    }
}
// ---------------------------------------------------------------------------
// Contiguous kernels for Householder application
//
// `householder::apply` runs once per reflector, so a QR panel executes the
// BLAS-2 product `w = C^H v` and the rank-1 update `C -= tau v w^H` many times
// on shrinking shapes. The generic loops recompute the conjugated coefficient
// for every element; the kernels below hoist it and use AVX2/FMA when the CPU
// provides them, matching the dispatch faer already uses for GEMM.
// ---------------------------------------------------------------------------

#[inline]
unsafe fn axpy_scalar<T: Scalar>(n: isize, alpha: T, x: *const T, y: *mut T) {
    for i in 0..n {
        *y.offset(i) = *y.offset(i) + alpha * *x.offset(i);
    }
}

/// `y += alpha * x` for contiguous length-`n` vectors. A zero length or a null
/// input is a no-op, so degenerate callers never dereference storage.
pub(crate) unsafe fn axpy<T: Scalar>(n: isize, alpha: T, x: *const T, y: *mut T) {
    if n <= 0 || alpha == T::z() || x.is_null() || y.is_null() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            match (T::COMPLEX, std::mem::size_of::<T>()) {
                (true, 16) => {
                    let (ar, ai) = alpha.reim();
                    axpy_c64(n as usize, ar, ai, x.cast(), y.cast());
                    return;
                }
                (false, 8) => {
                    axpy_f64(n as usize, alpha.re(), x.cast(), y.cast());
                    return;
                }
                (true, 8) => {
                    let (ar, ai) = alpha.reim();
                    axpy_c32(n as usize, ar as f32, ai as f32, x.cast(), y.cast());
                    return;
                }
                (false, 4) => {
                    axpy_f32(n as usize, alpha.re() as f32, x.cast(), y.cast());
                    return;
                }
                _ => {}
            }
        }
    }
    axpy_scalar(n, alpha, x, y);
}

/// `y[j] = sum_i conj(a[i,j]) * x[i]` for a column-major `m x n` matrix and a
/// contiguous `x`.
pub(crate) unsafe fn gemv_conj_contiguous<T: Scalar>(
    m: isize,
    n: isize,
    a: *const T,
    lda: isize,
    x: *const T,
    y: *mut T,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            match (T::COMPLEX, std::mem::size_of::<T>()) {
                (true, 16) => {
                    for j in 0..n {
                        let (re, im) = dot_conj_c64(m as usize, a.offset(j * lda).cast(), x.cast());
                        *y.offset(j) = T::from_parts(re, im);
                    }
                    return;
                }
                (false, 8) => {
                    for j in 0..n {
                        *y.offset(j) = T::from_parts(
                            dot_f64(m as usize, a.offset(j * lda).cast(), x.cast()),
                            0.0,
                        );
                    }
                    return;
                }
                (true, 8) => {
                    for j in 0..n {
                        let (re, im) = dot_conj_c32(m as usize, a.offset(j * lda).cast(), x.cast());
                        *y.offset(j) = T::from_parts(re as f64, im as f64);
                    }
                    return;
                }
                (false, 4) => {
                    for j in 0..n {
                        *y.offset(j) = T::from_parts(
                            dot_f32(m as usize, a.offset(j * lda).cast(), x.cast()) as f64,
                            0.0,
                        );
                    }
                    return;
                }
                _ => {}
            }
        }
    }
    for j in 0..n {
        let mut sum = T::z();
        for i in 0..m {
            sum = sum + (*a.offset(i + j * lda)).cj() * *x.offset(i);
        }
        *y.offset(j) = sum;
    }
}

/// `y[i] = sum_j a[i,j] * x[j]` for a column-major `m x n` matrix and a
/// contiguous `x`. Replaces `y` entirely (LAPACK's `beta = 0` form).
pub(crate) unsafe fn gemv_n_contiguous<T: Scalar>(
    m: isize,
    n: isize,
    a: *const T,
    lda: isize,
    x: *const T,
    y: *mut T,
) {
    if m <= 0 || n <= 0 || a.is_null() || x.is_null() || y.is_null() {
        return;
    }
    for i in 0..m {
        *y.offset(i) = T::z();
    }
    for j in 0..n {
        let xj = *x.offset(j);
        axpy(m, xj, a.offset(j * lda), y);
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_pd(v: __m256d) -> f64 {
        let lo = _mm256_castpd256_pd128(v);
        let hi = _mm256_extractf128_pd(v, 1);
        let s = _mm_add_pd(lo, hi);
        let s = _mm_hadd_pd(s, s);
        _mm_cvtsd_f64(s)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_f64(n: usize, alpha: f64, x: *const f64, y: *mut f64) {
        let a = _mm256_set1_pd(alpha);
        let mut i = 0;
        while i + 4 <= n {
            let xv = _mm256_loadu_pd(x.add(i));
            let yv = _mm256_loadu_pd(y.add(i));
            _mm256_storeu_pd(y.add(i), _mm256_fmadd_pd(a, xv, yv));
            i += 4;
        }
        while i < n {
            *y.add(i) += alpha * *x.add(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_f32(n: usize, alpha: f32, x: *const f32, y: *mut f32) {
        let a = _mm256_set1_ps(alpha);
        let mut i = 0;
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(x.add(i));
            let yv = _mm256_loadu_ps(y.add(i));
            _mm256_storeu_ps(y.add(i), _mm256_fmadd_ps(a, xv, yv));
            i += 8;
        }
        while i < n {
            *y.add(i) += alpha * *x.add(i);
            i += 1;
        }
    }

    /// `y += (ar + i ai) * x` for interleaved complex f64.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_c64(n: usize, ar: f64, ai: f64, x: *const f64, y: *mut f64) {
        let arv = _mm256_set1_pd(ar);
        let aiv = _mm256_set1_pd(ai);
        let mut i = 0;
        // Two complex values (four doubles) per iteration.
        while i + 4 <= 2 * n {
            let xv = _mm256_loadu_pd(x.add(i));
            let xs = _mm256_permute_pd(xv, 0b0101);
            let yv = _mm256_loadu_pd(y.add(i));
            // p + [ar*xr, ar*xi]; s = [ai*xi, ai*xr]; addsub => complex product.
            let p = _mm256_mul_pd(xv, arv);
            let s = _mm256_mul_pd(xs, aiv);
            let t = _mm256_addsub_pd(p, s);
            _mm256_storeu_pd(y.add(i), _mm256_add_pd(yv, t));
            i += 4;
        }
        while i < 2 * n {
            let xr = *x.add(i);
            let xi = *x.add(i + 1);
            *y.add(i) += ar * xr - ai * xi;
            *y.add(i + 1) += ar * xi + ai * xr;
            i += 2;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_c32(n: usize, ar: f32, ai: f32, x: *const f32, y: *mut f32) {
        let arv = _mm256_set1_ps(ar);
        let aiv = _mm256_set1_ps(ai);
        let mut i = 0;
        while i + 8 <= 2 * n {
            let xv = _mm256_loadu_ps(x.add(i));
            let xs = _mm256_permute_ps(xv, 0b1011_0001);
            let yv = _mm256_loadu_ps(y.add(i));
            let p = _mm256_mul_ps(xv, arv);
            let s = _mm256_mul_ps(xs, aiv);
            let t = _mm256_addsub_ps(p, s);
            _mm256_storeu_ps(y.add(i), _mm256_add_ps(yv, t));
            i += 8;
        }
        while i < 2 * n {
            let xr = *x.add(i);
            let xi = *x.add(i + 1);
            *y.add(i) += ar * xr - ai * xi;
            *y.add(i + 1) += ar * xi + ai * xr;
            i += 2;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f64(n: usize, x: *const f64, y: *const f64) -> f64 {
        let mut acc = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= n {
            acc = _mm256_fmadd_pd(_mm256_loadu_pd(x.add(i)), _mm256_loadu_pd(y.add(i)), acc);
            i += 4;
        }
        let mut sum = hsum_pd(acc);
        while i < n {
            sum += *x.add(i) * *y.add(i);
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f32(n: usize, x: *const f32, y: *const f32) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i)), acc);
            i += 8;
        }
        let mut sum = hsum_ps(acc);
        while i < n {
            sum += *x.add(i) * *y.add(i);
            i += 1;
        }
        sum
    }

    /// `sum_i conj(a_i) * x_i` for interleaved complex f64.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_conj_c64(n: usize, a: *const f64, x: *const f64) -> (f64, f64) {
        let mut pre0 = _mm256_setzero_pd();
        let mut pre1 = _mm256_setzero_pd();
        let mut qac0 = _mm256_setzero_pd();
        let mut qac1 = _mm256_setzero_pd();
        let mut i = 0;
        // Four complex values per iteration with independent accumulators.
        while i + 8 <= 2 * n {
            let av0 = _mm256_loadu_pd(a.add(i));
            let av1 = _mm256_loadu_pd(a.add(i + 4));
            let xv0 = _mm256_loadu_pd(x.add(i));
            let xv1 = _mm256_loadu_pd(x.add(i + 4));
            let xs0 = _mm256_permute_pd(xv0, 0b0101);
            let xs1 = _mm256_permute_pd(xv1, 0b0101);
            pre0 = _mm256_fmadd_pd(av0, xv0, pre0);
            pre1 = _mm256_fmadd_pd(av1, xv1, pre1);
            qac0 = _mm256_fmadd_pd(av0, xs0, qac0);
            qac1 = _mm256_fmadd_pd(av1, xs1, qac1);
            i += 8;
        }
        while i + 4 <= 2 * n {
            let av = _mm256_loadu_pd(a.add(i));
            let xv = _mm256_loadu_pd(x.add(i));
            let xs = _mm256_permute_pd(xv, 0b0101);
            pre0 = _mm256_fmadd_pd(av, xv, pre0);
            qac0 = _mm256_fmadd_pd(av, xs, qac0);
            i += 4;
        }
        let pre = _mm256_add_pd(pre0, pre1);
        let qac = _mm256_add_pd(qac0, qac1);
        let re = hsum_pd(pre);
        let sign = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0);
        let im = hsum_pd(_mm256_mul_pd(qac, sign));
        let (mut re, mut im) = (re, im);
        while i < 2 * n {
            let ar = *a.add(i);
            let ai = *a.add(i + 1);
            let xr = *x.add(i);
            let xi = *x.add(i + 1);
            re += ar * xr + ai * xi;
            im += ar * xi - ai * xr;
            i += 2;
        }
        (re, im)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_conj_c32(n: usize, a: *const f32, x: *const f32) -> (f32, f32) {
        let mut pre0 = _mm256_setzero_ps();
        let mut pre1 = _mm256_setzero_ps();
        let mut qac0 = _mm256_setzero_ps();
        let mut qac1 = _mm256_setzero_ps();
        let mut i = 0;
        // Eight complex values per iteration with independent accumulators.
        while i + 16 <= 2 * n {
            let av0 = _mm256_loadu_ps(a.add(i));
            let av1 = _mm256_loadu_ps(a.add(i + 8));
            let xv0 = _mm256_loadu_ps(x.add(i));
            let xv1 = _mm256_loadu_ps(x.add(i + 8));
            let xs0 = _mm256_permute_ps(xv0, 0b1011_0001);
            let xs1 = _mm256_permute_ps(xv1, 0b1011_0001);
            pre0 = _mm256_fmadd_ps(av0, xv0, pre0);
            pre1 = _mm256_fmadd_ps(av1, xv1, pre1);
            qac0 = _mm256_fmadd_ps(av0, xs0, qac0);
            qac1 = _mm256_fmadd_ps(av1, xs1, qac1);
            i += 16;
        }
        while i + 8 <= 2 * n {
            let av = _mm256_loadu_ps(a.add(i));
            let xv = _mm256_loadu_ps(x.add(i));
            let xs = _mm256_permute_ps(xv, 0b1011_0001);
            pre0 = _mm256_fmadd_ps(av, xv, pre0);
            qac0 = _mm256_fmadd_ps(av, xs, qac0);
            i += 8;
        }
        let pre = _mm256_add_ps(pre0, pre1);
        let qac = _mm256_add_ps(qac0, qac1);
        let re = hsum_ps(pre);
        let sign = _mm256_set_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        let im = hsum_ps(_mm256_mul_ps(qac, sign));
        let (mut re, mut im) = (re, im);
        while i < 2 * n {
            let ar = *a.add(i);
            let ai = *a.add(i + 1);
            let xr = *x.add(i);
            let xi = *x.add(i + 1);
            re += ar * xr + ai * xi;
            im += ar * xi - ai * xr;
            i += 2;
        }
        (re, im)
    }
}

#[cfg(target_arch = "x86_64")]
use simd::*;

pub unsafe fn rank1<T: Scalar>(
    u: u8,
    n: isize,
    alpha: T,
    x: *const T,
    ix: isize,
    a: *mut T,
    lda: isize,
    h: bool,
) {
    if n == 0 || alpha == T::z() {
        return;
    }
    let x = x.offset(start(n, ix));
    for j in 0..n {
        let v = *x.offset(j * ix);
        for i in 0..n {
            if (u == b'U' && i > j) || (u == b'L' && i < j) {
                continue;
            }
            let p = a.offset(i + j * lda);
            let old = if h && i == j { T::real((*p).re()) } else { *p };
            let v = old + alpha * *x.offset(i * ix) * if h { v.cj() } else { v };
            *p = if h && i == j { T::real(v.re()) } else { v };
        }
    }
}

pub unsafe fn symm<T: Scalar>(
    side: u8,
    u: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    b: *const T,
    ldb: isize,
    beta: T,
    c: *mut T,
    ldc: isize,
    h: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == T::z() {
        scale(m, n, beta, c, ldc);
        return;
    }
    let d = if side == b'L' { m } else { n };
    let mut full = vec![T::z(); (d * d) as usize];
    for j in 0..d {
        for i in 0..d {
            full[(i + j * d) as usize] = symmetric(a, lda, i, j, u, h);
        }
    }
    if side == b'L' {
        gemm(
            b'N',
            b'N',
            m,
            n,
            m,
            alpha,
            full.as_ptr(),
            d,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    } else {
        gemm(
            b'N',
            b'N',
            m,
            n,
            n,
            alpha,
            b,
            ldb,
            full.as_ptr(),
            d,
            beta,
            c,
            ldc,
        );
    }
}

pub unsafe fn rankk<T: Scalar>(
    u: u8,
    t: u8,
    n: isize,
    k: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    b: *const T,
    ldb: isize,
    beta: T,
    c: *mut T,
    ldc: isize,
    h: bool,
    two: bool,
) {
    if n == 0 || ((alpha == T::z() || k == 0) && beta == T::o()) {
        return;
    }
    let mut out = vec![T::z(); (n * n) as usize];
    if alpha != T::z() && k != 0 {
        let other = if t == b'N' {
            if h {
                b'C'
            } else {
                b'T'
            }
        } else {
            b'N'
        };
        let (rhs, ldr) = if two { (b, ldb) } else { (a, lda) };
        gemm(
            t,
            other,
            n,
            n,
            k,
            alpha,
            a,
            lda,
            rhs,
            ldr,
            T::z(),
            out.as_mut_ptr(),
            n,
        );
        if two {
            gemm(
                t,
                other,
                n,
                n,
                k,
                if h { alpha.cj() } else { alpha },
                b,
                ldb,
                a,
                lda,
                T::o(),
                out.as_mut_ptr(),
                n,
            );
        }
    }
    for j in 0..n {
        for i in 0..n {
            if (u == b'U' && i > j) || (u == b'L' && i < j) {
                continue;
            }
            let p = c.offset(i + j * ldc);
            let old = if beta == T::z() {
                T::z()
            } else if h && i == j {
                T::real((*p).re())
            } else {
                *p
            };
            let v = out[(i + j * n) as usize] + beta * old;
            *p = if h && i == j { T::real(v.re()) } else { v };
        }
    }
}

pub unsafe fn triangular<T: Scalar>(
    a: *const T,
    ld: isize,
    i: isize,
    j: isize,
    u: u8,
    t: u8,
    d: u8,
) -> T {
    let (r, c) = if matches!(t, b'N' | b'R') {
        (i, j)
    } else {
        (j, i)
    };
    if r == c && d == b'U' {
        return T::o();
    }
    if (u == b'U' && r > c) || (u == b'L' && r < c) {
        return T::z();
    }
    get(a, ld, i, j, t)
}
pub unsafe fn trmv<T: Scalar>(
    u: u8,
    t: u8,
    d: u8,
    n: isize,
    a: *const T,
    lda: isize,
    x: *mut T,
    inc: isize,
    solve: bool,
) {
    if n == 0 {
        return;
    }
    let x = x.offset(start(n, inc));
    if !solve {
        let input: Vec<T> = (0..n).map(|i| *x.offset(i * inc)).collect();
        for i in 0..n {
            let mut sum = T::z();
            for j in 0..n {
                if (matches!(t, b'N' | b'R')) == (u == b'U') {
                    if j < i {
                        continue;
                    }
                } else if j > i {
                    continue;
                }
                sum += triangular(a, lda, i, j, u, t, d) * input[j as usize];
            }
            *x.offset(i * inc) = sum;
        }
    } else {
        let upper = (u == b'U') == (matches!(t, b'N' | b'R'));
        for step in 0..n {
            let i = if upper { n - 1 - step } else { step };
            let mut v = *x.offset(i * inc);
            let range = if upper { i + 1..n } else { 0..i };
            for j in range {
                v -= triangular(a, lda, i, j, u, t, d) * *x.offset(j * inc);
            }
            *x.offset(i * inc) = if d == b'U' {
                v
            } else {
                v / triangular(a, lda, i, i, u, t, d)
            };
        }
    }
}
pub unsafe fn trmatrix<T: Scalar>(
    side: u8,
    u: u8,
    t: u8,
    d: u8,
    m: isize,
    n: isize,
    alpha: T,
    a: *const T,
    lda: isize,
    b: *mut T,
    ldb: isize,
    solve: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if alpha == T::z() {
        scale(m, n, T::z(), b, ldb);
        return;
    }
    if !solve {
        let dim = if side == b'L' { m } else { n };
        let mut full = vec![T::z(); (dim * dim) as usize];
        for j in 0..dim {
            for i in 0..dim {
                full[(i + j * dim) as usize] = triangular(a, lda, i, j, u, t, d);
            }
        }
        let mut out = vec![T::z(); (m * n) as usize];
        if side == b'L' {
            gemm(
                b'N',
                b'N',
                m,
                n,
                m,
                alpha,
                full.as_ptr(),
                dim,
                b,
                ldb,
                T::z(),
                out.as_mut_ptr(),
                m,
            );
        } else {
            gemm(
                b'N',
                b'N',
                m,
                n,
                n,
                alpha,
                b,
                ldb,
                full.as_ptr(),
                dim,
                T::z(),
                out.as_mut_ptr(),
                m,
            );
        }
        for j in 0..n {
            for i in 0..m {
                *b.offset(i + j * ldb) = out[(i + j * m) as usize];
            }
        }
    } else {
        scale(m, n, alpha, b, ldb);
        if side == b'L' {
            for j in 0..n {
                trmv(u, t, d, m, a, lda, b.offset(j * ldb), 1, true);
            }
        } else {
            // Solve transpose(op(A)) * transpose(X) = transpose(B).
            let dim = n;
            let mut full = vec![T::z(); (dim * dim) as usize];
            for j in 0..dim {
                for i in 0..dim {
                    full[(i + j * dim) as usize] = triangular(a, lda, j, i, u, t, d);
                }
            }
            let ut = if (u == b'U') == (matches!(t, b'N' | b'R')) {
                b'L'
            } else {
                b'U'
            };
            for i in 0..m {
                trmv(ut, b'N', d, n, full.as_ptr(), dim, b.offset(i), ldb, true);
            }
        }
    }
}

#[cfg(test)]
mod conj_gemv_tests {
    use super::*;

    fn state() -> u64 {
        0x9e37_79b9_7f4a_7c15
    }
    fn next(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// The `?laqps` F row against the textbook dot product, for every path the
    /// kernel picks: the generic GEMV (small blocks and real scalars) and the
    /// specialized complex kernels (4 columns, 8 columns, and the leftovers).
    #[test]
    fn conj_gemv_matches_dot() {
        for &(rows, cols) in &[
            (1usize, 1usize),
            (7, 3),
            (33, 17),
            (64, 40),
            (100, 70),
            // the panel's F column shapes: many rows, few columns
            (500, 1),
            (500, 2),
            (500, 3),
            (500, 5),
            (500, 7),
            (500, 15),
            (977, 13),
            (97, 7),
            // rows*cols*16 > 6e6: the eight-column kernel
            (1024, 401),
        ] {
            let mut s = state();
            let mut a = vec![c64::new(0.0, 0.0); rows * cols];
            for v in a.iter_mut() {
                *v = c64::new(next(&mut s), next(&mut s));
            }
            let mut x = vec![c64::new(0.0, 0.0); rows];
            for v in x.iter_mut() {
                *v = c64::new(next(&mut s), next(&mut s));
            }
            let alpha = c64::new(0.75, -1.25);
            let mut y = vec![c64::new(9.0, 9.0); cols];
            unsafe {
                conj_gemv(
                    y.as_mut_ptr(),
                    a.as_ptr(),
                    rows as isize,
                    x.as_ptr(),
                    rows,
                    cols,
                    alpha,
                    colmat::Par::Seq,
                );
            }
            let mut worst = 0.0f64;
            for c in 0..cols {
                let mut acc = c64::new(0.0, 0.0);
                for i in 0..rows {
                    acc += a[i + c * rows].conj() * x[i];
                }
                acc = alpha * acc;
                worst = worst.max((y[c] - acc).norm());
            }
            let scale = rows as f64;
            assert!(
                worst <= 1e-12 * scale,
                "({rows}, {cols}): max |y - dot| = {worst:e}"
            );
        }
    }

    /// The threaded split must produce the same F row as the sequential run.
    #[test]
    fn conj_gemv_threaded_matches_sequential() {
        let (rows, cols) = (256usize, 203usize);
        let mut s = state();
        let mut a = vec![c64::new(0.0, 0.0); rows * cols];
        for v in a.iter_mut() {
            *v = c64::new(next(&mut s), next(&mut s));
        }
        let mut x = vec![c64::new(0.0, 0.0); rows];
        for v in x.iter_mut() {
            *v = c64::new(next(&mut s), next(&mut s));
        }
        let mut seq = vec![c64::new(0.0, 0.0); cols];
        let mut par = vec![c64::new(0.0, 0.0); cols];
        colmat::team::with_team(4, |token| unsafe {
            conj_gemv(
                seq.as_mut_ptr(),
                a.as_ptr(),
                rows as isize,
                x.as_ptr(),
                rows,
                cols,
                c64::new(1.0, 0.0),
                colmat::Par::Seq,
            );
            conj_gemv(
                par.as_mut_ptr(),
                a.as_ptr(),
                rows as isize,
                x.as_ptr(),
                rows,
                cols,
                c64::new(1.0, 0.0),
                token,
            );
        });
        for c in 0..cols {
            assert!(
                (seq[c] - par[c]).norm() <= 1e-12 * rows as f64,
                "column {c}: {} vs {}",
                seq[c],
                par[c]
            );
        }
    }
}

#[cfg(test)]
mod col_axpy_tests {
    use super::*;

    fn next(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    fn check<const CONJ: bool>(rows: usize, cols: usize, lda: usize) {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut a = vec![c64::new(0.0, 0.0); lda * cols];
        for v in a.iter_mut() {
            *v = c64::new(next(&mut s), next(&mut s));
        }
        let mut x = vec![c64::new(0.0, 0.0); cols];
        for v in x.iter_mut() {
            *v = c64::new(next(&mut s), next(&mut s));
        }
        let alpha = c64::new(0.5, 1.5);
        let mut y = vec![c64::new(0.25, -0.75); rows];
        let want: Vec<c64> = (0..rows)
            .map(|i| {
                let mut acc = c64::new(0.0, 0.0);
                for c in 0..cols {
                    let v = a[i + c * lda];
                    acc += alpha * if CONJ { v.conj() } else { v } * x[c];
                }
                y[i] + acc
            })
            .collect();
        unsafe {
            col_axpy::<c64, CONJ>(
                y.as_mut_ptr(),
                a.as_ptr(),
                lda as isize,
                x.as_ptr(),
                rows,
                cols,
                alpha,
                colmat::Par::Seq,
            );
        }
        let mut worst = 0.0f64;
        for i in 0..rows {
            worst = worst.max((y[i] - want[i]).norm());
        }
        assert!(
            worst <= 1e-12 * cols as f64,
            "CONJ={CONJ} rows={rows} cols={cols} lda={lda}: {worst:e}"
        );
    }

    #[test]
    fn col_axpy_matches_manual() {
        for &(rows, cols, lda) in &[
            // below the SIMD threshold: the scalar conjugated fallback
            (7usize, 3usize, 7usize),
            (33, 16, 33),
            (63, 5, 63),
            (64usize, 1usize, 64usize),
            (64, 3, 64),
            (65, 7, 70),
            (100, 16, 100),
            (127, 17, 130),
            (300, 5, 300),
            (1000, 16, 1000),
            (511, 33, 511),
        ] {
            check::<false>(rows, cols, lda);
            check::<true>(rows, cols, lda);
        }
    }
}
