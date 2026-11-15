use colmat::{c32, c64, Accum, Conj, MatMut, MatRef};
use std::ops::{Add, Div, Mul, Neg, Sub};

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
    fn z() -> Self;
    fn o() -> Self;
    fn cj(self) -> Self;
    fn abs1(self) -> f64;
    fn magnitude(self) -> f64;
    fn re(self) -> f64;
    fn real(x: f64) -> Self;
}
macro_rules! real_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            const COMPLEX: bool = false;
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
            fn real(x: f64) -> Self {
                x as Self
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
            fn real(x: f64) -> Self {
                Self::new(x as $r, 0.0)
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
    let v = if t == b'N' {
        *a.offset(i + j * ld)
    } else {
        *a.offset(j + i * ld)
    };
    if t == b'C' {
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
        if ta == b'N' { 1 } else { lda },
        if ta == b'N' { lda } else { 1 },
    );
    let rhs = MatRef::from_raw_parts(
        b,
        k as usize,
        n as usize,
        if tb == b'N' { 1 } else { ldb },
        if tb == b'N' { ldb } else { 1 },
    );
    let acc = if beta == T::z() {
        Accum::Replace
    } else {
        scale(m, n, beta, c, ldc);
        Accum::Add
    };
    let dst = MatMut::from_raw_parts_mut(c, m as usize, n as usize, 1, ldc);
    crate::threading::with_par(|par| {
        colmat::linalg::matmul::matmul_with_conj(
            dst,
            acc,
            lhs,
            if ta == b'C' { Conj::Yes } else { Conj::No },
            rhs,
            if tb == b'C' { Conj::Yes } else { Conj::No },
            alpha,
            par,
        );
    });
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
    // Reference BLAS quick return also applies to a zero inner dimension.
    if m == 0 || n == 0 {
        return;
    }
    let (rows, cols) = if t == b'N' { (m, n) } else { (n, m) };
    let y = y.offset(start(rows, iy));
    if alpha == T::z() {
        for i in 0..rows {
            scale(1, 1, beta, y.offset(i * iy), 1);
        }
        return;
    }
    let x = x.offset(start(cols, ix));
    // A vector is a one-column matrix. Negative input/output strides are valid.
    // Stage the result to support strided output. The public buffers never
    // become Rust-owned and no borrowed view escapes this call.
    let mut result = vec![T::z(); rows as usize];
    if beta != T::z() {
        for i in 0..rows {
            result[i as usize] = beta * *y.offset(i * iy);
        }
    }
    let lhs = MatRef::from_raw_parts(
        a,
        rows as usize,
        cols as usize,
        if t == b'N' { 1 } else { lda },
        if t == b'N' { lda } else { 1 },
    );
    let rhs = MatRef::from_raw_parts(x, cols as usize, 1, ix, 0);
    let dst = MatMut::from_raw_parts_mut(result.as_mut_ptr(), rows as usize, 1, 1, rows);
    crate::threading::with_par(|par| {
        colmat::linalg::matmul::matmul_with_conj(
            dst,
            if beta == T::z() {
                Accum::Replace
            } else {
                Accum::Add
            },
            lhs,
            if t == b'C' { Conj::Yes } else { Conj::No },
            rhs,
            Conj::No,
            alpha,
            par,
        )
    });
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
pub unsafe fn nrm2<T: Scalar>(n: isize, x: *const T, inc: isize) -> f64 {
    if n <= 0 || inc <= 0 {
        return 0.0;
    }
    // hypot avoids overflow/underflow from squaring the entries directly.
    let mut norm = 0.0f64;
    for i in 0..n {
        norm = norm.hypot((*x.offset(i * inc)).magnitude());
    }
    norm
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
    let mut max = (*x).abs1();
    for i in 1..n {
        let v = (*x.offset(i * inc)).abs1();
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
    if n == 0 {
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
            sum += symmetric(a, ld, i, j, u, h) * *x.offset(j * ix);
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
    if n == 0 {
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
    let (r, c) = if t == b'N' { (i, j) } else { (j, i) };
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
                if (t == b'N') == (u == b'U') {
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
        let upper = (u == b'U') == (t == b'N');
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
            let ut = if (u == b'U') == (t == b'N') {
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
