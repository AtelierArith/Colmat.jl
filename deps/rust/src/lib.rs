//! Synchronous Float64 facade. Buffers are borrowed only during each call.
use faer::{linalg::solvers::Solve, Accum, Mat, MatRef, Par};
use rustcall_julia_macros::julia;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("faer panic");
        Err(message.to_owned().into())
    })
}

// Safety: callers supply contiguous column-major buffers with the declared
// dimensions. Julia owns and GC-preserves them; Rust never retains pointers.
unsafe fn input(ptr: *const f64, rows: usize, cols: usize) -> Result<Mat<f64>, String> {
    let len = rows.checked_mul(cols).ok_or("matrix size overflow")?;
    let values = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    };
    if values.iter().any(|x| !x.is_finite()) {
        return Err("input must be finite".into());
    }
    Ok(Mat::from_fn(rows, cols, |i, j| values[i + j * rows]))
}

unsafe fn output(value: MatRef<'_, f64>, ptr: *mut f64) {
    for j in 0..value.ncols() {
        for i in 0..value.nrows() {
            ptr.add(i + j * value.nrows()).write(value[(i, j)]);
        }
    }
}

#[julia]
pub fn matmul_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let a = input(a, m, k)?;
        let b = input(b, k, n)?;
        let mut c = Mat::<f64>::zeros(m, n);
        faer::linalg::matmul::matmul(
            c.as_mut(),
            Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            Par::Seq,
        );
        output(c.as_ref(), out);
        Ok(())
    })
}

#[julia]
pub fn solve_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    n: usize,
    nrhs: usize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let a = input(a, n, n)?;
        let b = input(b, n, nrhs)?;
        let lu = a.as_ref().partial_piv_lu(Par::Seq);
        if (0..n).any(|i| lu.U()[(i, i)] == 0.0) {
            return Err("singular matrix".into());
        }
        let x = lu.solve(b.as_ref(), Par::Seq);
        output(x.as_ref(), out);
        Ok(())
    })
}

#[julia]
pub fn svd_f64(
    a: *const f64,
    u: *mut f64,
    s: *mut f64,
    v: *mut f64,
    m: usize,
    n: usize,
) -> Result<(), String> {
    guarded(|| unsafe {
        let a = input(a, m, n)?;
        let svd = a
            .as_ref()
            .thin_svd(Par::Seq)
            .map_err(|e| format!("SVD failed: {e:?}"))?;
        output(svd.U(), u);
        output(svd.V(), v);
        for i in 0..m.min(n) {
            s.add(i).write(svd.S().column_vector()[i]);
        }
        Ok(())
    })
}

/// Opaque Rust-owned storage. No pointer into this matrix escapes to Julia.
#[julia]
pub struct OwnedMatrix {
    matrix: Mat<f64>,
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn new() -> Self {
        Self {
            matrix: Mat::zeros(0, 0),
        }
    }

    #[julia]
    pub fn load_buffer(
        &mut self,
        data: *const f64,
        rows: usize,
        cols: usize,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = unsafe { input(data, rows, cols)? };
            Ok(())
        })
    }

    #[julia]
    pub fn make_zeros(&mut self, rows: usize, cols: usize) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            self.matrix = Mat::zeros(rows, cols);
            Ok(())
        })
    }

    #[julia]
    pub fn duplicate(&self) -> Result<Self, String> {
        guarded(|| {
            Ok(Self {
                matrix: self.matrix.clone(),
            })
        })
    }

    #[julia]
    pub fn read_element(&self, row: usize, col: usize) -> Result<f64, String> {
        guarded(|| {
            if row >= self.matrix.nrows() || col >= self.matrix.ncols() {
                return Err("index out of bounds".into());
            }
            Ok(self.matrix[(row, col)])
        })
    }

    #[julia]
    pub fn write_element(&mut self, row: usize, col: usize, value: f64) -> Result<(), String> {
        guarded(|| {
            if row >= self.matrix.nrows() || col >= self.matrix.ncols() {
                return Err("index out of bounds".into());
            }
            self.matrix[(row, col)] = value;
            Ok(())
        })
    }

    #[julia]
    pub fn copy_buffer(&self, out: *mut f64, len: usize) -> Result<(), String> {
        guarded(|| {
            if len != self.matrix.nrows() * self.matrix.ncols() {
                return Err("output length mismatch".into());
            }
            unsafe {
                output(self.matrix.as_ref(), out);
            }
            Ok(())
        })
    }

    #[julia]
    pub fn assign(&mut self, other: &OwnedMatrix) -> Result<(), String> {
        guarded(|| {
            if self.matrix.shape() != other.matrix.shape() {
                return Err("matrix dimensions must match".into());
            }
            self.matrix.as_mut().copy_from(other.matrix.as_ref());
            Ok(())
        })
    }

    #[julia]
    pub fn fill_value(&mut self, value: f64) -> Result<(), String> {
        guarded(|| {
            self.matrix.as_mut().fill(value);
            Ok(())
        })
    }

    #[julia]
    pub fn add_from(&mut self, a: &OwnedMatrix, b: &OwnedMatrix, sign: f64) -> Result<(), String> {
        guarded(|| {
            if a.matrix.shape() != b.matrix.shape() {
                return Err("matrix dimensions must match".into());
            }
            self.matrix = Mat::from_fn(a.matrix.nrows(), a.matrix.ncols(), |i, j| {
                a.matrix[(i, j)] + sign * b.matrix[(i, j)]
            });
            Ok(())
        })
    }

    #[julia]
    pub fn scale_from(&mut self, a: &OwnedMatrix, factor: f64) -> Result<(), String> {
        guarded(|| {
            self.matrix = Mat::from_fn(a.matrix.nrows(), a.matrix.ncols(), |i, j| {
                factor * a.matrix[(i, j)]
            });
            Ok(())
        })
    }

    #[julia]
    pub fn multiply_from(&mut self, a: &OwnedMatrix, b: &OwnedMatrix) -> Result<(), String> {
        guarded(|| {
            if a.matrix.ncols() != b.matrix.nrows() {
                return Err("inner dimensions must match".into());
            }
            let mut matrix = Mat::zeros(a.matrix.nrows(), b.matrix.ncols());
            faer::linalg::matmul::matmul(
                matrix.as_mut(),
                Accum::Replace,
                a.matrix.as_ref(),
                b.matrix.as_ref(),
                1.0,
                Par::Seq,
            );
            self.matrix = matrix;
            Ok(())
        })
    }

    #[julia]
    pub fn transpose_from(&mut self, a: &OwnedMatrix) -> Result<(), String> {
        guarded(|| {
            self.matrix = Mat::from_fn(a.matrix.ncols(), a.matrix.nrows(), |i, j| a.matrix[(j, i)]);
            Ok(())
        })
    }

    #[julia]
    pub fn solve_from(&mut self, a: &OwnedMatrix, b: &OwnedMatrix) -> Result<(), String> {
        guarded(|| {
            let n = a.matrix.nrows();
            if n != a.matrix.ncols() || n != b.matrix.nrows() {
                return Err("solve dimensions must match".into());
            }
            if n == 0 {
                self.matrix = b.matrix.clone();
                return Ok(());
            }
            for mat in [&a.matrix, &b.matrix] {
                for j in 0..mat.ncols() {
                    for i in 0..mat.nrows() {
                        if !mat[(i, j)].is_finite() {
                            return Err("input must be finite".into());
                        }
                    }
                }
            }
            let lu = a.matrix.as_ref().partial_piv_lu(Par::Seq);
            if (0..n).any(|i| lu.U()[(i, i)] == 0.0) {
                return Err("singular matrix".into());
            }
            self.matrix = lu.solve(b.matrix.as_ref(), Par::Seq);
            Ok(())
        })
    }

    #[julia]
    pub fn svd_buffers(
        &self,
        u: *mut f64,
        s: *mut f64,
        v: *mut f64,
        full: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            check_finite(a)?;
            let (m, n) = (a.nrows(), a.ncols());
            if m.min(n) == 0 {
                // Empty full factors are identity matrices, as in Julia.
                if full {
                    for j in 0..m {
                        for i in 0..m {
                            unsafe {
                                u.add(i + j * m).write(if i == j { 1.0 } else { 0.0 });
                            }
                        }
                    }
                    for j in 0..n {
                        for i in 0..n {
                            unsafe {
                                v.add(i + j * n).write(if i == j { 1.0 } else { 0.0 });
                            }
                        }
                    }
                }
                return Ok(());
            }
            let factor = if full {
                a.svd(Par::Seq)
            } else {
                a.thin_svd(Par::Seq)
            }
            .map_err(|e| format!("SVD failed: {e:?}"))?;
            unsafe {
                output(factor.U(), u);
                output(factor.V(), v);
                for i in 0..m.min(n) {
                    s.add(i).write(factor.S().column_vector()[i]);
                }
            }
            Ok(())
        })
    }

    #[julia]
    pub fn singular_values_buffer(&self, s: *mut f64) -> Result<(), String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            check_finite(a)?;
            if a.nrows().min(a.ncols()) == 0 {
                return Ok(());
            }
            let values = a
                .singular_values(Par::Seq)
                .map_err(|e| format!("SVD failed: {e:?}"))?;
            for (i, value) in values.into_iter().enumerate() {
                unsafe {
                    s.add(i).write(value);
                }
            }
            Ok(())
        })
    }
}

fn check_finite(a: MatRef<'_, f64>) -> Result<(), String> {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if !a[(i, j)].is_finite() {
                return Err("input must be finite".into());
            }
        }
    }
    Ok(())
}

use faer::linalg::solvers::{
    ColPivQr, DenseSolveCore, FullPivLu, Lblt, Ldlt, Llt, PartialPivLu, Qr, ShapeCore, SolveLstsq,
    Svd,
};

#[derive(Clone)]
enum DenseFactor {
    Lu(PartialPivLu<f64>),
    FullLu(FullPivLu<f64>),
    Qr(Qr<f64>),
    PivotQr(ColPivQr<f64>),
    Llt(Llt<f64>),
    Ldlt(Ldlt<f64>),
    Lblt(Lblt<f64>),
    Svd(Svd<f64>),
}

macro_rules! dispatch_factor {
    ($factor:expr, $f:ident => $body:expr) => {
        match $factor {
            DenseFactor::Lu($f) => $body,
            DenseFactor::FullLu($f) => $body,
            DenseFactor::Qr($f) => $body,
            DenseFactor::PivotQr($f) => $body,
            DenseFactor::Llt($f) => $body,
            DenseFactor::Ldlt($f) => $body,
            DenseFactor::Lblt($f) => $body,
            DenseFactor::Svd($f) => $body,
        }
    };
}

/// A reusable factorization owns its numerical data independently of the input.
#[julia]
pub struct OwnedFactor {
    factor: Option<DenseFactor>,
    info: usize,
}

impl OwnedFactor {
    fn get(&self) -> Result<&DenseFactor, String> {
        self.factor
            .as_ref()
            .ok_or("factorization is not available".into())
    }
}

#[julia]
impl OwnedFactor {
    #[julia]
    pub fn new() -> Self {
        Self {
            factor: None,
            info: 0,
        }
    }

    #[julia]
    pub fn initialize(
        &mut self,
        source: &OwnedMatrix,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<usize, String> {
        guarded(|| self.initialize_with_par(source, kind, lower, Par::Seq, m, n, tag))
    }
    #[julia]
    pub fn initialize_team(
        &mut self,
        source: &OwnedMatrix,
        kind: i32,
        lower: bool,
        m: usize,
        n: usize,
        tag: i32,
        team: &crate::teams::OwnedTeam,
    ) -> Result<usize, String> {
        guarded(|| {
            team.execute(|par| self.initialize_with_par(source, kind, lower, par, m, n, tag))?
        })
    }
    #[julia]
    pub fn qr_blocksize(&self) -> Result<usize, String> {
        guarded(|| match self.get()? {
            DenseFactor::Qr(x) => Ok(x.Q_coeff().nrows()),
            DenseFactor::PivotQr(x) => Ok(x.Q_coeff().nrows()),
            _ => Err("not a QR factorization".into()),
        })
    }
    #[julia]
    pub fn clone_factor(&mut self, source: &OwnedFactor) -> Result<(), String> {
        guarded(|| {
            self.factor = source.factor.clone();
            self.info = source.info;
            Ok(())
        })
    }

    #[julia]
    pub fn permutation(&self, out: *mut usize, len: usize, column: bool) -> Result<(), String> {
        guarded(|| {
            let p = match self.get()? {
                DenseFactor::Lu(f) if !column => f.P(),
                DenseFactor::FullLu(f) => {
                    if column {
                        f.Q()
                    } else {
                        f.P()
                    }
                }
                DenseFactor::PivotQr(f) => f.P(),
                DenseFactor::Lblt(f) => f.P(),
                _ => return Err("factorization has no requested permutation".into()),
            };
            let indices = p.arrays().0;
            if len != indices.len() {
                return Err("permutation length mismatch".into());
            }
            for (i, &value) in indices.iter().enumerate() {
                unsafe {
                    out.add(i).write(value + 1);
                }
            }
            Ok(())
        })
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn factor_part(&mut self, source: &OwnedFactor, part: i32) -> Result<(), String> {
        guarded(|| {
            let f = source.get()?;
            self.matrix = match part {
                0 => dispatch_factor!(f, x => x.reconstruct(Par::Seq)),
                1 => match f {
                    DenseFactor::Lu(x) => x.L().to_owned(),
                    DenseFactor::FullLu(x) => x.L().to_owned(),
                    DenseFactor::Llt(x) => x.L().to_owned(),
                    DenseFactor::Ldlt(x) => x.L().to_owned(),
                    DenseFactor::Lblt(x) => x.L().to_owned(),
                    _ => return Err("factorization has no L factor".into()),
                },
                2 => match f {
                    DenseFactor::Lu(x) => x.U().to_owned(),
                    DenseFactor::FullLu(x) => x.U().to_owned(),
                    _ => return Err("factorization has no U factor".into()),
                },
                3 => match f {
                    DenseFactor::Qr(x) => x.compute_Q(Par::Seq),
                    DenseFactor::PivotQr(x) => x.compute_Q(Par::Seq),
                    _ => return Err("factorization has no Q factor".into()),
                },
                4 => match f {
                    DenseFactor::Qr(x) => x.thin_R().to_owned(),
                    DenseFactor::PivotQr(x) => x.thin_R().to_owned(),
                    _ => return Err("factorization has no R factor".into()),
                },
                5 => match f {
                    DenseFactor::Ldlt(x) => {
                        Mat::from_fn(
                            x.nrows(),
                            x.ncols(),
                            |i, j| if i == j { x.D()[i] } else { 0.0 },
                        )
                    }
                    DenseFactor::Lblt(x) => Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
                        if i == j {
                            x.B_diag()[i]
                        } else if i == j + 1 {
                            x.B_subdiag()[j]
                        } else if j == i + 1 {
                            x.B_subdiag()[i]
                        } else {
                            0.0
                        }
                    }),
                    _ => return Err("factorization has no D factor".into()),
                },
                6 => match f {
                    DenseFactor::Qr(x) => x.compute_thin_Q(Par::Seq),
                    DenseFactor::PivotQr(x) => x.compute_thin_Q(Par::Seq),
                    _ => return Err("factorization has no Q factor".into()),
                },
                7 => match f {
                    DenseFactor::Lu(x) => typed::packed(x.L(), x.U()),
                    DenseFactor::FullLu(x) => typed::packed(x.L(), x.U()),
                    DenseFactor::Qr(x) => typed::packed(x.Q_basis(), x.thin_R()),
                    DenseFactor::PivotQr(x) => typed::packed(x.Q_basis(), x.thin_R()),
                    DenseFactor::Llt(x) => x.L().to_owned(),
                    DenseFactor::Ldlt(x) => typed::packed_diagonal(x.L(), x.D()),
                    DenseFactor::Lblt(x) => typed::packed_diagonal(x.L(), x.B_diag()),
                    _ => return Err("no packed storage for factor".into()),
                },
                8 => match f {
                    DenseFactor::Qr(x) => x.Q_basis().to_owned(),
                    DenseFactor::PivotQr(x) => x.Q_basis().to_owned(),
                    _ => return Err("no QR basis".into()),
                },
                9 => match f {
                    DenseFactor::Qr(x) => x.Q_coeff().to_owned(),
                    DenseFactor::PivotQr(x) => x.Q_coeff().to_owned(),
                    _ => return Err("no QR coefficients".into()),
                },
                _ => return Err("unknown factor part".into()),
            };
            Ok(())
        })
    }

    #[julia]
    pub fn factor_solve(
        &mut self,
        factor: &OwnedFactor,
        rhs: &OwnedMatrix,
        transposed: bool,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = dense_factor_solve_with_par(factor, rhs, transposed, Par::Seq)?;
            Ok(())
        })
    }

    #[julia]
    pub fn factor_inverse(&mut self, source: &OwnedFactor) -> Result<(), String> {
        guarded(|| {
            let owner = source;
            if owner.info != 0 {
                return Err("singular factorization".into());
            }
            let f = owner.get()?;
            if dispatch_factor!(f, x => x.nrows()!=x.ncols()) {
                return Err("inverse requires a square matrix".into());
            }
            let singular = match f {
                DenseFactor::Qr(x) => numerically_singular_r(x.thin_R()),
                DenseFactor::PivotQr(x) => numerically_singular_r(x.thin_R()),
                _ => false,
            };
            if singular {
                return Err("singular factorization".into());
            }
            self.matrix = dispatch_factor!(f, x => x.inverse(Par::Seq));
            Ok(())
        })
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn reduction(&self, kind: i32) -> Result<f64, String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            Ok(match kind {
                0 => a.sum(),
                1 => a.norm_l2(),
                2 => a.norm_l1(),
                3 => {
                    if a.has_nan() {
                        f64::NAN
                    } else {
                        a.norm_max()
                    }
                }
                4 => a.squared_norm_l2(),
                5 => {
                    if a.nrows() != a.ncols() {
                        return Err("determinant requires a square matrix".into());
                    }
                    a.determinant(Par::Seq)
                }
                6 | 7 => real_extrema::extremum_f64(a, kind == 6)?,
                8 => {
                    if a.nrows() != a.ncols() {
                        return Err("trace requires a square matrix".into());
                    }
                    (0..a.nrows()).map(|i| a[(i, i)]).sum()
                }
                9 => (0..a.ncols())
                    .map(|j| (0..a.nrows()).map(|i| a[(i, j)].abs()).sum::<f64>())
                    .fold(0.0, f64::max),
                10 => (0..a.nrows())
                    .map(|i| (0..a.ncols()).map(|j| a[(i, j)].abs()).sum::<f64>())
                    .fold(0.0, f64::max),
                _ => return Err("unknown reduction".into()),
            })
        })
    }

    #[julia]
    pub fn predicate(&self, kind: i32) -> Result<bool, String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            Ok(match kind {
                0 => a.is_all_finite(),
                1 => a.has_nan(),
                2 => a.nrows() == a.ncols() && a.is_identity(),
                3 => {
                    a.nrows() == a.ncols()
                        && (0..a.ncols()).all(|j| (0..j).all(|i| a[(i, j)] == a[(j, i)]))
                }
                _ => return Err("unknown predicate".into()),
            })
        })
    }

    #[julia]
    pub fn inner_product(&self, other: &OwnedMatrix) -> Result<f64, String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            let b = other.matrix.as_ref();
            if a.shape() != b.shape() {
                return Err("inner product dimensions must match".into());
            }
            Ok((0..a.ncols())
                .map(|j| {
                    faer::linalg::matmul::dot::inner_prod(
                        a.col(j).transpose(),
                        faer::Conj::No,
                        b.col(j),
                        faer::Conj::No,
                    )
                })
                .sum())
        })
    }

    #[julia]
    pub fn kron_from(&mut self, lhs: &OwnedMatrix, rhs: &OwnedMatrix) -> Result<(), String> {
        guarded(|| {
            let a = lhs.matrix.as_ref();
            let b = rhs.matrix.as_ref();
            a.nrows()
                .checked_mul(b.nrows())
                .ok_or("Kronecker size overflow")?;
            a.ncols()
                .checked_mul(b.ncols())
                .ok_or("Kronecker size overflow")?;
            self.matrix = a.kron(b);
            Ok(())
        })
    }

    #[julia]
    pub fn sum_axis_from(&mut self, source: &OwnedMatrix, axis: i32) -> Result<(), String> {
        guarded(|| {
            let a = source.matrix.as_ref();
            self.matrix = match axis {
                1 => Mat::from_fn(1, a.ncols(), |_, j| a.col(j).sum()),
                2 => Mat::from_fn(a.nrows(), 1, |i, _| a.row(i).sum()),
                _ => return Err("sum axis must be 1 or 2".into()),
            };
            Ok(())
        })
    }
}

fn symmetric_copy(a: MatRef<'_, f64>, lower: bool) -> Result<Mat<f64>, String> {
    if a.nrows() != a.ncols() {
        return Err("self-adjoint input must be square".into());
    }
    let matrix = Mat::from_fn(a.nrows(), a.ncols(), |i, j| {
        if (lower && i >= j) || (!lower && i <= j) {
            a[(i, j)]
        } else {
            a[(j, i)]
        }
    });
    check_finite(matrix.as_ref())?;
    Ok(matrix)
}
unsafe fn output_complex(a: MatRef<'_, faer::c64>, out: *mut u8) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            out.cast::<faer::c64>()
                .add(i + j * a.nrows())
                .write(a[(i, j)]);
        }
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn eigen_buffers(
        &self,
        values: *mut u8,
        vectors: *mut u8,
        compute_vectors: bool,
        mode: i32,
        lower: bool,
        other: &OwnedMatrix,
        other_lower: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            let n = a.nrows();
            if n != a.ncols() {
                return Err("eigen input must be square".into());
            }
            let b = if mode >= 2 {
                let b = other.matrix.as_ref();
                if b.shape() != a.shape() {
                    return Err("generalized eigen inputs must have equal shapes".into());
                }
                Some(b)
            } else {
                None
            };
            if n == 0 {
                return Ok(());
            }
            match mode {
                0 => {
                    check_finite(a)?;
                    let eigenvalues = if compute_vectors {
                        let f = a.eigen(Par::Seq).map_err(|e| format!("{e:?}"))?;
                        unsafe {
                            output_complex(f.U(), vectors);
                        }
                        (0..n).map(|i| f.S()[i]).collect::<Vec<_>>()
                    } else {
                        a.eigenvalues(Par::Seq).map_err(|e| format!("{e:?}"))?
                    };
                    for (i, x) in eigenvalues.into_iter().enumerate() {
                        unsafe {
                            values.cast::<faer::c64>().add(i).write(x);
                        }
                    }
                }
                1 => {
                    let a = symmetric_copy(a, lower)?;
                    let eigenvalues = if compute_vectors {
                        let f = a
                            .as_ref()
                            .self_adjoint_eigen(faer::Side::Lower, Par::Seq)
                            .map_err(|e| format!("{e:?}"))?;
                        unsafe {
                            output(f.U(), vectors.cast::<f64>());
                        }
                        (0..n).map(|i| f.S()[i]).collect::<Vec<_>>()
                    } else {
                        a.as_ref()
                            .self_adjoint_eigenvalues(faer::Side::Lower, Par::Seq)
                            .map_err(|e| format!("{e:?}"))?
                    };
                    for (i, x) in eigenvalues.into_iter().enumerate() {
                        unsafe {
                            values.cast::<f64>().add(i).write(x);
                        }
                    }
                }
                2 => {
                    check_finite(a)?;
                    let b = b.unwrap();
                    check_finite(b)?;
                    let f = a
                        .generalized_eigen(b, Par::Seq)
                        .map_err(|e| format!("{e:?}"))?;
                    if compute_vectors {
                        unsafe {
                            output_complex(f.U(), vectors);
                        }
                    }
                    for i in 0..n {
                        unsafe {
                            values.cast::<faer::c64>().add(i).write(
                                if f.S_b()[i] == faer::c64::new(0.0, 0.0) {
                                    if f.S_a()[i] == faer::c64::new(0.0, 0.0) {
                                        faer::c64::new(f64::NAN, 0.0)
                                    } else {
                                        faer::c64::new(f64::INFINITY, 0.0)
                                    }
                                } else {
                                    f.S_a()[i] / f.S_b()[i]
                                },
                            );
                        }
                    }
                }
                3 => {
                    let a = symmetric_copy(a, lower)?;
                    let b = symmetric_copy(b.unwrap(), other_lower)?;
                    let eigenvalues = if compute_vectors {
                        let f = a
                            .as_ref()
                            .self_adjoint_generalized_eigen(b.as_ref(), Par::Seq)
                            .map_err(|e| format!("{e:?}"))?;
                        unsafe {
                            output(f.U(), vectors.cast::<f64>());
                        }
                        f.values().to_vec()
                    } else {
                        a.as_ref()
                            .self_adjoint_generalized_eigenvalues(b.as_ref(), Par::Seq)
                            .map_err(|e| format!("{e:?}"))?
                    };
                    for (i, x) in eigenvalues.into_iter().enumerate() {
                        unsafe {
                            values.cast::<f64>().add(i).write(x);
                        }
                    }
                }
                _ => return Err("unknown eigen mode".into()),
            }
            Ok(())
        })
    }

    #[julia]
    pub fn pfaffian_value(&self) -> Result<f64, String> {
        guarded(|| {
            let a = self.matrix.as_ref();
            if a.nrows() != a.ncols() {
                return Err("Pfaffian requires square input".into());
            }
            check_finite(a)?;
            for j in 0..a.ncols() {
                for i in j..a.nrows() {
                    if a[(i, j)] != -a[(j, i)] {
                        return Err("Pfaffian requires skew-symmetric input".into());
                    }
                }
            }
            let mut memory = faer::dyn_stack::MemBuffer::new(
                faer::linalg::pfaffian::pfaffian_scratch::<f64>(self.matrix.nrows()),
            );
            faer::linalg::pfaffian::pfaffian(
                self.matrix.as_ref(),
                Par::Seq,
                faer::dyn_stack::MemStack::new(&mut memory),
            )
            .map_err(|e| format!("{e:?}"))
        })
    }
}

fn numerically_singular_r(r: MatRef<'_, f64>) -> bool {
    let tolerance = f64::EPSILON * r.nrows().max(r.ncols()) as f64 * r.norm_max();
    (0..r.nrows().min(r.ncols())).any(|i| r[(i, i)].abs() <= tolerance)
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn triangular_solve_from(
        &mut self,
        source: &OwnedMatrix,
        rhs: &OwnedMatrix,
        lower: bool,
        unit: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.matrix.as_ref();
            let b = rhs.matrix.as_ref();
            check_triangular(a, lower, unit)?;
            check_finite(b)?;
            if b.nrows() != a.nrows() {
                return Err("triangular solve dimensions mismatch".into());
            }
            let mut x = b.to_owned();
            match (lower, unit) {
                (true, true) => a.solve_unit_lower_triangular_in_place(x.as_mut(), Par::Seq),
                (true, false) => a.solve_lower_triangular_in_place(x.as_mut(), Par::Seq),
                (false, true) => a.solve_unit_upper_triangular_in_place(x.as_mut(), Par::Seq),
                (false, false) => a.solve_upper_triangular_in_place(x.as_mut(), Par::Seq),
            }
            self.matrix = x;
            Ok(())
        })
    }

    #[julia]
    pub fn triangular_inverse_from(
        &mut self,
        source: &OwnedMatrix,
        lower: bool,
        unit: bool,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.matrix.as_ref();
            check_triangular(a, lower, unit)?;
            let mut x = Mat::zeros(a.nrows(), a.ncols());
            use faer::linalg::triangular_inverse::*;
            match (lower, unit) {
                (true, true) => invert_unit_lower_triangular(x.as_mut(), a, Par::Seq),
                (true, false) => invert_lower_triangular(x.as_mut(), a, Par::Seq),
                (false, true) => invert_unit_upper_triangular(x.as_mut(), a, Par::Seq),
                (false, false) => invert_upper_triangular(x.as_mut(), a, Par::Seq),
            }
            // The unit-triangular kernels write only the strict triangle.
            if unit {
                for i in 0..x.nrows() {
                    x[(i, i)] = 1.0;
                }
            }
            self.matrix = x;
            Ok(())
        })
    }
}
fn check_triangular(a: MatRef<'_, f64>, lower: bool, unit: bool) -> Result<(), String> {
    if a.nrows() != a.ncols() {
        return Err("triangular matrix must be square".into());
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if (if lower { i >= j } else { i <= j }) && !(unit && i == j) {
                if !a[(i, j)].is_finite() {
                    return Err("input triangle must be finite".into());
                }
                if i == j && a[(i, j)] == 0.0 {
                    return Err("singular triangular matrix".into());
                }
            }
        }
    }
    Ok(())
}

fn permutation_sign(p: &[usize]) -> f64 {
    let mut seen = vec![false; p.len()];
    let mut sign = 1.0;
    for start in 0..p.len() {
        if seen[start] {
            continue;
        }
        let mut i = start;
        let mut length = 0;
        while !seen[i] {
            seen[i] = true;
            i = p[i];
            length += 1;
        }
        if length % 2 == 0 {
            sign = -sign;
        }
    }
    sign
}

#[julia]
impl OwnedFactor {
    #[julia]
    pub fn logabsdet_buffer(&self, out: *mut f64) -> Result<(), String> {
        guarded(|| {
            let f = self.get()?;
            let n = dispatch_factor!(f,x=>x.nrows());
            if dispatch_factor!(f,x=>x.ncols()) != n {
                return Err("determinant requires a square factor".into());
            }
            let mut logabs = 0.0;
            let mut sign = 1.0;
            let mut accumulate = |x: f64| {
                logabs += x.abs().ln();
                sign *= x.signum();
                if x == 0.0 {
                    sign = 0.0;
                }
            };
            match f {
                DenseFactor::Lu(x) => {
                    for i in 0..n {
                        accumulate(x.U()[(i, i)]);
                    }
                    sign *= permutation_sign(x.P().arrays().0);
                }
                DenseFactor::FullLu(x) => {
                    for i in 0..n {
                        accumulate(x.U()[(i, i)]);
                    }
                    sign *= permutation_sign(x.P().arrays().0) * permutation_sign(x.Q().arrays().0);
                }
                DenseFactor::Llt(x) => {
                    for i in 0..n {
                        accumulate(x.L()[(i, i)]);
                        accumulate(x.L()[(i, i)]);
                    }
                }
                DenseFactor::Ldlt(x) => {
                    for i in 0..n {
                        accumulate(x.D()[i]);
                    }
                }
                DenseFactor::Lblt(x) => {
                    let mut i = 0;
                    while i < n {
                        if i + 1 < n && x.B_subdiag()[i] != 0.0 {
                            accumulate(
                                x.B_diag()[i] * x.B_diag()[i + 1]
                                    - x.B_subdiag()[i] * x.B_subdiag()[i],
                            );
                            i += 2;
                        } else {
                            accumulate(x.B_diag()[i]);
                            i += 1;
                        }
                    }
                }
                _ => accumulate(
                    dispatch_factor!(f,x=>x.reconstruct(Par::Seq))
                        .as_ref()
                        .determinant(Par::Seq),
                ),
            }
            unsafe {
                out.write(logabs);
                out.add(1).write(sign);
            }
            Ok(())
        })
    }
}

pub mod typed;

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn reserve_storage(&mut self, rows: usize, cols: usize) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix capacity overflow")?;
            self.matrix
                .try_reserve(rows, cols)
                .map_err(|e| format!("{e:?}"))
        })
    }
    #[julia]
    pub fn resize_storage(
        &mut self,
        rows: usize,
        cols: usize,
        value: *const u8,
    ) -> Result<(), String> {
        guarded(|| {
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            let x = unsafe { *value.cast::<f64>() };
            self.matrix.resize_with(rows, cols, |_, _| x);
            Ok(())
        })
    }
    #[julia]
    pub fn truncate_storage(&mut self, rows: usize, cols: usize) -> Result<(), String> {
        guarded(|| {
            if rows > self.matrix.nrows() || cols > self.matrix.ncols() {
                return Err("truncate cannot grow a matrix".into());
            }
            self.matrix.truncate(rows, cols);
            Ok(())
        })
    }
    #[julia]
    pub fn identity_storage(&mut self, rows: usize, cols: usize, tag: i32) -> Result<(), String> {
        guarded(|| {
            if tag != 1 {
                return Err("Float64 owner requires Float64 scalar tag".into());
            }
            rows.checked_mul(cols).ok_or("matrix size overflow")?;
            self.matrix = Mat::identity(rows, cols);
            Ok(())
        })
    }
}

pub mod collections;

pub mod views;

pub mod permutations;

pub mod pseudoinverse;

pub mod schur;

pub mod qz;

pub mod sparse;
pub mod sparse_triangular;

pub mod sparsity;

pub mod borrowed_sparse;

pub mod sparse_factors;

pub mod ordering;

pub mod rotations;

pub mod householder;

pub mod householder_sequence;

pub mod operators;

pub mod statistics;

pub mod view_reductions;

pub mod balance;

pub mod tridiagonal;

pub mod bidiagonal;

pub mod random;

pub mod npy;

pub mod structured_products;

pub mod diagonal_products;

pub mod pivoted_cholesky;

mod sparse_lu_components;
mod sparse_lu_owned;

pub mod teams;

pub mod workspace;

pub mod svd_workspace;

pub mod evd_workspace;

pub mod general_evd_workspace;

pub mod selfadj_gevd_workspace;

pub mod general_gevd_workspace;

pub mod spectral_factors;

pub mod spectral_solve;

pub mod sparse_qr_components;
mod sparse_qr_owned;
mod sparse_qr_wide;

impl OwnedFactor {
    fn initialize_with_par(
        &mut self,
        source: &OwnedMatrix,
        kind: i32,
        lower: bool,
        par: Par<'_>,
        m: usize,
        n: usize,
        tag: i32,
    ) -> Result<usize, String> {
        let a = source.matrix.as_ref();
        if tag != 1 || a.shape() != (m, n) {
            return Err(
                "factor input native scalar type/dimensions do not match requested metadata".into(),
            );
        }
        if (4..=6).contains(&kind) {
            if a.nrows() != a.ncols() {
                return Err("factorization requires square input".into());
            }
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    if (if lower { i >= j } else { i <= j }) && !a[(i, j)].is_finite() {
                        return Err("input triangle must be finite".into());
                    }
                }
            }
        } else {
            check_finite(a)?;
        }
        let side = if lower {
            faer::Side::Lower
        } else {
            faer::Side::Upper
        };
        self.info = 0;
        let factor = match kind {
            0 => DenseFactor::Lu(a.partial_piv_lu(par)),
            1 => DenseFactor::FullLu(a.full_piv_lu(par)),
            2 => DenseFactor::Qr(a.qr(par)),
            3 => DenseFactor::PivotQr(a.col_piv_qr(par)),
            4 => match a.llt(side, par) {
                Ok(f) => DenseFactor::Llt(f),
                Err(faer::linalg::solvers::LltError::NonPositivePivot { index }) => {
                    self.factor = None;
                    self.info = index + 1;
                    return Ok(self.info);
                }
            },
            5 => match a.ldlt(side, par) {
                Ok(f) => DenseFactor::Ldlt(f),
                Err(faer::linalg::solvers::LdltError::ZeroPivot { index }) => {
                    self.factor = None;
                    self.info = index + 1;
                    return Ok(self.info);
                }
            },
            6 => DenseFactor::Lblt(a.lblt(side, par)),
            7 => DenseFactor::Svd(a.thin_svd(par).map_err(|e| format!("{e:?}"))?),
            _ => return Err("unknown factorization kind".into()),
        };
        let diagonal = match &factor {
            DenseFactor::Lu(f) => Some(f.U()),
            DenseFactor::FullLu(f) => Some(f.U()),
            _ => None,
        };
        if let Some(u) = diagonal {
            self.info = (0..u.nrows().min(u.ncols()))
                .find(|&i| u[(i, i)] == 0.0)
                .map_or(0, |i| i + 1);
        }
        if let DenseFactor::Lblt(x) = &factor {
            let mut i = 0;
            while i < x.nrows() {
                let block = i + 1 < x.nrows() && x.B_subdiag()[i] != 0.0;
                let determinant = if block {
                    x.B_diag()[i] * x.B_diag()[i + 1] - x.B_subdiag()[i] * x.B_subdiag()[i]
                } else {
                    x.B_diag()[i]
                };
                if determinant == 0.0 {
                    self.info = i + 1;
                    break;
                }
                i += if block { 2 } else { 1 };
            }
        }
        self.factor = Some(factor);
        Ok(self.info)
    }
}

#[cfg(test)]
mod dense_team_tests {
    use super::*;
    #[test]
    fn team_factor_owns_results_and_rejects_closed_authority() {
        let a = OwnedMatrix {
            matrix: Mat::from_fn(4, 4, |i, j| if i == j { 4.0 + i as f64 } else { 0.1 }),
        };
        let mut team = teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        team.execute(|par| assert!(matches!(par, Par::Team(_))))
            .unwrap();
        let mut f = OwnedFactor::new();
        assert_eq!(f.initialize_team(&a, 0, true, 4, 4, 1, &team).unwrap(), 0);
        team.team_close().unwrap();
        assert!(f.initialize_team(&a, 0, true, 4, 4, 1, &team).is_err());
        match f.get().unwrap() {
            DenseFactor::Lu(x) => assert!((x.reconstruct(Par::Seq) - &a.matrix).norm_max() < 1e-12),
            _ => panic!("wrong factor kind"),
        }
        assert!(f.initialize(&a, 0, true, 5, 4, 1).is_err());
        assert!(f.initialize(&a, 0, true, 4, 4, 0).is_err());
    }
    #[test]
    fn typed_factor_checks_current_dtype_and_shape_before_construction() {
        let a = typed::TypedMatrix {
            data: Some(typed::MatrixData::C32(Mat::identity(3, 3))),
        };
        let mut team = teams::OwnedTeam::new();
        team.team_initialize(2).unwrap();
        let mut f = typed::TypedFactor::new();
        assert!(f.initialize_team(&a, 2, true, 3, 3, 0, &team).is_err());
        assert!(f.data().is_err());
        assert!(f.initialize_team(&a, 2, true, 4, 3, 2, &team).is_err());
        assert!(f.data().is_err());
        assert_eq!(f.initialize_team(&a, 2, true, 3, 3, 2, &team).unwrap(), 0);
        team.team_close().unwrap();
        assert!(f.data().is_ok());
    }
}

#[cfg(test)]
mod allocation_probe;

mod dense_solve_team;

fn dense_factor_solve_with_par(
    factor: &OwnedFactor,
    rhs: &OwnedMatrix,
    transposed: bool,
    mut par: Par<'_>,
) -> Result<Mat<f64>, String> {
    let owner = factor;
    let f = owner.get()?;
    if owner.info != 0 {
        return Err("singular factorization".into());
    }
    let b = rhs.matrix.as_ref();
    check_finite(b)?;
    let m = dispatch_factor!(f, x => x.nrows());
    let n = dispatch_factor!(f, x => x.ncols());
    if b.nrows() != if transposed { n } else { m } {
        return Err("solve dimensions mismatch".into());
    }
    let rank_deficient = match f {
        DenseFactor::Qr(x) => numerically_singular_r(x.thin_R()),
        DenseFactor::PivotQr(x) => numerically_singular_r(x.thin_R()),
        DenseFactor::Svd(_) => true,
        _ => false,
    };
    let result = if rank_deficient {
        let a = dispatch_factor!(f, x => x.reconstruct(par.reborrow()));
        let a = if transposed {
            a.as_ref().transpose().to_owned()
        } else {
            a
        };
        dense_pseudoinverse_solve(a.as_ref(), b, par.reborrow())?
    } else if m == n {
        if transposed {
            dispatch_factor!(f, x => x.solve_transpose(b, par.reborrow()))
        } else {
            dispatch_factor!(f, x => x.solve(b, par.reborrow()))
        }
    } else if !transposed && m >= n {
        match f {
            DenseFactor::Qr(x) => x.solve_lstsq(b, par.reborrow()),
            DenseFactor::PivotQr(x) => x.solve_lstsq(b, par.reborrow()),
            DenseFactor::Svd(x) => x.solve_lstsq(b, par.reborrow()),
            _ => return Err("rectangular solve requires QR or SVD".into()),
        }
    } else {
        let a = dispatch_factor!(f, x => x.reconstruct(par.reborrow()));
        let a = if transposed {
            a.as_ref().transpose().to_owned()
        } else {
            a
        };
        dense_pseudoinverse_solve(a.as_ref(), b, par.reborrow())?
    };
    Ok(result)
}
fn dense_pseudoinverse_solve(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    mut par: Par<'_>,
) -> Result<Mat<f64>, String> {
    if a.nrows().min(a.ncols()) == 0 {
        return Ok(Mat::zeros(a.ncols(), b.ncols()));
    }
    let inverse = a
        .thin_svd(par.reborrow())
        .map_err(|e| format!("{e:?}"))?
        .pseudoinverse(par.reborrow());
    let mut x = Mat::zeros(a.ncols(), b.ncols());
    faer::linalg::matmul::matmul(x.as_mut(), Accum::Replace, inverse.as_ref(), b, 1.0, par);
    Ok(x)
}

mod real_extrema;

mod storage_capacity;

mod storage_initializer;

mod filled_construction;
