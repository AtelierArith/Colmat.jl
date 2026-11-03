//! Synchronous Float64 facade. Buffers are borrowed only during each call.
use faer::{linalg::solvers::Solve, Accum, Mat, MatRef, Par};
use rustcall_julia_macros::julia;
use std::panic::{catch_unwind, AssertUnwindSafe};

type FfiResult = Result<(), String>;

fn guarded(f: impl FnOnce() -> FfiResult) -> Result<(), String> {
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
