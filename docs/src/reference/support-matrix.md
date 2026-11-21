# Measured Julia API mapping

Records, separately, whether calls on ordinary arrays succeed and whether Colmat provides their entry points.
Entry points are chosen from the Julia 1.13 wrapper implementations and cross-checked against the LBT registration targets. This is not a dynamic call-count measurement or a trace of library internals.

## Verification conditions

- Julia: `1.13.4` / `x86_64-linux-gnu`.
- Colmat source: `c7bf908e993bf1f4bacc1b0daa2985361fd66e05`.
- Rust colmat source: `e1c83482f5a61449d30e989808c3ba5f751ec391`.
- Julia BLAS integer ABI: `ilp64`.
- Colmat execution width: `8`.
- Element types in order: Float32 / Float64 / ComplexF32 / ComplexF64.

## Results

| Julia API | Float32 | Float64 | ComplexF32 | ComplexF64 | Entry-point registration | Rust-side implementation / connection |
|---|---|---|---|---|---|---|
| `A * B` | success | success | success | success | Colmat | matrix product |
| `mul!(C, A, B)` | success | success | success | success | Colmat | matrix product |
| `A * x` | success | success | success | success | Colmat | matrix product |
| `dot(x, x)` | success | success | success | success | Colmat | Rust compatibility layer |
| `norm(x)` | success | success | success | success | Colmat | Rust compatibility layer |
| `BLAS.axpy!` | success | success | success | success | Colmat | Rust compatibility layer |
| `UpperTriangular(A) \ x` | success | success | success | success | Colmat | Rust compatibility layer |
| `UpperTriangular(A) \ B` | success | success | success | success | Colmat | Rust compatibility layer |
| `lu(A)` | success | success | success | success | Colmat | partial_piv_lu (connected) |
| `lu(A) \ x` | success | success | success | success | Colmat | LU solve (connected) |
| `qr(A)` | success | success | success | success | Colmat | qr (connected) |
| `qr(A, ColumnNorm())` | success | success | success | success | Colmat | col_piv_qr (connected) |
| `svd(A)` | success | success | success | success | Colmat | svd (connected) |
| `svd(A; alg=LinearAlgebra.QRIteration())` | success | success | success | success | Colmat | svd (connected) |
| `svdvals(A)` | success | success | success | success | Colmat | SVD (connected) |
| `eigen(A)` | success | success | success | success | Colmat | eigen (connected) |
| `eigen(Hermitian(A))` | success | success | success | success | Colmat | self_adjoint_eigen (connected) |
| `cholesky(Hermitian(A))` | success | success | success | success | Colmat | llt (connected) |

## Entry symbols and actual libraries

Each row verifies the same API as the table above. Symbols follow the four types in order.

| Julia API | Verified entry points | LBT registration |
|---|---|---|
| `A * B` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `mul!(C, A, B)` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `A * x` | `sgemv_`, `dgemv_`, `cgemv_`, `zgemv_` | `libcolmat_blas.so` |
| `dot(x, x)` | `cblas_sdot`, `cblas_ddot`, `cblas_cdotc_sub`, `cblas_zdotc_sub` | `libcolmat_blas.so` |
| `norm(x)` | `snrm2_`, `dnrm2_`, `scnrm2_`, `dznrm2_` | `libcolmat_blas.so` |
| `BLAS.axpy!` | `saxpy_`, `daxpy_`, `caxpy_`, `zaxpy_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ x` | `strsv_`, `dtrsv_`, `ctrsv_`, `ztrsv_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ B` | `strsm_`, `dtrsm_`, `ctrsm_`, `ztrsm_` | `libcolmat_blas.so` |
| `lu(A)` | `sgetrf_`, `dgetrf_`, `cgetrf_`, `zgetrf_` | `libcolmat_blas.so` |
| `lu(A) \ x` | `sgetrs_`, `dgetrs_`, `cgetrs_`, `zgetrs_` | `libcolmat_blas.so` |
| `qr(A)` | `sgeqrt_`, `dgeqrt_`, `cgeqrt_`, `zgeqrt_` | `libcolmat_blas.so` |
| `qr(A, ColumnNorm())` | `sgeqp3_`, `dgeqp3_`, `cgeqp3_`, `zgeqp3_` | `libcolmat_blas.so` |
| `svd(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `svd(A; alg=LinearAlgebra.QRIteration())` | `sgesvd_`, `dgesvd_`, `cgesvd_`, `zgesvd_` | `libcolmat_blas.so` |
| `svdvals(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `eigen(A)` | `sgeevx_`, `dgeevx_`, `cgeevx_`, `zgeevx_` | `libcolmat_blas.so` |
| `eigen(Hermitian(A))` | `ssyevr_`, `dsyevr_`, `cheevr_`, `zheevr_` | `libcolmat_blas.so` |
| `cholesky(Hermitian(A))` | `spotrf_`, `dpotrf_`, `cpotrf_`, `zpotrf_` | `libcolmat_blas.so` |

## What the verdicts mean

- "Success" means the probe met the reference computation or factorization residual on its small finite inputs.
- Residuals of matrix products, solves, and factorization reconstruction are checked with scalar multiply-adds that do not call BLAS.
- A success on the "existing backend" does not mean the corresponding Rust colmat factorization is connected.
- The Rust-side feature names are correspondence candidates confirmed in `faer/src/linalg/solvers.rs` at the pinned revision. This probe does not claim to have executed unconnected Rust functions directly.
- This table is not an exhaustive BLAS / LAPACK verification. It does not cover singular matrices, non-finite values, generalized eigenvalues, or other operating systems.
- It verifies high-level Julia APIs, separate from the existing regression tests such as the direct LP64 ABI tests and negative strides.

## Regeneration

Run from the repository root in a fresh Julia process after building.

```sh
julia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md
```

The table is not overwritten when verification fails. The call cases and the verification method are in `test/support_matrix.jl`.
