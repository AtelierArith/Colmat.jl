# Performance fixes for QR and Cholesky

`Colmat.jl` is a loader: it opens `libcolmat_blas` and registers LBT forwards.
The Julia side does not implement these routines. The slow code measured by the
benchmark is the Rust adapter in `deps/rust/src`, which this repository owns.

## Column-pivoted QR (`?geqp3`)

`pivoted_qr` recomputed the norm of every remaining column from scratch at every
pivot step, and `kernels::nrm2` used one `hypot` call per element. At `n=512`
this was 22x (Float64) and 15x (ComplexF64) slower than OpenBLAS, and the extra
work grew with the order.

Changes:

- `kernels::add_sq` / `kernels::nrm2`: scaled sum of squares, the same update
  LAPACK uses, instead of per-element `hypot`.
- `pivoted_qr`: keep one residual norm per column and downdate it after each
  reflector (`norm *= sqrt(1 - (r/norm)^2)`), recomputing the norm when the
  cancellation test (`temp <= sqrt(eps)`) says the update is unreliable. Pivot
  selection and the final residual keep the previous semantics.
- `householder::generate`: the reflector tail uses the same scaled accumulator
  instead of two `hypot` calls per element.

## Block reflector application (`?gemqrt`, LARFB)

`householder::apply_block` formed the full `order x order` matrix
`H = I - V T V^H` with two GEMMs and then multiplied `C` by it. That is
`O(order^3)` work per panel and `O(order^2)` memory, while LAPACK applies the
same reflector in `O(order^2 k)` with three GEMMs.

Changes:

- `apply_block` now computes `W = V^H C`, `W <- T W`, `C -= V W` for the left
  side and the transposed analogue for the right side, never materializing `H`.
- `householder::factor` computes the pairwise Gram matrix `V^H V` with one GEMM
  instead of O(k^2 n) scalar dot products. This was the largest single cost of
  QR factorization at one thread.

## Threaded QR (`?geqrt`)

`compact_qr` requested a thread team for every single active column, and
`householder::apply` acquired a team for every reflector application. Both are
BLAS-2 operations that cannot use more than one participant, so 8 threads were
slower than 1 thread.

Changes:

- the per-column native `qr_in_place` runs with `Par::Seq`;
- the per-column path factors the caller's column in place (no `to_owned` copy)
  and only rescales for subnormal/overflow ranges;
- `householder::apply` uses a sequential GEMV below 2^18 elements.

## Cholesky (`?potrf`)

The adapter copied the selected triangle into a dense temporary through
`Mat::from_fn`, which calls a closure once per element (about 45% of the call at
`n=1024`). The native LLT also wrote the opposite triangle, so the copy could
not simply be removed.

Changes:

- the temporary is allocated with an all-zero `write_bytes` fill and filled with
  per-column `copy_nonoverlapping`; only the selected triangle is copied back;
- the LLT block size is 32 instead of the automatic 128. A sweep over
  block sizes 16/32/64/128/256 and recursion thresholds 16/32/64/128 on this host
  found 32 fastest (about 16-20% better at n=512/1024).

## Panel scratch and team acquisition

`compact_qr` allocated the basis `V`, the two `k x n` trailing-update buffers,
a 1x1 coefficient, and the native column-QR workspace on every panel/column, and
each of the three trailing-update GEMMs acquired a thread team. At n=1024 with
8 threads the `with_par`/threadteam scheduling was a visible part of the
profile, and QR factor barely scaled (36.6 ms at 1T, 30.6 ms at 8T).

Changes:

- `basis_into` fills a caller buffer; `factor_with_basis` and
  `apply_forward_column_block_adjoint_with_basis` consume it, so a panel builds
  `V` once and shares it between LARFT and the trailing update.
- the QR panel loop allocates the basis, projection, transformed, coefficient
  and native workspace once, outside the loop;
- `kernels::gemm_with_par` takes an existing `Par`; the trailing update wraps
  its three GEMMs in one `with_par`.
- `kernels::gemv` writes a contiguous output vector in place instead of staging
  through an owned buffer, and `householder::apply` treats updates below 2^18
  elements as sequential (the scalar update after GEMV is serial anyway).
- Cholesky only overrides the pinned adapter's untuned `128/64` LLT defaults; a
  newer colmat-rs that already picks `64/32` keeps its choice.

## Verification

`test/qr_pivoted.jl`, `test/qr_standard.jl`, `test/qr_compact.jl`,
`test/householder.jl`, `test/block_unitary.jl`, `test/tsqr.jl`,
`test/reverse_qr.jl`, `test/lq.jl`, the LAPACKE QR/TSQR suites,
`test/lapacke_cholesky.jl`, `test/band_cholesky.jl`,
`test/packed_cholesky.jl` and `test/pivoted_cholesky.jl` pass. Reconstruction
residuals stay around 1e-15 for Float64 and ComplexF64 through order 512.

Microbenchmark (minimum of five calls, milliseconds; colmat-rs `b376b3ce`):

| Case | Before | After |
|---|---:|---:|
| `qr(A, ColumnNorm())`, f64, n=512, 1T | 228.5 | 20.5 |
| `qr(A, ColumnNorm())`, c64, n=512, 1T | 332.9 | 65.6 |
| `qr(A, ColumnNorm())`, f64, n=512, 8T | 300.0 | 41.4 |
| `qr(A, ColumnNorm())`, c64, n=512, 8T | 404.6 | 66.3 |
| `qr(A, ColumnNorm())`, f64, n=1024, 1T | 1803.7 | 154.0 |
| `qr(A, ColumnNorm())`, c64, n=1024, 1T | 2710.6 | 622.5 |
| `qr(A)`, f64, n=1024, 1T | 47.5 | 32.6 |
| `qr(A)`, f64, n=1024, 8T | 75.7 | 21.8 |
| `Matrix(qr(A).Q)`, f64, n=1024, 1T | 493.3 | 78.7 |
| `Matrix(qr(A).Q)`, c64, n=1024, 1T | 1695.9 | 292.4 |
| `cholesky`, f64, n=1024, 1T | 18.7 | 14.4 |
| `cholesky`, c64, n=1024, 1T | 53.2 | 42.7 |

## What is left

- **Cholesky** is still about 1.5-1.7x slower than OpenBLAS at 1 thread
  (`n=1024`: 14.4 vs 8.6 ms f64, 42.7 vs 28.0 ms c64). The adapter's copy is
  ~2 ms at n=1024; the rest is the native LLT kernel. colmat-rs #257 tracks the
  Skylake block-size follow-up; the adapter forces 32 on AVX2-only parts.
- **SVD** at 1 thread is about 1.14x slower for `svd` and 1.2-1.4x for
  `svdvals`. Profiling showed ~97% of the call inside
  `colmat::linalg::svd::svdd`; colmat-rs #259 tracks the values-only gap.
  At 8 threads Colmat is faster than OpenBLAS (0.70-0.90x).
- **Pivoted QR** is still unblocked and does not scale at 8 threads: 2.25x
  (f64) and 3.23x (c64) at 1T, 4.7x/5.8x at 8T at n=1024. Blocking the pivot
  panel into compact-WY form is a separate change.
- **QR factor** is at parity or better at 1 thread (0.89x f64, 1.14x c64) and
  1.05x/0.50x at 8 threads, n=1024.
- **Cholesky / GEMM / LU** at 8 threads still vary with size; the geometric
  means in `summary.md` include the 32/64/96 sizes where adapter fixed costs
  dominate.

