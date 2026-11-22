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

- the temporary is isolated from caller storage and filled with per-column
  `copy_nonoverlapping`; only the selected triangle is copied back (it is now
  per-thread and reused, see the UPLO section below);
- the LLT block size is 32 instead of the automatic 128. A sweep over
  block sizes 16/32/64/128/256 and recursion thresholds 16/32/64/128 on this host
  found 32 fastest (about 16-20% better at n=512/1024).

## Cholesky UPLO='U', scratch reuse and the resident team

Two costs were outside the LLT kernel itself. The benchmark calls
`cholesky(Hermitian(A))`, whose default `UPLO='U'` took the adapter's
per-element strided transpose branch, with a `cj()` and a multiply-add per
element for both the copy in and the copy out. The adapter also allocated and
zeroed a fresh `n x n` temporary on every call. At `n=1024` the 8 MiB
allocation and its page faults showed up as jitter even though the temporary is
only borrowed for one factorization.

Changes:

- factor in place. LAPACK owns only the selected triangle; the native LLT writes
  some of the opposite triangle as scratch, so that part is saved and restored.
- for `UPLO='U'` there is **no save at all**. The lower triangle of the
  transposed view is the caller's upper triangle, and the native leaf stages a
  row-major view through a column-major copy of its *lower* triangle; it never
  writes the caller's lower (unselected) triangle. This was verified bit for bit
  at n=1..1024, widths 1/4/8, f64/c64. The factor is written where LAPACK
  expects it and no transpose copy remains.
- for `UPLO='L'` the strict upper triangle is still saved and restored: a
  column-major leaf writes 7,680 of 523,776 strict-upper entries at n=1024 f64
  (3,584 for c64), so that half cannot be left alone. The save buffer is
  `n * (n - 1) / 2` elements per calling thread and is reused.
- `threading::with_par` keeps one `threadteam::Team` resident per calling
  thread. Building a pool and reserving its members on every BLAS call cost
  about 3.8 ms at 8 threads (`into_team` makes every member acknowledge and be
  placed), more than the factorization. A token from the resident team is a
  lock and a dispatch. Setting the width to 1 drops the team, so a
  single-threaded process leaves no parked members behind.

The pinned `colmat-benchmark` harness measures the kernel itself, without this
adapter: at `n=1024`, f64, 8T, column-major, its `cholesky` cell is 1.45 ms
(best of five, `--prime-ms 250`, colmat-rs `b376b3ce`).

Dropping the U save brings the Julia API to parity or ahead of OpenBLAS at
`n=1024` (median ms, same-process alternating runs): f64 1T 9.9-10.3 vs
10.2-10.3 (mins 7.84 vs 7.91), f64 8T 4.2-4.4 vs 5.1-5.2; c64 1T 23.3-23.5 vs
25.9-28.5, c64 8T 5.1 vs 8.4-11.9. The `UPLO='L'` path is unchanged.

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

## LLT block-size validation (Skylake)

The adapter picks the 32-wide LLT block on AVX2-only x86-64 and otherwise keeps
colmat-rs's native choice (64 on current main). Re-verified on the benchmark
host (Intel iMac 2019, Core i9-9900K, `b376b3ce`, `Par::Seq`, min of 5-10):

| dtype | n | bs=32 | bs=64 | auto |
|---|---:|---:|---:|---:|
| f64 | 512 | **1.335** | 1.393 | 1.392 (64) |
| f64 | 1024 | **11.234** | 14.567 | 13.571 (64) |
| c64 | 512 | **4.564** | 5.878 | 5.963 (64) |
| c64 | 1024 | 51.427 | 50.342 | 51.499 (64) |
| f32 | 512 | 0.702 | **0.652** | 0.692 (128) |
| f32 | 1024 | **5.256** | 5.504 | 6.887 (128) |
| c32 | 512 | **2.323** | 2.482 | 3.127 (128) |
| c32 | 1024 | **18.453** | 24.390 | 23.513 (128) |

32 is the best single choice on this AVX2 part. Only c64 n=1024 (~2%) and f32
n=512 (~7%) prefer 64; f32 n=512 with 32 is still within 1.4% of auto. The
adapter therefore keeps the uniform 32 override rather than adding type/size
special cases. colmat-rs issue #257 tracks an ISA-aware default upstream.

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

Full backend suite for `cholesky` on the Zen 4 host, `n=1024`, median of five
(`benchmarks/backends/results/linux-amd-2026-11-21-final2`; the mask has
`threads + 1` CPUs):

| dtype | T | OpenBLAS | MKL | Colmat | Colmat/OpenBLAS |
|---|---:|---:|---:|---:|---:|
| f64 | 1 | 7.84 | 8.96 | 10.26 | 1.31 |
| f64 | 4 | 4.52 | 2.31 | 5.50 | 1.22 |
| f64 | 8 | 4.73 | 1.94 | 2.98 | 0.63 |
| c64 | 1 | 27.26 | 27.45 | 24.12 | 0.88 |
| c64 | 4 | 10.65 | 5.77 | 9.84 | 0.92 |
| c64 | 8 | 9.67 | 4.82 | 6.41 | 0.66 |

## Blocked column-pivoted QR (`?geqp3`)

The first fix removed the per-element norm recomputation but left the factor
unblocked and called `qr::standard_qr` for a single column at every pivot step.
That path allocates a basis and two panel buffers, performs a GEMM, and acquires
a thread team per column; at 4T/8T the tiny factorizations were about 20x slower
than sequential, and the trailing update stayed at BLAS-2.

Changes in `qr_pivoted.rs` and `kernels.rs`:

- the factorization now follows LAPACK `?laqps`: a global pivot search over the
  remaining columns, exact residual norms maintained through the panel `F`
  recurrence, one `m x n x panel` GEMM per panel, and one team acquisition per
  panel shared by the panel GEMV, `F` update, row update and trailing GEMM;
- single reflectors come from `householder::generate`, without a per-column
  compact-WY allocation or GEMM;
- small shapes (roughly `k <= 8` or `m*n < 50k`, complex weighted 5x) stay on
  the unblocked level-2 path, where the panel overhead is not amortized;
- `nrm2` accumulates real and imaginary components directly instead of calling
  `hypot` per element.

The pivot order is identical to the unblocked algorithm and to OpenBLAS on the
100x100 check; blocked widths 4 and 5 reproduce the unblocked (1) permutation
and reconstruction in the regression tests, and the `?geqp3rk` RHS column equals
`Q^H B`.

ComplexF64 still spent 15.1 ms in the per-column norm downdate at n=1024: one
`hypot`/`abs` and a branch per remaining column, reading the updated row through
stride `lda`. The row is now staged contiguously (`rowfull`), the row-update GEMM
writes contiguous memory, and the downdate is a branch-free `re^2 + im^2` pass
that marks columns needing the same `?nrm2` recompute LAPACK uses. The pass is
1.0 ms, and c64 `?geqp3` improves 0.79-0.94x at n=256/512/1024; the residual 1T
gap is the DRAM-bound `F`-row GEMV (39 vs OpenBLAS's 57 GB/s).

## Small-shape team entry

The resident team fixed the cost of *building* a pool, but entering it still
opens a root threadteam scope on every call: the leader reads and changes its
affinity, members acknowledge and place themselves, and the leader restores its
mask on exit. An empty `with_par` measures 7.9 us at 4 threads and 8.9 us at 8
threads on the Zen 4 host, independent of the operation. At n=32/64/96 that is
larger than the work itself, and it was the reason the 4T/8T geometric mean
looked far worse than the n>=256 throughput.

Changes in `threading.rs`, `kernels.rs`, `lapack.rs`,
`triangular_lapack.rs` and `householder.rs`:

- `threading::with_par_if(work, limit, f)` takes the resident team only when
  `work` exceeds `limit`; below it the call runs on `Par::Seq`. A thread-local
  depth counter keeps every nested `with_par` in that call sequential (and is
  released on unwind), so an outer sequential decision also covers panel
  helpers and GEMMs instead of re-acquiring a token per helper.
- the thresholds are expressed in real element updates, with four per update
  for a complex scalar (`threading::real_work`):

  | operation | work | limit |
  |---|---|---:|---:|
  | `gemm` | `m*n*k` | 700 000 |
  | `lu` (`?getrf`) | `m*n*min(m,n)` | 8 000 000 |
  | `cholesky` (`?potrf`) | `n^3` | 9 000 000 |
  | triangular solve (`?trtrs`) | `n^2*k` | 2 000 000 |
  | one QR panel (LARFT/LARFB) | `rows*cols*block` | 1 500 000 |

  The limits are the measured 4T/8T crossovers rounded to the sequential side,
  so n=32/64 in the benchmark use `Par::Seq` while n=96 and above stay threaded
  wherever the team is already faster.
- `householder::apply_block` now issues its three GEMMs under one team
  acquisition instead of three, matching the existing
  `apply_forward_column_block_adjoint_with_basis`. This also removes two team
  entries per block from the explicit-Q (`?gemqrt`) path at every size.

No factorization algorithm, pivot scheme or pinning setting changed. A call
above its limit still uses the same pinned resident team, so the `threads + 1`
CPU-mask requirement and the large-shape throughput are untouched.

Median microseconds (my harness, 15 samples, `taskset` mask `threads + 1`),
Float64, 4 threads:

| operation | n=32 before -> after | n=64 before -> after | n=96 before -> after |
|---|---:|---:|---:|
| `gemm` | 9.28 -> 0.98 | 15.02 -> 6.87 | 23.01 -> 22.85 |
| `lu` | 12.06 -> 3.33 | 22.15 -> 12.23 | 38.11 -> 29.24 |
| `cholesky` | 10.66 -> 2.06 | 16.22 -> 7.56 | 26.18 -> 17.10 |
| `qr` factor | 24.14 -> 14.97 | 73.15 -> 44.03 | 139.19 -> 88.82 |
| `Matrix(qr(A).Q)` | 54.29 -> 19.49 | 151.28 -> 67.00 | 262.88 -> 144.00 |
| triangular solve | 10.43 -> 1.76 | 17.61 -> 7.68 | 24.80 -> 23.60 |

The geometric mean over these six operations and n=32/64/96 is 2.24x at 4T and
2.32x at 8T for Float64, and 1.45x / 1.52x for ComplexF64. At n=256/512/1024
the same comparison is 1.01x (4T) and 1.02x (8T), within run-to-run noise.

Guard at n=1024, four alternating baseline/new processes per cell: minimum and
median of every touched operation (gemm, lu, cholesky, qr factor, explicit Q,
triangular solve, both dtypes, 4T/8T) stay within +-3%, except `Matrix(qr(A).Q)`
which is 3-4% faster from the consolidated block application. Untouched
operations (`qr_pivot`) show no systematic change.

## What is left

- **Cholesky** is at parity or faster than OpenBLAS at every thread count after
  dropping the `UPLO='U'` save. At `n=1024` (`median ms`, f64/c64, alternating
  runs): 1T 9.9-10.3 / 23.3-23.5 vs 10.2-10.3 / 25.9-28.5; 8T 4.2-4.4 / 5.1 vs
  5.1-5.2 / 8.4-11.9. The native AVX-512 LLT kernel still has headroom;
  colmat-rs #257 tracks the block-size follow-up.
- **SVD** at 1 thread is about 1.14x slower for `svd` and 1.2-1.4x for
  `svdvals`. Profiling showed ~97% of the call inside
  `colmat::linalg::svd::svdd`; colmat-rs #259 tracks the values-only gap.
  At 8 threads Colmat is faster than OpenBLAS (0.70-0.90x).
- **Pivoted QR** now uses the blocked `?laqps` scheme. At `n=1024` (median ms,
  f64/c64): 1T 59.8 / 224.2 vs OpenBLAS 56.8 / 171.9 (1.05x / 1.30x); 4T 31.5 /
  86.2 vs 35.9 / 68.4 (0.88x / 1.26x); 8T 27.0 / 62.5 vs 34.0 / 45.3 (0.79x /
  1.38x). The remaining complex gap is the serial panel work (F update, row
  update and norm downdate); f64 is at parity or faster from 4T.
- **QR factor** is at parity or better at 1 thread (0.89x f64, 1.14x c64) and
  1.05x/0.50x at 8 threads, n=1024.
- **Cholesky / GEMM / LU** at 8 threads still vary with size; the geometric
  means in `summary.md` include the 32/64/96 sizes where adapter fixed costs
  dominate.

## Small complex compact QR (`?geqrt`, `?gemqrt`)

At `n=32/64/96` the ComplexF64 `qr`/`Matrix(qr(A).Q)` cells were 1.2-1.5x slower
than OpenBLAS while Float64 was near parity. The gap was in the Householder
path, not in the compact-WY blocking: `householder::apply` recomputed
`tau * v(i) * conj(w(j))` for every element (two complex products per update),
the BLAS-2 product went through faer's GEMM packing for a one-column right-hand
side, and the T build formed a full `k x k` Gram matrix plus an allocation per
column. Reflector generation also divided twice per element.

Changes:

- `kernels::axpy`, `gemv_conj_contiguous` and `gemv_n_contiguous`: contiguous
  Householder kernels that hoist the conjugated coefficient out of the update
  loop. On x86-64 they dispatch to AVX2/FMA implementations (`is_x86_feature_detected!`,
  the same pattern faer uses for GEMM); other targets and CPUs keep the scalar
  loop. `householder::apply` uses them when `INC=1` and the update is below the
  existing 2^18-element threading threshold, so large updates keep the previous
  threaded GEMV.
- `householder::generate`: the tail norm uses the already-known maximum scale
  instead of a dynamic `?lassq` update (two divisions per element), and the
  final normalization uses reciprocals. Subnormal columns keep the division form
  because their reciprocal overflows.
- `householder::factor_with_basis`: for `DIRECT='F'` T is now built column by
  column as in LARFT (`w = V[:, :j]^H v_j` through `gemv_conj_contiguous`, then
  one contiguous axpy per previous column) instead of a `k x k` Gram GEMM plus a
  branchy triple loop. `DIRECT='B'` keeps the Gram matrix.
- `multiply_q` sizes the basis, dense `T`, and both block-update buffers once per
  GEMQRT call and reuses them for every block.

`benchmarks/backends` medians (25 samples, best of two alternating rounds,
Colmat/OpenBLAS; the host was shared, so the least-disturbed round of each cell
is used). Before the change the same cells measured 1.27-1.45x for the c64
factor (matching the reported 1.33-1.48x) and 1.13-1.38x for explicit `Q`.

| dtype | op | T | n=32 | n=64 | n=96 |
|---|---|---:|---:|---:|---:|
| c64 | `qr` (factor) | 1 | 0.60 | 0.70 | 0.78 |
| c64 | `qr` (factor) | 4 | 0.59 | 0.73 | 0.78 |
| c64 | `qr` (factor) | 8 | 0.63 | 0.66 | 0.76 |
| c64 | `Matrix(qr(A).Q)` | 1 | 0.79 | 0.78 | 0.85 |
| c64 | `Matrix(qr(A).Q)` | 4 | 0.83 | 0.92 | 0.94 |
| c64 | `Matrix(qr(A).Q)` | 8 | 0.81 | 0.88 | 0.87 |
| f64 | `qr` (factor) | 1 | 0.57 | 0.73 | 0.85 |
| f64 | `Matrix(qr(A).Q)` | 1 | 0.68 | 0.86 | 0.98 |
| f64 | `Matrix(qr(A).Q)` | 8 | 0.67 | 0.86 | 0.96 |

At `n=256` the new tree is 0.62-0.68x (c64) and 0.77-0.80x (f64) for the factor.
Guard at `n=512/1024`, 1T/8T: c64 improves 10-35%; f64 stays within +-4% of the
baseline (median ms, n=1024, 1T: factor 26.1 vs 26.1, Q 65.2 vs 65.3; 8T:
11.1 vs 10.9 and 19.7 vs 20.7). No threshold or semantics changed: compact `T`
storage, TAU, row-major LAPACKE paths, and both integer ABIs are untouched.


## Small general EVD (`?geevx`)

Small `eigen(A)` calls entered the resident team even though the Hessenberg and
Schur work was effectively sequential, and inverse balancing allocated a second
vector matrix and renormalized every column with a componentwise `hypot` even
when no scale changed. The adapter now:

- transforms the owned vector matrix in place during inverse balancing and skips
  renormalization when no row scale changed (the Schur eigenvector routine
  already returns unit-norm columns);
- gates the EVD with `with_par_if` and `GENERAL_EVD_WORK = 100_000_000`: n<=256
  runs `Par::Seq`, n>=512 keeps the team.

This removes a 2-4.4x small-n penalty at 4T/8T: n=32 0.63->0.21 ms, n=64
2.7->1.1 ms, n=96 5.6->2.3 ms, n=128 8.3->4.3 ms, n=256 48->22.7 ms. n=96/128
are now 0.81-0.95x OpenBLAS; n=32/64 remain ~1.4x because the pinned upstream
Schur/Hessenberg kernels are slower (standalone `hessenberg_schur` 145-153 us at
n=32 vs OpenBLAS `DHSEQR` 106 us). `SENSE=N` still takes the native path; expert
modes keep the saved fallback.

## Complex pivoted QR F-row (`?geqp3`)

The `?laqps` F-row GEMV dominated ComplexF64 pivoted QR (97 of 158 ms at n=1024,
1T). It is a transposed read of the trailing block, and the generic path walked
one output column per pass, re-fetching the reflector and deinterleaving each
complex operand separately (37-39 GB/s). `kernels::conj_gemv` keeps eight output
columns in flight (four when the block fits L3), one `vpermt2pd` deinterleave per
eight complex, with AVX-512 runtime dispatch and column-block threading;
`col_axpy` covers the column update, F accumulation and row update. The kernel
reaches 58 GB/s at 16 MB and 104 GB/s at 4 MB.

Julia-API c64 `qr(A, ColumnNorm())`, n=1024, min/median ms: 1T 186/201 vs
OpenBLAS 214/229, 4T 71/75 vs 75/76, 8T 49/53 vs 79/84. The isolated `?geqp3`
ccall still trails OpenBLAS's tuned ccall at 1T, but the path Julia users call is
at parity or ahead. f64 guard cells are unchanged within noise.
## Small pivoted QR (`?geqp3`, n<=96)

The small-shape `qr(A, ColumnNorm())` loss was fixed overhead, not the pivot
search: reflector application was 50-66% of the time and the generic f64 path
ran at SSE2 width, while c64 n<=96 chose the unblocked path by mistake and every
panel paid a ~8 us resident-team entry.

Changes:

- a runtime-dispatched AVX2/FMA `larf_left_f64_avx2` for non-threaded, unit-stride
  left applications (four accumulators for `w = C^H v`, one FMA per element for
  the update); other scalars/sides and threaded updates keep the scalar path;
- `panel_width` policy: real unblocked through `m*n <= 16_384`, panel 8 through
  `m*n < 60_000`, panel 16 above; complex unblocked through `m*n*5 <= 32_000`,
  panel 8 through `m*n*5 < 200_000`, panel 16 above; `k <= 8` still forces
  unblocked;
- pivoted panels use `with_par_if(..., 400_000, ...)`, so small panels stay
  `Par::Seq` and nested helpers see the serial depth guard instead of acquiring
  another team.

Julia API f64 `qr(A, ColumnNorm())` n=32/64/96 is now 11.1/33.9/90.3 us versus
OpenBLAS 13.3/39.2/102.0; c64 n=32/64 is a tie and n=96 at 8T is 183 us versus
308 before and 471 for OpenBLAS. n>=512 is unchanged within noise.

## Small `eigh` / `svd_thin`

`svd` now writes into LAPACK's U/VT buffers directly when the layout fits (VT is
the same buffer viewed with V's shape and transposed strides, with complex
entries conjugated in place); owned buffers, zero-fills, a memcpy and a
transpose copy remain only for `JOBZ='O'`, `r=0` and complex singular values.
`eigh`/`svd` use `with_par_if` with `SELFADJOINT_WORK`/`SVD_WORK = 100_000`
(complex 300_000), so 32-square calls stay sequential at every thread count.

f64 `svd_thin` n=32 at 4T/8T improves from 1.13/1.15x to 0.98/0.97x OpenBLAS;
c64 n=32 is 1.03-1.05x and n=64/96 beat it. The residual f64 n=64/96 1T gap
(1.07/1.03x) is in the pinned `dgesdd` core (307-324 vs 291-294 us at n=64),
not the adapter; the workspace query must keep the documented LAPACK minimum.

For small f64 `UPLO='U'` (`n < 128`) the row-major transposed view is now
replaced by a column-major temporary: copy the upper triangle into the temp's
lower with faer's tiled copy, factor, and write the factor back. The temp copy
wins at n=32/64/96 (core 1.17-1.27x, shim 1.08-1.22x, Julia API 7-14% at
n=64/96); from n=128 the in-place view wins and is kept, with a regression test
at n=127/128. c64 keeps the in-place path: its column-major core is only ~5%
faster and the shim/API see no gain, so the remaining c64 small/mid delta is the
Julia-side `Hermitian` copy/allocation, not layout.

## SVD / self-adjoint EVD native workspace reuse

`svd_native` and the self-adjoint driver allocated a fresh native workspace per
call. In a Julia process the first touch of that 254 KiB region costs about 12 us
of a 64-square call, and the workspace is dead after the factorization. Both now
use a per-calling-thread `Vec<u64>` scratch through `with_native_scratch`, which
re-aligns, grows on demand, falls back to a fresh `MemBuffer` on reentry, and
caps retention at 16 MiB: n=512/1024 keep the old allocate/free behaviour, while
n<=256 reuses the buffer. f64 `svd_thin` n=32 improves from 0.996x to 0.917x
OpenBLAS and n=96 from 1.015x to 1.000x; n=64 remains 1.077x because the pinned
divide-and-conquer core itself is 10% behind at the LAPACK boundary, not the
adapter. `eigh` improves about 1% at every size; n=512/1024 are within noise.
