# Backend comparison: OpenBLAS, MKL and Colmat

A self-contained Julia benchmark for the dense kernels Colmat exposes through
libblastrampoline. It does not use an external harness: `run.sh` builds/checks
the library, runs one process per backend and thread budget, and writes
`summary.csv` plus `summary.md`.

## Quick start

```sh
./benchmarks/run.sh                  # OpenBLAS, MKL, Colmat; 1/4/8 threads
./benchmarks/run.sh --quick          # tiny smoke run (~1 minute)
./benchmarks/run.sh --build          # rebuild Colmat first
./benchmarks/run.sh --threads 1 8 --backends openblas colmat
```

Results go to `benchmarks/backends/results/runs/<timestamp>/` by default. An
explicit `--out` directory must not already contain TOML results, so a report
cannot mix a new run with a stale baseline:

```
summary.md        tables and geometric means
summary.csv       long-format rows with min/median/mean, allocations, residual
<backend>_t<N>.toml
logs/<backend>_t<N>.log
```

Use `--out DIR` to choose the directory. The environment variables
`COLMAT_BENCH_BACKENDS`, `COLMAT_BENCH_THREADS_LIST`, `COLMAT_BENCH_SAMPLES`,
`COLMAT_BENCH_QUICK`, `COLMAT_BENCH_AFFINITY` and `COLMAT_BENCH_CPU_START`
provide the same controls.

## What is measured

Public Julia API calls on square matrices. Allocations and input copies done by
the public API are included.

| Case | Julia call | LAPACK entry on Julia 1.13 |
|---|---|---|
| `gemm` | `mul!(C, A, B)` | `?gemm` |
| `lu` | `lu(A)` | `?getrf` |
| `cholesky` | `cholesky(Hermitian(A*A' + nI))` | `?potrf` |
| `qr_factor` | `qr(A)` | `?geqrt` |
| `qr` | `Matrix(qr(A).Q)` | `?geqrt` + `?gemqrt` |
| `qr_pivot` | `qr(A, ColumnNorm())` | `?geqp3` |
| `svd_values` | `svdvals(A)` | `?gesdd` |
| `svd_thin` | `svd(A)` | `?gesdd` |
| `eigh` | `eigen(Hermitian(A*A' + nI))` | `?syevr` / `?heevr` |
| `eig` | `eigen(A)` | `?geevx` |

Sizes are 32/64/96 and 128/256/512, plus 768/1024 for GEMM and 1024 for the
factorizations/decompositions. General `eig` stops at 512. `--quick` uses
32/128 only. Every case uses fixed seeds, so all backends see identical inputs.

Each case runs BenchmarkTools with five samples. One sample bundles enough
calls to last about 20 ms for fast operations; slow operations use one call per
sample. `GC.gc()` runs once before each case. The Markdown report shows the
median table, a minimum-time table and geometric means for both: the minimum
removes most of the Julia GC/allocation variance that dominates microsecond
cells.

Every case has a probe outside the timed region: GEMM is checked with a plain
loop, factorizations by reconstruction/orthogonality residual, and `svd_values`
against `svd`. The residual is stored in the TOML/CSV.

## Backends

- `openblas`: Julia's bundled OpenBLAS, the baseline.
- `mkl`: Intel oneAPI MKL through `MKL.jl`/`MKL_jll` (pinned in
  `Project.toml`).
- `aocl`: AMD AOCL through `AOCL.jl`/`AOCL_jll` (5.3.x); the runner sets
  `OMP_NUM_THREADS`, `BLIS_NUM_THREADS` and `FLA_NUM_THREADS` for the process.
- `colmat`: this repository's Rust BLAS/LAPACK library.

Each backend runs in a separate process because libblastrampoline forwards are
process-global. The `openblas` process is the reference for the ratios.

## Affinity

On Linux the runner pins each process to `threads + threads/2` CPUs by default
(at least `threads + 1`), taken from the process's **allowed CPU list**
(`taskset -pc $$`), not from a hard-coded range: inside a container or scheduler
allocation the allowed set may be, for example, `8-15`. Colmat's thread team
places the caller apart from its members, and on an SMT host the spare CPU must
not be a sibling of a team member; 1T/4T/8T therefore use 2/6/12 CPUs on a
16-logical-CPU host. A mask exactly as wide as the budget makes runs
intermittently serial. Use `--no-affinity` to disable this. On macOS no
`taskset` is available and the runner runs unpinned.

## Environment

`benchmarks/backends/Project.toml` pins the benchmark-only dependencies
(BenchmarkTools, MKL, TOML, Statistics, ...) and loads Colmat from this
repository. `run.sh` instantiates it automatically.

## Caveats

These are single-machine numbers. The median includes Julia GC, so use the
minimum table for the steady-state kernel/adapter comparison and the median
table for the end-to-end public API. Do not run other heavy jobs on the same
CPUs while a run is in progress.
