# Backend comparison: OpenBLAS, MKL, AOCL, PureBLAS, Colmat and AppleAccelerate

A self-contained Julia benchmark for the dense kernels Colmat exposes through
libblastrampoline. It does not use an external harness: `run.sh` builds/checks
the library, runs one process per backend and thread budget, and writes
`summary.csv` plus `summary.md`.

## Quick start

```sh
./benchmarks/run.sh                  # OpenBLAS, MKL, AOCL (Linux), PureBLAS, Colmat; 1/4/8 threads
./benchmarks/run.sh --quick          # tiny smoke run (still a few minutes with PureBLAS)
./benchmarks/run.sh --build          # rebuild Colmat first
./benchmarks/run.sh --threads 1 8 --backends openblas colmat
```

On macOS, `--backends ... apple` adds AppleAccelerate.jl (0.7.0). It forwards
the Accelerate framework's ILP64/LP64 symbols. Its threading API is a
single/multi toggle, not an exact count: `threads=1` is single-threaded and
`threads=4`/`8` both use Accelerate-managed multi-threading (8 threads on the
benchmark host), recorded as `vendor_threads`. The threading API requires
macOS 15 or later; `run.jl` refuses the Apple arm on 13.4–14 because the setters
there only warn and return 1, which would mislabel a multi-threaded run as 1T.

MKL needs explicit LAPACK thread control: `BLAS.set_num_threads(n)` through
libblastrampoline sets the BLAS domain but not MKL's LAPACK threading. `run.jl`
also calls `MKL_Set_Num_Threads(n)`. Without it a "1 thread" MKL
`potrf`/`gesdd`/`syevd` cell still uses 8 threads: at n=1024, `cholesky` measured
1.57 ms with `BLAS.set_num_threads(1)` alone and 6.97 ms after
`MKL_Set_Num_Threads(1)`; `svdvals` 88.5 ms vs 141.1 ms.

PureBLAS also ignores `BLAS.set_num_threads(n)`: it registers no
libblastrampoline thread interface. Its pool is built from Julia threads, so
the runner launches this arm with `-t $threads` (all other arms stay at
`-t 1`), and `run.jl` calls `PureBLAS.set_num_threads(n)` before the first
measured call. The value that call returns is recorded as `vendor_threads`;
it is capped by `Threads.nthreads()` and, by default, by the physical cores in
the process affinity mask. The mask is `threads + threads/2` **logical** CPUs,
so on the 16-logical-CPU / 8-core SMT host a 4T run gets 6 logical CPUs = 3
physical cores and an 8T run gets 12 = 6 cores. Left alone, that default cap
would silently run the 4T/8T cells at 3T/6T while every other backend runs
4T/8T. The runner therefore sets `PUREBLAS_MT_ALL_THREADS=1` for this arm only,
so the requested budget is honored like the other arms. The trade-off is that
workers may then be placed on SMT siblings; that is the same CPU set the other
backends are given, so the comparison is like-for-like.

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
median table, a minimum-time table and geometric means for both, with ratios
written as `Colmat/<backend>`. The minimum removes most of the Julia
GC/allocation variance that dominates microsecond cells.

Every case has a probe outside the timed region: GEMM is checked with a plain
loop, factorizations by reconstruction/orthogonality residual, and `svd_values`
against `svd`. The residual is stored in the TOML/CSV.

## Backends

- `openblas`: Julia's bundled OpenBLAS, the baseline.
- `mkl`: Intel oneAPI MKL through `MKL.jl`/`MKL_jll` (pinned in
  `Project.toml`).
- `aocl`: AMD AOCL through `AOCL.jl`/`AOCL_jll` (5.3.x); the runner sets
  `OMP_NUM_THREADS`, `BLIS_NUM_THREADS` and `FLA_NUM_THREADS` for the process.
  AOCL ships Linux binaries only, so this arm is in the default list on Linux
  but omitted on macOS (`AOCL_jll` is unavailable there).
- `pureblas`: the pure-Julia `PureBLAS.jl` (0.1.2, pinned by `rev` to a commit
  reachable from upstream `master` in `Project.toml`/`Manifest.toml` because it
  is unregistered; Pkg cannot fetch a dangling SHA that no branch or tag
  contains). `run.jl` calls
  `PureBLAS.activate()`, which installs in-process libblastrampoline function
  pointers, so `BLAS.get_config()` still lists the originally loaded library.
  The library column records `PureBLAS.jl (LBT function pointers)` instead.
  Its load-time `@strict_function ... :no_spill` self-check of the generated
  gemm microkernel fails on this AVX-512 host (the tile spills), so the
  environment disables StrictMode checks; see Environment below.
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
`taskset` is available and the runner runs unpinned. The `pureblas` arm is the
one exception to `-t 1`: it gets `-t $threads` because its pool is built from
Julia threads, and it sets `PUREBLAS_MT_ALL_THREADS=1` so the mask's
physical-core count does not cap the requested budget (see above); the
affinity mask itself is unchanged.

## Environment

`benchmarks/backends/Project.toml` pins the benchmark-only dependencies
(BenchmarkTools, MKL, TOML, Statistics, ...) and loads Colmat from this
repository. `run.sh` instantiates it automatically.

`benchmarks/backends/LocalPreferences.toml` sets
`[StrictMode] checks_enabled = false`, and `StrictMode` is a direct dependency
of the benchmark project so that Julia reads the preference. PureBLAS checks
its generated gemm microkernel at load time with `@strict_function ...` and
throws when the tile spills vector registers; on this AVX-512 host that makes
`using PureBLAS` fail, before any benchmark runs. StrictMode guarantees are
compile-time assertions only — with the preference off the macros expand to
the bare definitions — so no measured code changes. Remove the preference to
run the check on a host where it passes. The first `Pkg.instantiate()` after a
fresh clone precompiles PureBLAS (several minutes).

## Caveats

These are single-machine numbers. The median includes Julia GC, so use the
minimum table for the steady-state kernel/adapter comparison and the median
table for the end-to-end public API. Do not run other heavy jobs on the same
CPUs while a run is in progress.
