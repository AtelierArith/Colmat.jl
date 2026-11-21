# Backend comparison: GEMM, LU, Cholesky, QR, SVD, eigen

Compares Colmat against OpenBLAS and MKL (`MKL_jll` 2023.2.0, the last release with
a macOS x86_64 build) at the Julia API. The families and thread budgets mirror
`tensor4all/colmat-benchmark`: `gemm`, `lu`, `cholesky`, `qr_factor`, `qr`
(explicit Q), `svd_values`, `svd_thin`, `eigh` and `eig`, `f64`/`c64`, 1/4/8
threads. Column-pivoted QR is kept as an extra family because its implementation
changed here.

Each backend runs in its own Julia process because libblastrampoline forwards are
process-global.

## Setup

Julia 1.13. Build Colmat first:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project=benchmarks/backends -e 'using Pkg; Pkg.instantiate()'
```

`benchmarks/backends/Project.toml` pins `MKL_jll = "=2023.2.0"` for macOS support
and loads Colmat from this repository.

## Running

Run one process per backend and thread count. Do not run them concurrently: the
measurements assume the machine is otherwise idle.

```sh
for threads in 1 4 8; do
  # Give the team leader a CPU of its own. Colmat is a libblastrampoline
  # backend whose threadteam places the caller apart from its members, so a
  # mask exactly as wide as the thread budget leaves no spare CPU and the run
  # becomes intermittently serial. OpenBLAS and MKL do not care, but compare
  # every backend on the same mask with at least threads + 1 CPUs.
  case $threads in 1) cpus=4;; 4) cpus=4-9;; 8) cpus=0-11;; esac
  for backend in openblas mkl colmat; do
    taskset -c $cpus env \
      COLMAT_BENCH_BACKEND=$backend \
      COLMAT_BENCH_THREADS=$threads \
      COLMAT_BENCH_OUTPUT=benchmarks/backends/results/after/${backend}_t${threads}.toml \
      julia --startup-file=no -t 1 --project=benchmarks/backends benchmarks/backends/run.jl
  done
done
```

The masks above are for a 16-CPU host; on another host keep at least
`threads + 1` CPUs in the mask and pin every backend to the same set.

Then summarize:

```sh
COLMAT_BENCH_RESULTS_DIR=benchmarks/backends/results/after \
  julia --startup-file=no --project=benchmarks/backends benchmarks/backends/summarize.jl
```

This writes `results/after/summary.md` and `results/after/summary.csv`.
`results/after/ENVIRONMENT.md` records the exact revisions and binaries;
`results/pre-fix/` keeps the measurements taken before the pivoted-QR and
threading fixes.

## Fixes measured here

`fix-notes.md` describes the three changes in `deps/rust/src`: LAPACK-style
column-norm downdating for GEQP3, a scaled sum-of-squares `nrm2`, sequential
per-column reflectors and small-update GEMV, and a factored block-reflector
apply that no longer materializes the full `order x order` matrix.

## What is measured

Public Julia API calls on square matrices, so allocations and copies done by the
wrapper are included:

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

Sizes: 32/64/96 and 128/256/512, plus 768/1024 for GEMM and 1024 for the
factorizations and decompositions. General `eig` stops at 512.

`summary.md` computes geometric means over every size. At 4T/8T the 32/64/96
cells are dominated by a fixed thread-team entry cost (about 10 us on the Zen 4
host), and OpenBLAS stops scaling past four threads there; the aggregate can
therefore look worse than the throughput a large matrix gets. Read the size
tables or the `n >= 128` rows for throughput, and keep the small sizes as a
guard on per-call overhead.

Each case runs BenchmarkTools with five samples. One sample bundles enough calls
to last roughly 20 ms for fast operations; slow operations use one call per
sample. Inputs come from fixed seeds, so every backend sees identical matrices.
`run.jl` records the library that each LAPACK symbol is forwarded to and checks a
reconstruction residual outside the timed region.

Only use the numbers if the target library is the backend under test (for example
`libcolmat_blas.dylib` for `colmat`).

## Caveats

These numbers describe one machine, one Julia version, and this repository
revision. They are not a general performance claim. In particular, MKL 2023.2 is
older than current oneAPI releases and has no current macOS build, so MKL numbers
here reflect that older build.

The public Julia API copies the input for the destructive decompositions. Those
copies are charged to every backend equally and are included in all numbers.
