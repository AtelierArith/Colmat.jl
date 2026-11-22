# Benchmarks

This directory is self-contained. It does not call an external benchmark
harness, fetch revisions, or require a second checkout.

## Backend comparison

```sh
./benchmarks/run.sh                  # OpenBLAS, MKL, AOCL, Colmat; 1/4/8 threads
./benchmarks/run.sh --quick          # tiny smoke run
./benchmarks/run.sh --help
```

`run.sh` checks/builds Colmat, instantiates `benchmarks/backends`, runs one
process per backend and thread budget, and writes:

```
benchmarks/backends/results/runs/<timestamp>/summary.md
benchmarks/backends/results/runs/<timestamp>/summary.csv
benchmarks/backends/results/runs/<timestamp>/<backend>_t<N>.toml
```

The suite covers `gemm`, `lu`, `cholesky`, `qr` (factor and explicit Q),
column-pivoted QR, `svdvals`, thin SVD, self-adjoint eigen and general eigen
for `Float64`/`ComplexF64` at 1/4/8 threads. See
[`backends/README.md`](backends/README.md) for the case table, sizes, sampling
and affinity policy, and the meaning of the median and minimum aggregates.

## General EVD audit (#318)

`general_evd.jl` is the separate, opt-in audit for the general eigensolver. It
uses `benchmarks/environment` (BenchmarkTools, JET, AllocCheck):

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project=benchmarks/environment -e 'using Pkg; Pkg.instantiate()'
COLMAT_EVD_OUTPUT=/tmp/evd.toml julia --startup-file=no \
  --project=benchmarks/environment benchmarks/general_evd.jl --quick
```

The audit records cold compile time, steady timing, allocations, residuals,
revisions and hashes; the full method, phase capture and limitations are in
[`docs/audits/issue-318-general-evd.md`](../docs/audits/issue-318-general-evd.md).
