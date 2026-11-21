# Colmat.jl

A Julia BLAS / LAPACK backend that uses Rust's [colmat](https://github.com/tensor4all/colmat-rs).
`using Colmat` registers its forwarding targets with libblastrampoline. Ordinary `Matrix`
and `LinearAlgebra` work as-is; no dedicated matrix types or RustCall are needed.

```julia
using Colmat, LinearAlgebra

A = randn(100, 80)
B = randn(80, 60)
C = A * B
mul!(C, A, B)

BLAS.get_config()           # shows Colmat and the existing fallbacks
BLAS.set_num_threads(2)     # sets both Colmat and the existing backends
```

Julia 1.13, Git, and Rust 1.89 or later are required. The first build needs a network connection.

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project examples/poc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

The build uses the Rust source recorded in `deps/upstream.toml` and `Cargo.lock`.
It does not automatically update the upstream HEAD during the build. Restart Julia after
rebuilding the shared library.

## Scope

LP64 / ILP64 and Float32 / Float64 / ComplexF32 / ComplexF64 are supported.

| Category | Routines forwarded to Colmat |
|---|---|
| Level 1 | copy, swap, scal, axpy, axpby, dot / dotu / dotc, nrm2, asum, iamax, rot |
| Level 2 | gemv, symv, hemv, ger / geru / gerc, syr, her, trmv, trsv |
| Level 3 | gemm, symm, hemm, syrk, syr2k, herk, her2k, trmm, trsm |

Matrix multiplication borrows Julia's memory and runs the Rust colmat kernels.
Symmetric, Hermitian, and triangular products allocate temporary storage as needed.
Some vector operations and triangular solves are implemented in this package's Rust
compatibility layer.
LU, Cholesky, solves, inverses, SVD, general/self-adjoint eigenvalues, Schur/QZ,
generalized Hessenberg, and least squares process the caller's memory without copying
the input arrays.
The initial thread count inherits `BLAS.get_num_threads()` at load time. The configured
parallel width applies to matrix multiplication.

The ordinary-array `lu`, `svd`, `eigen`, and `cholesky` entry points are connected to colmat-rs.

| LAPACK family | What Colmat handles |
|---|---|
| getrf / getrs / getri / gesv | LU, solves, inverses |
| potrf / potrs / potri | Cholesky, solves, inverses |
| gesdd / gesvd | SVD, full / thin / overwrite |
| geev / geevx | General eigenvalues, left/right eigenvectors, balancing |
| syev / heev / syevd / heevd / syevr / heevr | Self-adjoint eigenvalues and A/I/V range selection |

Unsupported BLAS / LAPACK routines keep the existing backend.
For `geevx`, `SENSE=N` is native and condition estimation is delegated to the existing backend.
Several SVD and self-adjoint EVD drivers connect to a common native factorization; the
package does not guarantee that each Netlib algorithm runs identically. Range selection
chooses from the full self-adjoint EVD and is not native MRRR.

The public API correspondence and restrictions for all 5,146 LBT symbols are recorded in
[the full routine mapping](docs/src/reference/lbt-inventory.md), and ordinary Julia API
execution results in [the measured mapping](docs/src/reference/support-matrix.md).

Switching on load applies to the whole Julia process. If another backend is loaded later,
`Colmat.load_backend()` re-selects Colmat. Switch backends before starting parallel computation.

Version 0.2 redesigned the package completely. The 0.1 APIs such as `FaerMatrix` are not provided.

[Manual](docs/src/index.md) / [Getting started](docs/src/getting-started.md) /
[BLAS coverage](docs/src/reference/coverage.md) / [API](docs/src/reference/api.md) /
[Development](docs/src/development.md) / [Troubleshooting](docs/src/troubleshooting.md)

How to consult the old version's material is described in [the 0.2 redesign](docs/src/migration.md).
