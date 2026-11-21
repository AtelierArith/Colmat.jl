# Colmat.jl

Colmat is a Julia BLAS / LAPACK backend that uses Rust's colmat.
`using Colmat` registers its forwarding targets with libblastrampoline (LBT), and
computation then runs through ordinary Julia arrays and the `LinearAlgebra` API.

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [4.0 1; 1 3]; B = [2.0 0; 0 2];

julia> A * B == [8.0 2; 2 6]
true

julia> C = similar(A); mul!(C, A, B); C == A * B
true
```

## Reading guide

- [Installation](@ref): required environment and the first build.
- [Getting started](@ref): matrix products, vector operations, and checking the forwarding targets.
- [Backends and fallbacks](@ref): process-wide switching and LAPACK handling.
- [Thread settings](@ref): parallel width for Colmat and the existing backends.
- [Full LBT and colmat-rs routine mapping](@ref): connection candidates for every symbol and unsupported APIs.
- [BLAS coverage](@ref): types, integer ABIs, and provided routines.
- [Measured Julia API mapping](@ref): execution results and the distinction between Colmat and the existing backends.
- [Troubleshooting](@ref): checks when a build or load fails.

## Supported computations

Float32 / Float64 / ComplexF32 / ComplexF64 and LP64 / ILP64 are supported.
GEMM, GEMV, and Level 3 operations built on matrix products use the Rust colmat kernels;
some vector operations and triangular solves run in this package's Rust compatibility layer.

`lu`, `svd`, general/self-adjoint `eigen`, and `cholesky` connect to colmat-rs
factorizations through the LAPACK ABI. Unsupported BLAS / LAPACK routines use the
existing backend. Some families, such as QR, generalized eigenvalues, and Schur, are not
connected yet even though the Rust side provides them. The existence of a public API and
the actual ABI connection are distinguished in the full routine mapping.

[The 0.2 redesign](@ref) removed the old API based on dedicated matrix types.
