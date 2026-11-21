# Getting started

Complete the build described in [Installation](@ref) and run the examples in a new Julia process.

## Computing with ordinary arrays

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [1.0 2; 3 4]; B = [2.0 1; 1 3];

julia> A * B == [4.0 7; 10 15]
true

julia> C = zeros(2, 2); mul!(C, A, B); C == A * B
true

julia> x = [2.0, 3.0]; A * x == [8.0, 18.0]
true

julia> dot(x, x) == 13.0
true
```

Inputs and results are ordinary `Matrix` / `Vector` values. There is no operation that
converts a matrix to a Colmat-specific type, and Julia still owns the matrix memory.

## Checking the forwarding targets

`BLAS.get_config()` shows the loaded libraries. It is normal for OpenBLAS and similar
libraries to appear together with Colmat because their fallbacks are retained.
You can check the forwarding target for an individual routine as follows.

```jldoctest
julia> using Colmat, LinearAlgebra

julia> interface = sizeof(LinearAlgebra.BlasInt) == 8 ? :ilp64 : :lp64;

julia> lib = BLAS.lbt_find_backing_library("dgemm_", interface);

julia> realpath(lib.libname) == realpath(Colmat.backend_library())
true
```

This check inspects the forwarding target of double-precision GEMM. For the status of
other routines, see [BLAS coverage](@ref).

## Operations that use LAPACK

```jldoctest
julia> using Colmat, LinearAlgebra

julia> A = [4.0 1; 1 3]; b = [1.0, 2.0];

julia> A * (A \ b) ≈ b
true

julia> F = svd(A); F.U * Diagonal(F.S) * F.Vt ≈ A
true
```

These LU and SVD drivers connect to colmat-rs factorizations and solves.
The forwarding boundary is described in [Backends and fallbacks](@ref).
