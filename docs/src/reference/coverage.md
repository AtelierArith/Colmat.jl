# BLAS coverage

The correspondence between the results of running the ordinary Julia API and the
registered targets is documented in [Measured Julia API mapping](@ref).

## Types and integer ABIs

| Julia element type | BLAS prefix |
|---|---|
| `Float32` | `s` |
| `Float64` | `d` |
| `ComplexF32` | `c` |
| `ComplexF64` | `z` |

The same shared library provides LP64 (32-bit integers) and ILP64 (64-bit integers).
LP64 Fortran symbols are exposed as `dgemm_` and ILP64 ones as `dgemm_64_`, for example.
LBT forwards to the entry point matching Julia's integer ABI.
The element precision and the width of integer arguments are independent.

## Provided routines

This list is based on the implementation in `deps/rust/src/abi.rs`.
"All four types" means the four types in the table above.

| Level | Family | Supported |
|---|---|---|
| 1 | `copy`, `swap`, `scal`, `axpy`, `axpby`, `iamax`, `rot`, `rotg` | All four types |
| 1 | `dot` | The two real types |
| 1 | `dotu`, `dotc` | The two complex types |
| 1 | `nrm2`, `asum` | All four types, returning the corresponding real type |
| 1 | `csscal`, `zdscal`, `csrot`, `zdrot` | Complex arrays with real coefficients |
| 1 | `amax`, `amin`, `max`, `min`, `sum` | The two real types |
| 1 | `iamin` | All four types; complex values use abs1 and Fortran 1-based indices |
| 2 | `gemv`, `symv`, `syr`, `syr2`, `trmv`, `trsv` | All four types |
| 2 | `ger` | The two real types |
| 2 | `geru`, `gerc`, `hemv`, `her`, `her2` | The two complex types |
| 3 | `gemm`, `gemmt`, `gemmtr`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm` | All four types |
| 3 | `gemm3m`, `hemm`, `herk`, `her2k` | The two complex types |

The actual symbol names for `iamax` are `isamax_` / `idamax_` / `icamax_` / `izamax_`,
and complex norms and absolute-value sums are `scnrm2_` / `dznrm2_` / `scasum_` / `dzasum_`.
CBLAS connects the 172 entry points from #271 to LP64 / ILP64 in addition to the six
pre-existing dot entry points. It converts row/column-major layouts, Cint enums, complex
coefficient pointers, and 0-based `size_t` indices. This does not mean that every LBT
CBLAS extension is provided.
CBLAS GEMM3M composes three native real GEMM operations and allocates temporary storage.
Intermediate sums can overflow or lose precision; no optimized 3M kernel or speedup is claimed.
Fortran GEMM3M remains arithmetic-compatible ordinary native complex GEMM.

In this configuration, LP64 and ILP64 each register 1480 symbols (374 BLAS and 1106
LAPACK) with LBT.
To inspect the registration, run:

```jldoctest
julia> using Colmat, LinearAlgebra

julia> cfg = BLAS.get_config();

julia> libs = filter(l -> realpath(l.libname) == realpath(Colmat.backend_library()), cfg.loaded_libs);

julia> Set(l.interface for l in libs) == Set([:lp64, :ilp64])
true

julia> all(length(BLAS.lbt_forwarded_funcs(cfg, l)) == 3116 for l in libs)
true
```

## What remains as a fallback

The tables on this page summarize the main families. The current connection status,
including additional band/packed families and LAPACK, is in
[Full LBT and colmat-rs routine mapping](@ref). Unregistered routines use the
existing backend.

## Computation paths and temporary storage

| Operation | Current path |
|---|---|
| GEMM | Borrows Julia buffers and runs the Rust colmat matrix product |
| GEMV | Runs the Rust colmat matrix product and copies the result from a temporary vector to the output |
| SYMM / HEMM / TRMM | Expands structured input into temporary storage and runs the matrix product |
| SYRK / SYR2K / HERK / HER2K | Computes the product in a temporary matrix and writes back to the selected triangle |
| Vector operations, SYMV / HEMV, GER / SYR / HER | Loops in the Rust compatibility layer |
| TRSV / TRSM | Triangular solves in the Rust compatibility layer; right-side TRSM expands the coefficient matrix temporarily |
| GETRF / POTRF | Borrows Julia buffers and factors in place |
| GETRS / GETRI / POTRS / POTRI | Borrows the factors and writes back only the solve or inverse result |
| GESDD / GESVD | Borrows and reads the input and writes singular values and vectors to the output |
| GEEV / GEEVX / SYEV / HEEV families | Borrows the input and computes the eigenvalue decomposition; balancing and Schur form construction also run in caller buffers |
| GGEV / GGHRD families | Borrow the input and write the Schur and Hessenberg forms back to caller buffers |
| HSEQR / GEES / HGEQZ / GGES families | Borrow the input Hessenberg or triangular matrix and run scaling and block transformations in place; Q/Z are written to the output region |
| TREXC / LAEXC / TRSEN / GEESX | Use public Schur block moves and the existing GEES, converting block indices, selection order, Q updates, and the intermediate state on rejection; S/SEP and RCONDE/RCONDV use the original ILP64 LAPACK, while integer arrays and callbacks use ABI conversion |
| TREVC / TREVC3 | Compose the public Schur eigenvector helper, the left side from a reversed transpose/adjoint form, selected columns, and back-transformation; native rejections use the saved original LAPACK |
| TGEXC / TGEX2 / TGSEN / GGESX | Public QZ reorder and eigenvalue extraction from saved factors; condition and separation estimates use the saved existing ILP64 LAPACK; allocate internal matrices, scratch, and space for integer ABI conversion |
| Band/packed tridiagonalization and Q formation/application | Compose native tridiagonalization and packed reflector application; AP/TAU interoperate with external LAPACK, and the band path also applies Q on the right of an existing X |
| GST/GV/GVD/GVX for SBG/HBG/SPG/HPG | Expand band/packed storage into owned matrices and compose triangular/split Cholesky, congruence transformations, full EVD, and range selection; positive ABSTOL, uncertain range boundaries, and native failures use the saved original LAPACK |
| GELS / GELSS / GELSD families | Borrow the coefficient matrix and pass it to SVD or QR; expand only the right-hand side into the scaling workspace |
| PSTRF / PSTF2 | Borrow the caller's triangle; allocate separate work for updating the residual matrix |

At present there is no guarantee that every routine runs without allocation or is faster
than the existing backend. When comparing performance, measure with the same forwarding
target, type, matrix size, and thread count.

## Native boundary

The BLAS ABI is a contract in which the caller passes valid pointers, shapes, leading
dimensions, strides, and output regions. In addition to the checks in Julia's ordinary
wrappers, the compatibility layer validates options, dimensions, leading dimensions, and
zero strides in Level 2/3 calls. Level 1 argument conventions differ by routine.

If argument validation finds a problem, the call returns without computing.
The native diagnostic function `colmat_blas_last_error()` reads and clears the last
diagnostic recorded for the calling OS thread. Zero means no diagnostic, and a positive
value is the number of the invalid argument.
For low-level GEEVX calls, it records -1 if a required original implementation is not
configured, and -2 if an integer output of an external implementation is outside the
caller ABI's range. These are not LAPACK INFO values.
This is not an API that converts Rust exceptions into Julia exceptions or checks the
safety of arbitrary raw pointers.


## Connected LAPACK

For all four types and LP64/ILP64, the following 19 families (76 canonical symbols in
total) are provided.

| Family | Native processing |
|---|---|
| GETRF / GETRF2 / GETF2 / GETRS / GETRI / GESV | Partial-pivoting LU and triangular solves |
| POTRF / POTRF2 / POTF2 / POTRS / POTRI / POSV | LLT and triangular solves |
| GESDD / GESVD | SVD with full/thin/overwrite conversion |
| GEEV / GEEVX | General EVD, balancing, left/right eigenvectors, Schur |
| GGEV / GGEV3 / GEGV / GGEVX | Generalized EVD, homogeneous α/β, left/right eigenvectors |
| SYEV / HEEV / SYEVD / HEEVD | Self-adjoint EVD |
| SYEVR / HEEVR | Range selection from the full self-adjoint EVD |

The real SY families and the complex HE families count as the same family per type.
GEEVX condition estimation is delegated to the saved original backend, converting
integers to ILP64 if no original LP64 implementation exists. An original ILP64 GEEVX for
all four types is required before registration. The native support is `SENSE=N`.
For GGEVX, `BALANC=N/SENSE=N` is native, and other modes are delegated to the existing
ILP64 backend.
Connection candidates, missing public APIs, and contract differences for all LBT symbols
are listed in [Full LBT and colmat-rs routine mapping](@ref).


## ROOK factor connections (#372)

The 20 symbols for the real SY / complex Hermitian HE `TRF_ROOK`, `TF2_ROOK`, `TRS_ROOK`,
`TRI_ROOK`, and `SV_ROOK` routines are connected to both integer ABIs.
They use public native Rook pivoting and conversion of the compact IPIV with two swap
indices, and verify mutual solve/inverse use with external LAPACK factors.
The minimum caller WORK and the query result for TRF/SV are 1, and internal scratch is
allocated separately. Subnormal rows receive a symmetric congruence scaling, and
subnormal D and right-hand sides receive a per-block common scaling. These also apply to
the shared ordinary and packed BK adapters.
This does not imply dedicated Fortran algorithms, identical pivot columns, or equivalent
performance.
