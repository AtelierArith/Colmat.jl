# Issue #271: CBLAS entry adapters

The frozen issue scope contains 172 canonical CBLAS names, in addition to six
pre-existing dot entries. The installed library forwards all 172 names through
both integer interfaces. The complete LBT inventory now records 1,480 forwards
per ABI: 374 BLAS and 1,106 LAPACK. Other CBLAS extensions remain outside this
scope. The native dependency stays pinned to
`4e6ed0e34659147a5c744ab6431f150e78db6f82`.

## Contracts and implementation

Signatures follow [OpenBLAS v0.3.30 cblas.h](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/cblas.h).
GEMMTR additionally follows the current official header blob
`89b2f99cde1c6c4f420f24af43816e16163a01eb`, whose four scalar declarations have
the GEMMT signature. Enums remain C int for both integer interfaces; dimensions
and strides use i32/i64. CBLAS_INDEX is size_t in both interfaces. Complex
coefficients are pointers, except the specified real Hermitian coefficients.
Complex dot functions return complex values. Index reductions return zero-based
indices and preserve the first tie, including single-precision complex abs1.

The adapters borrow caller buffers during the call and reuse existing native
products, reductions, triangular operations and storage adapters. Row-major
conversion preserves conjugation and selected triangles, including band and
packed layouts. ConjNoTrans is handled explicitly. Span validation reports
original C argument positions through the consuming `colmat_blas_last_error`
getter before pointer access. There is no Julia method piracy.

CBLAS GEMM3M stages the logical real/imaginary operands and computes
P=Ar*Br, Q=Ai*Bi, R=(Ar+Ai)*(Br+Bi), using three public native real GEMMs.
The complex product is (P-Q)+i(R-P-Q). Its explicit staging buffers hold
3*(m*k+k*n+m*n) real elements; internal native scratch is additional.
Intermediate sums can overflow or lose precision. This composition makes no
optimized-kernel or speed claim. The existing Fortran GEMM3M entry retains its
ordinary native complex GEMM arithmetic compatibility.

## Numerical and ABI checks

`test/cblas.jl` freezes all 172 names and verifies their actual LBT targets.
`test/cblas_calls.jl` executes named DLL or LBT trampoline entries and records
completion after the concrete call returns. The family tests require every name
in both ABIs and both routes: 688 distinct completed combinations. Independent
scalar references cover supported real/complex types, layouts, transpose and
conjugation options, leading dimensions, signed strides, selected triangles,
unit diagonals, zero coefficients and empty dimensions. Unused storage and
Hermitian imaginary diagonals are poisoned with NaNs; output guards are checked.
Triangular tests exchange products with standard Fortran BLAS in both directions
and verify that the Fortran routes also target this library.

`test/cblas_abi_worker.c` is compiled as an independent C caller by the ordinary
Julia suite. It checks complex value returns, size_t indices, Cint enums in
ILP64, real versus complex-pointer coefficients, row-major and structured
storage fixtures. Linux links libdl/libm; macOS uses its normal system runtime.

The execution artifact includes the source/native revisions, library/manifest
and all production/test/worker source hashes. CBLAS validation distinguishes
leading Cint enums from by-value integer dimensions, requires both routes and
rejects registration-only witnesses. Registration is separate from numerical
execution and independent-algorithm assessments.

## True no-op correction

An executed C HERK call with alpha=0, beta=1 changed an ignored diagonal imaginary
NaN to zero. The shared adapter kernel now returns before allocation, output
access or negative-stride pointer arithmetic for the reference quick-return
conditions in GEMV, HEMV/SYMV and rank-k operations. Regression tests preserve
NaN values and call legal no-ops with null arrays, checking that no
argument error was reported. The reference is LAPACK v3.12.1 BLAS
[ZHERK](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/BLAS/SRC/zherk.f),
[ZSYRK](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/BLAS/SRC/zsyrk.f),
[ZGEMV](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/BLAS/SRC/zgemv.f)
and [ZHEMV](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/BLAS/SRC/zhemv.f).
This correction is in Colmat.jl's adapter; the native repository is unchanged.

## Acceptance evidence

The full `Pkg.test(; allow_reresolve=false)` run passes at code revision
`863c18df1a15ee916cb2a007fb3692f12cc3f407` on Julia 1.13.3 / x86_64 macOS.
The imported artifact matches all 232 production/test/worker source hashes,
the dependency manifest, registration table and installed library hash. It
contains 4,469 call-site witnesses and validates 1,106 LAPACK plus 172 new CBLAS
canonical entries. CBLAS includes all 688 distinct ABI/route combinations.
The full-suite evidence gates pass 5 CBLAS and 240 LAPACK/provenance checks.

Rust formatting and 389 Rust tests pass. All 22 inventory/evidence Python tests,
15 actual LBT inventory checks, the README example, independent compiled C
caller and documentation build pass locally. The source snapshot includes the
C worker and every Julia numerical family test.

Merge additionally requires successful Linux/macOS tests and documentation
checks for the final PR commit. Their ABI artifacts must match the committed
source hashes, native pin, manifest and registration inventory. CI run and
artifact evidence is recorded in the PR description before merging.
