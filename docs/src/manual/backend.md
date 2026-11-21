# Backends and fallbacks

## Call flow

```text
Julia Matrix / Vector and LinearAlgebra
                  ↓
           libblastrampoline
                  ↓
        BLAS ABI of libcolmat_blas
                  ↓
       Rust colmat / Rust compatibility layer
```

`Colmat.__init__()` runs `Colmat.load_backend()`.
It checks that the library exists and the compatibility layer's ABI version, then
registers ILP64 and LP64, in that order, with `BLAS.lbt_forward`.

Rust borrows Julia arrays only for the duration of the call. No Rust matrix ownership or
lifetime-management handle is exposed on the Julia side. Mutation rules such as those of
`mul!` follow the Julia API.

## Handling unsupported routines

Registration uses `clear=false`. It replaces the symbols Colmat provides and retains the
forwarding targets of unsupported BLAS / LAPACK routines.

| Call | Where it runs |
|---|---|
| GEMM / GEMV and similar for supported types | Colmat |
| Band/packed storage or unsupported BLAS | The existing forwarding target |
| LAPACK LU / SVD / general and self-adjoint eigenvalues / Cholesky | Colmat |
| LAPACK such as QR / Schur / generalized eigenvalues that is not connected yet | The existing forwarding target |
| Types outside this BLAS, such as integers or BigFloat | Julia's existing per-type implementation |

Whether BLAS calls made from inside the fallback's shared library go through LBT depends
on that library's link configuration. Having an entry registered in LBT is not the same
as switching the internal computations of other unconnected factorizations to Colmat.
Loading this package alone also does not replace the choice of sparse solver.

## Scope and ordering of switching

Changes apply to the whole Julia process and also affect other packages that go through
LBT. They are not per-computation, per-array, or per-Julia-Task settings.

Loading another BLAS provider later may cause that provider to change the forwarding
targets. To re-select Colmat, run:

```julia
Colmat.load_backend()
```

This call also retains the current forwarding targets of unsupported routines. It does
not restore an original OpenBLAS configuration or pin the fallback to a specific library.

Finish switching backends before starting parallel BLAS computation.
Colmat's internal configuration lock does not synchronize other packages or direct LBT
operations.


## LAPACK conversion and unsupported options

Colmat converts the final LU permutation to LAPACK's IPIV and the SVD's V to VT.
The LU and Cholesky factorizations borrow the caller's array and run in place, and SVD,
general/self-adjoint EVD, Schur/QZ, generalized Hessenberg, and least squares also read
their input without copying it.
Workspace queries accept LAPACK's `lwork=-1` and similar forms, but the native workspace
is currently allocated inside the adapter. This is not an implementation that runs
without allocation using only the LAPACK work array.

General EVD uses native balancing, left/right eigenvector computation, and Schur
computation. It currently performs additional work to obtain the returned Schur form, so
there is a performance limitation.

For GEEVX, `SENSE=N` is native, and condition estimation for `SENSE=E/V/B` uses the saved
original implementation. An implementation with the same integer ABI is preferred; if
there is no original LP64 GEEVX, integers are converted and the call is delegated to the
saved ILP64 implementation. All four types require an ILP64 GEEVX before Colmat is
loaded. If it is missing, initialization stops with a configuration error before LBT is
changed. A valid expert mode is not reported as an invalid `SENSE`. Re-running
`load_backend()` does not save Colmat itself as an original backend and retains the
existing implementation.

GGEV/GGEV3/GEGV and GGEVX with `BALANC=N/SENSE=N` use native GEVD.
Other balance/condition modes of GGEVX are delegated to the saved ILP64 backend, with
integer conversion for LP64 calls. Therefore all four types require an ILP64 GGEVX before
Colmat is loaded. Ordinary Julia provides OpenBLAS. If it is missing, initialization
stops with a configuration error before changing the LBT registration.

SYEVR/HEEVR compute the full spectrum and then select the `RANGE=A/I/V` results.
This does not mean colmat-rs has a public MRRR or Sturm-count API.
Per-driver numerical implementations, storage formats, and unsupported features are
documented in [Full LBT and colmat-rs routine mapping](@ref).
