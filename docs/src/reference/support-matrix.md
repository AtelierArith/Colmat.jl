# Measured Julia API mapping

This records separately whether calls on ordinary arrays succeed and whether Colmat provides that entry point.
Entry points are selected from the Julia 1.13 wrapper implementation, and the LBT forwarding target is checked. This is not a measurement of dynamic call counts or a trace that follows the library internals.

## Verification conditions

- Julia: `1.13.4` / `x86_64-linux-gnu`.
- Colmat source: `08581c59f5d3f0b92faf6dcb1d87c6341efeeba8`.
- Rust colmat source: `e1c83482f5a61449d30e989808c3ba5f751ec391`.
- Julia BLAS integer ABI: `ilp64`.
- Colmat execution width: `8`.
- Element types are listed in the order Float32 / Float64 / ComplexF32 / ComplexF64.

## Results

| Julia API | Float32 | Float64 | ComplexF32 | ComplexF64 | Forwarding target of the main entry | Rust-side implementation / connection |
|---|---|---|---|---|---|---|
| `A * B` | Success | Success | Success | Success | Colmat | Matrix product |
| `mul!(C, A, B)` | Success | Success | Success | Success | Colmat | Matrix product |
| `A * x` | Success | Success | Success | Success | Colmat | Matrix product |
| `dot(x, x)` | Success | Success | Success | Success | Colmat | Rust compatibility layer |
| `norm(x)` | Success | Success | Success | Success | Colmat | Rust compatibility layer |
| `BLAS.axpy!` | Success | Success | Success | Success | Colmat | Rust compatibility layer |
| `UpperTriangular(A) \ x` | Success | Success | Success | Success | Colmat | Rust compatibility layer |
| `UpperTriangular(A) \ B` | Success | Success | Success | Success | Colmat | Rust compatibility layer |
| `lu(A)` | Success | Success | Success | Success | Colmat | partial_piv_lu (connected) |
| `lu(A) \ x` | Success | Success | Success | Success | Colmat | LU solve (connected) |
| `qr(A)` | Success | Success | Success | Success | Colmat | qr (connected) |
| `qr(A, ColumnNorm())` | Success | Success | Success | Success | Colmat | col_piv_qr (connected) |
| `svd(A)` | Success | Success | Success | Success | Colmat | svd (connected) |
| `svd(A; alg=LinearAlgebra.QRIteration())` | Success | Success | Success | Success | Colmat | svd (connected) |
| `svdvals(A)` | Success | Success | Success | Success | Colmat | SVD (connected) |
| `eigen(A)` | Success | Success | Success | Success | Colmat | eigen (connected) |
| `eigen(Hermitian(A))` | Success | Success | Success | Success | Colmat | self_adjoint_eigen (connected) |
| `cholesky(Hermitian(A))` | Success | Success | Success | Success | Colmat | llt (connected) |

## Entry symbols and the actual libraries

Each row examines the same API as the table above. Symbols follow the order of the four types above.

| Julia API | Entry examined | LBT forwarding target |
|---|---|---|
| `A * B` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `mul!(C, A, B)` | `sgemm_`, `dgemm_`, `cgemm_`, `zgemm_` | `libcolmat_blas.so` |
| `A * x` | `sgemv_`, `dgemv_`, `cgemv_`, `zgemv_` | `libcolmat_blas.so` |
| `dot(x, x)` | `cblas_sdot`, `cblas_ddot`, `cblas_cdotc_sub`, `cblas_zdotc_sub` | `libcolmat_blas.so` |
| `norm(x)` | `snrm2_`, `dnrm2_`, `scnrm2_`, `dznrm2_` | `libcolmat_blas.so` |
| `BLAS.axpy!` | `saxpy_`, `daxpy_`, `caxpy_`, `zaxpy_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ x` | `strsv_`, `dtrsv_`, `ctrsv_`, `ztrsv_` | `libcolmat_blas.so` |
| `UpperTriangular(A) \ B` | `strsm_`, `dtrsm_`, `ctrsm_`, `ztrsm_` | `libcolmat_blas.so` |
| `lu(A)` | `sgetrf_`, `dgetrf_`, `cgetrf_`, `zgetrf_` | `libcolmat_blas.so` |
| `lu(A) \ x` | `sgetrs_`, `dgetrs_`, `cgetrs_`, `zgetrs_` | `libcolmat_blas.so` |
| `qr(A)` | `sgeqrt_`, `dgeqrt_`, `cgeqrt_`, `zgeqrt_` | `libcolmat_blas.so` |
| `qr(A, ColumnNorm())` | `sgeqp3_`, `dgeqp3_`, `cgeqp3_`, `zgeqp3_` | `libcolmat_blas.so` |
| `svd(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `svd(A; alg=LinearAlgebra.QRIteration())` | `sgesvd_`, `dgesvd_`, `cgesvd_`, `zgesvd_` | `libcolmat_blas.so` |
| `svdvals(A)` | `sgesdd_`, `dgesdd_`, `cgesdd_`, `zgesdd_` | `libcolmat_blas.so` |
| `eigen(A)` | `sgeevx_`, `dgeevx_`, `cgeevx_`, `zgeevx_` | `libcolmat_blas.so` |
| `eigen(Hermitian(A))` | `ssyevr_`, `dsyevr_`, `cheevr_`, `zheevr_` | `libcolmat_blas.so` |
| `cholesky(Hermitian(A))` | `spotrf_`, `dpotrf_`, `cpotrf_`, `zpotrf_` | `libcolmat_blas.so` |

## Scope of the assessment

- "Success" means that the reference computation or the decomposition residual was satisfied for this probe's small finite inputs.
- Residuals for matrix products, solves, and decomposition reconstruction are checked with scalar sum-of-products that do not call BLAS.
- Success on the "existing backend" does not mean that the corresponding Rust colmat decomposition is connected.
- The Rust-side capability names are candidates confirmed in `faer/src/linalg/solvers.rs` at the pinned revision. This does not claim that the probe directly executed Rust functions that are not connected.
- This table is not exhaustive verification of all BLAS / LAPACK. It also does not include singular matrices, non-finite values, generalized eigenvalues, or results on other OSes.
- This is a check of the high-level Julia API, separate from the existing regression tests such as the direct LP64 ABI tests and negative-stride tests.

## Regeneration

Run this from the repository root in a fresh Julia process after building.

```sh
julia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md
```

If verification fails, the table is not overwritten. The call cases and verification methods are in `test/support_matrix.jl`.
## Householder ABI connection (#264)

This addendum is a verification separate from the existing Linux measurement table. It used Julia 1.13.3 /
`x86_64-apple-darwin` and colmat-rs
`e1c83482f5a61449d30e989808c3ba5f751ec391`.
`test/householder.jl` invokes the four-type, LP64/ILP64 shared-library entry points of `larf/larfx/larfy/larft/larfb/larfg/larfgp`
with concrete C signatures, and
confirms that the LBT forwarding target of the 28 canonical symbols is Colmat.

Generation is a Rust adapter matched to LAPACK's `I - tau*v*v'`, real beta, and the non-negative beta of LARFGP.
It is not a mere reciprocal conversion of colmat's generation API that preserves real coefficient and complex phase.
Application uses the existing native GEMV/SYMV/GEMM.
The implicit unit diagonal of F/B and C/R is expanded, and the triangular factor of LARFT is constructed.
LARFB also accepts external factors in LAPACK form.

Verification covers reflector products, unitarity, minimal/maximal scales, complex heads, zero coefficients, length 1, positive and negative strides, left/right, upper/lower triangular, and LARFT factors from external OpenBLAS.
For LARFX at small orders WORK is not referenced, and LARFY updates only the selected triangular region.
These auxiliary routines have no workspace query or INFO argument.
Diagnostics for invalid arguments use the existing `colmat_blas_last_error` convention.

LARFB internally allocates an explicit reflector matrix of size basis times order squared.
It runs with the caller's minimal legal WORK, but this does not guarantee an implementation that uses only the workspace, nor the same operation count and allocation volume as LAPACK.
This connection is limited to the Fortran ABI; LAPACKE is a separate task.

## Compact QR connection (#290)

With Julia 1.13.3 / x86_64-apple-darwin and the pinned colmat-rs revision above, ordinary `qr(A)`, `Matrix(F.Q)`, and left/right application of Q/Q' were connected to Colmat's GEQRT/GEMQRT entry points. The record of "not connected" in the existing Linux snapshot is retained as the result at that point in time.

`test/qr_compact.jl` verifies the four types, concrete C signatures for LP64/ILP64, NB=1/2/4, GEQRT2/GEQRT3, bidirectional interoperation with external OpenBLAS factors, leading dimension, empty matrices, zero and dependent columns, minimal scale, and invalid arguments.
These entry points have no LWORK query. WORK is run at the standard minimum length.

Native QR is applied to each active column, and LAPACK's T is constructed from the reciprocal of the real native coefficients. This preserves LAPACK's diagonal positions even when native rank handling compacts zero columns. The diagonal phase of a complex R is preserved, and the product of Q and R and reflector application by external LAPACK are verified.
GEQRT3 also uses the common numerical path and differs from Netlib's recursive algorithm.
Internal arrays are allocated, and GEMQRT expands each block explicitly.

## Standard QR connection (#289)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 36 symbols of GEQRF/GEQR2, the non-negative diagonal variants, ORGQR/UNGQR/ORG2R/UNG2R, ORMQR/UNMQR/ORM2R/UNM2R, and GEQRS were connected to both integer ABIs.
`test/qr_standard.jl` verifies reconstruction and left/right application of Q through the Julia wrapper, as well as direct execution on both ABIs, acceptance of external OpenBLAS factors and external application of native factors, non-negative diagonals, workspace query and minimal legal workspace, padded leading dimension, empty/rectangular/zero-column cases, and the least-squares residual of GEQRS.

The real standard version builds A/TAU from the block-size-one coefficients of native QR.
The complex standard version and the non-negative diagonal version use LAPACK gauge reflector generation and align real beta and complex TAU with Julia's standard convention.
Application of Q is connected to native GEMV, and the triangular solve of GEQRS to the native triangular kernel.
Because internal arrays are allocated, memory is used beyond the caller's WORK.
GEQRS follows the deprecated LAPACK contract and does not provide a workspace query.

## Column-pivoted QR connection (#299)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 12 symbols of GEQP3/GEQPF/GEQP3RK were connected for the four types and both integer ABIs.
`qr(A, ColumnNorm())` reaches native QR/reflector application from GEQP3.
Fixed JPVT columns are moved to the front while preserving the input order, and the output returns the original column numbers as 1-based values.

GEQP3RK stops on KMAX and on ABSTOL/RELTOL against the largest column 2-norm of the residual part.
Because column norms are recomputed at each step, the operation count and rounding path differ from rank determination by the diagonal values of native ColPivQr and from Netlib's partial norm updates. Attached NRHS columns are excluded from pivot selection, and Q' is applied. NaN/Inf INFO, zero and minimal tolerance, and the return of rank and residual norms are handled separately. Internal arrays are allocated.

`test/qr_pivoted.jl` verifies concrete C signatures for both ABIs, fixed columns, external OpenBLAS factors, reconstruction of a decomposition stopped midway, independently computed residual norms, attached right-hand sides, query and minimal legal workspace, leading dimension, empty/1×1/zero matrices, NaN/Inf diagnostics, and LBT registration. Because the Julia GEQP3 wrapper for an empty matrix does not call the native entry point, this is distinguished from the empty-matrix case in the direct ABI.

## LQ connection (#291)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 56 symbols of standard, blocked, short-wide, and header-form LQ, generation and left/right application of Q, and GELQS were connected for the four types and both integer ABIs.
Standard LQ uses the adjoint QR identity and constructs A/TAU and a row-preserving block T.
The short-wide form stores the factor of the first NB columns and the factors of the following [L B] panels in the same order as LAPACK. It accepts external T and headers.

GELQ chooses the legal MB=1, NB=N form, so the recommended values of the optimal and minimal queries coincide. GELQT3 is a common decomposition path and is not Netlib's recursive algorithm.
It internally allocates adjoint matrices, reflectors, selected panels, and so on. GELQS follows the deprecated LAPACK contract and does not provide a workspace query.

`test/lq.jl` verifies the four types and both ABIs, the ordinary Julia wrapper, minimal legal workspace, minimal header query, block boundaries, bidirectional interoperation with external OpenBLAS factors, left/right, transpose, and conjugate, leading-dimension padding, empty/rectangular/rank-deficient cases, the residual and minimum-norm property of GELQS, and LBT registration of the 56 symbols.

## RZ connection (#304)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 32 symbols of TZRZF/TZRQF/LATRZ, ORMRZ/UNMRZ, LARZ/LATZM, and LARZT/LARZB were connected for the four types and both integer ABIs. The upper trapezoid `A = [R 0] Z` is constructed from the trailing rows, and the sparse reflectors are applied with native GEMV/GEMM.
LARZT/LARZB handle LAPACK's backward and row-preserving forms and also convert the conjugates of a complex T.
The central region on which neither a single reflector nor a block reflector acts is preserved, and internal arrays are allocated.

`test/rz.jl` verifies direct calls on both ABIs, the Julia wrapper, interoperation with external OpenBLAS factors and application, query and minimal legal workspace, padding, positive and negative strides, separated C1/C2, L=0 and a tail overlapping the implicit components, input with NaN in the central region, orthogonality, reconstruction, conjugate application, and LBT registration.

## Least-squares driver connection (#310)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 28 symbols of GELS/GELST/GETSLS, GELSS/GELSD, and GELSY/GELSX were connected for the four types and both integer ABIs.
Full-rank systems handle tall/wide plus transpose and conjugate with QR/LQ and a triangular solve, scale A/B, and restore the scale of the solution, residual, and factors.
GELSS/GELSD determine RCOND and rank with the common native SVD.
GELSS returns row-wise right singular vectors in A. The GELSD query returns the required lengths of IWORK/RWORK in addition to WORK.

GELSY/GELSX is a complete orthogonal decomposition by column-pivoted QR with fixed JPVT, condition determination of the leading R, RZ, application of Q/Z, and inverse permutation. Because the singular values of the leading R are used for condition determination, the rank near a boundary may differ from Netlib's incremental condition estimation.
GELST/GETSLS use the common full-rank path and are not identical to Netlib's compact-WY/TS algorithm.
GETSLS is configured so that the optimal and minimal query values coincide. The full U of the SVD and intermediate matrices are allocated internally, and the name GELSD does not guarantee execution of the native divide-and-conquer.

`test/least_squares.jl` verifies concrete C signatures for both ABIs, comparison with external OpenBLAS, fixed columns, minimum-norm solutions for rank-deficient and underdetermined systems, singular values and right singular vectors, TRANS/conjugate, residual placement, minimal/maximal scales, padding, minimal workspace and query, empty/zero inputs, and LBT registration of the 28 symbols.

## Hessenberg connection (#293)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 16 symbols of GEHRD/GEHD2, ORGHR/UNGHR, and ORMHR/UNMHR were connected for the four types and both integer ABIs.
A/TAU is constructed by LAPACK gauge reflector generation and left/right application with native GEMV, and the complex subdiagonal is made real. GEHRD/GEHD2 use a common path with no difference in blocking.
The subrange ILO/IHI is supported, and Q is the identity outside that range.
Arrays for reflectors and Q generation are allocated internally.

`test/hessenberg.jl` verifies the Julia wrapper, concrete C signatures for both ABIs, interoperation with external OpenBLAS factors and Q, A=Q*H*Q', orthogonality, preservation outside the active range, empty/1×1, left/right, transpose, conjugate, padding, minimal workspace and query, invalid ranges, and LBT registration.

## Schur driver connection (#300)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 20 symbols of HSEQR/LAHQR/LAQR0/LAQR4/GEES were connected for the four types and both integer ABIs.
The active block of the Hessenberg form is passed to the public native Schur API, and coupling with the outside range and the existing Z are updated with native GEMM. COMPZ=N/I/V, JOB=E/S, and WANTT/WANTZ and ILOZ/IHIZ of the auxiliary entry points are handled. Even with JOB=E/WANTT=false it computes through to Schur form. Each entry point uses the same native QR path and does not reproduce the switching among the Netlib LAHQR/LAQR0/LAQR4 algorithms.

GEES composes Hessenberg reduction, native Schur, and native block reorder.
It normalizes the scale before reduction, iteration, and reordering, and passes eigenvalues in the original units to SELECT. The Fortran LOGICAL of SELECT is handled with the same width as the integer ABI, and for a real conjugate pair both are selected if one is selected. It returns SDIM and the forward placement of the selected spectrum, and a rejected swap and an inconsistency in reselection are INFO=N+1/N+2 respectively. SORT=N and workspace query do not call the callback.
Condition estimation of GEESX/TRSEN is not included in this connection.

The public native Schur API does not return a partial index for non-convergence. In that case it returns INFO=IHI treating the whole active range as non-converged, and does not falsely report partial convergence.
It allocates matrices, reflector coefficients, and native scratch internally and can run with the caller's minimal legal WORK. The query returns that required length.

`test/schur.jl` verifies concrete C signatures for the four types and both ABIs, the ordinary Julia wrapper, A=Z*T*Z', orthogonality, the active range, updating the existing Z and row ranges, real Schur conjugate pairs, SELECT/SDIM, minimal/maximal scales, padding, NaN in unreferenced regions, empty/1×1, query, invalid arguments, non-convergence INFO, and LBT registration of the 20 symbols.

## Generalized Hessenberg connection (#269)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 8 symbols of GGHRD/GGHD3 were connected for the four types and both integer ABIs.
The left and right bases of the active block returned by the public `generalized_hessenberg` are embedded into the full matrix, and coupling with the outside range and the existing Q/Z are updated with native GEMM.
COMPQ/COMPZ=N/I/V and ILO/IHI are handled, and the strict lower triangle of B is not read on input. GGHRD/GGHD3 use the same native path and allocate internal matrices and scratch.
GGHD3 requires exactly one element of the caller's WORK and returns 1 in the query as well.
This does not reproduce the separate Netlib GGHRD/GGHD3 algorithms or how they use external WORK.

`test/generalized_hessenberg.jl` verifies concrete C signatures for both ABIs, the four types, Q'*A*Z=H and Q'*B*Z=T, Hessenberg/triangular structure, unitarity, empty/1×1/2×2, partial ranges, initialization/update/unused state of Q/Z, the 40×40 block path, padding, NaN in unreferenced B/Q/Z, query, invalid arguments, and LBT registration of the 8 symbols.

## QZ driver connection (#294)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 16 symbols of HGEQZ/GGES/GGES3/GEGS were connected for the four types and both integer ABIs.
HGEQZ composes the native QZ of the active block with basis transformations and handles COMPQ/COMPZ=N/I/V and ILO/IHI.
Even with JOB=E it computes through to Schur form internally.
It makes the diagonal of β non-negative real and preserves the transformations of Q/Z and the pencil.
α/β are returned as homogeneous pairs, and β=0 is not replaced with a finite eigenvalue.

GGES/GGES3/GEGS compose triangularization of B by QR, generalized Hessenberg reduction, and native QZ. A/B are independently scale-normalized, and the units of α/β and S/T are restored.
The SELECT of GGES/GGES3 is called with the LP64/ILP64 LOGICAL width, and both members of a real conjugate pair are selected if one is selected. The selected blocks are moved forward with native reorder, and the QZ of the reordered pencil is re-evaluated to update α/β.
Query and SORT=N do not call the callback. A rejected swap is INFO=N+3 and an inconsistency in reselection is INFO=N+2. Condition estimation of GGESX/TGSEN is out of scope.

The public native QZ API does not return a non-convergence state or partial index.
It checks the finiteness of the returned Schur structure and of each component of the homogeneous α/β, and treats an incomplete result as non-convergence of the whole active range with INFO=IHI.
It allocates internal matrices and scratch and provides LAPACK's minimal legal WORK and a query for each entry point. GGES/GGES3/GEGS use a common path and do not reproduce the blocking or permutation balancing implementations of the individual Netlib drivers.

`test/qz.jl` verifies concrete C signatures for both ABIs, the four types, reconstruction of the pencil, Schur structure, unitarity, the active range, initialization/update/unused state of Q/Z, 40×40 input, independent OpenBLAS comparison of homogeneous values, β=0, real conjugate pairs, SELECT/SDIM, independent minimal/maximal scales, padding, NaN in unreferenced regions, empty/1×1/2×2, query and minimal WORK, invalid arguments, non-convergence INFO, the Julia GGES wrapper, and LBT registration of the 16 symbols.

## Generalized EVD driver connection (#301)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 16 symbols of GGEV/GGEV3/GEGV/GGEVX were connected for the four types and both integer ABIs.
The ordinary driver converts the homogeneous α/β and left/right eigenvectors of the public `gevd_real/cplx`. It preserves β=0 and the 0/0 singular pencil and returns a real conjugate pair in adjacent real-part and imaginary-part columns. The native real transpose-left convention is converted to LAPACK's adjoint-left convention by reversing the imaginary-part columns, and the largest |Re|+|Im| of each vector is normalized to 1.
Each entry point uses the same native path and internal scratch and returns the minimal WORK/query for that entry point.
When native GEVD returns non-convergence, INFO=N is set because there is no partial index.

`BALANC=N/SENSE=N` of GGEVX is native. It returns ILO/IHI=1/N, LSCALE/RSCALE=1, and ABNRM/BBNRM of the original matrices. The Schur output for A/B is returned to the original units by tracking the power-of-two prescale of the native implementation at the pinned revision. This conversion must be re-verified when dependencies are updated.

The other BALANC/SENSE combinations are delegated to the existing ILP64 expert backend.
On LP64 calls, integer scalars are converted to 64 bit, and IWORK/BWORK are allocated within the call so that 64-bit values are not written into 32-bit output areas. ILO/IHI/INFO are restored to the original width, and ILO/IHI are not modified during a query. Balance and condition estimation are not treated as natively supported.
A four-type ILP64 GGEVX is required before loading Colmat. Ordinary Julia provides OpenBLAS.
If it is missing, an explicit configuration error is returned before changing the LBT registration; legal modes are not reported as invalid arguments, and uncomputed condition estimates are not treated as success.
Re-registration does not save Colmat itself as the expert backend.

`test/generalized_eigen.jl` verifies the four types, both ABIs, and all four entry points; left/right/single-sided/values-only; homogeneous left and right residuals; β=0; 0/0; real conjugate pairs and normalization; independent OpenBLAS comparison; all BALANC/SENSE combinations of GGEVX; metadata; query; integer-width guards; padding; Schur output in the original scale; empty/1×1; insufficient workspace; and LBT registration.
A separate process also checks the pre-registration diagnostic when the expert backend is missing and re-registration.

## Pivoted Cholesky connection (#265)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 8 symbols of PSTRF/PSTF2 were connected for the four types and both integer ABIs.
The adapter controls the largest-diagonal pivot and stopping by TOL, and composes the native LLT of each pivot with the Schur complement update by native GEMM. Because the public one-shot pivoted LLT uses a fixed internal tolerance, this configuration handles arbitrary TOL and TOL=0.

TOL<0 uses N*DLAMCH(E)*largest diagonal, and candidates after taking the first positive pivot are compared with TOL. RANK<n gives INFO=1, and empty input gives RANK=0/INFO=0.
PIV returns the original row numbers at each position after permutation as 1-based values.
If the initial diagonal is non-positive, it is not factorized and RANK=0/INFO=1 is set. If computation stops midway, the completed factor, the not-yet-computed permuted triangular part, and the stopping diagonal are returned.
With UPLO=U, the internal lower-triangular factor is output conjugate-transposed.
The opposite triangle and padding are left unchanged, and the imaginary part of a complex diagonal is not read on input either.

PSTRF/PSTF2 use a common sequential pivot path and allocate an internal dense matrix and native scratch.
They do not depend on the contents of the caller's WORK. These LAPACK entry points have no workspace query. Because pivots and rank can change through rounding near the tolerance, the algebraic rank and the numerical rank of the floating-point Gram matrix are distinguished.

`test/pivoted_cholesky.jl` verifies direct signatures for the four types, both ABIs, and UPLO; PIV and rank; PSD/full-rank/strictly rank-deficient/zero and negative diagonals; reconstruction of partial factors; comparison with OpenBLAS factors and stopping states; the equality boundary of TOL and the automatic tolerance; TOL=0 over a large dynamic range; empty/1×1; padding and unreferenced triangles; the imaginary part of a complex diagonal; integer guards; invalid arguments; the ordinary Julia API; and LBT registration.

## Bunch–Kaufman connection (#266)

With Julia 1.13.3 / x86_64-apple-darwin and the same pinned colmat-rs revision, the 32 symbols of TRF/TF2, TRS/TRS2, TRI/TRI2/TRI2X, and SV for real SY / complex Hermitian HE were connected to both integer ABIs. Complex symmetric SY systems remain on the existing backend.
It uses the partial pivoting of native `cholesky::lblt` and the separated diagonal, subdiagonal, and final permutation, and converts to LAPACK's classic IPIV and to the storage format of A that includes the intermediate permutations.
It undoes the subsequent row swaps of native L to preserve the columns, and also converts back from external LAPACK A/IPIV to call the native solve/inverse.
UPLO=U is processed in a basis with reversed row and column order.

TRF uses the native default panel and TF2 the native unblocked path. TRS/TRS2 use a common solve, and TRI/TRI2/TRI2X a common inverse.
Matrices and native scratch are allocated internally. The TRF/SV query returns 1, and the TRI2 query on the common inverse path returns max(1,N).
The NB of TRI2X is accepted as a blocking hint and uses the same inverse.
The opposite triangle, padding, and the imaginary part of a complex diagonal are not read on input.
On output too, only the specified triangle is updated.

A 1×1 zero D in factorization/inverse returns a positive INFO at the LAPACK position, and SV does not execute a solve in that case.
A non-singular 2×2 block with a zero diagonal is not mistakenly judged singular.
The format of the input factor and INFO follow LAPACK, and the native final permutation is not simply returned as IPIV.

`test/bunch_kaufman.jl` verifies the four types, both ABIs, and UPLO; 1×1/2×2; the 80×80 panel path; reconstruction of PAP' and LDL'/UDU'; solve/inverse residuals; bidirectional interoperation of A/IPIV between native and OpenBLAS; each inverse and driver; singular INFO and preservation of B; query; minimal WORK; insufficient workspace; invalid arguments; padding, integer guards, and unreferenced regions; minimal/maximal scales; the ordinary Julia API; and LBT registration of the 32 symbols.
It also regression-checks that complex symmetric problems are not misrouted to HE.


## GEBAL/GEBAK connection (#267)

The 8 Fortran symbols of GEBAL/GEBAK for the four types were connected to LP64/ILP64.
GEBAL uses native `balance_in_place_with` and GEBAK uses `balance_eigenvectors_in_place`. JOB=N/P/S/B and SIDE=L/R are handled, and the final permutation is converted to LAPACK's ILO/IHI/SCALE swap encoding.
Even when all eigenvalues are isolated, a one-element active range is returned for a non-empty matrix, as in LAPACK.
This encoding and range handling are shared with the existing GEEVX.

GEBAK reads and writes only the rows it updates, and does not read SCALE entries unused by the selected JOB, unreferenced rows, or leading-dimension padding.
Internally it allocates matrices, permutations, scales, and native scratch, and does not renormalize the eigenvectors.

`test/balance.jl` verifies the four types and both ABIs, reconstruction of the similarity transformation, left and right inverse transformations, isolated eigenvalues/non-trivial permutations/scales, bidirectional interoperation of metadata with OpenBLAS, eigenvectors for empty/one-element/rectangular cases, unreferenced regions, invalid arguments and integer widths, the Julia LAPACK wrapper, and LBT registration.

## Tridiagonal EVD driver connection (#268)

The four types of STEQR/STEDC/PTEQR/STEGR/STEMR and the two real types of STERF/STEV/STEVD/STEVR/STEVX, 30 Fortran symbols in total, were connected to LP64/ILP64.
The normal path calls native `tridiagonal_self_adjoint_evd`.
STEQR and STERF choose the QR path, and STEDC/STEVD switch between QR and divide-and-conquer at the native default threshold of 128. COMPZ=V post-multiplies the existing Z by the eigenvectors.
PTEQR forms a bidiagonal Cholesky factor of the positive definite tridiagonal and uses the QR path of native `bidiagonal_svd` to preserve the relative accuracy of small positive eigenvalues.
PTEQR returns eigenvalues in descending order, and the other drivers in ascending order.

The normal path of the range-selection drivers selects A/I/V from the full spectrum. This does not mean that dedicated MRRR or bisection/inverse iteration implementations are provided.
STEMR's TRYRAC=true and the positive ABSTOL of STEVR/STEVX are delegated to the saved existing LAPACK.
STEGR's ABSTOL is not used, as in Reference LAPACK.
ISUPPZ is built from the support of the obtained eigenvectors and is not updated for a partial range of STEVR.
The STEMR NZC=-1 query returns the required number of columns in Z(1,1).

Because the public native API does not return intermediate D/E on non-convergence, if native computation fails the saved LAPACK is called from the original input, preserving each driver's INFO and partial-result contract.
For this reason, an ILP64 LAPACK backend for the 24 entry points is required before loading. Ordinary Julia provides OpenBLAS.
For LP64 calls, integer arguments, outputs, and integer scratch are converted, and re-registration does not save itself as the forwarding target.

Matrices, eigenvectors, and native scratch are allocated internally.
Because STEVR/STEMR/STEGR compute the full spectrum even for a narrow selection range, the operation count and allocation volume differ from a dedicated range-selection implementation.
The STEDC caller workspace uses the QR threshold of 25 for small problems and LAPACK's minimum for D&C, and native scratch is allocated separately.
Query does not modify the input matrix or eigenvectors.
However, the STEMR value-range column-count query reads D/E.

`test/tridiagonal_evd.jl` verifies the four types and both ABIs, 0/1/2 dimensions, repeated/split/interval endpoints, post-multiplication of the existing Z, support, each query and minimal workspace, guards and integer widths, 160-dimensional D&C, expert fallback, the ordinary Julia API, and registration of the 30 symbols.
In an isolated process it confirms the diagnostic for a missing fallback and that the normal path does not use the fallback.
STEBZ/STEIN are not included in this connection.

## LAPACK solve/inverse/product for triangular matrices (#270)

The four types of TRTRS/TRTRI/TRTI2/LAUUM/LAUU2, 20 Fortran symbols in total, were connected to both integer ABIs.
TRTRS applies native triangular solve to the borrowed A and B and converts UPLO/TRANS/DIAG.
TRTRI/TRTI2 use native triangular inverse, and LAUUM/LAUU2 native triangular matmul.
TRTI2 shares the inverse kernel with TRTRI, and LAUU2 shares the product kernel with LAUUM.
This does not mean that independent blocked/unblocked algorithms are provided for each name.

The opposite triangle of A and padding are not referenced on input and are preserved on output.
DIAG=U does not read the actual diagonal and does not change that diagonal in the inverse output either.
TRTRS/TRTRI return the first zero diagonal with a positive INFO and do not modify B/A.
TRTRS performs this singular-diagonal check even when NRHS=0.
TRTI2 assumes non-singularity as in Reference LAPACK and does not return a positive INFO.

LAUUM/LAUU2 compute U Uᴴ for upper triangular and Lᴴ L for lower triangular.
They compute the mathematical triangular product including the complex diagonal, and set the imaginary part of the output diagonal to zero.
Solve borrows its input and output. Inverse/product build an internal n×n output matrix from the borrowed input and write back only the specified triangle, without making a copy of the entire input.
These entry points have no caller workspace or workspace query.

`test/triangular_lapack.jl` verifies the four types, both ABIs, and UPLO/TRANS/DIAG; empty/one-element/80-dimensional; multiple/zero right-hand sides; unit diagonal and NaN in the opposite triangle; padding; interoperation with external inverse; left and right inverse residuals; triangular product; singular INFO; invalid arguments and integer widths; the ordinary Julia API; and LBT registration of the 20 symbols.
The existing BLAS TRSV/TRSM paths are outside the scope of this connection.

## Standard-form reduction of the generalized self-adjoint problem (#275)

The 8 Fortran symbols of real SYGST/SYGS2 and complex HEGST/HEGS2 were connected to both integer ABIs.
The input B must already be Cholesky-factorized; this entry point does not factorize it.
L for UPLO=L and Uᴴ for UPLO=U are treated as a common lower-triangular factor, and ITYPE=1 forms L⁻¹ A L⁻ᴴ while ITYPE=2/3 form Lᴴ A L.
The former uses native triangular solve and the latter native triangular matmul.
SYGS2/HEGS2 and SYGST/HEGST use a common kernel; this does not mean that independent blocked/unblocked algorithms are provided for each name.

Only the specified triangle of A/B is read as input. A is expanded into a symmetric/Hermitian matrix, and only the real part of a complex diagonal is read.
The real factor of B is borrowed, and the complex factor is converted to a lower-triangular representation with a real diagonal, so an internal matrix is allocated. ITYPE=2/3 also allocate a temporary matrix for the product.
B is not modified, and the opposite triangle of A and padding are preserved.
These entry points have no workspace argument or query.
Positive definiteness of B is not rechecked, and as in LAPACK INFO reports only invalid arguments.

`test/generalized_reduction.jl` verifies the four types, both ABIs, ITYPE, and UPLO; 0/1/3/40/80 dimensions; non-unit diagonal factors; comparison with both reference LAPACK algorithms; residuals of the three original generalized problems and normalization by B/B⁻¹; the imaginary part of a complex diagonal and NaN in the opposite triangle; preservation of unreferenced regions; invalid arguments and integer widths; ordinary arrays through LBT; external Cholesky factors; and wide scales.
The common standard-form reduction processing is also usable by the generalized self-adjoint EVD driver.


## Generalized self-adjoint positive definite EVD driver (#292)

The 12 Fortran symbols of real SYGV/SYGVD/SYGVX and complex HEGV/HEGVD/HEGVX were connected to both integer ABIs.
They compose native LLT, the standard-form reduction of #275, native self-adjoint EVD, and the inverse transformation of eigenvectors by triangular solve/matmul.
ITYPE=1 handles A X = B X Λ, ITYPE=2 handles A B X = X Λ, and ITYPE=3 handles B A X = X Λ; 1/2 are normalized to Xᴴ B X = I and 3 to Xᴴ B⁻¹ X = I.
Both UPLO values and JOBZ=N/V are handled. B returns the Cholesky factor in the specified triangle.
A positive definiteness failure gives INFO=N+k and does not update A or the eigenvalue/eigenvector output.

GV/GVD use the common native EVD and choose QR or D&C at the default threshold of 128.
This does not mean that independent QR/D&C implementations are provided for each legacy driver name.
GVX selects A/I/V from the full spectrum, and a positive ABSTOL is delegated to the saved ILP64 LAPACK.
This does not mean that a selection computation by dedicated bisection/inverse iteration is provided.
In normal computation eigenvalues are ascending, and GV/GVD return eigenvectors in A while GVX returns them in Z.
IFAIL sets its first M elements to zero on native success.

Because complex native LLT uses the opposite triangle as workspace, factorization is done in an internal matrix and only the specified triangle is written back to B.
The imaginary part of a complex diagonal is not read on input.
Factorization of a real B is performed in a borrowed buffer.
The standard form of A, the eigenvectors, native scratch, and a backup of the specified triangle of B for rerunning on native failure are allocated internally.
If native computation fails without returning intermediate eigenvalues/eigenvectors, B is restored and the saved LAPACK is called, maintaining each driver's INFO and intermediate results.
For this reason, an ILP64 backend for the 12 entry points is required before loading. Ordinary Julia provides OpenBLAS.
LP64 forwarding converts integer arguments and integer outputs and allocates integer scratch internally.
Re-registration does not save itself as the fallback.

Query does not modify the input or the value/vector outputs, and returns the minimal WORK accepted by this implementation.
The multiple workspace queries of GVD and the minimal and insufficient sizes of each caller workspace are verified.
This does not mean that all processing completes within the caller workspace alone.
The opposite triangles of A/B and padding are not referenced on input. The opposite triangle of B is preserved on output as well.
With JOBZ=V, all of A becomes the eigenvector output in GV/GVD, and the Z of GVX updates only the first M columns.

`test/generalized_selfadjoint.jl` verifies the four types, both ABIs, ITYPE, UPLO, and JOBZ; normalization and each residual; reference LAPACK; range endpoints and repeated values; empty/one-element/160-dimensional D&C; query and minimal work; positive definiteness INFO; invalid arguments and integer widths; unreferenced regions; precision specification and fallback on native failure; the ordinary Julia `eigen(Hermitian(A), Hermitian(B))`; and registration of the 12 symbols.
In an isolated process it detects a missing backend before LBT registration and confirms native execution of the normal path.

## Complete pivoting LU and scaled solve (#272)

The four types of GETC2/GESC2, 8 Fortran symbols in total, were connected to both integer ABIs.
GETC2 runs native full-pivot LU with SMIN specified as abs_tol.
Its final permutation is converted to LAPACK's sequential IPIV/JPIV; on the remaining Schur complement, complete pivot selection and small-pivot replacement are performed, and elimination continues with native triangular solve and matmul.
It is not a scheme that runs factorization to the end and then replaces only the diagonal.

SMIN is max(eps × the first largest absolute value, floatmin/eps); for a single element it is floatmin/eps.
Small pivots are replaced with the positive real SMIN, and INFO returns the last position that was perturbed.
Because the native prefix and reference LAPACK differ in how they choose among equal candidates and in the pivot score for complex numbers, the IPIV/JPIV values themselves are not guaranteed to match. The factor format and the reconstruction/solve contracts are preserved.
The Bignum constants and the final SMIN use the corresponding real precision.

GESC2 performs row permutation, unit lower solve, upper solve, and reverse-order column permutation, and returns a scaled solution satisfying A x = SCALE b for the original matrix.
It scales down the entire right-hand side before updates in forward substitution and before divisions and products in backward substitution to prevent overflow.
In backward substitution it normalizes the rows of U by the diagonal before using the native product, avoiding unnecessary SCALE reduction caused only by products with unnormalized columns becoming huge.
SCALE is updated every time in real precision, and if a product underflows to zero the right-hand side is also set to zero.
SCALE is not necessarily numerically identical to reference LAPACK; 0≤SCALE≤1 and the residual are verified.

A is borrowed and factorized, and GESC2 does not modify the input factor.
Internally it allocates native scratch, permutation arrays proportional to the matrix size, and normalized U rows, and does not make a backup of the entire input matrix.
There is no caller workspace or query.
N=0 returns without touching the arrays, and SCALE of GESC2 is 1.
The adapter checks for invalid N/LDA; GETC2 reports a negative INFO, and GESC2, which has no INFO argument, reports the argument number through `colmat_blas_last_error`.
GESC2 does not reference the last IPIV/JPIV element.

`test/complete_pivot_lu.jl` verifies the four types and both ABIs, reconstruction of row/column permutations, non-singular/singular/rank-deficient/subnormal/SMIN boundaries, bidirectional interoperation of external factors and solve, minimal/maximal scales, huge right-hand sides, a non-zero solution at dimension 112 and unavoidable SCALE underflow at dimension 300, residual/solution comparison with BigFloat, padding, logical submatrices, integer guards, unreferenced pivots, invalid arguments, and ordinary arrays through LBT.

## Selected SVD and QR-preprocessed SVD (#273)

The four types of GESVDX/GESVDQ, 8 Fortran symbols in total, were connected to both integer ABIs.
GESVDX generates singular values in descending order and U/VT from the native SVD and handles RANGE=A/I/V. The value interval is `(VL,VU]`, and the computed singular values are compared as they are.
The mathematical singular values at the interval endpoints may not coincide with the rounded singular values.
It writes only the NS outputs and handles the complex conjugate of VT and the LDVT according to the number of selected columns.

GESVDQ combines column-pivoted QR, rank determination, native SVD on the rank rows of R, application of Q, and inverse permutation.
It handles JOBA=A/M/H, JOBP=N/P, JOBR=N/T, JOBU=A/S/U/R/F/N, and JOBV=A/V/R/N, and the combination of JOBU=R with JOBV=A/V is an invalid argument as in the reference implementation.
With JOBU=F it preserves the Householder representation of A and the first N tau in WORK, and returns the left vectors before Q is applied.
The first N entries of IWORK are the column permutation, and with row pivoting the following M-1 entries are the sequential row swaps.
Condition estimation for JOBA=E is delegated to the saved ILP64 driver.

Because GEJSV/GESVJ require an independent Jacobi algorithm, this adapter does not export them and leaves them on the existing backend. They are not substituted with the ordinary SVD.
GESVDX/GESVDQ are delegated to the saved ILP64 driver from the pre-modification input even when native computation fails. LP64 entry points convert integer arrays.
The existing ILP64 8 drivers are required before loading, preventing self-reference on re-registration.

A workspace query and minimal legal workspace are provided, but computation within the caller workspace alone is not guaranteed.
Temporary areas for native SVD and QR are allocated internally, and GESVDQ saves the QR input in a temporary area in case of rerunning after failure.
The extra leading-dimension region of input A and unused vector outputs are preserved.

`test/advanced_svd.jl` verifies rectangular residuals for the four types and both ABIs, repeated/zero singular values and interval endpoints, query/minimal workspace, integer widths, rank determination, application of the factored Q, and agreement of JOBA=E with the reference backend.

## Range selection in SYEVX/HEEVX (#274)

The 4 symbols of real SYEVX and complex HEEVX were connected to LP64/ILP64.
For non-positive ABSTOL, RANGE=A/I/V is selected from the full spectrum of the native self-adjoint EVD, returning ascending W and the corresponding Z. The value interval is `(VL,VU]`.
This does not mean that independent algorithms for Sturm count, MRRR, or inverse iteration are provided.
A positive ABSTOL and native computation failure are delegated to the ILP64 reference driver saved before loading, converting LP64 integer workspace and outputs.

Only the specified part of UPLO is read into a temporary matrix, and the imaginary part of a complex diagonal is not read.
On the native path, input A is preserved, and the caller's leading-dimension slack and unused outputs are preserved.
When delegating to the reference driver, the specified triangular part of A is overwritten within the extent LAPACK permits.
On success, the first M IFAIL entries for JOBZ=V are 0.
Z(1,1) for N=1 is written as 1 as in the reference implementation even when it falls outside the range.
Query does not modify M, the matrix, or the eigenvalues/eigenvectors.

The minimal LWORK is 1 for N<=1, and otherwise 8N for real and 2N for complex. The query returns this legal minimum as the recommended value.
Because the native full eigenvalue computation uses a separate internal allocation, the caller workspace does not restrict all temporary memory.

`test/selected_selfadjoint.jl` verifies the four types and both ABIs, UPLO, A/I/V, repeated/zero/interval endpoints, workspace, INFO, IFAIL, invalid arguments, empty arrays, integer widths, minimal/maximal scales, ignored regions of a complex diagonal, LBT registration, and numerical calls.

## Dense BLAS extensions and rank-2 update (#276)

The four types of GEMMT/GEMMTR and SYR2, the two types of complex HER2 and GEMM3M, 16 Fortran symbols in total, were connected to both integer ABIs.
GEMMT/GEMMTR use the native triangular-output matmul and update only the specified triangular part.
This is not a method that allocates the full matrix product and then slices it out; the opposite triangular part and leading-dimension slack are preserved.
N/T/C are handled independently, and with beta=0 the old values of the output are not read.
With alpha=0 or K=0, A/B are not read and only the beta multiple of the specified triangular part is performed.

SYR2/HER2 apply the native rank-1 triangular product twice and borrow positive and negative vector strides directly.
SYR2 is a transposed product even for complex numbers, and only HER2 uses conjugation.
HER2 with non-zero alpha reads only the real part of the old diagonal and sets the imaginary part of the output to 0.
With alpha=0 it neither reads nor writes the input vectors or A, and the diagonal is preserved.
Invalid arguments record the argument position in `colmat_blas_last_error()`, as in the existing BLAS.

GEMM3M is connected to the ordinary native GEMM and provides the same arithmetic contract for the sum of products.
This does not mean that it provides the intrinsic algorithm built from three real multiplications, or operation-order or bit-for-bit agreement with other implementations.
Connection of the common CBLAS entry points is a separate Issue, #271.

`test/dense_extensions.jl` verifies the four types and both ABIs, independent scalar sums of products, N/T/C, complex coefficients, empty/rectangular, positive and negative strides, alpha/beta=0, NaN in ignored regions, the real diagonal, argument diagnostics, integer widths, the LBT forwarding target, and actual calls.
Rust null pointer tests also confirm paths that do not reference unnecessary operands or C when alpha=0/beta=1.

## BLAS value reductions and IAMIN (#277)

The two types of real AMAX/AMIN/MAX/MIN/SUM and the four types of IAMIN, 14 Fortran symbols in total, were connected to both integer ABIs.
MAX/MIN/SUM borrow positive-stride vectors into native `MatRef::max/min/sum`.
Extreme absolute values and IAMIN build real comparison values in a temporary array and reduce with native min/max.
IAMIN finds the first position of the minimum and returns the Fortran 1-based logical index, not the physical offset.

Complex IAMIN compares `abs(real(x))+abs(imag(x))`, not the modulus.
Because the sum is rounded to the original real precision before comparison, the first one is chosen even when rounding makes values tie under ComplexF32.
Empty input gives 0 for both the value and the index.
The stride conventions for positive N are as follows.

| INCX | MAX/MIN/AMAX/AMIN | SUM | IAMIN |
| :--- | :--- | :--- | :--- |
| Positive | native reduction | native sum | first minimal index of abs1 |
| 0 | first value / absolute value | 0, X is not read | 0, X is not read |
| Negative | 0, X is not read | 0, X is not read | 0, X is not read |

Negative strides are fixed to the same non-referencing convention as the [OpenBLAS 0.3.30 generic AMIN kernel](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.30/kernel/arm/amin.c) and others.
The behavior of some optimized backends that read before the beginning with a negative stride is not reproduced.
Differential verification against external backends targets positive strides and the defined zero-stride behavior, and negative strides are verified with independent expected values and null pointer tests.
Operation order of SUM, signed zero, and bit-for-bit agreement of NaN payloads with other backends are not guaranteed.

`test/reduction_extensions.jl` verifies both real precisions and the relevant complex types, both ABIs, strides, ties, the difference between abs1 and modulus, Float32 rounding, empty input, non-finite scalars, integer-width checks that allocate no huge arrays, LBT registration, and actual calls.

## Givens rotation generation and array application (#278)

The four types of ROTG/LARTG/LARGV/LARTV/LAR2V/LASR and the two real types LARTGP/LARTGS, 28 Fortran symbols in total, were connected to both integer ABIs.
The sign and conjugation of the sine in native JacobiRotation are converted to LAPACK's convention.
real ROTG/LARGV adopt the sign of the larger input, LARTG the sign of a non-zero F, and LARTGP a non-negative R.
real ROTG returns R in A and the auxiliary value Z in B, and complex ROTG preserves B.

Because the real generation routines normalize their input before calling native ROTG, they preserve finite unit-length coefficients even when R itself exceeds the range of the type.
LARTGS constructs a shift column and generates a rotation with the order of LARTGP changed.
If an intermediate product would overflow, the ratio is obtained with products and quotients using separated exponents so that a representable small sine is not lost.
LAPACK's branches for small X and zero shifts are also preserved.

LARGV generates array coefficients and R from positive-stride X/Y.
LARTV converts individual coefficients into the native two-component application, and LAR2V into two-sided application of a Hermitian 2x2 matrix.
complex LAR2V reads only the real parts of X/Y and outputs a real diagonal.
LASR uses native left/right application with row/column views and handles V/T/B pivots and F/B order.
Identity rotations do not reference A, and leading-dimension slack is preserved.

Positive increments are a usage condition for the array-coefficient routines.
Invalid increments for positive N and invalid arguments to LASR record the argument position in `colmat_blas_last_error()`.
N<=0 for the array-coefficient routines returns without referencing the coefficients, arrays, or increments.
No workspace is required, and borrowing of inputs is limited to the duration of the call.
This does not mean that the modified Givens routines ROTM/ROTMG are provided.

`test/givens.jl` verifies both ABIs and the four types, independent BigFloat generation formulas, minimal/maximal values, zero, complex phase, output quantization of R overflow and subnormals, array coefficients and strides, two-sided Hermitian application, left/right, pivot, and order in LASR, guards, argument diagnostics, integer widths, and numerical LBT calls of all 28 symbols.
In normal ranges it also checks against the existing LAPACK.
Because the existing OpenBLAS complex ROTG has extreme inputs that do not satisfy the unit-length condition, correctness in that range is judged from BigFloat and residuals.
Rust tests also verify null pointers for empty/identity operations and preservation of a small sine.

## Matrix norms for dense and structured storage (#279)

The 54 Fortran symbols of LANGE/LANGB/LANGT/LANHS, dense/band/packed/RFP/tridiagonal for symmetric and Hermitian, and triangular/trapezoidal/triangular band and packed, were connected to both integer ABIs.
M, 1/O, I, F/E and the lowercase synonymous specifications are handled.
The 1-norm is the maximum column sum and I the maximum row sum; it is not the elementwise l1 norm of the whole matrix.
The modulus is used for each complex element.
The native complex norm_max (largest component) and norm_l1 (abs1) are not identified as-is with the LAPACK definitions; conversion to real absolute values, native sums, and maximum are composed.

F/E preserve the complex components, and the value normalized by the largest component is reduced with native norm_l2 and then restored.
Because the pinned native's fixed three-stage sum of squares makes the representable norm 0/∞ for the smallest subnormal or near MAX, this normalization is performed.
The [standalone reproducer crate](https://github.com/AtelierArith/Colmat.jl/tree/main/reproducers/native-norm-l2) retains the dependency git revision and Cargo.lock and is a diagnostic that shows the 8 native failures with exit 101.
This does not mean that the ordinary package tests fail.
An addition to the related [native report #36](https://github.com/tensor4all/t4a-faer/issues/36) was rejected because the repository is archived/read-only, and the unpublished draft is left in the reproducer crate.

Only the stored elements that are referenced are traversed; the unit diagonal is synthesized as 1 and its stored value is not read.
The Hermitian diagonal reads only the real part, and the complex diagonal of a symmetric format handles both components.
For complex `Inf+NaN*i` in M/1/I, modulus ∞, and for F/E component-wise NaN, the LAPACK convention is followed.
NaN in another element propagates to each norm.
RFP converts the even/odd, UPLO, and real N/T versus complex N/C correspondences.

Input arrays and WORK are preserved.
The internal temporary area for non-empty input is proportional to the number of stored elements, and band/tridiagonal matrices are not expanded into dense matrices.
Group management for 1/I also uses an area proportional to the number of rows/columns.
There is no guarantee that the caller WORK limits all temporary memory; these entry points have no LWORK query.
An empty m or n returns 0 without reading the arrays.
Invalid arguments return 0 and record the argument position in `colmat_blas_last_error()`.
This does not introduce a LAPACK INFO argument.

`test/matrix_norms.jl` verifies the four types and both ABIs, independent scalar/BigFloat norms, all storage formats, NaN in ignored regions, workspace guards, unit/Hermitian diagonals, minimal/maximal, NaN/Inf, argument diagnostics and integer widths, and numerical LBT calls of all 54 entry points.
RFP is checked against data independently packed by the existing TRTTF.
Rust also verifies null pointers for empty/implicit diagonals, non-overlapping placement of all RFP elements, the complex modulus, and the range of Frobenius values.


## Copy, permutation, scaling, precision conversion, and equilibration (#280)

112 Fortran canonical symbols were added to LP64/ILP64. They handle the corresponding four types; CLACGV/ZLACGV, real-to-complex copies and mixed products, and the 6 kinds of precision conversion are provided only for the defined types.
CBLAS/LAPACKE entry points are a separate task. The supported count is 834 for each ABI (BLAS 158, LAPACK 676).

The 24 entry points of TRTTP/TPTTR/TRTTF/TFTTR/TPTTF/TFTTP use native copies of triangular columns and convert packed/RFP positions, strides, and folded conjugates.
Even/odd RFP, UPLO, real N/T, and complex N/C are handled, and the unreferenced triangular side and leading-dimension padding are preserved.

LACPY/LASET copy/fill only the selected region. For the upper-triangular copy of LACPY, only the min(M,N) rows of the output are required, as with Julia's rectangular copy.
LASWP applies the pivots of `K1+(row-K1)*abs(INCX)` in the specified order and reverses the order for a negative INCX.
LAPMR/LAPMT perform forward/backward 1-based permutations with native row and column swaps and preserve K. Fortran LOGICAL has the width of each integer ABI. LACGV handles negative and zero strides.
LASCL applies a staged safe-range multiplier to the 7 storage formats.
For complex it uses the complex multiplication formula of the reference Fortran, and NaN propagation and signed zero are also verified.
The results for non-finite values differ from a scheme that multiplies real and imaginary parts independently.

SLAG2D/CLAG2Z are widening conversions, and DLAG2S/ZLAG2C/DLAT2S/ZLAT2C are narrowing conversions.
Narrowing checks whether each component exceeds the largest finite value of single precision and returns INFO=1 at the first out-of-range value.
NaN is converted without being treated as out of range by the comparison.
LAT2 references only the specified triangle.
On success it writes INFO=0.
Because some versions of the reference LAPACK LAT2 preserve INFO on success, the differential test side initializes the reference INFO to 0.

GEADD does not read the input when alpha=0 or the old output when beta=0.
OMATCOPY/IMATCOPY handle row/column major, N/T/R/C, and input and output leading dimensions that differ for rectangular matrices.
IMATCOPY allocates a native owned snapshot.
alpha=0 for the real matcopy does not read the input, and the complex matcopy is evaluated with the complex multiplication formula of the reference kernel. LACP2 sets the imaginary part to positive zero.
LACRM/LARCM apply two real GEMMs to strided real-part and imaginary-part views, avoiding propagation of Inf/NaN in one component to the other through products with zero components of the complex coefficient.

GEEQU/GBEQU and the radix versions compute complex abs1 at each precision and restrict the row and column factors to a safe range.
A zero row gives INFO=i, and a zero column after row scaling is applied gives INFO=M+j.
The AMAX of GEEQUB/GBEQUB is based on the row maximum after quantization.
POEQU/POEQUB/PBEQU/PPEQU read only the real diagonal and report a non-positive diagonal with INFO=i.
POEQUB uses an integer power of the radix, and SCOND is the ratio of the square roots of the original diagonal.
The band scaling computation maps only the valid column interval and does not expand into a full dense matrix.

SYEQUB/HEEQUB compose the BIN iteration for matrices with finite components using native GEMV and vector maps.
Both complex SY/HE use the same abs1 as the reference implementation, and the diagonal of HEEQUB uses this definition too.
The internal iteration is computed in f64, and overall normalization by logarithms plus symmetric row normalization when needed handles cases where the smallest subnormal and the largest finite value coexist. f32 output is converted at the end.
Up to 100 iterations per normalization, quantization to the radix, and SCOND are implemented. The order of iterations and sums and the safe normalization tolerate a case where the coefficient differs from the existing LAPACK by one radix step, and it is also verified independently with condition numbers and row sums after scaling.
A zero row terminates with INFO=i. At this boundary it differs from the path where the reference source proceeds to a non-finite value by division.
Numerical termination of the iteration uses INFO=-1 as in the reference algorithm.

All entry points borrow only for the duration of the call.
BIN allocates an O(n²) native matrix and internal workspace, and preserves the caller WORK.
There is no guarantee that the caller WORK limits the internal allocation.
Aliasing of input and output is not generally allowed; IMATCOPY and permutation, which have in-place contracts, are handled by dedicated paths.
Invalid arguments use the existing INFO/TLS diagnostics and never unwind across the C ABI.

The 10 added focused tests verify direct Int32/Int64 calls of each entry point, guards, empty/rectangular, independent scalar/BigFloat references, round trips of the storage formats, NaN/signed zero, extreme finite values, and the existing LAPACK.
`test/matrix_transforms_lbt.jl` numerically executes all 112 canonical symbols through the LBT entry points of both ABIs.


## Band/packed/RFP BLAS (#281)

The colmat-rs used for verification is `4e6ed0e34659147a5c744ab6431f150e78db6f82`, updated to follow the dependency update on main.

48 Fortran canonical symbols were added to both integer ABIs.
GBMV, SBMV/SPMV, SPR/SPR2, and TBMV/TBSV/TPMV/TPSV handle the four types, and HBMV/HPMV/HPR/HPR2 the defined two complex types.
The four entry points SSFRK/DSFRK/CHFRK/ZHFRK are also included.
The supported count is 882 for each ABI (BLAS 202, LAPACK 680).
Additions for CBLAS/LAPACKE are handled in their respective Issues.

GBMV passes the valid column interval of the band to native outer and inner products and converts transpose, conjugate, and negative strides.
Symmetric and Hermitian band/packed products also compute contributions to both sides from the stored columns only.
The imaginary part of the Hermitian diagonal is not read.
Complex SBMV/SPMV and SPR/SPR2 are complex symmetric and are not replaced with the Hermitian conjugation rule.
The alpha of HPR is real and that of HPR2 complex.
Packed rank updates update only the stored triangle, and when HPR/HPR2 are executed the imaginary part of the diagonal is reset to 0.
With alpha=0 the arrays are not touched and the diagonal is not normalized either.

Triangular band/packed products allocate an O(n) native X snapshot, and solve uses the solved components in order.
no-transpose skips zero columns as in reference Netlib and does not read the matrix elements of those columns.
For transpose and conjugate, the stored columns are used in native inner products, and rows needed for no-transpose are gathered into an internal native area.
The unit diagonal does not read stored values. Width-0 unit band and single-element unit packed operations also preserve component bits such as NaN and signed zero.
There is no pre-check of a non-unit singular diagonal returning a proprietary INFO; it is handled as BLAS floating-point arithmetic.

These band/packed operations do not expand into full dense matrices.
The operation count is O(m+n*(kl+ku+1)) for general band where m is the output vector length, O(n*(k+1)) for triangular and symmetric band, and O(n²) for packed.
The required snapshot/gather is proportional to the vector or valid-row length, and the native kernels themselves allocate workspace according to stride and execution plan.
This is not a guarantee that there is no internal allocation.

SFRK/HFRK expand the RFP storage format into an O(n²) native triangular matrix, perform selected-triangle rank-k multiplication, and return it to RFP.
They handle even/odd RFP, UPLO, real N/T and complex N/C, and transposition of A.
beta=0 does not read the old C, and alpha=0 or k=0 does not read A.
A zero product with beta=1 preserves C, and a zero product with beta=0 sets all physical RFP elements to 0.
For other HFRK cases only the real part of the diagonal is read, and the imaginary part of the physical output diagonal is set to positive zero.
These have no caller workspace argument and are not an API in which the caller specifies the internal memory allocation.

The tests verify all 48 entry points with direct Int32/Int64 C signatures and numerical LBT calls.
They check independent packing of the storage formats, scalar references, solve reconstruction, empty/width 0/rectangular/upper and lower/transpose/conjugate, negative strides, padding/guards, NaN in unit/Hermitian diagonals, and extreme integer widths.
Normal finite inputs are also checked against the existing OpenBLAS.
For huge empty inputs and singleton increments, Rust tests that allocate no large arrays are also provided.

OpenBLAS 0.3.30 complex SPR/SPR2 have a difference in the starting position for negative strides, and a [standalone reproducer](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-packed-negative-stride.jl) is retained.
Colmat's negative-stride calls are verified as they are, and on the differential reference side the same logical vector is arranged with a positive stride for comparison.
For no-transpose triangular operations too, NaN in a column multiplied by zero may propagate in some optimized providers.
Colmat preserves the zero-column skipping of reference Netlib, and the differential reference for that test uses a matrix with finite columns that do not change the computational result.
On Linux OpenBLAS, a difference was also confirmed where NaN propagates to an off-diagonal block from an unused A with alpha=0, beta=0.5, and NaN in SFRK.
Colmat uses the NaN in the unused A as-is and checks against an independent scalar reference, replacing only the provider comparison with a finite A that gives the same zero product.
The [standalone reproducer that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-sfrk-zero-product.jl) retains a diagnostic that outputs the results for an A with NaN and for a finite A.

## Band and general tridiagonal LU and solve (#282)

Using the native LU, multiplication, and triangular solve of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 32 Fortran canonical symbols of GBTRF/GBTF2/GBTRS/GBSV and GTTRF/GTTRS/GTTS2/GTSV were connected to both integer ABIs.
The supported count is 914 for each ABI (BLAS 202, LAPACK 712). The four types are handled; CBLAS/LAPACKE are a separate task.

Band factorization passes the stored valid columns to native partial-pivoting LU.
The pivot of that column is applied to the remaining valid columns, and the region within the skew-strided band is updated with native outer products.
Because subsequent row swaps are not applied to the old columns of L, LAPACK's stepwise IPIV and fill-in are preserved.
GBTF2 and GBTRF are the same unblocked composition and do not claim to provide a dedicated blocked band LU algorithm.
Rectangular cases, bandwidth 0, widths larger than the dimension, singular INFO, and unused padding are handled.

Solve applies the pivots and elimination of L in order and solves U with the native product over the valid interval and a diagonal solve.
For transpose and conjugate transpose, the order of U, L, and row swaps is reversed, and the conjugates of both factors are handled.
The path that solves OpenBLAS factors with Colmat and the reverse are verified.

The tridiagonal is converted into a four-row band workspace, and the multipliers in DL, D, DU, DU2, and the adjacent pivots are stored.
GTTRF records singular positions in INFO and continues factorization.
GTSV stops at the first singular pivot and returns B eliminated up to that point.
The DL of GTSV holds the second superdiagonal of U rather than multipliers, and the last DL element is preserved.
Even with NRHS=0, factorization and detection of singular INFO are performed, and B is not read or written.
GTTS2 has no INFO argument and treats ITRANS=0/1/2 as N/T/C.
Invalid arguments are recorded in the same thread-local diagnostics as other auxiliary entry points.
These entry points have no caller workspace/query argument.

There is no expansion into a full dense matrix.
Column processing in band factorization allocates native LU workspace proportional to the bandwidth and an O(min(m,n)) pivot area.
Solve uses an O(n) pivot area and gather areas for the needed valid rows.
Factorization, solve, and drivers for the tridiagonal additionally allocate an O(n) four-row band area.
The native kernel execution plan also has internal allocations, so absence of allocation is not guaranteed.

The tests cover the four types and both ABIs, independent reconstruction from the factors, known solutions and residuals, boundary pivots, intermediate output of singular drivers, interoperation with external factors, numerical LBT calls of all 32 entry points, INFO/integer widths, argument verification order, empty input, and guards for stored regions.

OpenBLAS 0.3.30 GTSV may perform backward substitution into the first column of B even when NRHS=0.
A [standalone reproducer that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-gtsv-zero-rhs.jl) is retained.
The differential reference always allocates a dummy column to avoid out-of-range access.
In this case the reference comparison targets the factors and INFO, and Colmat's B is independently verified to be unchanged in all elements.

## Band Cholesky and split Cholesky (#283)

Using colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 20 Fortran canonical symbols of PBTRF/PBTF2/PBTRS/PBSV/PBSTF were connected to both integer ABIs.
The supported count is 934 for each ABI (BLAS 202, LAPACK 732). The four types are handled; the C ABI is a separate task.

PBTRF and PBTF2 are the same unblocked composition. They use only the valid band interval and apply, in order, the native LLT of the diagonal, scaling by native triangular solve, and a selected-triangle native rank update.
This does not claim that a dedicated blocked band kernel exists.
They do not expand into a full dense matrix and handle upper/lower storage, width 0, and declared widths larger than the dimension.
The imaginary part of the Hermitian diagonal is not read on input, and the imaginary part of the diagonal after the rank update is reset to 0 as well.
Even when stopped by the non-positive definiteness INFO of a later pivot, the contract for this updated part is maintained.
As with the reference unblocked processing of PBTF2/PBSTF, a NaN real diagonal is propagated rather than judged non-positive definite.

PBTRS solves L, L**H in the lower case and U**H, U in the upper case, in that order.
The path that solves OpenBLAS factors with Colmat and the reverse are verified.
PBSV factorizes even with NRHS=0 and returns non-positive definiteness INFO.
B is unchanged on factorization failure and with NRHS=0.
These entry points have no caller workspace/query argument.

PBSTF is not a simple replacement by ordinary Cholesky.
It eliminates the second half in reverse order, reflects the contribution into the first half, and factorizes the updated first half in the forward direction.
It stores S consisting of the upper-triangular first half, the lower-triangular second half, and the part spanning the boundary, and forms A=S**H*S.
UPLO specifies the storage locations of input and output, and lower storage does not change the meaning of this split factor.

When the split point m=(n+kd)/2 exceeds n, the reference PBSTF also processes columns beyond the declared N.
The [standalone reproducer that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-pbstf-redundant-width.jl) uses safely allocated dummy columns and shows the behavior of returning INFO=2 for N=1.
Colmat sets m=min(n,(n+kd)/2) and uses only columns that actually exist.
In this case no external comparison is performed; S**H*S is independently reconstructed over the full width and the unused columns are verified to be unchanged.

Factorization uses O(n*min(kd,n)²) operations and an O(min(kd,n)) vector snapshot.
Even when the unused half of a skew-strided update destination overlaps the original vector, the snapshot avoids aliasing between the input and output views of the native multiplication.
Solve uses gather areas for the valid rows.
The native kernel execution plan also has internal allocations, so absence of allocation is not guaranteed.

The tests check the four types, both ABIs, upper/lower, width 0, and empty input; independent reconstruction of ordinary and split forms; known solutions and interoperation with external factors; non-positive definiteness INFO and intermediate output; NaN diagonals; padding/unused columns; numerical LBT calls of all 20 entry points; argument verification order; and INFO/integer widths.

## Packed/RFP Cholesky, solve, and inverse (#284)

Using the native LLT and triangular solve of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 28 Fortran canonical symbols of PPTRF/PPTRS/PPTRI/PPSV and PFTRF/PFTRS/PFTRI were connected to both integer ABIs.
The supported count is 962 for each ABI (BLAS 202, LAPACK 760). The four types are handled; the C ABI is a separate task.

The stored triangle of packed/RFP is moved into an owned native working matrix, factorized with native LLT, and only the stored triangle is returned.
Native LLT, which uses the opposite triangle as scratch, does not rewrite the caller's storage.
UPLO, even/odd RFP, real N/T and complex N/C, and the conjugate of a folded block are converted.
The imaginary part of the Hermitian diagonal is not read on input.
Non-positive definiteness INFO and the factor output of the valid leading part are also verified.

Solve passes the converted factor to native Cholesky solve and borrows and overwrites an ordinary B array.
Inverse solves the identity with native solve and returns the specified triangle to packed/RFP.
This does not claim that a dedicated packed/RFP inverse kernel exists.
An inverse with a zero diagonal returns that position in INFO and does not rewrite the factor.
The imaginary part of the output diagonal is set to 0.
Interoperation of solve and inverse with external factors is confirmed in both directions.

PPSV performs factorization and detection of non-positive definiteness INFO even with NRHS=0, and unused B is unchanged.
Solve does not read the factor when N=0 or NRHS=0.
Verification of the leading dimension and arguments is performed first.
These entry points have no caller workspace/query argument.

Internal areas are O(n²) for factorization and solve, and O(n²) for the factor and identity in inverse.
The native kernel plan also has internal allocations.
There is no guarantee of processing with only memory proportional to the storage format, nor of zero allocation.
If an internal dense area exceeds an addressable size, it is diagnosed as an invalid argument for N before touching the arrays.
A solve with zero RHS does not perform this allocation either.

The tests use independent OpenBLAS TR/TP/TF conversions to check the four types, both ABIs, and all RFP layouts.
They verify reconstruction of the original problem by scalar products, known solutions, and inverse residuals; interoperation with external factors; non-positive definiteness/zero diagonal INFO; empty/zero RHS; padding/guards; numerical LBT calls of all 28 entry points; argument verification order; and INFO/integer widths.

## Positive definite tridiagonal LDL and solve (#285)

Using the native LDL, multiplication, and triangular solve of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 16 Fortran canonical symbols of PTTRF/PTTRS/PTTS2/PTSV were connected to both integer ABIs.
The supported count is 978 for each ABI (BLAS 202, LAPACK 776). The four types are handled, and D is a real array even at the complex entry points.
The C ABI is a separate task.

Factorization stores a real D and a real/complex E.
Each elimination step normalizes the column by a power of two and takes the multiplier from the finalized first column of the native LDL panel.
The tentative remaining diagonal is not treated as the pivot of the original problem.
The update of D is computed with native real multiplications using the real and imaginary parts of complex E, and is always returned to a real array.
INFO is returned at the first non-positive definite pivot, and E and B from then on are not modified.
At non-finite boundaries, component-wise IEEE division is handled in the conversion stage, and NaN propagation is also checked against the reference.

Solve builds an O(n) unit lower band and composes a native unit triangular solve, real diagonal solve, and conjugate-transpose solve.
For complex UPLO=U it starts from the conjugate of E and preserves the meaning of U**H*D*U.
UPLO=L is L*D*L**H.
The complex IUPLO of PTTS2 uses 1 for upper and 0 for lower, and the real entry points have no UPLO argument.
The diagonal solve also normalizes the components of D and B by the same power of two, avoiding overflow of the reciprocal for a representable minimal pivot.
The real and imaginary parts of complex B are processed by real kernels.
Interoperation of solve with external factors is confirmed in both directions.

PTSV factorizes even with NRHS=0 and detects non-positive definiteness INFO. It does not touch B.
Solve does not read D/E/B when N=0 or NRHS=0. Argument verification is performed first.
PTTS2 has no INFO argument and records invalid arguments in the same thread-local diagnostics as other auxiliary entry points.
These entry points have no caller workspace/query argument.

No full dense expansion is performed.
The factorization workspace is a fixed-size panel, and solve allocates an O(n) band area.
The operation count is O(n) for factorization and O(n*NRHS) for solve.
The native kernels also allocate internally, so absence of allocation is not guaranteed.
If an internal area or the stride of complex components exceeds an addressable size, the arguments are diagnosed before touching the arrays.
With 1 RHS, no computation is performed with a huge column stride that is not used.

OpenBLAS 0.3.30 complex PTTS2 may rewrite the first column even when NRHS=0.
In addition, a difference was confirmed where PTTS2 at N=1, through the 1/D scaling, turns a solution representable with a minimal D into Inf.
The [standalone reproducer that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-ptts2-boundaries.jl) retains safely allocated dummy columns and a one-element problem.
Dummy columns are passed only to the differential reference, and Colmat's unused B is verified to be unchanged in all elements.
Minimal/huge values are checked against independent expected values and BigFloat.

The tests check the four types and both ABIs, reconstruction of real D and complex E, upper/lower and multiple RHS, reuse of external factors, non-positive definiteness INFO and intermediate D/E/B, n=0/1, subnormals and huge multipliers, non-finite values, guards, numerical LBT calls of all 16 entry points, argument verification order, and INFO/integer widths and huge unused strides.

## band/packed self-adjoint EVD (#286)

Using the full-spectrum self-adjoint EVD of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 24 Fortran canonical symbols BEV/BEVD/BEVX and PEV/PEVD/PEVX for real SB/SP and complex HB/HP were connected to both integer ABIs.
The supported count is 1002 for each ABI (BLAS 202, LAPACK 800). The C ABI is a separate task.

Only the specified triangle, bandwidth, and packed interval are read, and they are expanded into an owned full dense Hermitian matrix.
The imaginary part of a complex diagonal and unused band padding are not read.
BEV/BEVD and PEV/PEVD use the same native full EVD.
This does not claim that a dedicated band/packed eigensolver or a per-driver QR/DC algorithm is implemented.
Even for values-only and partial ranges, an O(n²) internal matrix and workspace are allocated and the full spectrum is computed.
Even if the minimal caller WORK length is specified, this internal allocation is required separately.

For non-positive ABSTOL, VX selects A/I/V from all eigenvalues. The value range is (VL,VU], and indices are 1-based.
band VX with n>1 returns all eigenvectors in Q and all eigenvalues on the diagonal of AB, and sets the stored off-diagonal to 0.
This diagonal matrix is also a tridiagonal matrix T and satisfies the original A=Q*T*Qᴴ.
Only the selected columns are returned in Z, so Q is not identical to the partial Z.
The packed output also becomes a diagonalized storage format after native success.
JOBZ=N does not reference Q/Z/IFAIL and preserves W/Z/IFAIL after the selection result and padding/guards.
packed VX with N=1 returns Z(1,1)=1 even when the range is empty, and band VX writes only when selected.
The singleton quick return of both does not modify Q/IFAIL.

The absolute-accuracy contract of a positive ABSTOL is not substituted by the native ordinary EVD.
Native failure, or a case where a range endpoint is within `64*n*eps(R)*max(abs.(λ))` of a native eigenvalue, is passed to the saved ILP64 expert driver before modifying the original input.
This guard is a conservative decision to avoid mistaking endpoints because of rounding, and does not mean a rigorous proof of error or an implementation of a native Sturm count. LP64 converts integer scalars and IWORK/IFAIL to ILP64.
An expert backend is required before loading, and it does not save itself on reload.
Native failure of ordinary BEV/BEVD and PEV/PEVD is reported with positive INFO=1, and non-converged internal indices are not exposed.

BEVD/PEVD report the Reference-LAPACK minimal lengths of WORK/RWORK/IWORK, and if any length is -1 it is treated as a query of all areas.
Query and invalid arguments do not read the mathematical input.
Single-precision recommended lengths are rounded in the direction that does not fall below the required integer length.
A minimal length that cannot be represented, a recommended value that does not fit in the LP64 IWORK(1), and an internal area whose address cannot be computed are detected, and integers are not silently truncated.

BEVD in Reference-LAPACK 3.12.1 and OpenBLAS 0.3.30, on the quick return for N=1, UPLO=U, KD>0, reads AB(1,1) instead of the specified AB(KD+1,1).
A [standalone reproducer for the four types that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-bevd-singleton-upper.jl) is retained, and this boundary is verified with an independent one-element eigenvalue and residual.
Colmat reads the specified diagonal.

## Application of block orthogonal and unitary matrices (#287)

With the native structured matrix products of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 4 Fortran symbols SORM22/DORM22/CUNM22/ZUNM22 were connected to both integer ABIs.
The supported count is 1006 for each ABI (BLAS 202, LAPACK 804). The C ABI is a separate task.

The row partition of Q is (N1,N2) and the column partition is (N2,N1).
Q11/Q22 are applied as rectangular and Q12 as lower and Q21 as upper with native rectangular/triangular matmul.
Triangles that are not specified are not read.
Left/right application, real N/T, complex N/C, and empty blocks are handled; with an empty block the remaining single triangular product is executed.
An O(m*n) result of the same shape is owned, and after computing all blocks only the valid region of C is written back.
Q is not expanded into a full dense matrix.
Even with minimal caller WORK, native internal allocation is required separately.

Query does not touch Q/C and returns m*n in WORK(1), rounding in single precision in the direction that does not fall short.
The minimum length for actual computation is NQ (M for left, N for right) matching the Reference-LAPACK executable code, and is 1 if either block is empty.
The quick return for M=0 or N=0 also performs argument verification first and returns WORK(1)=1.
When either block is empty, WORK(1)=1 after actual computation, and after normal actual computation m*n is returned.
If the address computation for an area or recommended length cannot be represented, the corresponding argument is diagnosed before borrowing the arrays.

The prose of Reference-LAPACK 3.12.1 states the M/N of the minimum length in the opposite way from the executable code.
In the [standalone reproducer for the four types that does not load Colmat](https://github.com/AtelierArith/Colmat.jl/blob/main/reproducers/openblas-orm22-workspace.jl), OpenBLAS 0.3.30 returns INFO=-12 for LWORK=2 and success for LWORK=5 in both left and right with NQ=5.
The tests use this execution contract and verify different block widths, NaN in unused triangles, unitary permutation and phase matrices plus general structured matrices, independent scalar products, guards on C/Q/WORK/INFO, and numerical LBT calls of all 4 entry points.

## Bidiagonalization and left/right Householder (#295)

Using the native bidiagonal panel and matrix products of colmat-rs `4e6ed0e34659147a5c744ab6431f150e78db6f82`, the 20 Fortran canonical symbols GEBD2/GEBRD/LABRD and real ORGBR/ORMBR plus complex UNGBR/UNMBR were connected to both integer ABIs.
The supported count is 1026 for each ABI (BLAS 202, LAPACK 824). The C ABI is a separate task.

Each active column/row is normalized by its largest component and passed to a one-column panel of native `bidiag_in_place`.
Because native uses Hermitian reflectors and a complex beta while LAPACK uses a real beta and complex tau, the tail of the reflector vector is multiplied by `(alpha-beta_native)/(alpha-beta_LAPACK)` to convert the storage format.
Computation uses the normalized components, preserving a LAPACK tail that is representable even under the native tiny-tail decision.
Real D/E are distinguished from real and complex TAUQ/TAUP, and the stored vectors on the right side have the necessary conjugates restored.
For non-finite input, the existing Householder adapter's IEEE arithmetic path is used in the scalar conversion stage.

GEBD2/GEBRD are a common unblocked composition that applies this panel and native GEMV/rank-one updates alternately.
M>=N returns an upper bidiagonal and M<N a lower bidiagonal.
This does not claim that GEBRD uses a dedicated blocked bidiagonalization.
The minimal and recommended WORK of GEBRD are 1 for an empty problem and max(M,N) otherwise, and query does not read A/D/E/TAU.
GEBD2 preserves the caller WORK and allocates an internal update area separately.
The reflector panel and the explicit update area are O(max(M,N)), and workspace is allocated inside native as well.

LABRD reduces the first NB steps in an owned O(M*N) working matrix and writes back only the first NB rows/columns of the caller.
Caller A22 is preserved without update, and the lower interval of X/Y needed for `A22 -= V2*Y2ᴴ + X2*U2ᴴ` is returned.
The leading scratch and padding of X/Y are preserved.
The nonexistent last reflector factor and subdiagonal output are also left unchanged.
It also distinguishes the case in Reference-LAPACK's complex singleton/final step where the head remains at the original value rather than the real beta.
LABRD has no INFO/query argument, and invalid arguments can be obtained from the same thread-local diagnostics as other auxiliary entry points.

Generation and application of Q/P distinguish the different reflector start positions for upper and lower bidiagonal, real N/T and complex N/C, and SIDE=L/R.
ORGBR/UNGBR return the full/thin slice of Q or Pᴴ in A.
Only the tails of stored non-zero reflectors are saved to an internal area and A is replaced by the output; with zero tau, unreferenced A is not read.
ORMBR/UNMBR can also use external LAPACK factors.
The reflector snapshot is O(NQ*min(NQ,K)) and is allocated separately from the minimal caller WORK length.
Query and empty problems do not read the mathematical input.
Minimum-length integers are rounded in single precision in the direction that does not fall short, and areas whose address cannot be computed are detected before borrowing the arrays.

Verification confirms the four types and both ABIs, rectangular/rank-deficient/zero matrices, A≈Q*B*Pᴴ and orthogonality, full/thin Q/P, left/right and conjugate, bidirectional interoperation with OpenBLAS factors/generation/application, partial reduction of LABRD and the X/Y update product, query and minimal WORK, INFO and auxiliary diagnostics, padding/guards, and numerical LBT calls of all 20 entry points.
For minimal/huge scales, normalized BigFloat reconstruction is also checked independently.
Huge values of LDA/LDC/LDX/LDY that are not used with one column are accepted, and checks are made over the area up to the last referenced element.
A zero-tau reflector does not read A, and if all reflectors are zero the application target C is not referenced either.

## Self-adjoint tridiagonalization and Q generation/application

Issue #296 connected the 20 Fortran symbols SYTRD/HETRD, SYTD2/HETD2, LATRD, ORGTR/UNGTR, and ORMTR/UNMTR to LP64/ILP64. Each ABI has 1046 symbols (BLAS 202, LAPACK 844).
The colmat-rs revision used is `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

Real reduction at normal scale uses native `tridiag_in_place`.
For the gauge conversion of complex numbers to a real E and for extreme real scales, a normalized native Householder panel and native GEMV/Hermitian rank-2 updates are composed.
Conversion is made to the reflector storage format of each of the upper and lower triangles, and the unreferenced triangular part and the imaginary part of the Hermitian diagonal are not read.
Q is generated and applied from the stored reflectors, and LATRD returns the W needed for the subsequent symmetric rank-2k update.

Working matrices are owned internally. The minimal legal LWORK is 1 for TRD, `max(1,n-1)` for ORGTR/UNGTR, `max(1,n)` for ORMTR/UNMTR with left application, and `max(1,m)` with right application.
A workspace query is implemented, but the same blocked algorithm and performance as the reference implementation are not claimed.

`test/selfadjoint_reduction.jl` verifies the four types and both integer ABIs, the selected triangle and array slack, reconstruction of the tridiagonal decomposition, left/right application and transpose/conjugate transpose of Q, interoperation with OpenBLAS factors, query and argument diagnostics, and the partial panel and deferred update of LATRD.


## Packed Bunch–Kaufman

Issue #297 connects the 16 Fortran symbols of real SPTRF/SPTRS/SPTRI/SPSV and complex HPTRF/HPTRS/HPTRI/HPSV to both integer ABIs. Each ABI has 1062 symbols (BLAS 202, LAPACK 860).
The colmat-rs revision is `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

Only the selected packed triangular part is converted to an internal dense array, sharing the existing dense Bunch–Kaufman and native LDL and 1×1/2×2 IPIV conversion.
solve/inverse also support external LAPACK packed factors. For complex numbers only the Hermitian series is targeted.
The imaginary part of the Hermitian diagonal is not referenced, and the output diagonal is real.
Storage-format conversion and native processing scratch are owned internally, and nothing is written to packed arrays or right-hand-side slack.
No workspace query is defined for these routines.

`test/packed_bunch_kaufman.jl` verifies the four types and LP64/ILP64, upper/lower packed, 2×2 pivots, singular INFO, empty matrices, 0/1/multiple right-hand sides, bidirectional interoperation with OpenBLAS factors, solve/inverse residuals, LBT drivers, argument diagnostics, and integer outputs and guard values around arrays.

## Band/packed/RFP triangular solve/inverse

Issue #298 connects the 20 Fortran symbols of TBTRS/TPTRS/TPTRI/TFTRI/TFSM for the four types and both integer ABIs. Each ABI has 1082 symbols (BLAS 202, LAPACK 880).
The colmat-rs revision is `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

The selected band/packed/RFP triangular part is converted to an internally owned dense array, and native triangular solve/inverse is used. The memory for conversion and workspace is O(n²).
Right application of TFSM uses transposed views of the matrix and right-hand side, with the complex conjugate specified separately.
Odd and even RFP sizes and real N/T and complex N/C storage are handled.
The unit diagonal is not read, and the original stored values are preserved in inverse as well.
TBTRS/TPTRS check the singularity of a non-unit diagonal even with zero right-hand-side columns and return a positive INFO.
TFSM, like BLAS, does not define an INFO for singularity checks, and with alpha=0 it does not read A or the original B.
No workspace query is defined for these routines.

`test/structured_triangular_lapack.jl` verifies the four types and both integer ABIs, storage formats, UPLO/DIAG/TRANS/TRANSR/SIDE, empty matrices, redundant band widths, and multiple right-hand sides, solve/inverse residuals, comparison with external LAPACK, invalid arguments, singular INFO, alpha=0, array slack and integer output guards, and calls through LBT of all 20 symbols.


## GEQR/GEMQR and the TSQR storage format

Issue #302 connects the 28 Fortran symbols of GEQR/GEMQR, LATSQR, ORGTSQR/UNGTSQR and the ROW versions, ORHR_COL/UNHR_COL, and GETSQRHRT for the four types and both integer ABIs. Each ABI has 1110 symbols (BLAS 202, LAPACK 908).
The colmat-rs revision is `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

LATSQR factorizes the first row panel with native QR, and in each subsequent row block factorizes `[R; B]` combining the upper-triangular R with the additional rows B. Earlier reflectors are not read as components of R, and the T of each row block is stored separately.
Generation and application of Q gather the non-contiguous rows/columns of this storage format and apply native block Householder.

GEQR implements a five-element header and normal/minimal queries of TSIZE/LWORK (`-1`/`-2`), and switches to the NB=1 format with the minimal work array.
Normally it uses up to 32 columns, and for tall-skinny it uses row blocks up to N+32 rows. GEMQR also handles the MB/NB metadata of an external GEQR.
ORHR_COL/UNHR_COL compose the modified LU of `Q-[D;0]` with native matrix updates and triangular solve, and GETSQRHRT combines TSQR, thin Q, Householder reconstruction, and `D*R`.

Working arrays are owned internally. The ROW-version Q generation also uses the common explicit-Q path, and identity with the reference implementation's in-place row algorithm, memory volume, or performance is not claimed.
`test/tsqr.jl` verifies the four types and both ABIs, multiple row/column blocks, minimal workspace and query, rank-deficient/zero matrices/rectangular shapes, bidirectional mutual use of external factors, left/right application and conjugate of Q, reconstruction/orthogonality, argument diagnostics and integer and array guards, and LBT calls of all 28 symbols.
It also verifies small/large uniform scales, the ROW version's minimal LWORK=1, unused huge leading dimensions, and a huge logical row count with 0 columns.

## QL/RQ factors and Q generation/application

Issue #303 connects the 48 Fortran symbols of GEQLF/GEQL2, GERQF/GERQ2, and each Q generation/application series for the four types and LP64/ILP64.
Each ABI has 1158 symbols (BLAS 202, LAPACK 956).
The colmat-rs revision is `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

Existing standard QR/LQ is called on a working array with both axes reversed, and the placement of the triangular part and the order of TAU are restored.
The Q of QL is `H(k)...H(1)`, the Q of complex RQ is `H(1)ᴴ...H(k)ᴴ`, and the row storage of RQ preserves the complex conjugate of the vectors.
The thin Q is constructed from the last columns in QL and from the last rows in RQ.

Before generating Q, the necessary reflector tails are copied internally, and implicit unit components and unused triangular parts are not read. In Q application, zero coefficients are skipped and only the prefix referenced by non-zero coefficients is checked.
Leading dimensions that become unused with a length-1 reflector are also not referenced.
The blocked/unblocked names use a common unblocked composition path, and identity with a dedicated native QL/RQ API or the reference implementation's blocked algorithm or performance is not claimed. Working arrays are owned internally.

`test/reverse_qr.jl` verifies the four types and both ABIs, rectangular/empty/zero/rank-deficient cases, QL/RQ reconstruction and orthogonality of Q, thin/full Q, bidirectional mutual use with OpenBLAS factors, left/right application and conjugate, workspace query/minimal work/argument diagnostics, guards on arrays and integer outputs, unused huge strides, and LBT calls of all 48 symbols.

## Triangular-pentagonal QR/LQ and Q application

The 28 Fortran symbols TPQRT/TPQRT2, TPLQT/TPLQT2, TPMQRT/TPMLQT, and TPRFB are connected for the four types and LP64/ILP64. Each ABI has 1186 symbols (BLAS 202, LAPACK 984).
They compose QR stacking the upper triangle and pentagonal tail, LQ using its adjoint, and generation/application of compact Householder.
This does not mean a dedicated native triangular-pentagonal API or the same blocking performance as the reference implementation.
Internal matrices and scratch are allocated separately, and the caller WORK is preserved.

TPRFB supports left/right, normal/adjoint, forward/backward, and column/row storage.
It does not read or write unspecified triangles, structural zeros, or leading-dimension padding.
This series has no LWORK/workspace query, and TPQRT2/TPLQT2 also have no WORK argument.
TPRFB has no INFO and reports invalid arguments through the backend's diagnostic code.

`test/pentagonal.jl` verifies mutual use of factors and T with reference LAPACK, reconstruction, orthogonality of Q, and the numerical results of TPRFB obtained independently from elementary Householder products.
Because row-forward-left TPRFB in reference OpenBLAS 0.3.30 has a defect that uses LDB for the WORK stride, the reference comparison sets LDWORK=LDB, and Native is verified with different strides as well.
An independent safe reproduction is in `reproducers/openblas-tprfb-workspace.jl`.

## Moving and selecting generalized Schur blocks

Issue #307 connects the 16 Fortran symbols of TGEXC/TGEX2/TGSEN/GGESX for the four types and LP64/ILP64. Each ABI has 1202 symbols (BLAS 202, LAPACK 1000).
The native revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

TGEXC/TGEX2 and TGSEN IJOB=0 use the public generalized Schur reorder.
Selecting either member of a real conjugate pair moves the whole 2×2 block, and even when a swap is rejected the factors and positions after the last accepted transformation are preserved.
The homogeneous α/β of TGSEN are extracted from the stored factors with the public `schur_eigenvalues_real` and are not re-decomposed.
Infinite eigenvalues with β=0 are preserved, and the normalization of the reference routines is distinguished from the early exit of the complex routines.
If eigenvalue extraction is rejected, the unmodified original factors are passed to the saved LAPACK.

GGESX SENSE=N uses the existing native dense QZ/selection path.
The condition and separation estimation of TGSEN IJOB=1:5 and GGESX SENSE=E/V/B is delegated to the ILP64 LAPACK from before registration.
This does not claim that native provides the same estimates.
LP64 converts SELECT and integer workspace and preserves unused output areas.
The selection callback of GGESX is also converted to 64-bit LOGICAL, so that callbacks of different calls do not mix through thread-local state and restoration from nesting.
Because internal matrices and scratch are owned, this does not imply allocation-free processing or the same operation count and performance as a dedicated native expert driver.

`test/qz_reorder.jl` verifies both ABIs, movement of real 1×1/2×2 blocks, Q/Z updates to external factors, array slack and unused factors, comparison of condition estimation with reference LAPACK, query and minimal WORK, and the LBT forwarding target and actual calls.

## Band/packed generalized self-adjoint EVD and standard-form reduction

Issue #308 connects the 32 Fortran symbols GST/GV/GVD/GVX of SBG/HBG/SPG/HPG for the four types and LP64/ILP64. Each ABI has 1234 symbols (BLAS 202, LAPACK 1032).
The native revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

packed GST takes an existing triangular Cholesky factor and performs the congruence transformation for ITYPE=1/2/3.
band GST restores the split factor `S = [U 0; M L]` of PBSTF and composes `C=XᴴAX` and `XᴴBX=I` from native block triangular solve and reflector transformations.
It stores within the original KA bandwidth and does not modify BB, unspecified storage, or padding.
It is a composition path using dense owned matrices and scratch, and does not claim the same operation count or performance as a dedicated band bulge chase.

GV/GVD share Cholesky, standard-form reduction, native full EVD, and inverse transformation.
The band BB output is a split factor rather than an ordinary triangular Cholesky.
A non-positive definite B in band returns `INFO=N+i` and preserves the partial factor including reverse-order pivots.
Unprocessed complex diagonals are not read or written, and only processed/updated diagonals are returned to real values.
GVD preserves query and each entry point's minimal WORK/RWORK/IWORK, but it allocates internal areas separately and does not mean execution of the divide-and-conquer algorithm of the same name.

GVX selects A/I/V from the native full spectrum.
The Q of band is the full generalized eigenbasis, giving `QᴴAQ=diag(λ)` and `QᴴBQ=I`, and the selected columns are returned in Z.
Because the diagonal form is also tridiagonal, AB stores the corresponding diagonal tridiagonal matrix.
This is not a dedicated partial-eigenvalue or Sturm count algorithm.
A positive ABSTOL, native rejection, and range endpoints within the accuracy-based guard are passed to the original ILP64 LAPACK saved before modifying the input.
LP64 integer work arrays and IFAIL restore only the slots that were written.
Entries of W/Z/IFAIL outside the selection, Q/Z with JOBZ=N, and matrix padding are preserved.
It also distinguishes the PEVX behavior of packed N=1 and empty value ranges returning Z=1 and preserving IFAIL.
Some versions of the reference implementation of complex HPGVX return with M unset when N=0, but the adapter returns M=0.

`test/structured_generalized*.jl` verifies mutual use with external factors, each ITYPE, UPLO, and range, residuals and B normalization, zero/rank-1/repeated spectra, the Q transformation formulas, query, the exact minimal area, diagnostics on insufficiency, type/integer ABI/storage guards, and actual LBT forwarding.
Residuals for a positive ABSTOL are judged including the requested absolute accuracy, and the fixed tolerance for the default precision and a strict positive ABSTOL is not relaxed.

Real SBGVD in reference OpenBLAS 0.3.30 writes outside the declared minimal LWORK=1 for N=1/JOBZ=V and returns unnormalized vectors that depend on the initial contents of WORK.
An independent reproduction in a safe spare area was saved in `reproducers/openblas-sbgvd-singleton-workspace.jl`.
Only the corresponding reference-comparison path is given slack, and native is verified with the minimal area and independent residuals and B normalization.


## band/packed tridiagonalization and packed Q (Issue #309)

The 16 Fortran symbols of SBTRD/HBTRD/SPTRD/HPTRD and OPGTR/UPGTR/OPMTR/UPMTR are connected for the four types and LP64/ILP64. Each ABI has 1250 symbols (BLAS 202, LAPACK 1048).
The native revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

SPTRD/HPTRD expand only the selected triangle into an owned matrix and convert the real beta gauge of the existing native reduction back to LAPACK's packed AP/TAU format. The imaginary part of a complex diagonal is not read.
Generation and application of Q share the existing native Householder application and gather the packed tail directly.
The implicit 1, the diagonal, tridiagonal components, and the tails of zero-coefficient reflectors are not read.
AP/TAU can also be used mutually with factors generated by reference LAPACK, and OPMTR/UPMTR support left/right application and transpose/conjugate transpose.

SBTRD/HBTRD also expand into an owned matrix and use the same reduction, returning D/E and the band diagonal and adjacent diagonals.
VECT=V returns Q, VECT=U returns X*Q for the caller's X, and VECT=N does not reference Q.
Complex E reflects the column phase into Q and returns non-negative real values.
The real path with KD=0/1 preserves the original tridiagonal form and the identity Q.
It does not claim the same operation count or performance as a dedicated band bulge chase, and the owned matrix area is O(n²).

The minimal area of the fixed WORK, INFO, empty/single element, zero/rank-1/repeated spectra, leading dimension, unused band regions, and Q and matrix padding are verified.
`test/structured_tridiagonal_*.jl` confirms A≈Q*T*Qᴴ, QᴴQ≈I, mutual use with external AP/TAU, updating of existing matrices, and actual LBT forwarding.

## Left and right eigenvectors from Schur form (Issue #311)

The 8 Fortran symbols of TREVC/TREVC3 are connected for the four types and LP64/ILP64. Each ABI has 1258 symbols (BLAS 202, LAPACK 1056), and the native revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

The right side uses the public `eigenvectors_from_schur_real/cplx`. For the left side, the inverted transpose/adjoint of the Schur form is passed to the same helper as an upper-triangular form, and rows and columns are restored to the original order.
For real 2x2 blocks, the order of the column pair corresponding to the positive imaginary part and the conjugation on the left side are made explicit.
The Schur form is not recomputed; only the specified triangle of the input T and the real first subdiagonal are read.

SIDE=L/R/B, HOWMNY=A/B/S, standardization of SELECT for real pairs, M/MM, and the back-transform of Q*X and Q*Y are preserved.
The largest abs1 component of a single column and the largest |re|+|im| of a real pair are normalized to 1.
VL/VR outside the specification, columns outside the selection, and T and padding are not modified.
SELECT rewrites only real pairs with HOWMNY=S, and complex SELECT is input-only.

TREVC3 shares the same native helper and matrix products and does not mean that Netlib's own blocked back-substitution algorithm is implemented.
The fixed WORK and minimal legal LWORK/LRWORK, collective query, and the different check order for real and complex are preserved.
Query returns a minimal area sufficient for this path, and there is no guarantee that it matches the recommended value based on the block size of reference LAPACK.
Owned matrices and native scratch are O(n²).
On native rejection, the input is passed to the saved original ILP64 TREVC/TREVC3 before modifying VL/VR, and LP64 SELECT/M/INFO are converted in a dedicated integer area.

`test/schur_vectors*.jl` verifies independent left and right residuals, packing and normalization of real conjugate pairs, M/MM, SELECT, back-transform, empty/single element/zero/repeated eigenvalues, minimal/maximal scales, the exact minimal work and query and INFO on insufficiency, storage guards, and actual LBT forwarding.

## Reordering of ordinary Schur blocks and condition estimation (Issue #312)

The 14 Fortran symbols of TREXC/LAEXC/TRSEN/GEESX are connected to LP64/ILP64. TREXC/TRSEN/GEESX handle the four types and LAEXC the two real types.
Each ABI has 1272 symbols (BLAS 202, LAPACK 1070), and the native revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

TREXC/LAEXC use the public native checked Schur move and preserve 1x1/2x2 block exchange, accumulation of the transformation into the existing Q, the leading-row adjustment of real IFST/ILST, and the final position.
The IFST/ILST of complex TREXC are preserved as input-only.
A rejected exchange is undone, and T/Q up to the last successful exchange and the current position are returned.
LAEXC does not reference data when N or the block width is zero.
Internally it allocates an owned Schur matrix and the necessary Q.

TRSEN JOB=N does not modify SELECT, selects real conjugate pairs together stably, and returns M and the eigenvalues and the necessary Q.
For the imaginary part, the square roots of the off-diagonals are computed separately so that a small imaginary part next to a large real part is not lost through relative scaling.
Only LWORK=-1 is a query, and LIWORK=-1 alone is an insufficient argument in the real version.
The minimal WORK for real JOB=N is max(1,N), for complex JOB=N it is 1, and real IWORK is 1.

GEESX SENSE=N uses the same numerical path as the existing native GEES.
The common GEES processing was also fixed so that SDIM on a rejected swap returns the total number of selected columns rather than the moved prefix.
SORT=N and query do not call SELECT.
Real GEESX treats either LWORK or LIWORK being -1 as a collective query, and the complex version treats only LWORK=-1 as a query.
The query on the native path returns a minimal WORK sufficient for that path and is not necessarily the recommended value based on the block size of the reference implementation.

Because there is no public native numerical API for S/SEP and RCONDE/RCONDV, TRSEN JOB=E/V/B and GEESX SENSE=E/V/B use the saved original ILP64 routines.
On LP64, SELECT, M, SDIM, INFO, and the integer and logical work arrays are converted, and IWORK/BWORK restore only the slots that were written.
The LOGICAL return value of the SELECT callback is converted through a thread-local guard, and the outer callback is restored after a nested call.
The invalid-argument INFO of the original routine is also recorded in the adapter diagnostics, and a positive numerical rejection is not treated as an invalid argument.

`test/ordinary_schur_*.jl` verifies the four types and both ABIs, all block sizes and movement directions, selection order, the similarity transformation, orthogonality/unitarity of Q, rejection with finite values and the intermediate state, the total number of selected columns, minimal workspace and query, the original condition estimates, the callback's original units, real pairs, and nested calls, unreferenced outputs, padding, integer guards, and the actual LBT forwarding of each entry point.

## Generalized QR and equality-constrained drivers (Issue #314)

All 16 Fortran symbols of `GGQRF/GGRQF/GGLSE/GGGLM` are connected for the four real and complex types and both LP64/ILP64.
The dependency revision remains `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

GGQRF composes `A=Q R, B=Q T Z` and GGRQF `A=R Q, B=Z T Q` from the existing native-backed QR/RQ and Q application.
The returned reflectors can also generate Q from the original LAPACK ORG/UNGQR/RQ, and both matrices of rectangular/empty/zero/rank-1 input can be reconstructed.
For complex Q application, the adjoint rather than the transpose is used.
Triangular solve uses the public native solve, avoiding the path where direct division of a complex scalar squares the denominator and makes extreme values NaN.

GGLSE solves `min ||c-Ax||₂, Bx=d` and GGGLM solves `min ||y||₂, Ax+By=d`. They check the dimension constraints `0≤P≤N≤M+P` and `0≤M≤N≤M+P` respectively.
If the triangular factor of the first constraint system is exactly singular, `INFO=1`; if the triangular factor of the remaining free variables is exactly singular, `INFO=2`.
This is not a numerical rank estimate using a rank-revealing tolerance.
On failure, the factorization and right-hand-side transformations already performed are preserved, and solutions not yet obtained are not returned as success values.
The residual and intermediate-solution payloads of C/D are also updated destructively.

`LWORK=-1` is a query and does not modify the input matrices, right-hand sides, solutions, or TAU.
The minimal WORK is `max(1,N,M,P)` for GGQRF/GGRQF, and normally `M+N+P` for GGLSE/GGGLM, or 1 if N of the system being solved is 0.
Query returns the adapter's legal minimum and does not require agreement with the recommended value based on Netlib's block size.
Auxiliary matrices, coefficients, and native scratch are allocated separately within the adapter.
This does not mean providing a dedicated colmat-rs driver or the same algorithm and performance as Netlib.

In addition to comparing solutions against the original OpenBLAS driver as an independent reference, it checks constraint residuals, stationarity in the free directions, the minimum-norm property, reconstruction with external Q, the boundaries of WORK/INFO/TAU and padding, query, argument order, and exact rank failure.
This includes checks that actually obtain the LBT provider and function pointers and call each of the 16 entry points on both ABIs.
The driver update order of the implementation follows Reference LAPACK v3.12.1 `SRC/dgglse.f` / `SRC/dggglm.f`, and the source and BSD-3-Clause license are recorded in `deps/rust/src/constrained_least_squares.rs` and `deps/rust/licenses/`.

## GEEVX expert mode and capability verification (Issue #316)

The native path of `SENSE=N` is preserved, and `SENSE=E/V/B` is delegated to the saved original implementation.
If an LP64 original implementation exists, it is called as is; otherwise the saved ILP64 implementation is used.
It extends `N/LDA/LDVL/LDVR/LWORK` and returns `ILO/IHI/INFO` and real IWORK with checks.
For a query, an unreferenced IWORK slot, or an output not written by the original implementation, the caller's value is preserved.
Complex RWORK is passed as scratch of the same real type. The internal TREVC3 query may update RWORK.

`load_backend()` checks the original ILP64 GEEVX for the four types before changing LBT, and returns a configuration error if it is missing.
Even in an ordinary Julia environment without an existing LP64 implementation, a legal expert mode is available through the ILP64 implementation.
`SENSE='?'` and combinations with E/B that do not request both left and right eigenvectors are treated as an invalid argument 4.
Re-registration does not save Colmat's own GEEVX as the fallback.

When using the low-level DLL directly, the original implementation must be registered before an expert call.
`colmat_lapack_has_expert_fallback(id,bits)` accepts id=0/1/2/3 (s/d/c/z) and bits=32/64, returning 1 if available.
On LP64 the saved ILP64 path is also included in availability.
If a legal expert call is made while unconfigured, INFO and the computed outputs are left unchanged and configuration diagnostic -1 is recorded in `colmat_blas_last_error()`.
If an external integer output is outside the ABI range, the diagnostic is -2.
In ordinary Julia, the check at registration prevents this unconfigured state.
No proprietary LAPACK INFO value is added, no success value is returned without computing the condition estimate, and no exception unwinds across the C ABI.

It checks the four types and both integer ABIs, normal computation/query, balance, left and right eigenvectors, all SENSE, minimal WORK, invalid argument order, and padding/integer guards.
It compares left and right residuals, non-zero normalization, and condition estimates with an independent reference, and checks direct LP64 delegation also with the test-only OpenBLAS32_jll.
It confirms, in independent processes, the configuration that handles LP64 with ILP64 only, removal and recovery of the original implementation, and absence before the first registration.
The dependency revision of the numerical API is `4e6ed0e34659147a5c744ab6431f150e78db6f82`, and the registration count at the time of this #316 verification is 1,288 for each ABI.


## Direct LP64/ILP64 execution evidence (#317)

The normal full suite checks completed project-owned foreign calls for all
1,106 forwarded LAPACK canonical entries in each integer ABI. The
[execution snapshot](../assets/lapack-abi-execution.json) records exact source
and native revisions, platform, Julia version, binary hash, source hashes and
individual DLL/LBT routes and signatures. Linux/macOS CI saves separate
artifacts from the same gate.

The original 76-symbol driver baseline additionally tests guarded integer and
floating outputs, padded leading dimensions, empty/rectangular/singular cases,
query and exact minimum work, solves/inverses, SVD full/thin/overwrite modes,
left/right eigenvectors and selfadjoint index/value selections. OpenBLAS calls
use independent ILP64 signatures; reconstruction, residuals and orthogonality
are checked against original inputs. The isolated C caller checks wrong-width
INFO writes in both directions, with a real protected-page crash confined to a
child process.

`direct_call_witness` means a concrete entry call returned. It does not claim
all options, the identity of Netlib algorithms or native sanitizer coverage.
The importer rejects stale production/worker/test sources, native revisions,
manifests, registration inventories and missing integer ABIs. Other symbols
remain `not_catalogued`, which does not mean they were never tested.


## ROOK factors and two swap indices (#372)

The 20 symbols `SY/HE TRF_ROOK, TF2_ROOK, TRS_ROOK, TRI_ROOK, SV_ROOK` are verified for the four types, both integer ABIs, and both UPLO directions.
The DLL and the actual LBT stub are called, passing native factors to the original OpenBLAS solve/inverse and original factors to native.
This includes chains in which both swaps of a two-column pivot are non-trivial, reconstruction by elementary permutation and triangular matrix products, singular INFO, query/minimum WORK, and integer and padding boundaries.

Native reciprocal computation of subnormals may produce non-finite values, and the executed MWE and an improvement proposal are recorded in
[tensor4all/colmat-rs#212](https://github.com/tensor4all/colmat-rs/issues/212).
The adapter protects this through composition of the public numerical API and scaling. Shared with ordinary and packed BK, non-zero L spanning different units and finite known solutions are also verified.
Under conditions with many subnormal blocks where the original OpenBLAS classic factorization itself fails, its output is not taken as correct; checks use element-wise reconstruction and the identity-matrix solve.
No dedicated algorithm, identical pivot selection, performance, or a finite inverse outside the representable range is claimed.
