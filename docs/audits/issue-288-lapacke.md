# Issue #288: implemented LAPACKE numerical subset

The current candidate connects **1,196 original-scope canonical numerical
entries**, including the core dense LU/Cholesky/eigendecomposition drivers and,
among the later stages, the 32 GGQRF/GGRQF/GGLSE/GGGLM, 24 LACPY/LASET/LASWP, 8
TGEVC, 24 LANGE/LANSY/LANTR, 16 LAPMR/LAPMT, 32 ORGTR/UNGTR/ORMTR/UNMTR and
ORGBR/ORMBR/ORGHR/ORMHR, 16 GGHRD/GGHD3, 8 HGEQZ, 16 PTTRF/PTTRS, 16
PBTRF/PBTRS, 16 PPTRF/PPTRS, 24 GBTRF/GBTRS/GBSV, 24 GTTRF/GTTRS/GTSV, 24
SYTRF/SYTRS/SYSV and HETRF/HETRS/HESV, 16 GEBAL/GEBAK, 24 TBTRS/TPTRS/TPTRI, 24
GEHRD/GEBRD/LARFX, 16 GEEQU/GEEQUB, 16 GGES/GGES3, 8 HSEQR, 32
GEQR/GELQ/GEMQR/GEMLQ, 24 PBEQU/PBSTF/PBSV, 24 PFTRF/PFTRI/PFTRS, 48
TRTTF/TFTTR/TFTTP/TPTTF/TPTTR/TRTTP, 40 POEQU/POEQUB/PPEQU/PPSV/PPTRI and 24
TFSM/TFTRI/PTSV high/work entries. Its immutable full suite validates 446
source files, 2488 canonical entries and 9463 completed witnesses, with 2692
registered forwards per integer ABI.
See [generalized QR contracts and evidence](issue-288-generalized-qr-contract-plan.md)
and [matrix auxiliary contracts](issue-288-matrix-aux-contract-plan.md).

**Issue #288 remains open.** Its original 1,616-entry acceptance scope is frozen
in `issue-288-lapacke-scope.json`; 420 of those entries remain beyond this stage.
Issue #372 additionally coordinates 20 LAPACKE Rook entries with #288. Those
additions are separate from the original scope and are also still pending.
Registration alone, successful fallback computation and the current subset's
passing tests do not establish completion of either scope.

## Reference and contracts

Numerical revision: `4e6ed0e34659147a5c744ab6431f150e78db6f82`.
C declarations and wrapper contracts: Reference-LAPACK tag `v3.12.1`, header
`LAPACKE/include/lapacke.h`, blob `82cc4e6c1e4d29ae81b88342fa07ece9339124bb`.
Representative inspected wrapper blobs under `LAPACKE/src`:

| Source | Blob |
| --- | --- |
| lapacke_dgetrf_work.c | 62e6fa502734cebb5363462c842e839792816d33 |
| lapacke_dgetrf.c | 6232721b4c595495494b978e4148e00af4de16e8 |
| lapacke_zgetrs_work.c | b9dde38658e59cb295b9248938e6f46513d14574 |
| lapacke_zgetrs.c | d76e323c14473665e2f5fca1814d0b38f2660cda |
| lapacke_dgesv_work.c | ec48c27e12ac4264e1e4ec4b6cf95623ee034440 |
| lapacke_zgetri_work.c | 93c19c59e8c71d16f307cff6d0ac13fb683c536c |
| lapacke_zgetri.c | 0af2ecfd5aed1df5dd7dc55808a834d3b24ae88c |
| lapacke_zpotrf_work.c | a490213a9a106b040e97013c692750bcef28fab2 |
| lapacke_zpotrf2_work.c | 60300e5fc65a2cc5a572ba7805fecaa5a98af430 |
| lapacke_zpotri_work.c | f98a8467f2b196d483f85d7308e73b2182042741 |
| lapacke_zpotrs_work.c | d6e43a69c42b299cbc9db6c16e611c982f8a95cf |
| lapacke_zposv_work.c | b990b5f1b26ae35a76ca36e389ce570dc8d87c28 |

Layout is C int and option characters are C char; dimensions, pivots and INFO
follow lapack_int. Negative Fortran INFO shifts by one for the layout argument.
Row-major leading-dimension errors precede numerical argument validation as in
the reference wrappers. Logical entries are transformed without reading or
writing padding. Cholesky transforms and NaN checks touch only the requested
triangle; POTRS preserves its input factor. N/T/C and lowercase flags, empty
shapes, singular/nonpositive INFO and factor interoperability are exercised.

GETRI queries do not read A/IPIV or allocate a transpose. The work entry borrows
caller work; the high-level entry queries and allocates it. Minimum legal work
is max(1,N) for this native-backed implementation. Work and transpose allocation
errors use -1010 and -1011. Native numerical scratch is still allocated
internally, including the identity RHS for inversion; zero allocation and
vendor LAPACK workspace/performance equivalence are not claimed. GESV/POSV
still factor A when NRHS=0 and preserve B on positive factorization failure.
POTRF2 retains the existing native Fortran factor alias; no separate recursive
algorithm or performance equivalence is claimed.

NaN policy follows the inspected `lapacke_nancheck.c` and dense/triangle helper
contracts: enabled by default, one-time `LAPACKE_NANCHECK` initialization,
explicit setter override and shared state across suffixed helper aliases.
High-level checks distinguish A and B error indices. They inspect complex
diagonal imaginary NaNs, whereas work/Fortran Hermitian factorization assumes
those imaginary parts are zero. Padding and unselected triangles are excluded.

## Existing Fortran boundary corrections

The old Fortran POTRF adapter exposed writes from the native in-place call into
the caller's unselected triangle. An executed ComplexF64, N=3, LDA=5 MWE changed
ignored positions 6, 11 and 12. A separate executed variant with diagonal
imaginary NaNs returned INFO=0 with nonfinite factor entries. These were
Colmat.jl LAPACK boundary defects; colmat-rs was not changed.

`factor_cholesky` now copies the selected triangle, projects the diagonal to
its real part and copies only the selected factor back, including failure data.
Upper storage is converted through the lower Hermitian representation. This
also fixes Fortran POTF2/POTRF2 and POSV through their existing POTRF dispatch.
It introduces explicit N-by-N scratch; row-major conversion can require an
additional N-by-N temporary. `reproducers/potrf_ignored_storage.jl` now reports
INFO=0, no changed ignored positions and finite factor entries.

### Hermitian diagonal projection

A 3-by-3 HEEV input with imaginary-only diagonal NaNs returned INFO=1 and
nonfinite vectors while independent OpenBLAS returned INFO=0. LAPACK assumes
those imaginary components are zero. The shared `selfadj` boundary now detects
a diagonal that differs from its real projection, copies only the selected
lower view and normalizes the diagonal. Ordinary real-diagonal inputs keep
borrowing their buffer. Normalization adds N-by-N scratch only when needed,
in addition to any row-layout temporary. The reproducer
`reproducers/heev_ignored_diagonal.jl` now agrees with OpenBLAS; the extended
full suite includes both layouts, both ABIs and imaginary-only diagonal NaNs.

## Execution evidence

The successful full `Pkg.test(; allow_reresolve=false)` ran source revision
`443d349b46686011c2889333f64f0ab6b630de92` with 257 production/test/worker sources,
manifest, registration table and release DLL frozen before execution. The
resulting artifact matched every captured hash and revision. Its 4,959 raw
witness rows independently validate **1,402 canonical entries**: the existing
1,106 Fortran LAPACK and 172 CBLAS targets plus 124 LAPACKE targets. The formal
snapshot is `docs/src/assets/lapack-abi-execution.json`; scope and provenance
validation rejects stale, partial or failed-suite artifacts.

The 124 LAPACKE entries contribute 496 completed call-site observations, covering
both ABIs and both actual routes. Each recorded argument and INFO return type
is checked against the current concrete C contract. Registered unreviewed
families are rejected until their real signature is added; future matrix-norm
returns must not be assumed to be integer INFO. Policy helpers have separate C
policy tests and do not count as numerical-entry witnesses.

Optimized export aliases can share an address. Numerical tests therefore
qualify caller-selected alias names where needed. Qualification creates no
witness: the actual returned ccall must match the selected symbol, ABI and
route in the address table. Selection is included in deduplication and restored
on exceptions. The recorder behavior tests pass 24 checks, and the LAPACKE
completed-call gate passes 2,486 checks. Independent C workers exercise the
concrete ABI and compiler's complex layout.

## Validation and remaining work

- Successful full Pkg.test, including ordinary standard BLAS/LAPACK regression.
- 428 Rust tests and Rust formatting checks passed.
- The LU/Cholesky numerical/C stage passed 33,320 checks before tracing gates;
  complex SYR adds 13,316 numerical checks and two C-runner checks;
  SVD adds 61,128 numerical checks and two independent C-runner checks;
  the combined SYEV/HEEV and SYEVD/HEEVD suite passes 77,200 numerical checks
  and two independent C-runner checks;
  separate existing Fortran driver/direct-ABI/boundary regressions passed 7,200.
- Independent OpenBLAS factors feed GETRS/GETRI/POTRS/POTRI; standard OpenBLAS
  solves consume generated LU and Cholesky factors. Scalar loops construct RHS,
  reconstruct factors, and verify left/right inverse products.
- C checks cover C complex buffer layout, signed INFO/pivot widths, guarded
  outputs, NULL query/zero-RHS inputs, policy initialization, imaginary-only
  diagonal NaNs, an ILP64 query above 32-bit range and Float32 work rounding.
- The Documenter build (`julia --project=docs docs/make.jl`) passed.
- After importing the full artifact, all 28 Python documentation/audit tests and
  all 15 actual inventory checks passed. Registration is 1,606 per ABI, with
  5,146 canonical symbols. The candidate audit preserves the original 2,910
  scope and separately tracked 40 Rook additions.

Remaining families need their own layout, workspace, callback, option and
numerical-contract validation. The original 1,616 declarations include 36
real-valued returns and 40 entries without matrix_layout; the implemented
families do not establish those contracts. Native-dependent BDS/TGEVC work and
coordinated Rook C entries remain pending. Issue #288 and parent #263 remain open until their respective requirements are verified.


## GETF2/GETRF2 extension (PR #380 historical checkpoint)

PR #379 delivered the first 72 entries. This extension adds 16 canonical
high/work LU entries and preserves the original scope and separately listed
Rook coordination targets. It uses the same public native LU numerical
contract as the already-connected Fortran aliases; unblocked/recursive algorithm
and performance equivalence are not claimed. Reference-LAPACK v3.12.1 work
wrapper blobs: `lapacke_cgetf2_work.c`
`53643cd7c30238179338ce19aff2e337b1696857` and `lapacke_cgetrf2_work.c`
`2586244a3ed00a210567e216b08864554664a7f2`.

C compiler tests cover both 3-by-2 and 2-by-3 matrices, nonreal complex values,
guarded pivots, both layouts, singular INFO and invalid dimensions. The Julia
LU checks pass 2,128 checks; all 88 numerical entries produce 352 actual
name/ABI/route observations with independently validated argument and return
types. Alias qualification prevents release code merging from dropping or
inventing executed names. Both C runners pass eight checks. Rust still passes
398 tests; inventory checks pass 15 tests and independent evidence fixtures
pass 15 tests.

The PR #380 full Pkg.test succeeded at source revision
`b72cd8767ddc92b56317917a02431f5c3555943e`. Its artifact matched all 245 captured
source hashes, manifest, release DLL and the exact
1,570-forward-per-ABI inventory, then independently validated 1,366 entries.
That successful full artifact replaced the earlier 72-entry snapshot. The
subsequent complex SYR and SVD full runs supply the current 124-entry snapshot;
focused artifacts are not published as full-suite evidence.

## SYTRI/HETRI, TRI2 and TRI2X extension

S/D SYTRI/SYTRI2/SYTRI2X and C/Z HETRI/HETRI2/HETRI2X high/work adapters
(24 symbols, 48 ABI entries) over the native Bunch-Kaufman inverse. Factors and
IPIV from SYTRF/HETRF are consumed directly. The high-level NaN scan covers only
the UPLO triangle (Hermitian diagonal imaginary parts ignored); row-major results
write back only that triangle. TRI2 supports the LWORK=-1 query; TRI2X allocates
(N+NB+1)*(NB+3) elements like the reference wrapper, NB being a hint only.
Complex symmetric CSYTRI/ZSYTRI remain outside this slice. The full-suite
artifact validates 1,220 LAPACKE canonical entries.

## Packed HP/SP TRI and SYTRS2/HETRS2 extension

SPTRI (S/D) and HPTRF/HPTRS/HPSV/HPTRI (C/Z) high/work adapters share the
packed Bunch-Kaufman macro; row-major AP is reordered with the reference
pp_trans permutation and the Hermitian diagonal imaginary part is excluded from
the NaN scan. SYTRS2/HETRS2 read the factor without modifying it and allocate
the N-element WORK array that the native solve does not use. The full-suite
artifact validates 1,248 LAPACKE canonical entries.

## Tridiagonal eigen driver extension

STERF (S/D, no matrix_layout), STEQR and STEDC (S/D/C/Z), and STEV/STEVD (S/D)
high/work adapters over the native tridiagonal eigensolvers. Row-major Z is
read only for COMPZ='V' and written back for COMPZ/JOBZ other than 'N'; the
high-level STEDC/STEVD entries query WORK/IWORK (and RWORK) before allocating.
Complex STEDC checks unitary starting vectors. The full-suite artifact
validates 1,276 LAPACKE canonical entries.

## Band/packed self-adjoint EVD extension

SBEV/HBEV, SPEV/HPEV, SBEVD/HBEVD and SPEVD/HPEVD high/work adapters over the
native structured eigensolvers. Row-major band AB (KD+1 rows) and packed AP are
converted to column-major (the band/packed triangle is written back), Z is
transposed for JOBZ='V', and the *EVD high-level entries query WORK/IWORK
(and RWORK) first. NaN checks cover only the stored band/packed triangle. The
full-suite artifact validates 1,308 LAPACKE canonical entries.

## SFRK/HFRK extension and the GEMLQ NaN-scan fix

SFRK (S/D) and HFRK (C/Z) high/work adapters over the native RFP rank-k update.
alpha and beta are passed by value; the RFP rectangle and the row-major A are
converted, and a K=0 call still scales C by beta. The routine returns no INFO,
so column-major argument errors reach only the native checker and the LAPACKE
status stays zero; row-major reports LDA < KA (-9).

The high-level GEMQR/GEMLQ NaN check scanned an R-by-K footprint for both
routines although GEMLQ stores a K-by-R reflector block, so it read past a
tight buffer and occasionally returned -7 in place of a later diagnostic. The
scan now follows the stored block, with a regression test at the last K-by-R
element. The full-suite artifact validates 1,316 LAPACKE canonical entries.

## Rook Bunch-Kaufman extension

SYTRF/SYTRS/SYSV_ROOK (S/D) and HETRF/HETRS_ROOK (C/Z) high/work adapters over
the native rook Bunch-Kaufman factor and solve; factors and IPIV (including
negative two-by-two pivots) interoperate between layouts. The complex
symmetric CSYTRF/CSYTRS/CSYSV_ROOK family has a different LDL^T contract and
stays out of scope, as does HESV_ROOK, which has no LAPACKE inventory entry.
The full-suite artifact validates 1,336 LAPACKE canonical entries.

## TGEXC extension

TGEXC (S/D/C/Z) high/work adapters over the native generalized Schur
reordering. Real IFST/ILST are in/out pointers with a WORK/LWORK query; complex
IFST/ILST are by value and the work entry takes no workspace. A, B, Q and Z are
transposed for row-major, and the workspace query is forwarded to the native
solver untransposed. The full-suite artifact validates 1,344 LAPACKE canonical
entries.

## Dense generalized self-adjoint extension

SYGST/HEGST, SYGV/HEGV and SYGVD/HEGVD (S/D/C/Z) high/work adapters over the
native generalized reduction and drivers, for ITYPE 1-3. The NaN scan covers
only the UPLO triangle of A and B; row-major A is written back whole for
JOBZ='V' (eigenvectors) and as a triangle otherwise, B as a triangle, and the
SYGST/HEGST B argument stays read only. The *GVD high-level entries query
WORK/IWORK (and RWORK). The full-suite artifact validates 1,368 LAPACKE
canonical entries.

## Band/packed generalized self-adjoint extension

SBGST/HBGST, SPGST/HPGST, SBGV/HBGV, SPGV/HPGV, SBGVD/HBGVD and SPGVD/HPGVD
(S/D/C/Z) high/work adapters over the native structured generalized solvers.
Row-major band AB/BB (KA+1 and KB+1 rows) and packed AP/BP are converted to
column-major and the stored triangles written back; Z or X is transposed for
JOBZ/VECT='V'. The GST routines read the split or plain Cholesky factor in B
without modifying it, and the *GVD high-level entries query WORK/IWORK (and
RWORK). The full-suite artifact validates 1,416 LAPACKE canonical entries.

## Band/packed selected eigenvalues and packed tridiagonal reduction

SBEVX/HBEVX and SPEVX/HPEVX (S/D/C/Z) and SPTRD/HPTRD high/work adapters over
the native structured selected eigensolvers and packed tridiagonal reduction.
Row-major band AB and packed AP are converted to column-major and written
back; for JOBZ='V' the band Q is transposed and only the returned Z columns
are written. RANGE='A'/'V'/'I' selections, ABSTOL, M and IFAIL pass through
and WORK/IWORK(/RWORK) use the reference fixed lengths. The full-suite
artifact validates 1,440 LAPACKE canonical entries.

## Dense, band and packed selected eigenpairs

SYEVX/HEEVX, SYGVX/HEGVX, SBGVX/HBGVX and SPGVX/HPGVX (S/D/C/Z) high/work
adapters over the native selected eigensolvers. The dense entries query
WORK/LWORK; the band and packed entries use the reference fixed lengths. Only
the stored triangles are scanned for NaN, row-major A/B are written back as
triangles, and Z is written only for the returned columns (the band Q only for
JOBZ='V'). RANGE='A'/'V'/'I', ABSTOL, M and IFAIL pass through. The full-suite
artifact validates 1,472 LAPACKE canonical entries.

## Tridiagonal selected eigenpairs

STEVX and STEVR (S/D) and STEGR and STEMR (S/D/C/Z) high/work adapters over the
native tridiagonal selected eigensolvers. D and E are scanned for NaN (E has
N-1 entries, N for STEMR), the row-major Z is written only for the returned
columns, and STEVR/STEGR/STEMR high-level entries query WORK/IWORK. STEMR
passes NZC and TRYRAC through unchanged. The full-suite artifact validates
1,496 LAPACKE canonical entries.

## SYTRD/HETRD and SYEQUB/HEEQUB

SYTRD/HETRD (S/D/C/Z), SYEQUB (S/D/C/Z) and HEEQUB (C/Z) high/work adapters
over the native dense tridiagonal reduction and symmetric scaling. Only the
UPLO triangle is scanned for NaN; row-major SYTRD/HETRD results are written
back as a triangle and the high-level entries query WORK/LWORK. SYEQUB/HEEQUB
read A without modifying it, and complex magnitudes follow LAPACK's CABS1. The
full-suite artifact validates 1,516 LAPACKE canonical entries.

## Mixed-precision conversion, LACRM/LARCM and PSTRF

DLAG2S, SLAG2D, ZLAG2C and CLAG2Z, CLACRM/ZLACRM, CLARCM/ZLARCM and
SPSTRF/DPSTRF/CPSTRF/ZPSTRF high/work adapters over the native conversion,
complex-by-real product and pivoted Cholesky kernels. The conversions report
overflow as a positive INFO; LACRM/LARCM return no INFO, so column-major
argument errors reach only the native checker; PSTRF writes back the UPLO
triangle and passes PIV and RANK through. Row-major A/B/C are transposed with
leading-dimension checks. The full-suite artifact validates 1,540 LAPACKE
canonical entries.

## Tall-skinny least squares and TSQR reconstruction

GETSLS, GETSQRHRT, ORGTSQR_ROW/UNGTSQR_ROW and ORHR_COL/UNHR_COL (S/D/C/Z)
high/work adapters over the native tall-skinny least-squares solve and the TSQR
Householder reconstruction. T is an NB-by-(N*slots) array, so the row-major
LDT must cover its column count; A, B and T are transposed and written back,
and GETSLS/GETSQRHRT/ORGTSQR_ROW query WORK/LWORK. ORHR_COL/UNHR_COL have no
workspace argument (their work entry is the same call). The tests compare the
reflectors, signs and T blocks against the independent reference LAPACK and
verify explicit Q from reference LATSQR factors. The full-suite artifact
validates 1,572 LAPACKE canonical entries.

## Bidiagonal SVD (BDSQR, BDSDC, BDSVDX)

BDSQR (S/D/C/Z) and BDSDC and BDSVDX (S/D) high/work adapters over the native
bidiagonal SVD. D/E are scanned for NaN (E has N-1 entries); VT, U and C are
transposed for row-major and written back only when present; BDSVDX Z has 2N
rows holding [U; V] and only the returned columns are written. BDSDC
COMPQ='N' and 'I' are verified end to end; COMPQ='P' (the compact IQ/Q form)
is passed through to the native compact driver and is not separately verified
in row-major. The full-suite artifact validates 1,588 LAPACKE canonical
entries.

## GESVDX

GESVDX (S/D/C/Z) high/work adapters over the native selected SVD. RANGE
'A'/'V'/'I' selections pass through; the row-major U is M-by-NS and VT is
NS-by-N and only the NS selected triplets are written back. The high-level
entry queries WORK/LWORK and uses the caller's SUPERB array as the integer
workspace (12*min(M,N) entries). The full-suite artifact validates 1,596
LAPACKE canonical entries.

## TGSEN

TGSEN (S/D/C/Z) high/work adapters over the native generalized Schur cluster
reordering with optional condition estimates (IJOB). SELECT is a LOGICAL
(integer) array; row-major A, B, Q and Z are transposed and written back whole
and WORK/LWORK/LIWORK are queried. LAPACK's TGSYL step (IJOB >= 1) needs one
element more LWORK than the reported minimum (the query value leaves
LWORK-2*M*(N-M) = 0, which LAPACK reports as an illegal TGSYL argument and
the native routine then corrupts the heap), so the high-level entries pad the
queried workspace and callers of the work entry should do the same. The
full-suite artifact validates 1,604 LAPACKE canonical entries.

## GGESX

GGESX (S/D/C/Z) high/work adapters over the native generalized Schur
factorization with ordering and condition estimates. SELECTCTG is handed to the
native sorter; row-major A, B, VSL and VSR are transposed and written back
whole; WORK/LWORK/LIWORK are queried, and (as for TGSEN) the high-level entries
pad LWORK because the SENSE estimates run LAPACK's TGSYL step. The full-suite
artifact validates 1,612 LAPACKE canonical entries.

## GESVDQ

GESVDQ (S/D/C/Z) high/work adapters over the native QR-preconditioned SVD.
Row-major A is transposed and written back (it holds the QR factors); U is
written at its logical size (M-by-M for JOBU='A', N-by-N for 'F', M-by-N
otherwise, NUMRANK columns for 'R') and V at NUMRANK or N rows by N columns.
The high-level entry queries IWORK/WORK/RWORK. The full-suite artifact
validates 1,620 LAPACKE canonical entries.
