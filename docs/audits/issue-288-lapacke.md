# Issue #288: implemented LAPACKE numerical subset

The current candidate connects **508 original-scope canonical numerical entries**,
including the 32 GGQRF/GGRQF/GGLSE/GGGLM, 24 LACPY/LASET/LASWP, 8 TGEVC and 24
LANGE/LANSY/LANTR high/work entries. Its immutable full suite validates 357
source files, 1790 canonical entries and 6627 completed witnesses, with 1994
registered forwards per integer ABI.
See [generalized QR contracts and evidence](issue-288-generalized-qr-contract-plan.md)
and [matrix auxiliary contracts](issue-288-matrix-aux-contract-plan.md).

**Issue #288 remains open.** Its original 1,616-entry acceptance scope is frozen
in `issue-288-lapacke-scope.json`; 892 of those entries remain beyond this stage.
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
