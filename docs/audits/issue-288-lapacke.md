# Issue #288: dense LU and Cholesky LAPACKE entries

This stage connects **72 canonical numerical entries**: high-level and `_work`
forms of GETRF, GETRS, GESV, GETRI, POTRF, POTRF2, POTRI, POTRS and POSV, for
Float32, Float64, ComplexF32 and ComplexF64. Both integer ABIs and actual DLL/LBT
routes are tested. Two additional NaN policy helpers are shared between ABIs;
their arguments and return values retain C-int width.

**Issue #288 remains open.** Its original 1,616-entry acceptance scope is frozen
in `issue-288-lapacke-scope.json`; 1,544 of those entries remain beyond this stage.
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

## Execution evidence

The successful full `Pkg.test(; allow_reresolve=false)` ran source revision
`021e74301570b9433d0e4f94476f3c60204b0846` with 245 production/test/worker sources,
manifest, registration table and release DLL frozen before execution. The
resulting artifact matched every captured hash and revision. Its 4,751 raw
witness rows independently validate **1,350 canonical entries**: the existing
1,106 Fortran LAPACK and 172 CBLAS targets plus 72 LAPACKE targets. The formal
snapshot is `docs/src/assets/lapack-abi-execution.json`; scope and provenance
validation rejects stale, partial or failed-suite artifacts.

The 72 LAPACKE entries contribute 288 completed call-site observations, covering
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
completed-call gate passes 1,446 checks. Independent C workers exercise the
concrete ABI and compiler's complex layout.

## Validation and remaining work

- Successful full Pkg.test, including ordinary standard BLAS/LAPACK regression.
- 398 Rust tests and Rust formatting checks passed.
- Focused LAPACKE numerical/C checks passed 31,912 checks before tracing gates;
  separate existing Fortran driver/direct-ABI/boundary regressions passed 7,200.
- Independent OpenBLAS factors feed GETRS/GETRI/POTRS/POTRI; standard OpenBLAS
  solves consume generated LU and Cholesky factors. Scalar loops construct RHS,
  reconstruct factors, and verify left/right inverse products.
- C checks cover C complex buffer layout, signed INFO/pivot widths, guarded
  outputs, NULL query/zero-RHS inputs, policy initialization, imaginary-only
  diagonal NaNs, an ILP64 query above 32-bit range and Float32 work rounding.
- The Documenter build (`julia --project=docs docs/make.jl`) passed.
- After importing the full artifact, all 26 Python documentation/audit tests and
  all 15 actual inventory checks passed. Registration is 1,554 per ABI, with
  5,146 canonical symbols. The candidate audit preserves the original 2,910
  scope and separately tracked 40 Rook additions.

Remaining families need their own layout, workspace, callback, option and
numerical-contract validation. The original 1,616 declarations include 36
real-valued returns and 40 entries without matrix_layout; the current nine
families do not establish those contracts. Native-dependent BDS/TGEVC work and
coordinated Rook C entries remain pending. This stage must not close #288 or
mark parent #263 complete.
