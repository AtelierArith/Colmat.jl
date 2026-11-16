# Issue 288: GESVD / GESDD LAPACKE contracts

Status: optimized-release C/Julia checks and immutable full-suite evidence pass.
Platform CI evidence remains pending. Issue #288 remains open; this numerical
subset does not establish the original scope's complete acceptance conditions.

This extension adds sixteen canonical names: S/D/C/Z GESVD and GESDD, each
high-level and `_work`, with LP64 and ILP64 exports. The total candidate LAPACKE subset is 108 numerical names,
leaving 1,508 original-scope entries and the separate 20 Rook targets pending.

## Reference contracts

Reference-LAPACK v3.12.1 source under `LAPACKE/src/` is the baseline:

| File | Git blob |
| --- | --- |
| lapacke_zgesvd_work.c | 71af24e0ae487fa94c843792b24fe3cd38c8960b |
| lapacke_zgesvd.c | c4af6ed620328d9500201e3477ce997924a55adc |
| lapacke_dgesvd_work.c | 99e27cfbdab1e110af68b3347c537d16eee4ec4f |
| lapacke_dgesvd.c | f2102cecb7793323d41bfc8d6de53b05681e132f |
| lapacke_zgesdd_work.c | ad9745fd825897c4f7a6c413ec55b6147763530c |
| lapacke_zgesdd.c | 412690cf786dbfc16cae0025a61e9238226533f1 |
| lapacke_dgesdd_work.c | 733fbd9fce0c1b7cc8fb53d589149d2ad2051b1b |
| lapacke_dgesdd.c | 76584d5786d50c8bc89d367055740287b8422186 |

Column-major work wrappers shift a negative Fortran INFO by minus one.
Row-major work wrappers check leading dimensions before workspace queries.
GESVD returns -7 for LDA < N, -10 for LDU below the selected U column count,
and -12 for LDVT below the selected VT column count. Unused U and VT each
retain a minimum row stride of one.

GESDD uses different positions: -6 for LDA < N, -9 for LDU below the
selected U column count, and -11 for LDVT < N. The last check also applies
to JOBZ=N. Do not share GESVD's unused-VT stride rule with GESDD.

Queries use temporary column-major leading dimensions and return without
transposing matrix storage. Executing row-major calls packs A and the selected
U/VT outputs, invokes Fortran, then copies logical A and selected outputs back.
JOB=O writes the selected singular vectors into A. Both rectangular orientations
are necessary to cover GESDD's change of overwrite destination.

High-level GESVD has a real SUPERB output of length min(M,N)-1. Reference real
wrappers copy WORK[1..min-1]; complex wrappers copy RWORK[0..min-2]. High-level
wrappers must initialize internal scratch before exposing any of it. A successful
result does not establish compatibility of failure diagnostics in SUPERB.

## Existing local backend limits to preserve explicitly

`deps/rust/src/lapack.rs::svd_driver` uses the native SVD implementation for both
Fortran driver names. It validates and reports WORK query/minimum sizes, but
complex RWORK and GESDD IWORK are currently unused. This does not establish
algorithm identity with Reference-LAPACK's GESVD or divide-and-conquer GESDD.
No native repository modification or pull request is authorized for this work.
Any demonstrated missing native capability must be reported there only as an
Issue containing an executed MWE and an improvement proposal.

## Required implementation evidence

- Exact C prototypes, including Cint layout, char flags, integer-width-specific
  dimensions and INFO return, real S/SUPERB/RWORK, and integer IWORK.
- Independent C caller and completed Julia calls through the DLL and LBT,
  with explicit symbol selection to disambiguate aliases.
- Both layouts, both integer widths, four scalar types, tall/wide/empty inputs,
  all legal A/S/N/O job combinations, and rejected double-overwrite GESVD.
- Reconstruction and orthogonality with nonreal complex data, ordered singular
  values, and an independent reference implementation; do not compare vector
  signs or phases directly.
- WORK query and minimum-size boundaries, invalid flags/dimensions/strides,
  NaN policy, untouched padding, null unused outputs, and guarded buffers.
- Immutable full-suite source/DLL evidence before importing documentation
  snapshots, followed by both platform CI artifact checks before merge.


## Executed local checks

The candidate implements all sixteen high/work numerical names in both
integer widths. The Rust suite passes 408 tests. The adapter distinguishes a
legal minimum from a rounded/optimal query and rejects unrepresentable native
workspace calculations before entering the backend.

Julia numerical tests pass 61,128 checks through direct DLL and actual LBT
routes. An independently resolved OpenBLAS Fortran GESVD supplies singular-value
comparisons; reconstruction, orthogonality, sorted nonnegative values, padding,
all legal GESVD A/S/N/O job pairs, GESDD overwrite orientations, empty matrices
and null unused outputs pass.

The independent C11 worker uses exact real/complex prototypes and both integer
widths. High/work queries and execution, lower-case flags, invalid flags,
negative dimensions, row-major leading dimensions, workspace errors, NaN input
and ignored padding pass. ILP64 calls use negative dimensions and valid query
strides beyond 32-bit range. Its Julia runner passes two checks.

Focused completed-call tracing validates 108 current LAPACKE names and all 432
ABI/route observations against independent signature/return contracts. The
completed-call gate passes 2,166 checks and the evidence-validator fixtures pass
17 tests, including real SUPERB/RWORK and integer IWORK rejection cases. The
focused artifact explicitly has `full_suite_passed=false`; it is not published
as full-suite evidence. Registration is 1,590 forwards per ABI. The optimized release repeats all
61,128 SVD numerical checks and the C worker successfully; its focused trace
again validates every one of the 432 ABI/route observations.

## Immutable full-suite evidence (SVD checkpoint)

Full `Pkg.test(; allow_reresolve=false)` succeeded at public source revision
`e7768503a2c72e001cdb2a26bf5ff54cc1059614`, Julia 1.13.3. All 253
production/test/worker source hashes, manifest, native revision, registration
digest and optimized DLL match preflight. The DLL SHA-256 is
`9998d6956c6ac104c4c989d030abfaf21641a8fbcf8f8d40f5d847d6b7e78315`.

Its 4,895 witnesses validate 1,386 canonical names: 1,106 Fortran LAPACK,
172 CBLAS and 108 LAPACKE. The latter have all 432 completed ABI/route
observations with independent concrete argument and return-type validation.
Only this successful full artifact replaces the prior 92-entry snapshot.
Focused traces are not published as full-suite evidence.

The subsequent SYEV/HEEV full suite supplies the current 116-entry snapshot;
the counts above preserve the SVD stage's own checkpoint.
