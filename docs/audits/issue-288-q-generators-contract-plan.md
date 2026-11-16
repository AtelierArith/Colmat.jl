# Issue #288: explicit Q generation C adapters

Status: 32 C adapters pass focused independent numerical and policy checks.
Independent C and typed tracing also pass; optimized and full-suite proof
remain pending.
Issue #288 remains open.

The canonical inventory contains 32 high/work names across real ORGQR/ORGLQ/
ORGQL/ORGRQ and complex UNGQR/UNGLQ/UNGQL/UNGRQ. It contains no corresponding
unblocked ORG2R/UNG2R/ORG2L/UNG2L C names. This stage follows the 228-name
factorization candidate and would bring the candidate to 260 names, leaving
1356 original-scope names plus the separate 20 Rook C targets. Staged counts
are not merged-main counts.

Reference LAPACK v3.12.1
[ORGQR work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dorgqr_work.c)
uses layout,m,n,k,A,lda,const TAU,WORK,lwork; high omits WORK/lwork.
Row-major LDA<N returns -6 before querying or allocation, and row queries call
the Fortran routine using temporary column-major LDA=max(1,M) without reading
A. Negative Fortran INFO shifts by one. QR/QL produce orthonormal columns,
while LQ/RQ produce orthonormal rows; valid rectangular dimension/K bounds and
family-specific WORK minima must be checked independently.

The high-level
[UNGRQ implementation](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zungrq.c)
scans the entire M-by-N matrix for NaNs, returning -5, and K TAU entries,
returning -7. This differs from the numerical generator's ignored triangular
storage: high-level NaN checks and low-level numerical reads need separate
tests. Const TAU, layouts, padding and guarded outputs must be preserved.

Required evidence includes independently produced reflectors for all four
orientations, full/thin Q and Q reconstruction, real/complex types, both integer
ABIs, both layouts, direct/LBT and high/work, empty dimensions, K=0/partial/full,
ignored storage, active/inactive NaNs, minimal/query WORK, diagnostic precedence,
independent C prototypes and ILP64 dimensions outside int32. Completed typed
calls, optimized repetition and immutable full package proof are required
before publishing an implementation PR. No full-suite proof is claimed.

The implementation shares checked row/column packing while validating the
extra K argument and orientation-specific rectangle bounds. Row-major LDA
returns -6 before dimensions. High-level A and TAU NaN checks return -5/-7;
low-level generators retain their numerical ignored-storage contract. Const
TAU and caller padding are preserved. Independent OpenBLAS output comparisons
pass 33888 assertions for QR/LQ/QL/RQ, all scalar types and both integer ABIs,
layouts, high/work and DLL/LBT, thin/full matrices and K=0/partial/full.
Analytic unitary reflectors include complex TAU with nonzero phase. A further
1216 checks verify diagnostics, ignored diagonal NaNs, inactive TAU NaNs,
K=0 TAU preservation and negative-INFO output retention. Python execution
fixtures pass 17 tests and reject wrong K/LWORK integer widths. The numerical
candidate now includes 260 C entries; full-suite proof remains pending.

The independent C11 caller passes both runner checks for all 32 names. It
compares outputs with explicit ILP64 OpenBLAS Fortran prototypes for all
orientations, scalar types, both own ABIs, layouts and high/work, empty/thin/
full shapes and every K through the available extent. It checks const TAU,
padding/WORK guards, query immutability, diagnostic precedence, A/TAU NaNs
and ILP64 dimensions outside int32. Rust full-suite checks pass 460 tests.
Typed tracing, optimized repetition and immutable full proof remain required.

Completed-call tracing verifies all 260 names with 1040 typed name/ABI/route
observations and 5206 gate checks. Focused evidence explicitly records
`full_suite_passed=false`; optimized repetition and immutable full package
proof remain required.
