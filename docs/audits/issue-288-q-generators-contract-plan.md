# Issue #288: explicit Q generation C adapters

Status: canonical scope and reference contract investigation; implementation
and full-suite evidence pending. Issue #288 remains open.

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
