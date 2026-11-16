# Issue #288: GEQRF / GEQR2 C adapters

Status: adapters and focused numerical/C ABI checks pass. No full-suite
evidence has been imported. Issue #288 remains open.

Sixteen high/work S/D/C/Z names extend the preceding 172-name candidate to 188
original-scope names. The original 1616-name scope retains 1428 names beyond
this candidate, plus the separate 20 Rook C targets.

Reference LAPACK v3.12.1
[GEQRF work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqrf_work.c)
and [GEQR2 work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqr2_work.c)
use C-int layout, lapack_int dimensions/INFO and scalar-specific A/TAU/WORK.
GEQRF work has LWORK and supports queries; GEQR2 work has only the WORK pointer.
High-level calls own initialized scratch. Row LDA is checked against N at
argument 5 before dimensions; column LDA is max(1,M). GEQRF minimum WORK is
max(1,N), with C argument 8 for insufficient length. High-level NaN A is -4.

Row conversion preserves all compact Householder entries and copies TAU without
conjugation. Queries and empty calls do not pack/read A. Negative INFO does not
copy row scratch back. Padding and TAU bounds require independent guards.

The executed `reproducers/geqrf_empty_workspace.jl` exposed an existing own
Fortran empty-result difference: M=0,N=3 returned WORK(1)=3 versus independent
OpenBLAS 1. The blocked boundary now sets WORK(1)=1 after validation/query for
empty execution. Unblocked empty calls return before the complex path allocates
an unused N-element temporary. Shared positive-diagonal variants receive the
same empty-return correction and need their existing regressions repeated.

Rust numerical checks reconstruct Q from stored Householder vectors and TAU,
then check A≈QR and QᴴQ≈I for real/complex, both integer ABIs and layouts,
high/work, empty/scalar/tall/wide/square and deficient matrices. They also cover
query immutability, minimum WORK, empty WORK(1) and poisoned padding/TAU guards.
Independent OpenBLAS and cross-consumer factor checks pass 78,336 Julia
assertions, including scalar/empty/tall/wide/square and deficient inputs, small
and large scales, both integer ABIs, scalar types, layouts, high/work and
DLL/LBT. Own factors are consumed by independent ORGQR/UNGQR; independent factors
are consumed by the own routines. Exact native/reference pointer distinction,
padding, TAU and WORK guards, query preservation and empty WORK(1) are checked.

Independent C11 declarations pass all sixteen names, including reconstruction,
orthogonality, complex TAU, distinct work signatures, argument diagnostics, NaN
A and ILP64 arguments outside int32. Rust's full suite passes 460 tests and
Python's execution fixtures pass 17. Standard and compact QR regressions,
including positive-diagonal variants, also pass after the shared empty fix.
The independent C Julia runner passes two checks. Completed-call tracing
validates all 188 names and 752 name/ABI/route observations, with 3766 gate
checks. Focused witnesses remain explicitly full_suite_passed=false.
Immutable full-suite evidence remains required.

Optimized build and repeated numerical, independent C and signature checks pass.
The frozen optimized DLL SHA256 is
`cd24fc168ff8f788a1ca4b6dae89bb7822b9c4694ba2450dcf5bced8e09cae21`.
Registration is 1670 forwards per integer ABI. Full-suite proof remains pending;
focused evidence is not imported as a full-suite snapshot.
