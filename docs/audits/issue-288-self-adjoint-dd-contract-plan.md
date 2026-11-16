# Issue 288: SYEVD / HEEVD contract preparation

Status: high/work DD adapters and workspace diagnostic correction implemented.
Numerical and independent C ABI checks pass; completed-call tracing and optimized
release pass. Immutable full-suite evidence remains pending. Issue #288 remains
open. The preceding 116-entry worktree has passed its immutable full package run.

The extension adds eight canonical names: S/D SYEVD and C/Z HEEVD,
high/work forms, both integer ABIs. The original 1,616-name acceptance scope
and separate 20 Rook coordination targets remain intact.

## Primary references

Reference-LAPACK v3.12.1:

| Source | Git blob |
| --- | --- |
| LAPACKE/src/lapacke_dsyevd_work.c | 1dead1b00bb921bff78cc0a14b3f918de935422d |
| LAPACKE/src/lapacke_zheevd_work.c | f3ae213337986713ce62f5bd3e987030cb7c7338 |
| LAPACKE/src/lapacke_dsyevd.c | 79beddbd6facad59111aed8b74ba895e01fbcbf2 |
| LAPACKE/src/lapacke_zheevd.c | a70e26e10bae491a170449f9769f82e71e52e07d |
| SRC/zheevd.f | 559624bc13671cc351d4a1c567d7bcf99df7d95f |
| SRC/zheevr.f | 56761a5249870c0f544cb0431d816086512686b8 |

The high-level signature matches SYEV/HEEV. Work forms additionally expose
integer IWORK/LIWORK; complex forms also expose real RWORK/LRWORK. A query in
any workspace length activates the query path for all workspaces, without
reading matrix data. Row-major leading-dimension errors precede queries.
Selected-triangle packing and JOBZ-dependent output follow the preceding
SYEV/HEEV layout contracts. Main, real and integer workspace minima and query
rounding need separate validation; do not compare a legal minimum to an optimal
or rounded query size.

## Existing Fortran diagnostic defect

For ZHEEVD, N=3/JOBZ=V/LWORK=32/LRWORK=0/LIWORK=0, the current main DLL
returned INFO=-12 while independent OpenBLAS returned INFO=-10. The reference
checks main WORK, then real RWORK, then integer IWORK. The generated complex
HEEVD and HEEVR entries previously tested IWORK first. The generator now emits
checks in argument order, and the generated Rust changes only those checks.

Regression cases cover complex Float32/Float64 and LP64/ILP64, simultaneous
invalid lengths, each individual auxiliary failure, main-workspace precedence,
and queries activated by either auxiliary length with other lengths zero.
`reproducers/heevd_workspace_order.jl` retains the executed minimal comparison.
This is a Colmat.jl boundary defect; no native repository change is made.

The four regression tests pass, and the full Rust suite passes 420 tests.
The executed minimal comparison now returns INFO=-10 for both Colmat and
OpenBLAS. Those results preserve the earlier diagnostic-only checkpoint.


## Initial DD adapter validation

All eight DD high/work names now export LP64 and ILP64 entry points. Shared
selected-triangle packing retains the preceding SYEV/HEEV behavior. The DD
adapter checks main, real and integer workspace minima separately, in argument
order; a query activated by any workspace length returns without reading A.
Unrepresentable native workspace calculations are rejected before dispatch.

The combined SYEV/HEEV and SYEVD/HEEVD Julia numerical suite passes 77,200 checks
against independently resolved OpenBLAS eigenvalues, with both actual DLL and
LBT routes. Cases include upper/lower and lowercase flags, both integer widths,
N=0/1/3, N/V jobs, padding, guards and imaginary-only Hermitian diagonal NaNs.

The independent C11 worker now exercises all sixteen classic/DD names and both
widths. Auxiliary-only queries with other lengths zero, exact first-invalid
workspace diagnostics, and guarded integer query outputs pass. Initializing
integer query guards to -99 exposes partial 32-bit writes into an ILP64 value.
Rust's full suite passes 428 tests; the evidence fixtures pass 17 tests,
including rejection of incorrect real/integer workspace pointer types.


Completed-call tracing independently validates all 124 current LAPACKE names
and all 496 ABI/route observations; the completed-call gate passes 2,486 checks.
The optimized release repeats the numerical and independent C caller checks
successfully. Registration is 1,606 forwards per integer ABI. Focused traces
remain explicitly non-full-suite results.

The successful 116-entry snapshot is backed up and retained as stale candidate
evidence during the next immutable full package run. Only a successful full
artifact matching all 257 production/test/worker hashes, manifest, registration
and optimized DLL may replace it. The original scope retains 1,492 entries
beyond this 124-entry candidate and the separate 20 Rook coordination targets.
