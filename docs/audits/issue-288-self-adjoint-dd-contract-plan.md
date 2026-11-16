# Issue 288: SYEVD / HEEVD contract preparation

Status: workspace diagnostic correction implemented; LAPACKE DD adapters and
their numerical/C ABI/full-suite evidence remain pending. Issue #288 remains
open. The preceding 116-entry worktree is frozen during its full package run.

The planned extension adds eight canonical names: S/D SYEVD and C/Z HEEVD,
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
OpenBLAS. LAPACKE DD exports remain unimplemented at this checkpoint.
