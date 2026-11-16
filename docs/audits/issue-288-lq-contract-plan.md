# Issue #288: GELQF/GELQ2 C adapters

Status: own Fortran empty-work correction and sixteen C adapters pass focused
independent Julia checks. C caller, tracing and full-suite evidence remain pending.
Issue #288 remains open. The sixteen S/D/C/Z high/work names extend 196 candidate names to 212, leaving
1404 original-scope names plus the separate 20 Rook C targets. These staged
candidates are not all merged yet.

Reference LAPACK v3.12.1 [DGELQF](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgelqf.f)
requires WORK length one for empty matrices and returns query/output WORK(1)=1.
For nonempty input the minimum is M. Its
[C work adapter](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dgelqf_work.c)
checks row-major LDA against N before allocating or querying, translates
negative Fortran INFO by one, and queries using temporary column-major LDA.

Executed `reproducers/gelqf_empty_workspace.jl` at M=3,N=0 against the preceding
candidate DLL and independent OpenBLAS. The own Fortran boundary returns query
3 and rejects LWORK=1 with INFO=-7 while leaving WORK=777. OpenBLAS returns
query 1 and accepts LWORK=1 with INFO=0 and WORK=1. This is a Colmat.jl boundary
correction; no upstream native source changes are needed for this reproducer.

Required checks include both integer ABIs/layouts, high/work and direct/LBT,
compact factors consumed independently by ORGLQ/UNGLQ, reconstruction and
orthogonality, complex TAU and conjugation, empty/rectangular/scaled/deficient
matrices, NaN policy, diagnostics, padding and independent C prototypes.
Completed-call tracing, optimized runs and immutable full package evidence
are required before publishing an implementation PR.

The corrected own Fortran boundary uses minimum WORK=1 when N=0 and returns
query/output WORK(1)=1 for either empty dimension. Its nonempty minimum remains
M. All four Reference LAPACK scalar implementations use this same empty rule.
The reproducer now matches independent OpenBLAS. The new 576 assertions compare
all four scalar types and both own integer ABIs with independent ILP64 OpenBLAS,
including query, invalid zero WORK, minimal/oversized WORK, unchanged A/TAU and
WORK guards. Existing LQ testsets also pass (240/480/544/264/176/32/144).

The C adapters share checked geometry, high-level NaN policy and layout copying
with ordinary QR, with separate M-based WORK checks for blocked LQ. GELQ2 has
only a WORK pointer and owns an M-sized high-level scratch array. Row queries
avoid reading A, transpose-back skips negative INFO, and empty blocked queries
return one. Independent factors are mutually consumed by ORGLQ/UNGLQ to verify
A=LQ and Q orthogonality. Numerical checks pass 78,912 assertions for all four
scalar types, both integer ABIs and layouts, direct/LBT and high/work, empty,
rectangular, deficient and very small/large inputs. Python signature fixtures
pass 17 tests with distinct GELQF/GELQ2 work signatures. No full-suite evidence
has been imported for this candidate.
