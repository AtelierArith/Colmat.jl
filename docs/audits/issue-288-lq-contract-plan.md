# Issue #288: GELQF/GELQ2 C adapters

Status: current immutable full package validation passes; source-bound
evidence is imported. CI and main integration remain pending.

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
pass 17 tests with distinct GELQF/GELQ2 work signatures. The subsequent full-suite evidence for this candidate is imported.

An independent C11 caller verifies all sixteen names, real/complex reflector
reconstruction, orthogonality, empty/minimal workspace, guarded TAU/WORK,
diagnostics, ignored padding, NaN policy and ILP64 values outside int32. Its
Julia runner passes two checks. Tracing validates all 212 candidate names with
848 completed typed name/ABI/route observations. Focused evidence explicitly
records `full_suite_passed=false`; it is not a full-suite snapshot.

Optimized repetition passes the numerical, independent C and all 848 typed
observations. The call-site gate passes 4246 checks. The optimized DLL SHA-256
is `87a0030b69a5044a1ea4df1c9f15e7e477d18feffa5759bb25f385be8558b146`.
Registration is 1694 forwards per integer ABI. The subsequent immutable full package run passes; source-bound evidence
is imported.

Full-suite proof is bound to `616d98a1220c26350e9f6d044f155584b59a9b30`
and all 292 frozen sources, manifest, registration and optimized DLL. The
imported proof validates 1490 canonical entries: 1106 Fortran LAPACK, 172 CBLAS
and 212 LAPACKE, with 5351 completed witnesses. Python 28, inventory 15 and
documentation builds pass. The repository evidence test now expects the
current 212-entry scope; this metadata-only fix preserves every full-suite
production/test source hash. CI and main integration remain required.

The inherited QR, positive-QR and LQ C fixtures now use explicit outer loop
braces for GCC -Werror. LQ invalid-work probes assert and consume their own
thread-local diagnostics at the origin, preventing contamination of later
tests. The LQ suite and all three independent C runners pass again. These
test-source changes require fresh immutable full evidence; the earlier
616d98a proof is historical and is not claimed for the updated source set.

Current immutable full proof is bound to `b42c920fd6c69c96e5d2515b35e51b17aeb08f37`
and all 292 frozen execution sources. HEAD, optimized DLL, manifest and
registration remained unchanged through terminal-zero Pkg.test. The imported
proof validates 1490 canonical entries (1106 Fortran LAPACK, 172 CBLAS,
212 LAPACKE), with 5351 completed witnesses and 1694 forwards per ABI.
The corrected portable fixtures and diagnostic isolation are covered by this
current proof. CI artifacts and main integration remain required.
