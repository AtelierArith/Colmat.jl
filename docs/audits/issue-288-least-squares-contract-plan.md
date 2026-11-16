# Issue #288: least-squares C contracts

Status: all 32 planned C entries pass focused debug numerical checks and an
independent C caller. Full-stage policy, typed tracing, optimized repetition
and immutable full proof remain pending.
Issue #288 remains open.

The canonical inventory contains eight high/work S/D/C/Z names each for
GELS, GELSY, GELSS and GELSD: 32 entries in total. Following the staged
388-name pentagonal candidate this would reach 420 original names, leaving
1196 plus the separate 20 Rook C targets. These are staged counts, not
merged-main coverage. This branch will integrate its parent's final frozen
stress tests and evidence before publication.

Primary contracts use Reference LAPACK v3.12.1:

- [GELS work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgels_work.c): layout,TRANS,M,N,NRHS,A,LDA,B,LDB,WORK,LWORK;
  row LDA<N and LDB<NRHS return -7 and -9 before numerical headers.
- [GELSY work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgelsy_work.c): layout,M,N,NRHS,A,LDA,B,LDB,JPVT,RCOND,RANK,WORK,LWORK;
  complex adds real RWORK. JPVT and RANK follow the selected lapack_int;
  RCOND is a by-value real scalar. Row LDA/LDB errors are -6/-8.
- [GELSS work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgelss_work.c): replaces JPVT with real singular values S;
  complex adds real RWORK. RCOND and RANK retain their respective contracts.
- [GELSD work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgelsd_work.c): adds selected-width IWORK and, for complex types,
  real RWORK. Multiple workspace outputs and minima require independent checks.

Logical A is M-by-N; B storage has max(M,N) rows and NRHS columns, independent
of TRANS. The input rows that are mathematically read, residual rows on output,
minimum-norm solutions, queries, insufficient work and empty dimensions must
be tested separately before selecting any packing or NaN policy. Queries
must not pack or modify A/B/JPVT/S/RANK. High-level temporary allocations must
use checked arithmetic and preserve independent leading dimensions/guards.

The existing Colmat.jl Fortran drivers are in deps/rust/src/least_squares.rs.
GELS uses QR/LQ; GELSY uses complete orthogonal factorization; GELSS/GELSD
share an existing SVD solver. Exposing those contracts does not prove a
separate dedicated divide-and-conquer implementation for GELSD. Keep algorithm
classification distinct from successful numerical and C ABI execution.

Executed MWE reproducers/least_squares_empty_workspace.py compares the existing
native optimized DLL with independent OpenBLAS 0.3.30. All 384 observations
(192 own/reference pairs) agree on INFO and RANK. For GELS/GELSY/GELSS with
LWORK=0, native WORK remains sentinel 777 while the reference publishes its
query value before reporting insufficient work. This is a publication-order
difference. The proposed improvement publishes the native implementation's
own query value after validating ordinary headers and before rejecting WORK;
it does not copy another implementation's optimal size. Nonempty optimum,
RWORK and IWORK size differences alone are implementation observations.
Selected real/complex observations and exact source/library/reproducer/output
hashes are in issue-288-least-squares-workspace-reproduction.json. No new
least-squares C numerical result or full-suite proof is claimed.

Required evidence includes independent residual/minimum-norm solutions,
rank-deficient and scaled inputs, fixed/free pivot constraints, singular values,
A/B overwrite semantics, real/complex TRANS, both integer ABIs/layouts/routes,
high/work, empty and NRHS=0 cases, formal minima versus queries, publication
and INFO priorities, const/read regions, independent C declarations, typed
completed calls, debug/optimized repetition and immutable full package proof.
Native tensor4all/colmat-rs edits and PRs are not authorized; upstream reports
must contain an executed MWE and improvement proposal only.

The Colmat.jl Fortran boundary now publishes its own query value after ordinary
header validation and before rejecting insufficient LWORK for GELS, GELSY and
GELSS. Repeating the MWE confirms all 48 insufficient-work cases publish that
value, with all 192 native/reference pairs still matching INFO and RANK. Rust
passes 461 tests, including a new real/complex, LP64/ILP64 regression that checks
query/error publication, buffer guards and earlier-header output preservation.
Exact hashes are in issue-288-least-squares-workspace-corrected.json. This is
focused Fortran boundary evidence only.

Eight GELS C high/work entries are drafted with checked layout headers,
query/error bypass of row packing and distinct RHS input/solution extents.
High-level NaN scans follow the complete reference A/B storage contract;
numerical work reads only input RHS rows. C numerical validation is running.
The other 24 planned C entries and full-stage evidence remain pending.

Initial GELS C testing exposed a debug abort in the shared row_to_column
helper: a valid empty matrix computed unsigned M-1 before copying zero entries.
The helper now returns an empty allocation before any span arithmetic or read.
Two regressions cover empty/null conversion and actual high/work GELS C calls
with zero dimensions/NRHS, both layouts, all scalar types and integer ABIs,
checking zero RHS output and caller guards. Rust passes 463 tests. Exact
before/after library, source and log hashes are recorded in
issue-288-least-squares-empty-layout.json. The larger GELS C numerical test is
rerunning after this correction; no passing numerical result is claimed yet.

GELS C numerical checks now pass 25088 assertions, covering all scalar types,
integer ABIs, layouts, high/work and actual DLL/LBT routes; empty/NRHS=0,
tall/wide/square and zero systems; query/error output preservation; A overwrite,
solution/residual comparisons and padding guards. The original random systems
are retained. A Float32 forward-comparison failure is checked with a wide
independent Fortran oracle and a condition-dependent forward bound, while
normal-equation residual checks use a stricter 100*epsilon scaled bound.
Both low-precision solvers are compared separately with the wide oracle.
Source/library/log hashes are in issue-288-least-squares-gels-focused.json.
These are focused eight-name results, not full 32-name stage validation.

Eight GELSY C entries now pass 66240 focused debug assertions across all
scalar types, both integer ABIs/layouts, high/work and actual DLL/LBT calls.
Independent Fortran and wide-precision references check rank, minimum-norm
solutions and normal-equation residuals; native Fortran output comparisons
check A overwrite. Fixed/free/all-fixed columns, full/zero/rank-one systems,
empty and NRHS=0 cases, query/error preservation, selected-width JPVT/RANK,
real-valued RCOND, complex RWORK and guards are covered. Rust passes 463
checks after adding these entries. Exact hashes are in
issue-288-least-squares-gelsy-focused.json. GELSS/GELSD and full-stage proof
remain pending; these results do not establish a completed 32-name stage.

Sixteen GELSS/GELSD C entries pass 138528 focused debug assertions for
singular values, rank and minimum-norm solutions, across scalar types, integer
ABIs, layouts, high/work and DLL/LBT. Full/zero/rank-one, empty and NRHS=0
inputs, powers +/-40 or +/-400, query/error publication, selected-width IWORK,
complex RWORK, guarded S/RANK and overwrite regions are covered. NRHS=0
SVD execution still converts A, unlike the early-return GELS/GELSY paths.
Rust passes 463 tests; the independent C runner passes both checks with all
32 declarations, analytic one-column solutions, transpose/conjugate GELS,
auxiliary workspaces, guards and signed ILP64 dimensions beyond int32.
Exact hashes are in issue-288-least-squares-svd-focused.json.

A separate executed MWE, reproducers/gelsd_zero_rhs.py, uses a known diagonal
matrix with singular values (2,1) and legal NRHS=0. All four native GELSD
variants return INFO=0/RANK=2; independent GELSS agrees. The independent
OpenBLAS GELSD query succeeds but execution returns INFO=-4 from LALSD and
leaves RANK untouched. This is recorded in
issue-288-gelsd-zero-rhs-reference.json. The
[primary contract](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zgelsd.f)
permits NRHS=0 and still specifies S/RANK. Keep these numerical cases and
use independent GELSS as their oracle; other GELSD cases retain independent
GELSD comparison. A reference improvement should handle zero RHS without
passing an invalid zero count into LALSD while preserving S/RANK. No upstream
issue or PR is created by this stage. Earlier pending-SVD statements are
superseded by these focused results; full-stage validation remains required.
