# Issue #288: generalized QR/RQ and constrained least-squares C contracts

Status: all 32 new C entries pass focused debug numerical, diagnostic policy
and independent C ABI checks. Rust 463 and Python ABI fixtures 17 pass.
Full-stage typed tracing, optimized repetition and immutable full package
proof remain pending.
Issue #288 remains open.

Canonical scope: S/D/C/Z high/work GGQRF, GGRQF, GGLSE and GGGLM (32 names).
After the staged 420-name least-squares candidate this would reach 452 names,
leaving 1164 original-scope names plus separate 20 Rook C targets. These are
candidate counts, not merged-main coverage.

Pinned Reference LAPACK v3.12.1 high/work C sources for all four scalar types
are saved in /tmp/colmat-288-generalized-qr-reference, with source URLs and
SHA-256 hashes in provenance.json. Primary complex work contracts:
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zggqrf_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zggrqf_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgglse_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zggglm_work.c

GGQRF takes layout,N,M,P,A,LDA,TAUA,B,LDB,TAUB, then WORK,LWORK for work.
A is N-by-M and B N-by-P; row LDA<M is -6, row LDB<P is -9. GGRQF uses
layout,M,P,N with A M-by-N and B P-by-N; both row strides require N. It has
the same argument positions. High-level A/B NaNs report -5/-8, TAUA/TAUB
are output. Both complex TAU arrays must match LAPACK reflector conventions.

GGLSE takes layout,M,N,P,A,LDA,B,LDB,C,D,X, then WORK,LWORK for work.
A is M-by-N, B P-by-N, C length M, D length P and X length N. Both row
strides require N; errors are -6/-8. High NaN order is A(-5), B(-7), C(-9),
D(-10), with X output only. Feasibility requires 0<=P<=N<=M+P.

GGGLM takes layout,N,M,P,A,LDA,B,LDB,D,X,Y, then WORK,LWORK for work.
A is N-by-M, B N-by-P, D length N, X length M, Y length P. Row strides
require M/P and report -6/-8. High NaN order is A(-5), B(-7), D(-9), with
X/Y output only. Feasibility requires 0<=M<=N<=M+P.

Existing Fortran compositions are generalized_qr.rs and
constrained_least_squares.rs in Colmat.jl's Rust boundary. Pair minimum WORK
is max(1,first,second,third). Driver minimum is 1 for zero solution order,
otherwise first+second+P, with checked addition. Query, ordinary header and
insufficient-WORK publication order need an executed MWE: pair_driver writes
WORK before its ordinary header checks, unlike the constrained driver.
No correction is justified by inspection alone. Do not compare implementation
optimal sizes as though they were required to equal OpenBLAS sizes.

Required tests: all scalar types, integer ABIs, layouts, high/work and actual
DLL/LBT calls; independent factor reconstruction with external Q consumers;
complex conjugation; equality constraints, normal residuals and minimum Y
norm; exact INFO=1/2 rank failures; destructive A/B/C/D payload; query/error
output preservation; selected-width signed dimensions beyond int32; NaN
order and ignored output NaNs; empty dimensions and padded/guarded arrays;
independent C prototypes; typed completed calls; debug/optimized repetition;
full immutable Pkg.test evidence. Keep composition and algorithm assessments
separate from C ABI execution.

Native tensor4all/colmat-rs edits and PRs are not authorized. Upstream reports
may contain only an executed MWE and improvement proposal.

Executed guarded Fortran workspace probe (128 observations) now confirms
GGQRF/GGRQF native and independent OpenBLAS agree on INFO for query,
insufficient WORK, negative first dimension and bad LDA, across all scalar
types and both native integer ABIs. Both implementations publish their own
WORK query value even before ordinary-header errors in these cases. This
publication order must be preserved; no correction is warranted. Native
optimal WORK=4 and reference WORK=128 are implementation choices. Integer
diagnostics are consumed and recorded at origin, and INFO/WORK guards remain
untouched. Evidence is preparation only, not new C adapter execution.

The 32 adapter exports compile in debug mode. The independent C caller also
compiles under C11 -Wall -Wextra -Werror; runtime is pending in dyld startup.
The pending runtime and full least-squares suite are preserved. Do not treat
successful compilation as numerical or completed-call evidence. The executed
Fortran workspace probe is stored in issue-288-generalized-qr-workspace.json.

The original independent executable C caller completed successfully after
macOS dyld startup resumed. It checks all 32 high/work declarations, all four
scalar types, both integer ABIs/layouts, query preservation, output guards,
negative ILP64 dimensions and analytic one-dimensional complex constrained
solutions. Pair factors are smoke-checked only in this caller; reconstruction
and orthogonality require the separate independent Q-consumer tests. Exact
source/library/log hashes and limitations are recorded in
issue-288-generalized-qr-c-abi.json. Python ABI evidence fixtures pass 17 tests,
including all 32 exact prototypes and wrong layout/WORK/scalar-type rejection.
The full 452-name focused tracing run is currently live; no completion is
claimed. Earlier pending-C-caller statements are superseded by this result.

Focused debug evidence now passes 178560 independent pair reconstruction,
130368 constrained-driver and 6656 policy assertions. An original Float32
fully constrained case exposed a test-conditioning error: B has condition
about 27463, while [A;B] has condition about 18. Native C and Fortran outputs
agree; independent single precision also differs from the wide solution.
The exact original matrices are preserved in
reproducers/generalized_constraint_conditioning.jl. Independent wide LAPACK
confirms scaled backward errors about 6.09e-8 native and 2.17e-8 reference.
Condition estimates now use independent wide SVD for the constraints,
projected free variables and coupling. Both single-precision solvers are
checked against the wide solution, and backward residual bounds are ten times
stricter than the original tests. All original matrices and random seeds are
retained. The previous full-stage trace was interrupted after this demonstrated
failure and its temporary assets restored. It will be repeated after the
correction; no interrupted trace or focused run is a full-suite proof.
Exact source/library/log hashes and limits are recorded in
issue-288-generalized-qr-debug-numerical.json.
