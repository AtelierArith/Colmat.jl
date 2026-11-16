# Issue #288: pivoted QR C adapter contracts

Status: primary contract investigation only; adapters, independent C ABI,
optimized repetition and full-suite proof remain pending. Issue #288 remains open.

Canonical scope contains sixteen S/D/C/Z high/work GEQP3 and GEQPF names.
Following the 324-name compact QR candidate, this would reach 340 original
names, leaving 1276 plus the separate 20 Rook C targets. Candidate counts do
not imply merged-main coverage.

Reference LAPACK v3.12.1 primary sources:
- [complex GEQP3 work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqp3_work.c)
- [complex GEQP3 high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqp3.c)
- [complex GEQPF work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqpf_work.c)
- [real GEQP3](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgeqp3.f)
- [complex GEQP3](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zgeqp3.f)

High signatures are layout,M,N,A,LDA,JPVT,TAU. JPVT is an in/out pointer to
the selected integer ABI, not a scalar or fixed-width pivot array. GEQP3 work
adds WORK,LWORK and, for complex scalars only, real RWORK. GEQPF work adds
WORK and complex-only real RWORK, with no LWORK or query argument. Row LDA<N
returns -5 before Fortran dimension checks. Row queries bypass array packing.
High NaN A diagnostics return -4 before workspace calls. Complex high GEQP3
allocates at least 2*N real elements before its query. Own allocation sizes
must be checked for integer and byte overflow.

GEQP3 minimum work is 3*N+1 for nonempty real matrices and N+1 for complex;
when min(M,N)=0 it is one. Query and execution output WORK(1) need separate
checks: reference may report implementation-dependent optimal sizes. Execution
still processes initial JPVT flags even at M=0. Negative and non-unit flag
values mean fixed columns. Tests must verify fixed-column ordering, a valid
one-based output permutation, A[:,JPVT]=Q*R and mutual independent Q consumers
rather than require identical free-column pivot choices at norm ties.

Required checks include real/complex, both integer ABIs, both layouts,
high/work and DLL/LBT, empty/tall/wide/deficient and scaled inputs, mixed
fixed/free columns, integer pivot guards and unused TAU/work/padding, negative
INFO precedence, logical NaN boundaries, and ILP64 arguments beyond int32.
Investigate failures with executed MWEs before changing existing Fortran
boundaries. No native colmat-rs edits or PRs are authorized by this stage.
