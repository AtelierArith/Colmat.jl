# Issue #288: pivoted QR C adapter contracts

Status: merged into main in PR #397 (e4b862143aaf4cc6b39cb59b78c1997d13b127ef).
CI 43938478349 (Linux/macOS) and Documentation 43938477818 passed. Both
downloaded artifacts validate 318 source hashes, 1618 canonical entries and
5927 witnesses at test merge 449268c3f122b3566f5d5eda3856366bdab36f69.
Its parents and tree match the reviewed head and base. The merged tree,
verified optimized DLL and Python 28 post-merge checks pass. Local inventory
verification remains live in macOS dyld startup.

Issue #288 remains open.

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

Executed `geqp3_error_workspace.jl` at M=N=1,LWORK=0: own INFO=-8 left
WORK(1)=777 while independent OpenBLAS returned -8 and published a workspace
size. The own boundary now publishes its own optimal value after dimension
validation and before the LWORK rejection. Optimal values depend on the
implementation and are not required to equal OpenBLAS. Independent guarded
checks pass 688 assertions across all scalar types and both own ABIs,
including empty queries and insufficient work, with no A/TAU/JPVT/RWORK
changes. Expected thread-local diagnostics are asserted and consumed at
the origin. This is focused evidence only, not full validation.

The sixteen adapters are implemented. Row packing preserves JPVT and TAU
pointers of the caller ABI, and insufficient-work paths publish WORK without
packing A. Complex RWORK uses the underlying real scalar type; checked high
allocations distinguish GEQP3 query scratch from fixed GEQPF work. Focused
numerical checks pass 211680 assertions across all scalar types, integer ABIs,
layouts, high/work and DLL/LBT. Independent OpenBLAS factors feed own Q
generators and own C factors feed reference Q generators. Fixed/free column
patterns (including negative flags), output permutation, reconstruction,
orthogonality, empty/rectangular/deficient and scaled inputs and guards pass.
Signature validation fixtures pass 17 tests, rejecting wrong pivot widths,
complex RWORK types and a spurious GEQPF LWORK. Independent C, policy, typed
completed calls, optimized repetition and full proof remain required.

Policy checks pass 1336 assertions for header precedence, signed ILP64
values beyond int32, insufficient-work publication and consumed diagnostics,
NaN rejection without mutation and ignored NaN padding. The independent C11
caller passes with explicit real/complex high/work prototypes for all sixteen
names and both integer ABIs. It checks mixed fixed columns, guarded integer
pivots, explicit Householder-product reconstruction and orthogonality, real
diagonals, workspace queries/errors, empty/rectangular cases and ILP64 values.
Rust 460 tests pass after the adapter additions. Typed tracing, optimized
repetition and immutable full-suite proof remain pending.

Optimized repetition passes 211680 numerical assertions, 1336 policy checks,
the independent C caller and the 6806-call gate. Typed tracing covers all
340 candidate names with 1360 completed ABI/route observations and records
full_suite_passed=false. The optimized DLL SHA-256 is
a92e83c58f003c6cc3749a21662e431ae7aa6c591194e740d6496105fb92412a.
Parent integration preserves all 318 execution-source hashes. Fresh immutable
full-suite proof is required before publishing this extended scope.

Current full proof is bound to 09aedc796c0348e37486583cc03636b60d5ecdb2 and all 318 frozen
execution sources. Cloud job success, ZIP/TOML digests, complete source map,
manifest, registration and all typed witnesses were verified before import.
The snapshot validates 1618 canonical entries with 5927 completed witnesses.
A separate publication tree preserves the original local full-suite source
and HEAD. Its copied optimized DLL remains byte-identical to the frozen DLL.
The published full run platform is Linux; final Linux/macOS PR CI remains
required. Provenance is stored in the adjacent cloud-provenance JSON.

Publication checks pass Python 28, inventory 15 and documentation build.
All execution-source hashes and the copied optimized local DLL remain
unchanged. Preceding staged candidates and final PR CI/main merge remain
required; this does not claim merged-main coverage for the candidate.
