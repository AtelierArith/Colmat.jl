# Issue #288: GEQRFP C adapters

Status: immutable full package validation passes and source-bound execution
evidence is imported. CI and integration are pending.
Issue #288 remains open. Eight S/D/C/Z high/work names extend the preceding
188-name candidate to 196 original-scope names. The remaining original scope is
1420 names, with the separate 20 Rook C targets still additional work.

Reference LAPACK v3.12.1
[GEQRFP work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqrfp_work.c)
has the same C argument types as GEQRF work, but its numerical contract requires
real nonnegative R diagonals. The
[Fortran implementation](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zgeqrfp.f)
also sets minimal and queried WORK to one whenever min(M,N)=0. This differs
from ordinary GEQRF's N-based minimum/query. For nonempty GEQRFP, minimum WORK
is N. Row-major LDA diagnostics remain argument 5 before dimensions; insufficient
WORK is -8 and high-level NaN A is -4. Layout conversion shares the preceding
QR adapter and preserves compact Householder factors, complex TAU and padding.

The executed `reproducers/geqrfp_empty_workspace.jl` exposed the existing shared
Fortran boundary returning query 3 and rejecting LWORK=1 at M=0,N=3, while
independent OpenBLAS returns/accepts one. The shared blocked export now selects
the one-element empty minimum/query only for positive-diagonal variants.
Ordinary GEQRF retains its own minimum/query. No native upstream source changed.

Focused independent comparisons pass 51,264 Julia assertions for real/complex,
both integer ABIs and layouts, high/work and DLL/LBT, empty/scalar/tall/wide,
deficient and small/large-scale inputs. Native factors are consumed by reference
ORGQR/UNGQR and reference factors by native Q generators; reconstruction,
orthogonality and real nonnegative diagonals are checked. Exact empty query checks pass in the traced run.

An independent C11 caller verifies all eight positive names with explicit
prototypes, reconstruction, diagonal sign, complex TAU, guards, minimal work,
empty query exactly one, diagnostic positions, NaN policy and ILP64 dimensions
outside int32; it also repeats the baseline GEQR2 checks. Python execution
fixtures pass 17 tests, including rejected TAU scalar and LWORK integer types.
Typed completed-call tracing covers all 196 candidate names with 784 completed
observations across both ABIs and routes; its call-site gate passes 3,926 checks.
The Rust suite passes 460 tests. This focused evidence explicitly records
`full_suite_passed=false`. Optimized repetition also passes the 51,264 numerical assertions, 3,926 call-site
checks and all 784 typed observations. The subsequent immutable full package run passes and its source-bound
evidence is imported.

Optimized DLL SHA-256: `004dc672b0e3b06f92a94973e254267f86aa6d8cd84ff2e429cde272883bd707`.

The terminal-zero full Pkg.test run is bound to source commit
`6abf88063a921606f7ea5fb6192a357cdf29a3b0` and all 288 frozen source files.
HEAD, every source, optimized DLL, manifest and registration remained unchanged.
The imported proof validates 1474 canonical entries: 1106 Fortran LAPACK,
172 CBLAS and 196 LAPACKE, with 5275 completed witnesses. Python 28,
inventory 15 and documentation build checks pass. CI and main integration
remain required.

Main integration after PR #390 preserves every production source and the
optimized GEQRFP DLL. The inherited QR C worker and positive-QR C worker now
use explicit outer initialization-loop braces for GCC -Werror. Both independent
C runners pass again. These two test-source changes require a new immutable
full run before current-source evidence can be published; the earlier source
proof remains historical and is not claimed for the changed fixtures.
