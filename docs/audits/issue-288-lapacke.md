# Issue #288: LAPACKE connection audit (in progress)

The full acceptance scope remains the 1,616 canonical entries in
`issue-288-lapacke-scope.json`. That file is derived from the original #262
inventory, not the shrinking current set of unconnected entries. This issue
must not be closed upon completing only the initial numerical families.

## Reference contract

Reference-LAPACK tag `v3.12.1`:

- `LAPACKE/include/lapacke.h`, blob `82cc4e6c1e4d29ae81b88342fa07ece9339124bb`.
- `LAPACKE/src/lapacke_dgetrf_work.c`, blob `62e6fa502734cebb5363462c842e839792816d33`.
- `LAPACKE/src/lapacke_dgetrf.c`, blob `6232721b4c595495494b978e4148e00af4de16e8`.

`matrix_layout` remains C `int` in both ABIs. Dimensions, pivot outputs and
returned INFO follow `lapack_int`. Negative Fortran INFO shifts by one for
the layout argument. The row-major GETRF work entry checks LDA before the
underlying Fortran parameter checks; row-major empty matrices permit LDA=0
when N=0. Positive numerical INFO is preserved.

## Initial implementation

`deps/rust/src/lapacke.rs` introduces GETRF work entry adapters for the four
scalar types and both integer ABIs. Column-major dispatches to this adapter's
existing native-backed Fortran entry. Row-major copies logical entries to a
column-major temporary, factors there and copies back without touching row
padding. Transpose allocation failure/address overflow returns -1011.

GETRF high-level entries now implement optional NaN checking. The shared atomic
policy defaults to enabled, reads `LAPACKE_NANCHECK` only on first getter access,
and allows explicit setter overrides. NaNs in either complex component count;
padding is excluded. The reference's scan-before-numerical-validation ordering
and bounded leading-dimension scan are preserved. Reference source:
`LAPACKE/src/lapacke_nancheck.c`, `LAPACKE/utils/lapacke_dge_nancheck.c`
(blob `c47fc5b9de76bad64a529e8c41ab6f8e30e51ba8`) and complex counterpart
(blob `94cb051e0d2a7bc06293b617e8c197aeacceb360`).

LBT automatically forwards these exported entry names when this worktree's
library is loaded. NaN getter/setter aliases with `64_` retain C-int flags and
return values and share the unsuffixed helpers' policy state. These helpers
are additions outside the frozen 1,616 scope; they do not replace scope entries.
Inventory assets and source-bound full-suite execution evidence are still
pending and must be regenerated before a PR can be marked ready.

Other families require their own workspace, layout and callback contracts;
GETRF conversion is not a blanket implementation for them.

## Remaining gates

- Independent C ABI calls and actual LBT calls for every new registered entry.
- Source-bound execution evidence for both integer ABIs and dispatch routes.
- High-level versus work entry tests, NaN policy and argument precedence.
- Layout/stride/padding, complex data and external Fortran interoperability.
- Full Rust and Julia regressions, documentation and inventory regeneration.
- Remaining full frozen scope, including numerical dependencies #306/#313.

Native numerical revision: `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

## Local verification

The targeted Rust run passed all nine GETRF work tests, including nonzero imaginary matrix entries for both complex precisions and
ABIs. Tests reconstruct P*A=L*U with scalar arithmetic, compare padded row/column
results and pivot indices, preserve padding, propagate singular INFO and reject
invalid/overflowing dimensions before dereferencing null buffers. This is
adapter-level evidence; independent C and LBT evidence is described below.

The full local Rust suite passed: 398 unit tests, zero failures; formatting
checks passed. Full Julia regression and snapshot regeneration remain pending.

The independent C worker checks all 16 GETRF high/work scalar/ABI exports,
row/column layouts, padded storage, guarded pivot arrays and singular INFO using
scalar P*A=L*U reconstruction. It also checks high-level NaN error precedence,
imaginary NaNs, ignored padding, setter/getter state, and environment parsing
and one-time initialization in fresh subprocesses. `test/lapacke_abi.jl` passed
six checks with the final shared-state suffixed helper aliases.

`test/lapacke_getrf.jl` passed 720 checks (including helper state/ABI checks): all eight canonical GETRF high/work
entries, both ABIs, direct-library and actual LBT trampoline calls, both layouts,
independent scalar factor reconstruction and padded/guarded output. It asserts
the exact LBT forwarding pointer before every numerical call. These observations
are not yet imported as a source-bound execution snapshot.

After the final helper change, the full Rust suite again passed all 398 unit
tests. The focused Julia/C checks passed 726 tests (720 + 6). Actual registered
counts were 1,490 for each ABI: the prior 1,480 plus eight GETRF canonical entries
and two policy helpers. The documentation inventory still records the previous
main baseline; it must be regenerated alongside source-bound execution evidence.
No full Julia package test or completion claim is made for this intermediate
commit. The full frozen scope remains 1,616 entries.

## GETRS high/work adapters

Reference-LAPACK v3.12.1 sources: `LAPACKE/src/lapacke_zgetrs_work.c`
(blob `b9dde38658e59cb295b9248938e6f46513d14574`) and
`LAPACKE/src/lapacke_zgetrs.c`
(blob `d76e323c14473665e2f5fca1814d0b38f2660cda`).

GETRS high/work entries now cover four scalar types and both ABIs. Row-major
copies the read-only LU factors and mutable RHS independently; only logical RHS
entries are copied back. N/T/C and lowercase flags are passed to the existing
native-backed Fortran solve. Row-major LDA/LDB errors precede numerical argument
checks as in the reference wrapper. High-level NaN diagnostics distinguish the
factor argument (-5) from the RHS (-8).

`test/lapacke_getrs.jl` passed 4,232 checks. Factors and sequential pivot arrays
come from direct calls into the separate ILP64 OpenBLAS library, not Colmat or
its LBT trampoline. The factors are reused for both integer ABIs. RHS data is
constructed by scalar multiplication from a known solution; tests cover real
and nontrivial complex N/T/C, lowercase flags, both layouts and both direct DLL
and actual LBT trampoline routes. Every call asserts its forwarding pointer;
factors and pivots remain unchanged, RHS padding remains untouched.

The independent C worker additionally exercises all 16 GETRS scalar/ABI/high/work
exports with manually specified standard LAPACK LU factors and verifies scalar
solutions, original factor/pivot preservation, empty calls, invalid flags,
row-major leading-dimension precedence, imaginary NaNs, distinct high-level
NaN error indices and a negative ILP64 dimension beyond 32-bit range. Its Julia
runner passed all six checks. The full Rust suite still passed 398 tests.

GETRS adds eight canonical entries to the earlier eight GETRF entries. The
frozen full scope and remaining completion gates are unchanged; focused checks
are not a replacement for the final source-bound full Julia regression.

Combined GETRF/GETRS/C validation in one fresh Julia process passed 4,958 checks
(720 + 4,232 + 6). Observed registration counts were 1,498 per ABI. Inventory
assets still describe main's 1,480 baseline and require regeneration before
full-package validation and PR readiness.

## GESV high/work adapters

Reference-LAPACK v3.12.1 sources: `LAPACKE/src/lapacke_dgesv_work.c`
(blob `ec48c27e12ac4264e1e4ec4b6cf95623ee034440`) and
`LAPACKE/src/lapacke_dgesv.c`
(blob `c1384262418cd3ef72f3ccf00bc5503967e2b586`).

GESV now covers four scalar types, both ABIs and high/work entries. The row-major
adapter allocates both temporary matrices before modifying caller data, calls
the existing native Fortran driver and copies back only logical entries. As in
the reference, row-major leading-dimension errors precede numerical argument
checks. High-level NaN diagnostics distinguish A (-4) from B (-7). NRHS=0 still
factors A and returns sequential pivots and numerical INFO; it does not perform
a blanket no-op. Positive singular INFO is preserved and B remains unchanged.

`test/lapacke_gesv.jl` passed 1,216 checks over direct DLL and LBT trampoline
routes, both layouts, four types and both integer ABIs. RHS values are built
with scalar multiplication from a known real/complex solution. Standard
OpenBLAS GETRS then consumes each produced LU/IPIV pair to solve a second RHS,
verifying the outgoing factor contract. Zero-RHS factorization must match the
ordinary driver's factors and pivots. Singular inputs must preserve B.

The independent C worker additionally checks all 16 GESV exports, including
nonreal complex coefficients, guarded pivot outputs, padding, singular INFO,
zero RHS with NULL B, empty dimensions and row-major error precedence. It checks
high-level NaN diagnostics for A/B and negative ILP64 dimensions outside 32-bit
range. The full Rust suite passed 398 tests.

Combined focused Julia/C validation passed 6,174 checks (720 + 4,232 + 1,216 + 6).
Actual registration counts are 1,506 per ABI: 24 frozen-scope LAPACKE canonical
entries and two additional NaN policy helpers beyond the prior 1,480 baseline.
The frozen issue scope remains 1,616 entries, with no completion claim for the
remaining entries or the still-pending inventory/provenance/full-Julia gates.

## GETRI high/work adapters

Reference-LAPACK v3.12.1 sources: `LAPACKE/src/lapacke_zgetri_work.c`
(blob `93c19c59e8c71d16f307cff6d0ac13fb683c536c`) and
`LAPACKE/src/lapacke_zgetri.c`
(blob `0af2ecfd5aed1df5dd7dc55808a834d3b24ae88c`).

GETRI high/work entries now cover four scalar types and both integer ABIs.
Row-major queries call the existing Fortran query directly with a legal
column-major leading dimension, without reading A/IPIV or allocating a
transpose. Nonquery row-major calls transform the LU factor storage and copy
back only logical entries. Caller workspace is borrowed for work entries;
high entries query and allocate their own work, returning -1010 on failed work
allocation. Transpose allocation failures return -1011. The numerical inverse
uses the existing native solve on an identity RHS; no blocked-GETRI performance
or identical workspace recommendation to vendor LAPACK is claimed.

`test/lapacke_getri.jl` passed 1,768 checks using independently generated
OpenBLAS LU factors and sequential pivots. Both DLL and actual LBT trampoline
routes, both layouts, four scalar types, both ABIs, query, minimum N work,
insufficient work (-7), work guards, singular diagonal INFO and padding are
covered. Scalar loops verify both A*inverse(A) and inverse(A)*A. Queries and
invalid workspace leave factors/pivots unchanged; singular inputs retain their
storage.

The independent C worker tests all 16 GETRI high/work exports, known inverses
of nonreal complex matrices, guarded minimum work, NULL A/IPIV on query, empty
calls, singular INFO and high-level NaN INFO=-3. Additional actual ILP64 query
calls use N=4,294,967,297 to detect truncation and N=16,777,217 in Float32 to
verify upward rounding of the returned workspace size. No corresponding
matrix allocation or dereference occurs during these queries.

The full local Rust suite passed 398 unit tests after GETRI was added. Inventory
regeneration, source-bound execution evidence and a full Julia package run
remain pending for the full LAPACKE work. The frozen scope stays 1,616 entries.

Combined focused Julia/C regression passed 7,942 checks (720 + 4,232 + 1,216 +
1,768 + 6). Actual forwarding counts were 1,514 per ABI: 32 frozen-scope
canonical entries and two policy helpers beyond the prior main baseline.

## Dense Cholesky high/work adapters

The additional families are POTRF, POTRF2, POTRI, POTRS and POSV: 40 canonical
entries, 80 scalar/integer-ABI exports. All 40 names were checked against the
full frozen 1,616-entry scope. Reference-LAPACK v3.12.1 work-entry sources:

| Source under LAPACKE/src | Git blob |
| --- | --- |
| lapacke_zpotrf_work.c | a490213a9a106b040e97013c692750bcef28fab2 |
| lapacke_zpotrf2_work.c | 60300e5fc65a2cc5a572ba7805fecaa5a98af430 |
| lapacke_zpotri_work.c | f98a8467f2b196d483f85d7308e73b2182042741 |
| lapacke_zpotrs_work.c | d6e43a69c42b299cbc9db6c16e611c982f8a95cf |
| lapacke_zposv_work.c | b990b5f1b26ae35a76ca36e389ce570dc8d87c28 |

The corresponding high-level entry sources were also inspected, along with
`lapacke_zpo_nancheck.c` (66740ae874951c3e8472d979db95a56a9b2170fe),
`lapacke_ztr_nancheck.c` (5c870a6461ab4237429c9a0851bd6703249971c0),
and the PO/TR transpose helpers. Only the selected triangle is copied or
scanned. High-level NaN scanning includes imaginary diagonal components;
work/Fortran POTRF/POTRF2/POSV ignore those components as the Hermitian
numerical contract requires. NaN scanning respects UPLO and leading dimension.
Row-major leading-dimension validation precedes numerical option validation.
POSV still factors with zero RHS and preserves B on positive failure INFO.
POTRS never writes the supplied factor.

POTRF2 uses the already-documented native Fortran factor alias; no separate
recursive algorithm or reference-LAPACK POTRF2 performance is claimed. Work
entries have no caller work-array arguments for these families. Row-major
transposes and numerical scratch remain internal allocations.

### Existing Fortran boundary fixes discovered by these tests

The direct Fortran adapter previously exposed writes from the public native
in-place Cholesky call into the caller's unselected triangle. An executed
ComplexF64, N=3, LDA=5 MWE changed ignored positions 6, 11 and 12. A separate
executed variant with NaNs only in diagonal imaginary components returned
INFO=0 with nonfinite factor entries. These are LAPACK boundary defects in
Colmat.jl; no change or report was made to colmat-rs.

`factor_cholesky` now copies only the selected triangle into native-owned
scratch, projects the diagonal to its real part, and copies only the selected
factor back, including numerical failure data. Upper storage is conjugated
into the lower Hermitian representation and back. This also fixes the Fortran
POTF2/POTRF2 aliases and POSV through their existing POTRF dispatch. The extra
N-by-N scratch allocation is explicit; zero-copy factorization and matching
vendor performance are not claimed.

`reproducers/potrf_ignored_storage.jl` is an executed standalone regression.
The corrected library returns INFO=0, no changed ignored positions and finite
factor entries. Direct Fortran tests cover four types, both ABIs, upper/lower
and lowercase flags, unused/padding NaNs, imaginary diagonal NaNs, and external
OpenBLAS consumption of the generated factors.

### Validation

`test/lapacke_cholesky.jl` passed 23,968 checks. Tests use scalar construction
and reconstruction of positive-definite matrices, independent OpenBLAS factors
for POTRS/POTRI, external OpenBLAS solves with produced POTRF/POTRF2/POSV factors,
and both actual DLL and LBT trampoline routes. They verify selected-triangle
and padding preservation, zero RHS, failure INFO, B preservation, and policy
versus numerical handling of diagonal imaginary NaNs.

The separate C compiler worker calls all 80 exports with concrete C signatures,
C-int layout/char options and matching 32/64-bit dimensions and INFO. It checks
nonreal complex matrices, poisoned unselected storage, exact known factors,
solutions and inverses, zero/empty calls, row-major validation precedence,
positive INFO, NULL B for zero RHS, imaginary-only diagonal NaNs and high-level
NaN diagnostics. Its Julia runner passed two checks.

After the final boundary correction, combined LAPACKE Julia/C validation
passed 31,912 checks. Separate existing Fortran driver/direct-ABI/boundary
regressions passed 7,200 checks, and the complete Rust suite passed 398 tests.
Actual LBT forwarding counts are 1,554 per ABI: 72 frozen-scope canonical
entries plus the two policy helpers beyond main's 1,480 baseline. The fixed
scope, inventory/provenance gates, full Pkg.test requirement and remaining
1,544 canonical entries are unchanged; #288 is not complete.
