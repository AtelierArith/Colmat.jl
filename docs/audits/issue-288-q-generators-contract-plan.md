# Issue #288: explicit Q generation C adapters

Status: merged to main through PR #394. Full source-bound package proof,
Linux/macOS PR CI, documentation and artifact provenance checks pass.

Issue #288 remains open.

The canonical inventory contains 32 high/work names across real ORGQR/ORGLQ/
ORGQL/ORGRQ and complex UNGQR/UNGLQ/UNGQL/UNGRQ. It contains no corresponding
unblocked ORG2R/UNG2R/ORG2L/UNG2L C names. This stage follows the 228-name
factorization candidate and would bring the candidate to 260 names, leaving
1356 original-scope names plus the separate 20 Rook C targets. Staged counts
are not merged-main counts.

Reference LAPACK v3.12.1
[ORGQR work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dorgqr_work.c)
uses layout,m,n,k,A,lda,const TAU,WORK,lwork; high omits WORK/lwork.
Row-major LDA<N returns -6 before querying or allocation, and row queries call
the Fortran routine using temporary column-major LDA=max(1,M) without reading
A. Negative Fortran INFO shifts by one. QR/QL produce orthonormal columns,
while LQ/RQ produce orthonormal rows; valid rectangular dimension/K bounds and
family-specific WORK minima must be checked independently.

The high-level
[UNGRQ implementation](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zungrq.c)
scans the entire M-by-N matrix for NaNs, returning -5, and K TAU entries,
returning -7. This differs from the numerical generator's ignored triangular
storage: high-level NaN checks and low-level numerical reads need separate
tests. Const TAU, layouts, padding and guarded outputs must be preserved.

Required evidence includes independently produced reflectors for all four
orientations, full/thin Q and Q reconstruction, real/complex types, both integer
ABIs, both layouts, direct/LBT and high/work, empty dimensions, K=0/partial/full,
ignored storage, active/inactive NaNs, minimal/query WORK, diagnostic precedence,
independent C prototypes and ILP64 dimensions outside int32. Completed typed
calls, optimized repetition and immutable full package proof are required
before publishing an implementation PR. No full-suite proof is claimed.

The implementation shares checked row/column packing while validating the
extra K argument and orientation-specific rectangle bounds. Row-major LDA
returns -6 before dimensions. High-level A and TAU NaN checks return -5/-7;
low-level generators retain their numerical ignored-storage contract. Const
TAU and caller padding are preserved. Independent OpenBLAS output comparisons
pass 33888 assertions for QR/LQ/QL/RQ, all scalar types and both integer ABIs,
layouts, high/work and DLL/LBT, thin/full matrices and K=0/partial/full.
Analytic unitary reflectors include complex TAU with nonzero phase. A further
1216 checks verify diagnostics, ignored diagonal NaNs, inactive TAU NaNs,
K=0 TAU preservation and negative-INFO output retention. Python execution
fixtures pass 17 tests and reject wrong K/LWORK integer widths. The numerical
candidate now includes 260 C entries; full-suite proof remains pending.

The independent C11 caller passes both runner checks for all 32 names. It
compares outputs with explicit ILP64 OpenBLAS Fortran prototypes for all
orientations, scalar types, both own ABIs, layouts and high/work, empty/thin/
full shapes and every K through the available extent. It checks const TAU,
padding/WORK guards, query immutability, diagnostic precedence, A/TAU NaNs
and ILP64 dimensions outside int32. Rust full-suite checks pass 460 tests.
Typed tracing, optimized repetition and immutable full proof remain required.

Completed-call tracing verifies all 260 names with 1040 typed name/ABI/route
observations and 5206 gate checks. Focused evidence explicitly records
`full_suite_passed=false`; optimized repetition and immutable full package
proof remain required.

Explicit outer braces now delimit the independent C initialization loops for
GCC -Werror portability; both C runner checks pass again. The incoming parent
fix also consumes QL/RQ diagnostic state between boundary cases. These test
fixes must precede this candidate full-suite freeze.

Optimized repetition after all C portability and diagnostic-isolation fixes
passes the numerical, policy, independent C and all 1040 typed observations
with 5206 gate checks. Optimized DLL SHA-256:
`b2deaef4792bc5210984dd85a675c004ac792752aebd6abec81980efe11ba926`.
Registration is 1742 forwards per integer ABI. The source is being frozen for
full package validation; no full-suite proof is claimed for this candidate.

Full-suite proof is bound to `3321360eff7c89cf7c1eea8a69d88401f79283fe` and all
300 frozen sources, optimized DLL, manifest and registration. The imported
proof validates 1538 canonical entries: 1106 Fortran LAPACK, 172 CBLAS and
260 LAPACKE, with 5567 completed witnesses. Python 28, inventory 15 and
documentation build checks pass. All frozen source hashes and DLL remain
unchanged. CI and main integration remain required.

Parent integration preserves all production sources and the optimized DLL.
Only test/lq.jl changes among the 300 execution sources, adding validated
assertions/consumption of expected diagnostics. The 3321360 full proof is
historical; fresh full validation will cover the current test source set.

Current full proof is bound to 9c2c495b0205021fdedc1030e51f98be5126979b and all 300 frozen
execution sources. Cloud job success, ZIP/TOML digests, complete source map,
manifest, registration and all typed witnesses were verified before import.
The snapshot validates 1538 canonical entries with 5567 completed witnesses.
A separate publication tree preserves the original local full-suite source
and HEAD. Its copied optimized DLL remains byte-identical to the frozen DLL.
The published full run platform is Linux; final Linux/macOS PR CI remains
required. Provenance is stored in the adjacent cloud-provenance JSON.

Publication checks pass Python 28, inventory 15 and documentation build.
All execution-source hashes and the copied optimized local DLL remain
unchanged. Preceding staged candidates and final PR CI/main merge remain
required; this does not claim merged-main coverage for the candidate.

Current-main integration after PR #393 uses a normal merge from main
882dcbc0ca417da2067d1c71ccdfd46583721678. Twelve expected conflicts were
resolved: current main supplies the already merged LQ and QL/RQ audit
records; this candidate retains its runtime/test/evidence/registration files.
All 300 frozen execution-source hashes and the optimized DLL are unchanged.
The 228-name factorization parent is now merged; this stage adds 32 names.
Issue #288 remains open.

PR #394 merged at 72a7706e333692a7429aa5e9fab70bc41e9110dd after CI
43922508148 and documentation 43922508113 passed. Both OS artifacts validate
all 300 source hashes, manifest, registration, 1538 entries and 5567 witnesses
against test merge 021f37bf2c02868226f373d963c06d7cf551c05d. Its parents and
tree match the guarded base/head. Main tree and optimized DLL match the merge
gate; post-merge Python 28 and inventory 15 pass. Issue #288 remains open.
