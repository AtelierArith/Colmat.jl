# Issue #288: Q application C adapters

Status: current-source immutable full package validation passes; source-bound
evidence is imported. Publication checks pass; final PR CI and merge remain pending.

Issue #288 remains open.

The canonical inventory contains 32 high/work names for real ORMQR/ORMLQ/
ORMQL/ORMRQ and complex UNMQR/UNMLQ/UNMQL/UNMRQ. These follow the 260-name
Q-generation candidate and would bring the candidate to 292 names, leaving
1324 original-scope names plus the separate 20 Rook C targets. Candidate
counts are not merged-main counts.

Reference LAPACK v3.12.1
[ORMQR work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dormqr_work.c)
uses layout, SIDE, TRANS, M,N,K, const A,LDA,const TAU,C,LDC,WORK,LWORK.
High omits WORK/LWORK. Row-major LDA and LDC checks (-8/-11) precede flag
and dimension checks; query calls use temporary column-major strides without
reading A or C. QR/QL store A as order-by-K; LQ/RQ store K-by-order, where
order is M for SIDE=L and N for SIDE=R. SIDE/TRANS are C chars and numerical
integers follow lapack_int. Real transpose and complex adjoint options need
separate checks, as do query and minimal WORK.

High-level NaN scans visit A, then C, then TAU and return -7/-10/-9.
Primary source investigation also found a reference asymmetry requiring an
explicit independent contract probe before choosing behavior: real ORMLQ uses
K-by-order, but complex
[UNMLQ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmlq.c)
uses K-by-M; real and complex
[ORMRQ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dormrq.c)
also use K-by-M. These differ for SIDE=R. Tests must distinguish addressable
padded inputs, skipped NaNs, precedence and numerical reflector dimensions;
a source observation alone does not establish a safe complete contract.

Required evidence includes independent OpenBLAS applications and explicit Q
products, all orientations, scalar types, ABIs/layouts, DLL/LBT and high/work,
SIDE=L/R and legal transpose flags, empty/rectangular inputs and K=0/partial/
full, complex phase TAU, const A/TAU, C padding, query immutability, diagnostics,
active/inactive NaNs and signed ILP64 values outside int32. Independent C,
typed completed-call traces, optimized repetition and immutable full package
proof are required before publishing an implementation PR. No native source
changes or upstream PRs are part of this stage.

Executed `reproducers/lapacke_q_apply_nan_extent.py`: unchanged v3.12.1 high
wrappers for all 16 scalar/family combinations, both layouts and two right-side
rectangles produce 64 verified observations. Numerical work is explicitly
stubbed; this is a policy-only probe, not a numerical reproduction. Every
primary source SHA-256 and observation is stored in the adjacent reference
policy JSON. The buffers are padded for either convention, so the probe itself
performs no out-of-bounds access.

For SIDE=R,M=3,N=5,K=2, real ORMLQ scans 2x5 and finds a NaN in column 5;
complex UNMLQ and all ORMRQ/UNMRQ scan 2x3 and miss it. At M=5,N=3 the latter
scan 2x5 and reject a NaN beyond the logical 2x3 reflector matrix. The planned
Colmat.jl high boundary uses the actual Q order for all orientations. This
intentional compatibility correction follows Issue #288's explicit requirement
not to read unused storage and checks all active reflectors. Tests must retain
both observed reference cases and the logical-storage expectation; no family,
SIDE or scalar case is excluded. NaN diagnostic precedence remains A, C, TAU.

A separate executed numerical-boundary MWE,
`reproducers/q_apply_empty_workspace.jl`, calls own and independent OpenBLAS
Fortran entries for all four orientations, double real/complex, at SIDE=L,
M=0,N=3,K=0. All calls return INFO=0 and preserve the guarded WORK tail.
After execution with LWORK=3 the own boundary returns WORK(1)=3, while all eight
reference variants return one. This requires a Colmat.jl Fortran empty-return
correction. Query results are algorithm-specific (own three, OpenBLAS 4256 in
this configuration); query size differences are not treated as this defect.
Full scalar/ABI/shape coverage must accompany the correction before publication.

The own Fortran correction returns WORK(1)=1 after an empty execution for all
four orientations. QR/LQ also return one for nonempty K=0; QL/RQ retain their
own algorithm workspace value for nonempty K=0. Query behavior and minimum
WORK validation remain separate. Reference QL/RQ empty queries return one
even when their argument validation still requires NW; tests record this
query/minimum distinction rather than assuming queried size is the minimum.
Independent boundary checks pass 19265 assertions across all four scalar
types, both own integer ABIs, all orientations, SIDE=L/R and legal transpose
flags, empty/nonempty K=0, query, insufficient/minimal/oversized WORK, INFO,
queryable diagnostics and guarded unused A/TAU/C/WORK. Rust full-suite checks
pass 460 tests. Existing Fortran QR/LQ/QL/RQ regressions are being repeated.
C application adapters and immutable full-suite proof remain pending.

The combined regression uncovered an older unconsumed GELQS error 10 before
the new empty-LQ checks. GELQS now asserts/consumes that diagnostic at its
origin; every own and reference empty-LQ probe also checks diagnostic state.
LQ then QL/RQ then Q-application boundary tests pass together: GELQS 40,
empty LQ 768, QL/RQ factor interoperability 162624, diagnostics 2240, LBT 352,
empty QL/RQ 1536 and Q-application boundaries 19265. This strengthens error
checks and prevents prior calls contaminating later independent comparisons.

The 32 C application adapters now pass 502208 independent numerical checks.
Own results match both direct OpenBLAS application and explicit products with
independently generated unitary Q. Coverage includes all scalar types, both
ABIs/layouts, DLL/LBT, high/work, all four orientations, SIDE=L/R, legal
transpose/adjoint flags, empty/scalar/rectangular C, K=0/partial/full, complex
phase TAU, large/small C scales, const A/TAU, padding, query immutability and
minimum WORK. Row conversion copies A and C without conjugation and copies
back only C after nonnegative INFO. High NaN scans use logical Q order and
preserve A/C/TAU diagnostic precedence. Independent C, typed tracing, optimized
and immutable full-suite proof remain pending; no full proof is claimed.

Logical NaN-bound and diagnostic checks pass 8960 assertions. They verify
row LDA/LDC precedence, flags and dimension INFO, negative-INFO C retention,
case-insensitive flags, stored diagonal NaNs (high rejects, work ignores),
NaN-check toggling, A/C/TAU error precedence, inactive TAU entries, K=0 unused
inputs, and ILP64 values outside int32. Both right-side rectangles from the
reference policy probe are covered: every logical reflector entry is scanned,
and padded entries beyond Q order are ignored for all orientations and scalar
types. The numerical 502208 checks also pass again in the standalone policy
run. Independent C and full-suite proof for this candidate remain pending.

The independent C11 caller passes both runner checks for all 32 names with
explicit C and ILP64 reference Fortran prototypes (including character lengths).
It compares all orientations, SIDE=L/R and legal transpose/adjoint flags,
scalar types, both own integer ABIs, layouts and high/work, empty/scalar/
rectangular C and every K up to Q order. It verifies const A/TAU against byte
copies, C/WORK guards, query and error immutability, INFO precedence, A/C/TAU
NaN order and signed ILP64 values outside int32. Typed completed-call tracing,
optimized repetition and immutable full package proof remain required.

Completed-call tracing verifies all 292 candidate names with 1168 typed
name/ABI/route observations and 5846 gate checks. Focused evidence explicitly
records `full_suite_passed=false`. High allocation respects both the query and
SIDE-dependent minimum WORK; reference QL/RQ empty queries can report one
below their argument-validation minimum, so the two quantities are not
interchangeable. Optimized repetition and immutable full proof remain required.

Optimized repetition passes all numerical, policy, independent C and 1168
typed observations with 5846 gate checks. Optimized DLL SHA-256:
`0bc1b452ee357cc43a5df9b87b482ddcfda20afc853c78bb146df7886658364b`.
Registration is 1774 forwards per integer ABI. The source is being frozen for
full package validation; no full-suite proof is claimed for this candidate.

Current full-suite proof is bound to `39aa06e36fd6d4e6342596c40342d92d9f3c9018` and all
306 frozen sources, optimized DLL, manifest and registration. The imported
proof validates 1570 canonical entries: 1106 Fortran LAPACK, 172 CBLAS and
292 LAPACKE, with 5727 completed witnesses. Python 28, inventory 15 and
documentation build checks pass. All frozen source hashes and DLL remain
unchanged. CI artifact provenance and main integration remain required.

Normal integration with main after PR #394 preserves all 306 frozen execution
sources and the optimized DLL. Fourteen expected conflicts keep this candidate's
runtime/tests/evidence and take current-main LQ, QL/RQ and Q-generator audits.
Python 28, registration inventory 15 and documentation build pass after this
source-preserving merge. The 260-name Q-generator parent is now merged.
Issue #288 remains open.
