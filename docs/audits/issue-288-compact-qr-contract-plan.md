# Issue #288: compact QR C adapters

Status: source-bound immutable full package proof and publication checks pass.
Final PR CI, artifact provenance checks and merge remain pending.

Issue #288 remains open.

The canonical inventory contains 32 high/work entries for S/D/C/Z GEQRT,
GEQRT2, GEQRT3 and GEMQRT. It contains no GELQT/GELQT3/GEMLQT C entries.
This stage follows the 292-name Q-application candidate and would reach 324
names, leaving 1292 original-scope names plus the separate 20 Rook C targets.
Candidate counts are not merged-main counts.

Reference LAPACK v3.12.1
[GEQRT work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqrt_work.c)
uses layout,M,N,NB,A,LDA,T,LDT,WORK without LWORK/query. Row LDA<N returns -6
before LDT<min(M,N) returns -8. High scans logical A and returns -5 for NaNs;
its scratch is max(1,NB)*max(1,N). T is an NB-by-min(M,N) block factor.
[GEQRT2 work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeqrt2_work.c)
and GEQRT3 have the same arguments as high: layout,M,N,A,LDA,T,LDT, with no
WORK pointer/query. Row LDA<N and LDT<N return -5/-7; high A NaNs return -4.
These variants produce an N-by-N T and require separate dimension checks.

[GEMQRT work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgemqrt_work.c)
uses layout,SIDE,TRANS,M,N,K,NB,const V,LDV,const T,LDT,C,LDC,WORK without a
query parameter. Its row checks are LDC<N (-13), LDT<NB (-11), then LDV<K
(-9). High NaN precedence is C (-12), T (-10), then V (-8); WORK allocation
is max(1,M)*max(1,NB). Flags, side-specific minimum scratch and logical V/T
extents must be checked independently rather than inferred from Q application.

A primary-source row-packing observation requires an executed contract probe:
the reference GEMQRT work wrapper transposes V using caller LDV as the row
count and T using LDT-by-NB, while the numerical V/T contract uses Q-order-by-K
and NB-by-K. It also preserves caller LDV/LDT as temporary column strides.
Tests must expose stride padding, both sides, K>NB and last partial blocks on
addressable padded buffers before selecting compatibility behavior. As in
Issue #288, unused storage must not be read and read-only inputs must remain
unchanged. Source observations alone do not establish numerical correctness.

Required evidence includes independent OpenBLAS A/T factors and mutual GEMQRT
consumers, reconstruction and orthogonality, all scalar types and integer ABIs,
both layouts, high/work and DLL/LBT, empty/rectangular/rank-deficient inputs,
block sizes including partial tails, legal flags, active/inactive NaNs,
INFO precedence, factor/output padding, independent C prototypes and signed
ILP64 values outside int32. Typed completed calls, optimized repetition and
immutable full package proof are required before publication. No upstream
native edits or PRs are part of this stage.

Executed `reproducers/lapacke_gemqrt_row_packing.py` with unchanged v3.12.1
S/D/C/Z GEMQRT work wrappers, both layouts/sides and M,N=(3,5)/(5,3): all
32 observations verify the source copy extents and forwarded strides. The
numerical Fortran function is stubbed; this is explicitly a packing/header
probe, not numerical GEMQRT validation. Caller buffers are padded for all
observed copies. Source SHA-256 hashes and observations are stored in the
adjacent reference-packing JSON.

For row-major K=2,NB=1,LDV=LDT=4 the wrapper copies V as 4x2 and T as 4x1,
although the logical shapes are Q-order-by-2 and 1x2. It forwards LDV=4,
which is below the numerical leading-dimension requirement when Q order=5.
When order=3 it copies an extra V row; T both copies extra rows and omits an
active block column. The planned own wrapper uses logical V/T extents and
derived column-major strides, preserving const inputs and caller padding.
Full numerical comparisons and block-factor interoperability are still
required before claiming this correction is implemented or complete.

Executed `geqrt_dimension_precedence.jl` before correction: at M=-1,N=-2 the
own double GEQRT2/3 boundaries return -1 while independent OpenBLAS returns
-2, with A/T unchanged. The own validation now checks N first, matching the
primary Fortran routines and preserving M<N as argument 1. New independent
checks pass 720 assertions across all scalar types, both own ABIs and both
families, including header precedence, guarded INFO and untouched A/T.
Existing compact QR checks also pass 368 Julia-wrapper, 1000 external-factor
interoperability and eight subnormal-scale checks.

`geqrt3_empty_reference.jl` was run only in an isolated subprocess with core
dumps disabled and a 20-second limit. Own GEQRT3 M=N=0 returns INFO=0 and
preserves A/T; this OpenBLAS reference raises StackOverflowError and exits 1.
The evidence JSON records the observed library hash and explicitly does not
claim full-suite proof. Normal tests use independent GEQRT2 as the empty-case
oracle for GEQRT3 and still test every own GEQRT3 empty entry. The reference
crash is not executed inside the package suite. Nonempty GEQRT3 comparisons
remain against the actual reference routine.

The own C adapters now pack logical V as Q-order-by-K and T as NB-by-K,
using derived column-major strides. Row-major T requires LDT>=max(NB,K):
NB preserves the reference header minimum while K prevents overlapping
logical row storage. High GEMQRT allocates NB times the side-dependent
workspace dimension (N for left, M for right).

Debug numerical checks pass 150000 factor/consumer assertions and 32256
rectangular application assertions. They cover all four scalar types, both
integer ABIs, layouts, high/work and DLL/LBT routes; left/right application,
transpose/conjugate transpose, empty and K=0 cases, and NaN padding with
immutable V/T. Independent OpenBLAS factors and block consumers establish
both directions of interoperability. These are focused checks only; no
full-suite proof or merged-main coverage is claimed for this stage.

Independent C factor testing exposed complex GEQRT R diagonals with imaginary
parts about 0.002 on a finite 5-by-3 matrix. The executed
`reproducers/complex_geqrt_diagonal.jl` records the before/after comparison
with independent OpenBLAS in the adjacent complex-phase JSON. Native complex
QR uses a different reflector phase; the own Fortran compact boundary now
uses the existing LAPACK Householder generator for complex scalars and forms
block T from those complex coefficients. Real scalars retain the native path.
The MWE now has exactly real diagonals matching OpenBLAS to rounding error.
Focused numerical checks pass 159600 factor assertions (including real R
diagonals) plus 32256 rectangular application and 1040 boundary assertions.
Rust 460 tests pass after this correction. Full immutable evidence is pending.

Optimized repetition passes the same 191856 numerical, 1040 policy, independent
C caller and 6486 completed-call gate checks. Typed tracing covers all 324
candidate names with 1296 completed ABI/route observations and explicitly
records full_suite_passed=false. The optimized DLL SHA-256 is
23ee0b32576db80489f651ce99d16dd44997f1de39169a3cfa43c6bfe6b487b1.
Normal main/parent integration preserves all 312 execution-source hashes.
A fresh full package run remains required before publishing this scope.

Current terminal-zero full Pkg.test proof is bound to
`1cc2018faf9bff9bd4831a5c7eb057fc0a8685af` and all 312 frozen sources.
HEAD, every source, optimized DLL, manifest and registration were unchanged
through the run. The imported proof validates 1602 canonical entries:
1106 Fortran LAPACK, 172 CBLAS and 324 LAPACKE, with 5855 completed
witnesses and 1806 forwards per ABI. This full proof includes the complex
real-diagonal correction and all independent numerical/C/policy tests.
CI artifact provenance and final main integration remain required.

After parent/main integration, all 312 validated source hashes and the DLL
remain unchanged. Python 28, inventory 15 and documentation build pass. The
inherited coverage doctest now checks 1806 forwards per ABI. CI provenance
and final main merge remain required.

Normal integration with main after PR #395 preserves all 312 frozen execution
sources and optimized DLL. Thirteen expected conflicts retain this candidate's
runtime/tests/evidence and take current-main LQ, QL/RQ, Q-generation and
Q-application audits. Python 28, registration inventory 15 and documentation
build pass after this source-preserving merge. Issue #288 remains open.
