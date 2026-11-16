# Issue #288: QL/RQ C factorization adapters

Status: Fortran empty-work correction passes independent boundary regressions;
C adapters pass focused numerical and independent C checks. Typed tracing,
optimized repetition and full-suite evidence remain pending.
Issue #288 remains open. The planned GEQLF/GERQF high/work entries
add 16 names to the preceding 212-name candidate. This is staged scope; main
has fewer entries until each validated candidate is merged.

Reference LAPACK v3.12.1
[GEQLF](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgeqlf.f)
and [GERQF](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgerqf.f)
return query WORK(1)=1 for empty matrices. GEQLF permits minimum WORK=1 when
M=0 and otherwise requires max(1,N). GERQF permits one when N=0 and otherwise
requires max(1,M). These rules require separate workspace validation from
ordinary GEQRF, even though the C argument types coincide.

Executed `reproducers/reverse_qr_empty_workspace.jl` with independent OpenBLAS.
At GEQLF M=0,N=3 and GERQF M=3,N=0 the own Fortran boundary returns query 3 and
rejects LWORK=1 with INFO=-7, leaving WORK=777. Independent OpenBLAS returns
query 1 and accepts LWORK=1 with INFO=0 and WORK=1 for both routines. This
requires a Colmat.jl boundary correction; no native upstream edits are needed.

Required evidence includes real/complex reflectors, both integer ABIs and
layouts, high/work and DLL/LBT, independent mutual Q consumers, A=QL or A=RQ
reconstruction and orthogonality, offset triangular storage in rectangular
matrices, empty dimensions, minimal/query WORK, padding/TAU guards, diagnostics,
NaN policy and independent C declarations. Typed completed-call traces,
optimized repetition and immutable full package proof are required before
publishing an implementation PR. Original scope and the separate 20 Rook C
targets remain in place.

The corrected Fortran factor driver chooses one for the empty case, writes
WORK before checking LWORK as the reference does, and keeps nonempty family
minima separate. The MWE now matches independent OpenBLAS. New checks pass
1152 assertions across both families, all scalar types, both own ABIs and
independent ILP64 OpenBLAS, including LWORK=-1/0/1/3 and guarded unused arrays.
Existing factor/Q interoperability passes 162624 checks, diagnostics and
prefix storage pass 2240, and LBT calls pass 352.

The authoritative canonical LBT inventory contains GEQLF/GERQF high/work
entries, but no LAPACKE GEQL2/GERQ2 entries. C implementation and tracing
therefore cover only the sixteen canonical names. The existing unblocked
Fortran routines and their independent regressions remain in scope.

Sixteen GEQLF/GERQF C adapters share checked layout conversion and NaN policy
with QR while delegating family-specific WORK validation to the corrected
Fortran boundary, preserving its WORK write on LWORK errors. Numerical checks
pass 87552 assertions for all scalar types, both layouts and integer ABIs,
high/work and direct/LBT, empty/scalar/rectangular/deficient inputs and extreme
scales. Independently generated Q consumes own factors and own Q consumes
reference factors; QL/RQ triangular offsets and orthogonality are verified.
The independent C11 caller verifies all sixteen names with explicit prototypes
and OpenBLAS Q consumers, padding/TAU/WORK guards, empty queries, diagnostics,
NaN policy and signed ILP64 values outside int32. Its Julia runner passes two
checks. The candidate extends 212 names to 228; 1388 original-scope names plus
the separate 20 Rook C targets remain. No full-suite proof is claimed yet.

Rust full-suite checks pass 460 tests. Completed-call tracing validates all
228 names with 912 typed name/ABI/route observations and 4566 gate checks.
The focused evidence explicitly records `full_suite_passed=false`. Optimized
repetition and immutable full package evidence remain required.
