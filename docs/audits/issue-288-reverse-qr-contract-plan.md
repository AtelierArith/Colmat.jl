# Issue #288: QL/RQ C factorization adapters

Status: executed empty-work MWE; implementation and full-suite evidence pending.
Issue #288 remains open. The planned GEQLF/GEQL2/GERQF/GERQ2 high/work entries
add 32 names to the preceding 212-name candidate. This is staged scope; main
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
