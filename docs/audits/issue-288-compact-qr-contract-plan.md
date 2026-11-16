# Issue #288: compact QR C adapters

Status: canonical scope and primary contract investigation; implementation
and full-suite evidence pending. Issue #288 remains open.

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
