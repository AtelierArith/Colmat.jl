# Issue #288: PFTRF/PFTRI/PFTRS C contracts

Status: the 24 S/D/C/Z high/work entries pass focused RFP-layout,
factorization, inversion, solve, diagnostic and independent-call checks. The
immutable full suite records 440 sources, 2,376 canonical entries and 8,983
witnesses (2,580 forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{pftrf,pftri,pftrs}` and `_work` forms (24 entries).

## Contracts

- The rectangular full packed (RFP) array is a ROW-by-COL rectangle with
  ROW*COL = N(N+1)/2 elements. For TRANSR='N' the rectangle is (N+1)x(N/2) for
  even N and N x (N+1)/2 for odd N; TRANSR='T'/'C' transposes those
  dimensions.
- Row-major storage is the plain dense transpose of that rectangle, matching
  the reference dtf_trans mapping. The factor and the inverse round-trip
  through the same transpose.
- PFTRF factors in place; PFTRI inverts the factor in place; PFTRS solves with
  the factor. Row-major requires LDB >= NRHS (-8). B is converted over the
  logical N-by-NRHS rectangle only.
- High-level NaN order: PFTRF/PFTRI A (-5); PFTRS A (-6), B (-7).
- Negative native INFO is shifted once for the layout argument; positive
  non-positive-definite INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, TRANSR N and T/C, UPLO U/L, N odd and even, factor and inverse
  equivalence through the RFP transpose, PFTRS solve residual, padding guards
  and diagnostics.
- The RFP input buffers are built with the native TRTTF entry so the packing
  follows the production converter.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
