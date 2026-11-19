# Issue #288: TRTTF/TFTTR/TFTTP/TPTTF/TPTTR/TRTTP C contracts

Status: the 48 S/D/C/Z high/work entries pass focused conversion, layout,
diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{trttf,tfttr,tfttp,tpttf,tpttr,trttp}` and `_work` forms
(48 entries). High and work share the same signature because these conversions
need no workspace.

## Contracts

- TRTTF packs a full triangle into RFP; TFTTR unpacks RFP into a full triangle
  (writing only the selected UPLO triangle); TFTTP/TPTTF convert between RFP
  and packed; TPTTR/TRTTP convert between packed and full.
- Row-major RFP storage is the dense transpose of the ROW-by-COL rectangle,
  matching the reference dtf_trans mapping. Row-major packed AP uses the
  reference pp_trans permutation. Dense A conversions cover the logical N-by-N
  rectangle only.
- TRANSR `N` selects the normal rectangle and `T`/`C` the transposed one for
  complex types.
- Row-major leading-dimension errors: TRTTF LDA >= N (-6), TFTTR (-7), TPTTR
  (-6), TRTTP (-5).
- High-level NaN order: TRTTF A (-5, selected triangle), TFTTR/TFTTP ARF (-5,
  all N(N+1)/2 entries), TPTTF AP (-5), TPTTR AP (-4), TRTTP A (-4, selected
  triangle).
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, TRANSR N and T/C, UPLO U/L, N odd and even, equality with the native
  TRTTF reference packing, full/RFP/packed round trips through every pair of
  formats, triangle-only writes for TFTTR/TPTTR, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
