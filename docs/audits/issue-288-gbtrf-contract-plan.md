# Issue #288: GBTRF/GBTRS/GBSV C contracts

Status: the 24 S/D/C/Z high/work entries pass focused band-layout, LU, solve,
driver, diagnostic and independent-call checks. The immutable full suite records
412 sources, 2,104 canonical entries and 7,871 witnesses (2,308 forwards per
integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gbtrf,gbtrs,gbsv}` and `_work` forms (24 entries).

## Contracts

- General band storage with KL subdiagonals and KU superdiagonals; column-major
  uses `LDAB >= 2*KL+KU+1` band rows, row-major uses LDAB as the row stride of a
  `(2*KL+KU+1)`-by-`LDAB` buffer and requires LDAB >= N.
- Band entries are reordered with the reference `gb_trans` index mapping; the LU
  write-back touches only band entries. IPIV is shared across layouts.
- GBTRS takes TRANS `N`/`T`/`C`; row-major requires LDAB >= N (-8) and
  LDB >= NRHS (-11). GBSV requires LDAB >= N (-7) and LDB >= NRHS (-10).
  GBTRF row-major requires LDAB >= N (-7).
- High-level NaN order: GBTRF AB (-6); GBTRS AB (-7), B (-10); GBSV AB (-6),
  B (-9).
- Negative native INFO is shifted once for the layout argument; positive
  singularity INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, rectangular and square band shapes, KL/KU combinations, factor and
  IPIV equivalence across layouts, TRANS variants, solve residual, GBSV
  end-to-end equivalence, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
