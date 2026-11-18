# Issue #288: GEBAL/GEBAK C contracts

Status: the 16 S/D/C/Z high/work entries pass focused balancing,
back-transform, layout, diagnostic and independent-call checks. The immutable
full suite records 422 sources, 2,196 canonical entries and 8,239 witnesses
(2,400 forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gebal,gebak}` and `_work` forms (16 entries).

## Contracts

- GEBAL takes JOB `N`/`P`/`S`/`B` and returns ILO, IHI and the real SCALE
  vector; the balanced matrix overwrites A. Row-major requires LDA >= N (-5).
- GEBAK back-transforms an N-by-M matrix V with SIDE `L`/`R` using the same
  JOB, ILO, IHI and SCALE. Row-major requires LDV >= M (-10).
- High-level NaN order: GEBAL A (-4, only for JOB B/P/S); GEBAK SCALE (-7),
  V (-9).
- A and V are converted over their logical rectangles only; padding is
  preserved. SCALE stays real for every scalar type.
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, JOB N/P/S/B, SIDE L/R, an isolated leading row/column to exercise
  permutation, ILO/IHI and SCALE equality across layouts, GEBAK equivalence,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
