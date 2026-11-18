# Issue #288: LARFT C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, DIRECT/STOREV,
diagnostic and independent-call checks. The immutable full suite records 381
sources, 1,920 canonical entries and 7,127 witnesses (2,124 forwards per
integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}larft` and `_work` forms (8 entries).

## Contracts

- DIRECT `F`/`B`, STOREV `C`/`R`; V is N-by-K for `C` and K-by-N for `R`.
- Row-major requires LDV >= the logical V column count (-7) and LDT >= K
  (-10). V is converted over the logical rectangle only; T is written with the
  reference transpose convention for row-major output.
- High-level NaN order: V (-6), tau (-8). K=0/N=0 returns without reading V.
- The Fortran owner has no INFO output; invalid arguments surface through the
  shared diagnostic channel.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, DIRECT F/B, STOREV C/R, row/column-major T transpose equivalence and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

LARFB and the rest of the 1,616-entry #288 scope are not implied by this stage.
