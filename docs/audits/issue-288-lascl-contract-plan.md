# Issue #288: LASCL C contracts

Status: the 8 S/D/C/Z high/work entries pass focused type, layout, diagnostic
and independent-call checks. The immutable full suite is recorded in
`docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}lascl` and `_work` forms (8 entries).

## Contracts

- TYPE `G` (full), `L`/`U` (triangles), `H` (Hessenberg), `B`/`Q` (symmetric
  bands) and `Z` (GBTRF band storage).
- CFROM/CTO are real by-value scalars; for complex routines they remain real
  (`float` for c, `double` for z), matching the Reference LAPACKE headers.
- Row-major requires LDA >= N (-9). The logical block has `NR` rows where NR is
  M for G/L/U/H, KL+1 for B, KU+1 for Q and 2*KL+KU+1 for Z. Only that block is
  converted; padding is preserved.
- High-level NaN checks the logical block and returns -9. Fortran negative
  INFO shifts by one for the layout argument.
- Zero M/N calls still validate the leading dimension and flags.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, all seven TYPE values, band row counts, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
