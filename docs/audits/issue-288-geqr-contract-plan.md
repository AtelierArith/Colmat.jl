# Issue #288: GEQR/GELQ/GEMQR/GEMLQ C contracts

Status: the 32 S/D/C/Z high/work entries pass focused layout, blocked-reflector,
round-trip, diagnostic and independent-call checks. The immutable full suite
records 436 sources, 2,328 canonical entries and 8,783 witnesses (2,532
forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{geqr,gelq,gemqr,gemlq}` and `_work` forms (32 entries).

## Contracts

- GEQR/GELQ produce a compact reflector block T (with a small descriptor the
  native query writes, so the query buffer must hold several elements) and
  overwrite A with the packed factor. The high-level entry queries LWORK,
  allocates WORK and passes the caller's T buffer through so the matching
  GEMQR/GEMLQ entry can consume it, as in the reference.
- GEMQR reads an R-by-K reflector block; GEMLQ reads a K-by-R block, where
  R = M for SIDE='L' and R = N for SIDE='R'. Row-major requires LDA >= K for
  GEMQR (-8) and LDA >= R for GEMLQ (-8), plus LDC >= N (-11).
- High-level NaN order: factor A (-4) with row-major LDA >= N (-5); apply C
  (-10), A (-7), T (-9) where T is a compact array checked over TSIZE entries.
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, tall and wide shapes matched to each family's reflector geometry,
  factor equality across layouts, GEMQR/GEMLQ round trips on both layouts,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
