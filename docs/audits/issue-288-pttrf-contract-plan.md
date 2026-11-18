# Issue #288: PTTRF/PTTRS C contracts

Status: the 16 S/D/C/Z high/work entries pass focused factorization/solve,
layout, diagnostic and independent-call checks. The immutable full suite records
402 sources, 2,032 canonical entries and 7,583 witnesses (2,236 forwards per
integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{pttrf,pttrs}` and `_work` forms (16 entries).

## Contracts

- PTTRF has no matrix-layout argument: `(n, d, e)`. D holds the N diagonal
  entries; E holds the N-1 subdiagonal entries. Complex routines keep D real
  and E complex.
- PTTRS solves with the factors. Real signature `(layout, n, nrhs, d, e, b,
  ldb)`; complex adds UPLO `U`/`L`.
- Row-major requires LDB >= NRHS (real -7, complex -8). B is converted over the
  logical N-by-NRHS rectangle only; padding is preserved.
- High-level NaN order: PTTRF D (-2), E (-3); PTTRS B (-6/-7), D (-4/-5), E
  (-5/-6). Work entries have no workspace argument.
- Negative native INFO is shifted by one for the layout argument; PTTRF
  reports the native index directly because it has no layout.

## Evidence

- Four scalar types, both integer ABIs, both layouts (PTTRS), high/work, DLL
  and LBT routes, N=1/N, NRHS=1/3, factor-buffer equality and residual checks,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
