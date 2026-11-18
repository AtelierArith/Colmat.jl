# Issue #288: GTTRF/GTTRS/GTSV C contracts

Status: the 24 S/D/C/Z high/work entries pass focused factorization/solve,
driver, layout, diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gttrf,gttrs,gtsv}` and `_work` forms (24 entries).

## Contracts

- GTTRF has no matrix-layout argument: `(n, dl, d, du, du2, ipiv)`. DL/D/DU hold
  the subdiagonal, diagonal and superdiagonal; DU2 holds the second
  superdiagonal created by pivoting and IPIV the pivot record.
- GTTRS solves with the factors and TRANS `N`/`T`/`C`; row-major requires
  LDB >= NRHS (-11). B is converted over the logical N-by-NRHS rectangle only.
- GTSV factors and solves in one call; row-major requires LDB >= NRHS (-8) and
  returns the destroyed DL/D/DU identically across layouts.
- High-level NaN order: GTTRF D (-3), DL (-2), DU (-4); GTTRS B (-10), D (-6),
  DL (-5), DU (-7), DU2 (-8); GTSV B (-7), D (-5), DL (-4), DU (-6).
- Negative native INFO is shifted once for the layout argument; GTTRF reports
  the native index directly because it has no layout.

## Evidence

- Four scalar types, both integer ABIs, both layouts (GTTRS/GTSV), high/work,
  DLL and LBT routes, N=1/N, TRANS variants, factor-buffer and IPIV equality,
  solve residuals, GTSV end-to-end equivalence, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
