# Issue #288: PPTRF/PPTRS C contracts

Status: the 16 S/D/C/Z high/work entries pass focused packed-layout,
factorization/solve, diagnostic and independent-call checks. Issue #288 remains
open.

## Connected names

`LAPACKE_{s,d,c,z}{pptrf,pptrs}` and `_work` forms (16 entries).

## Contracts

- Packed symmetric/Hermitian storage with UPLO `U`/`L`; AP holds N(N+1)/2
  values and the public C signature has no LDAP argument, so the row-major
  buffer is tightly packed.
- Row-major AP is reordered with the reference `pp_trans` permutation. The
  captured factor follows the same convention, and PPTRS converts it back
  through the same map before solving.
- PPTRS takes `(layout, uplo, n, nrhs, ap, b, ldb)`; row-major requires
  LDB >= NRHS (-7). B is converted over the logical N-by-NRHS rectangle only.
- High-level NaN order: PPTRF AP (-4); PPTRS AP (-5), B (-6).
- Negative native INFO is shifted once for the layout argument; positive
  non-positive-definite INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, factor equivalence through the pp_trans permutation, solve
  residual, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
