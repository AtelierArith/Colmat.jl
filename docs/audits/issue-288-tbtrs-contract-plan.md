# Issue #288: TBTRS/TPTRS/TPTRI C contracts

Status: the 24 S/D/C/Z high/work entries pass focused triangular band and
packed solve/inverse, layout, diagnostic and independent-call checks. Issue #288
remains open.

## Connected names

`LAPACKE_{s,d,c,z}{tbtrs,tptrs,tptri}` and `_work` forms (24 entries).

## Contracts

- TBTRS solves a triangular band system with UPLO `U`/`L`, TRANS `N`/`T`/`C` and
  DIAG `N`/`U` over KD band rows (AB(KD+1+i-j, j) convention). Row-major
  requires LDAB >= N (-9) and LDB >= NRHS (-11).
- TPTRS solves a packed triangular system with the same UPLO/TRANS/DIAG; the
  packed AP holds N(N+1)/2 values with no LDAP argument. Row-major requires
  LDB >= NRHS (-9).
- TPTRI inverts a packed triangular matrix in place; the row-major result is the
  `tp_trans` permutation of the column-major result, and TPTRS consumes that
  convention through the same map.
- High-level NaN order: TBTRS AB (-8), B (-10); TPTRS AP (-7), B (-8); TPTRI
  AP (-5).
- Negative native INFO is shifted once for the layout argument; positive
  singularity INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, TRANS N/T/C, DIAG N/U, KD=1/2, solve residuals, TPTRI
  round trips through the tp_trans permutation and inverse identities, padding
  guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
