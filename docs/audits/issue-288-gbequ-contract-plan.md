# Issue #288: GBEQU/GBEQUB C contracts

Status: the 16 S/D/C/Z high/work entries pass focused band-layout, equilibration
and diagnostic checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gbequ,gbequb}` and `_work` forms (16 entries).

## Contracts

- Analysis band convention AB(KU+1+i-j, j) with LDAB >= KL+KU+1 (unlike the
  2*KL+KU+1 factor convention used by GBTRF).
- Row-major AB uses LDAB as the row stride of a `(KL+KU+1)`-by-`LDAB` buffer and
  requires LDAB >= N (-7).
- Outputs R (M), C (N), ROWCND, COLCND and AMAX are real for every scalar type
  and are identical across layouts. GBEQUB returns radix-binned scaling factors.
- High-level NaN order: AB (-6), checked over the logical band entries only.
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, square and rectangular band shapes, KL/KU combinations, R/C/condition
  and AMAX equivalence, an equilibration sanity bound, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
