# Issue #288: LARFB C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, option, diagnostic
and independent-call checks. The immutable full suite records 383 sources,
1,928 canonical entries and 7,159 witnesses (2,132 forwards per integer ABI).
Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}larfb` and `_work` forms (8 entries).

## Contracts

- SIDE `L`/`R`, TRANS `N`/`T` (`N`/`C` for complex), DIRECT `F`/`B`, STOREV
  `C`/`R`; V dimensions follow SIDE and STOREV, T is K-by-K and C is M-by-N.
- Row-major requires LDV >= the logical V column count (-10), LDT >= K (-12)
  and LDC >= N (-14). V/T/C are converted over their logical rectangles only;
  C is written back with the reference row-major layout.
- High-level NaN order: V (-8), T (-11), C (-13). K=0/M=0/N=0 returns without
  reading V/T/C.
- The work entry uses LDWORK=K as in the native adapter.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, SIDE L/R, TRANS N/T or N/C, DIRECT F/B, STOREV C/R, rectangular C,
  row/column-major equivalence and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
