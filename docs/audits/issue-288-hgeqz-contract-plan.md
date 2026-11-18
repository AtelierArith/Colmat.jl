# Issue #288: HGEQZ C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, eigenvalue/vector,
diagnostic and independent-call checks for JOB='E'. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}hgeqz` and `_work` forms (8 entries).

## Contracts

- COMPQ/COMPZ `N`/`V`; ILO/IHI follow the reference range. H is upper
  Hessenberg and T upper triangular.
- Real entries carry ALPHAR/ALPHAI/BETA; complex entries carry ALPHA/BETA and
  real RWORK.
- Row-major requires LDH >= N (-9), LDT >= N (-11), LDQ >= N (real -16,
  complex -15) and LDZ >= N (real -18, complex -17), matching the reference
  work entries. H/T/Q/Z are converted over the logical N-by-N rectangle only;
  padding is preserved.
- High-level NaN order: H (-8), Q (-15/-14, only when V), T (-10), Z
  (-17/-16, only when V).
- Work queries LWORK=-1, allocates the requested workspace (and N real RWORK
  for complex) and calls the work entry.
- The native generalized Schur kernel accepts JOB='E'/'S'; JOB='N'/'B' remain
  outside this stage and are documented as such rather than silently mapped.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, COMPQ/COMPZ N/V, eigenvalue and vector row/column-major equivalence,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

HGEQZ JOB N/B and the rest of the 1,616-entry #288 scope are not implied by
this stage.
