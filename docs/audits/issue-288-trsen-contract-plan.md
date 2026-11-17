# Issue #288: TRSEN C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, selection,
diagnostic and independent-call checks for JOB='N'. The immutable full suite is
recorded in `docs/src/assets/lapack-abi-execution.json`. Issue #288 remains
open.

## Connected names

`LAPACKE_{s,d,c,z}trsen` and `_work` forms (8 entries).

## Contracts

- JOB='N' selects and moves blocks with the native checked Schur reordering.
  JOB E/V/B condition estimates are delegated to the saved original LAPACK
  implementation, as in the Fortran TRSEN boundary.
- COMPQ `N`/`V`; SELECT is `lapack_logical` (integer width); M is the moved
  block count.
- Real signatures carry WR/WI and IWORK/LIWORK; the complex signature carries
  W and has no integer workspace. S/SEP remain real in both.
- Row-major requires LDT >= N (-7) and LDQ >= N (-9). T/Q are converted over
  the logical N-by-N rectangle; padding is preserved.
- High-level NaN order: T (-6), Q (-8). Work queries LWORK=-1 (and LIWORK=-1
  for real), allocates the requested arrays and calls the work entry.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL/LBT
  routes, COMPQ N/V, SELECT/M, row/column-major equivalence, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

GEES/GEESX and the rest of the 1,616-entry #288 scope are not implied by this
stage.
