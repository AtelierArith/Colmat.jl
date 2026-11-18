# Issue #288: GEES C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, eigenvalue/vector,
diagnostic and independent-call checks for SORT='N'. The immutable full suite
is recorded in `docs/src/assets/lapack-abi-execution.json`. Issue #288 remains
open.

## Connected names

`LAPACKE_{s,d,c,z}gees` and `_work` forms (8 entries).

## Contracts

- JOBVS `N`/`V`, SORT `N`/`S`; SELECT is passed through to the native Schur
  driver with the reference C callback signature. BWORK is supplied when
  SORT='S'.
- Real entries carry WR/WI; the complex entry carries W. SDIM is returned in
  both.
- Row-major requires LDA >= N (-7) and LDVS >= N (real -12, complex -11). A and
  VS are converted over the logical N-by-N rectangle; eigenvalues use the
  original indexing.
- High-level NaN checks A (-6). Work queries LWORK=-1, allocates the requested
  workspace and calls the work entry.
- The Fortran owner performs the native Schur reduction and selection; the
  LAPACKE layer owns only representation and workspace conversion.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, JOBVS N/V, SORT N, row/column-major A/VS equivalence and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

GEESX and the rest of the 1,616-entry #288 scope are not implied by this stage.
