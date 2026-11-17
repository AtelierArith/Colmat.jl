# Issue #288: TREXC C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, index, diagnostic
and independent-call checks. The immutable full suite records 371 sources,
1,880 canonical entries and 6,967 witnesses (2,084 forwards per integer ABI).
Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}trexc` and `_work` forms (8 entries).

## Contracts

- COMPQ `N`/`V`; T is N-by-N quasi-triangular and Q is N-by-N when COMPQ='V'.
- The real C entry takes IFST/ILST by pointer; the complex C entry takes them by
  value, matching the Reference LAPACKE headers. The Fortran owner always
  receives pointers.
- Row-major requires LDT >= N (-5) and, for COMPQ='V', LDQ >= N (-7). T and Q
  are converted over the logical N-by-N rectangle; padding is preserved.
- High-level NaN order: T (-4), Q (-6) when COMPQ='V'. Real work allocates N
  scalars; complex work has no work argument.
- Invalid layout/COMPQ/dimensions follow the reference negative indices with
  the LAPACKE layout shift.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, COMPQ N/V, row/column-major equivalence, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

TRSEN/GEES and the rest of the 1,616-entry #288 scope are not implied by this
stage.
