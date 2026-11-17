# Issue #288: ORGTR/UNGTR and ORMTR/UNMTR C contracts

Status: the 16 S/D/C/Z high/work entries pass focused layout, diagnostic and
independent-call checks. The immutable full suite is recorded in
`docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{orgtr,ungtr,ormtr,unmtr}` and their `_work` forms (16
entries). The generators use the native tridiagonal reduction vectors; the
applicators use the native Q application.

## Contracts

- ORGTR/UNGTR take UPLO and return the orthogonal/unitary Q from the
  tridiagonal reduction. Only the selected triangle is read; tau has N-1
  entries. Row-major requires LDA >= N (-5).
- ORMTR/UNMTR take SIDE (`L`/`R`), UPLO and TRANS (`N`/`T`, or `N`/`C` for the
  complex routines). The reflector order is R = M for SIDE='L' and R = N for
  SIDE='R'. Row-major requires LDA >= R (-8) and LDC >= N (-11).
- High-level NaN order: generator A (-4), tau (-6); applicator A (-7), C (-10),
  tau (-9). Work-level leading dimensions, UPLO/SIDE/TRANS and workspace errors
  follow the reference negative indices.
- The high-level entries query LWORK=-1, allocate the requested workspace, and
  call the work entry. Zero dimensions do not read A, C or tau.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, row/column-major equivalence for generators and all SIDE/TRANS
  applicator combinations, non-square C, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The ORGBR/ORMBR and ORGHR/ORMHR families and the rest of the 1,616-entry #288
scope are not implied by this stage.
