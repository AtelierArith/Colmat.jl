# Issue #288: GEESX C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, eigenvalue/vector,
diagnostic and independent-call checks for SORT='N', SENSE='N'. The immutable
full suite records 389 sources, 1,952 canonical entries and 7,255 witnesses
(2,156 forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}geesx` and `_work` forms (8 entries).

## Contracts

- JOBVS `N`/`V`, SORT `N`/`S`, SENSE `N`/`E`/`V`/`B`; SELECT is passed through
  with the reference C callback signature. SENSE other than N uses the saved
  expert backend as in the Fortran boundary.
- Real entries carry WR/WI, IWORK and RCONDE/RCONDV; complex entries carry W,
  RWORK and real RCONDE/RCONDV.
- Row-major requires LDA >= N (-8) and LDVS >= N (real -13, complex -12). A and
  VS are converted over the logical N-by-N rectangle.
- High-level NaN checks A (-7). Work queries LWORK/LIWORK, allocates them and
  calls the work entry.
- RCONDE/RCONDV semantics for SENSE N/E/V/B are those of the underlying native
  or saved driver; the LAPACKE layer does not fabricate condition estimates.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, JOBVS N/V, SORT/SENSE N, row/column-major A/VS equivalence and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
