# Issue #288: GGEVX C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, eigenvalue/vector,
diagnostic and independent-call checks for BALANC='N', SENSE='N'. The immutable
full suite records 393 sources, 1,976 canonical entries and 7,351 witnesses
(2,180 forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}ggevx` and `_work` forms (8 entries).

## Contracts

- BALANC `N`/`P`/`S`/`B`, JOBVL/JOBVR `N`/`V`, SENSE `N`/`E`/`V`/`B`.
  BALANC/SENSE other than N use the saved expert backend as in the Fortran
  boundary.
- Outputs include real ALPHAR/ALPHAI/BETA or complex ALPHA/BETA, ILO/IHI,
  LSCALE/RSCALE, ABNRM/BBNRM and RCONDE/RCONDV.
- Row-major requires LDA/LDB >= N (-8/-10) and LDVL/LDVR >= N (real -15/-17,
  complex -14/-16). A/B/VL/VR are converted over the logical rectangle.
- High-level NaN checks A (-7), B (-9). Work queries LWORK=-1 and allocates
  complex/real work plus IWORK and BWORK.
- Condition estimates are owned by the native or saved expert driver; the
  LAPACKE layer does not synthesize them.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, BALANC/SENSE N, alpha/beta, vector and row/column-major equivalence.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
