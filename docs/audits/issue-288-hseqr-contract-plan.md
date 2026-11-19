# Issue #288: HSEQR C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, COMPZ,
eigenvalue/vector, Schur-form, diagnostic and independent-call checks. The
immutable full suite records 434 sources, 2,296 canonical entries and 8,655
witnesses (2,500 forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}hseqr` and `_work` forms (8 entries).

## Contracts

- JOB `E` (eigenvalues only) or `S` (eigenvalues plus Schur form); COMPZ `N`
  (no Z), `I` (Z initialized to the identity) or `V` (Z accumulated). ILO/IHI
  follow the reference range.
- Real entries carry WR/WI (WI real); the complex entry carries W.
- Row-major requires LDH >= N (-8) and LDZ >= N (-12). H is converted over the
  logical N-by-N rectangle; Z is converted and written back when COMPZ is
  I or V. Padding is preserved.
- High-level NaN order: H (-7), Z (-11, only when COMPZ is I or V). The
  high-level entries query LWORK=-1, allocate the requested workspace and call
  the work entry.
- Negative native INFO is shifted once for the layout argument; positive
  convergence-failure INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, JOB E/S, COMPZ N/I/V, WR/WI or W equality, Schur form verification
  (upper triangular for complex, quasi-upper triangular for real), Z
  orthogonality and the Z*H*Z' = H0 identity, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
