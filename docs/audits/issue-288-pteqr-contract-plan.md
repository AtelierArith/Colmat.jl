# Issue #288: PTEQR C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, COMPZ, eigenvector,
diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}pteqr` and `_work` forms (8 entries).

## Contracts

- COMPZ `N` (eigenvalues only), `V` (Z in/out) and `I` (Z initialized to the
  identity by the native routine). D and E are real for every scalar type.
- Row-major requires LDZ >= N (-7); Z is converted over the logical N-by-N
  rectangle only and written back for COMPZ V/I. Padding is preserved.
- High-level NaN order: D (-4), E (-5), Z (-6, only when COMPZ='V').
- Workspace is real and sized 1 for COMPZ='N', otherwise MAX(1, 4N-4), matching
  the reference high-level allocation.
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, COMPZ N/V/I, D/E and Z equivalence across layouts, orthogonality,
  residual A*Z = Z*diag(D), padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
