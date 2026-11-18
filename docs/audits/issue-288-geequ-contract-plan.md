# Issue #288: GEEQU/GEEQUB C contracts

Status: the 16 S/D/C/Z high/work entries pass focused equilibration, layout,
diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{geequ,geequb}` and `_work` forms (16 entries).

## Contracts

- Dense equilibration with R (M), C (N), ROWCND, COLCND and AMAX real outputs
  for every scalar type.
- Row-major requires LDA >= N (-5); A is converted over the logical M-by-N
  rectangle only and padding is preserved.
- High-level NaN order: A (-4). GEEQUB returns radix-binned powers of two.
- Native info is returned directly: positive values report a zero row/column,
  and negative values are shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, square and rectangular shapes, R/C/condition/AMAX equality across
  layouts, a near-unity equilibrated row maximum (modulus for real, the native
  component L1 magnitude for complex; within a factor of two for GEEQUB),
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
