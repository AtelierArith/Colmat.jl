# Issue #288: LANGE/LANSY/LANTR C contracts

Status: the 24 S/D/C/Z high/work entries pass focused numerical, layout,
diagnostic and independent C ABI checks. The immutable full suite is recorded
in `docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{lange,lansy,lantr}` and `_work` forms (24 entries). They call
the existing native Fortran matrix-norm entries; no new numerical kernel is
introduced.

## Reference contracts

Reference-LAPACK tag `v3.12.1`, LAPACKE/src/{s,d,c,z}{lange,lansy,lantr}{,_work}.c.

- `M`: max absolute entry.
- `1`/`O`: max absolute column sum.
- `I`: max absolute row sum.
- `F`/`E`: Frobenius norm.
- LANSY uses one selected triangle and mirrors it; the complex routine is
  complex symmetric, not Hermitian.
- LANTR selects one triangle and optionally treats the diagonal as unit.

The C return value is a real scalar (`float` for s/c, `double` for d/z), not an
integer INFO. On invalid layout or leading dimension the reference returns the
negative argument number converted to that real type; high-level NaN checks use
-5 for LANGE/LANSY and -7 for LANTR.

Row-major LANGE and LANTR are evaluated on the transposed storage view, swapping
1/infinity norm names (and dimensions/UPLO for LANTR). LANSY swaps UPLO. This is
the reference's physical interpretation and touches no padding.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, all norm modes, symmetric/triangular selection and unit diagonal.
- Padding guards, NaN diagnostics, leading-dimension diagnostics and invalid
  layouts.
- Independent C11 `-Wall -Wextra -Werror` caller for all 24 symbol/ABI variants.
- The immutable full-suite artifact records the actual return type, both
  integer ABIs and both routes.

## Remaining scope

LAPACKE HEEV/band/packed norm variants and the rest of the 1,616-entry #288
scope are not implied by this stage.
