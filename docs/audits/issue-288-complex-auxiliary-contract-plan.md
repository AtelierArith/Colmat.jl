# Issue #288: LACGV/LACP2 C contracts

Status: the 8 C/Z high/work entries pass focused layout, stride, diagnostic and
independent-call checks. The immutable full suite records 375 sources, 1,896
canonical entries and 7,031 witnesses (2,100 forwards per integer ABI). Issue
#288 remains open.

## Connected names

`LAPACKE_{c,z}{lacgv,lacp2}` and `_work` forms (8 entries).

## Contracts

- LACGV has no matrix-layout argument: `(n, x, incx)`. It conjugates the N
  entries at the signed stride; padding is untouched. High-level NaN checking
  uses the actual INCX and returns -2.
- LACP2 copies the selected triangle of real A into complex B. Row-major
  requires LDA >= N (-6) and LDB >= N (-8). Only the logical M-by-N rectangle
  is converted; B padding is preserved.
- LACP2 high-level NaN checks A (-5). The Fortran owner has no INFO output and
  reports invalid arguments through the shared diagnostic channel; the LAPACKE
  entries return 0 after the work call.

## Evidence

- ComplexF32/ComplexF64, both integer ABIs, both layouts (LACP2), high/work,
  DLL and LBT routes, positive/negative strides, LACP2 triangles, padding
  guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
