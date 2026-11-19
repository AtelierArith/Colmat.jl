# Issue #288: TFSM/TFTRI/PTSV C contracts

Status: the 24 S/D/C/Z high/work entries pass focused RFP triangular
solve/multiply/inverse, tridiagonal solve, layout, diagnostic and
independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{tfsm,tftri,ptsv}` and `_work` forms (24 entries). TFSM high
and work share the same signature.

## Contracts

- TFSM solves op(A)*X = alpha*B (SIDE `L`, A m-by-m) or X*op(A) = alpha*B
  (SIDE `R`, A n-by-n) with A triangular in RFP storage, TRANS `N`/`T`/`C`,
  DIAG `N`/`U`. The native entry reports no INFO, so the column-major path
  returns 0. Row-major requires LDB >= M (-12, reproducing the reference
  check); B is converted over the logical M-by-N rectangle and the RFP factor
  over the rectangle sized by SIDE.
- TFTRI inverts a triangular RFP matrix in place; row-major storage is the
  dense transpose of the ROW-by-COL rectangle and round-trips through it.
- PTSV factors and solves a positive-definite tridiagonal system with real D
  and T-typed E destroyed in place identically across layouts. Row-major
  requires LDB >= NRHS (-7).
- High-level NaN order: TFSM A (-10, only when alpha is nonzero), alpha (-9),
  B (-11, only when alpha is nonzero); TFTRI A (-6, all N(N+1)/2 entries);
  PTSV B (-6), D (-4), E (-5).
- Negative native INFO is shifted once for the layout argument; positive
  non-positive-definite INFO from PTSV is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, TRANSR N and T/C, UPLO U/L, N odd and even, SIDE L/R, TRANS N/T/C,
  DIAG N/U, the op(A)*X = alpha*B and X*op(A) = alpha*B identities on both
  layouts, TFTRI inverse identities inv(T)*T = I through the RFP round trip,
  PTSV end-to-end residuals with identical destroyed D/E, padding guards and
  diagnostics.
- RFP input buffers are built with the native TRTTF entry and unpacked with
  the native TFTTR entry.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
