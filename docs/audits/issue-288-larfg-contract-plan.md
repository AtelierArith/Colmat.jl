# Issue #288: LARFG C contracts

Status: the 8 S/D/C/Z high/work entries pass focused stride, diagnostic and
independent-call checks. The immutable full suite records 379 sources, 1,912
canonical entries and 7,095 witnesses (2,116 forwards per integer ABI). Issue
#288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}larfg` and `_work` forms (8 entries).

## Contracts

- LARFG has no matrix-layout argument: `(n, alpha, x, incx, tau)`.
- Alpha, x and tau are in/out; the signed INCX stride is honored and unused
  padding is not read beyond the actual entries for NaN checks.
- High-level NaN order: alpha (-2), x (-3). Work returns 0 after the native
  Householder generator; invalid arguments surface through the shared
  diagnostic channel.
- N=0 returns without dereferencing alpha/x/tau.

## Evidence

- Four scalar types, both integer ABIs, high/work, DLL and LBT routes, N=0/1/N,
  positive/negative strides and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

LARFT/LARFB and the rest of the 1,616-entry #288 scope are not implied by this
stage.
