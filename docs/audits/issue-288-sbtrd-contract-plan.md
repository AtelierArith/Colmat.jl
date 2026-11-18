# Issue #288: SBTRD/HBTRD C contracts

Status: the 16 S/D/C/Z high/work entries pass focused band-layout, reduction,
vector, diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{sbtrd,hbtrd}` and `_work` forms (16 entries).

## Contracts

- Symmetric/Hermitian band storage with UPLO `U`/`L`, bandwidth KD and KD+1
  band rows; VECT `N`/`V` selects whether Q is accumulated.
- Row-major AB uses `LDAB` as the row stride of a `(KD+1)`-by-`LDAB` buffer and
  requires LDAB >= N (-7). VECT='V' requires LDQ >= N (-11). Column-major uses
  `LDAB >= KD+1`.
- Band entries are reordered with the reference `dpb_trans`/`zpb_trans` index
  mapping; the reduction writes the reflector band back to AB and D/E stay
  real. Q is converted over the logical N-by-N rectangle only.
- High-level NaN order: AB (-6), Q (-10, only when VECT='V').
- Negative native INFO is shifted once for the layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, KD=1/2, VECT N/V, D/E and AB equivalence across layouts,
  Q orthogonality, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
