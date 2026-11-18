# Issue #288: UPGTR/UPMTR C contracts

Status: the 8 C/Z high/work entries pass focused packed-layout, diagnostic and
independent-call checks. The immutable full suite records 399 sources, 2,008
canonical entries and 7,483 witnesses (2,212 forwards per integer ABI). Issue
#288 remains open.

## Connected names

`LAPACKE_{c,z}{upgtr,upmtr}` and `_work` forms (8 entries).

## Contracts

- Packed Hermitian storage with UPLO `U`/`L`; AP holds N(N+1)/2 values.
- UPGTR takes the packed reflectors from HPTRD, TAU (N-1) and produces the
  N-by-N unitary Q. Row-major requires LDQ >= N (-7).
- UPMTR applies Q with SIDE `L`/`R` and TRANS `N`/`C`; the reflector order is
  R = M for SIDE='L' and R = N for SIDE='R'. Row-major requires LDC >= N (-10).
- Row-major packed AP is reordered into column-major packed storage over the
  logical triangle only; no conjugation is applied, matching the reference
  `cpp_trans` permutation used by LAPACKE for these routines.
- High-level NaN order: UPGTR AP (-4), TAU (-5); UPMTR AP (-7), C (-9), TAU
  (-8).
- The work entries take WORK without an LWORK argument, as in the reference
  headers, and the high-level entries allocate it.

## Evidence

- ComplexF32/ComplexF64, both integer ABIs, both layouts, high/work, DLL and
  LBT routes, UPLO U/L, SIDE L/R, TRANS N/C, rectangular C, packing round
  trips, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
