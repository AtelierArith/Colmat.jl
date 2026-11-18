# Issue #288: OPGTR/OPMTR C contracts

Status: the 8 S/D high/work entries pass focused packed-layout, diagnostic and
independent-call checks. The immutable full suite records 400 sources, 2,016
canonical entries and 7,519 witnesses (2,220 forwards per integer ABI). Issue
#288 remains open.

## Connected names

`LAPACKE_{s,d}{opgtr,opmtr}` and `_work` forms (8 entries).

## Contracts

- Packed symmetric storage with UPLO `U`/`L`; AP holds N(N+1)/2 values.
- OPGTR takes the packed reflectors from SSPTRD/DSPTRD, TAU (N-1) and produces
  the N-by-N orthogonal Q. Row-major requires LDQ >= N (-7).
- OPMTR applies Q with SIDE `L`/`R` and TRANS `N`/`T`; the reflector order is
  R = M for SIDE='L' and R = N for SIDE='R'. Row-major requires LDC >= N (-10).
- Row-major packed AP is reordered into column-major packed storage over the
  logical triangle only, using the same permutation as the reference
  `spp_trans`/`dpp_trans` helpers.
- High-level NaN order: OPGTR AP (-4), TAU (-5); OPMTR AP (-7), C (-9), TAU
  (-8).
- The work entries take WORK without LWORK, matching the reference headers.

## Evidence

- Float32/Float64, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, SIDE L/R, TRANS N/T, rectangular C, packing round trips,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
