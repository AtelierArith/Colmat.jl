# Issue #288: PBTRF/PBTRS C contracts

Status: the 16 S/D/C/Z high/work entries pass focused band-layout,
factorization/solve, diagnostic and independent-call checks. The immutable full
suite records 404 sources, 2,048 canonical entries and 7,647 witnesses (2,252
forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{pbtrf,pbtrs}` and `_work` forms (16 entries).

## Contracts

- Symmetric/Hermitian positive-definite band storage with UPLO `U`/`L`,
  bandwidth KD and `KD+1` band rows.
- Row-major AB uses `LDAB` as the row stride of a `(KD+1)`-by-`LDAB` buffer and
  requires LDAB >= N (-6 for PBTRF, -7 for PBTRS). Column-major uses
  `LDAB >= KD+1`.
- Row-major PBTRS requires LDB >= NRHS (-9). B is converted over the logical
  N-by-NRHS rectangle only; padding is preserved.
- Band entries are reordered with the same index mapping as the reference
  `dpb_trans`/`zpb_trans` helpers; factorization write-back touches only band
  entries.
- High-level NaN order: PBTRF AB (-5); PBTRS AB (-6), B (-8).
- Negative native INFO is shifted once for the layout argument; positive
  non-positive-definite INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, KD=1/2, factor equality across layouts, round trips, solve
  residual in double precision, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
