# Issue #288: PBEQU/PBSTF/PBSV C contracts

Status: the 24 S/D/C/Z high/work entries pass focused band-layout, equilibration,
split-factor, driver, diagnostic and independent-call checks. The immutable full
suite records 438 sources, 2,352 canonical entries and 8,879 witnesses (2,556
forwards per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{pbequ,pbstf,pbsv}` and `_work` forms (24 entries).

## Contracts

- PBEQU returns the real S (N), SCOND and AMAX from the band diagonal; S is the
  reciprocal square root of the diagonal entries. Row-major requires
  LDAB >= N (-6).
- PBSTF computes the split Cholesky factor of the positive-definite band; the
  factor overwrites BB in place and round-trips through the row-major band
  conversion. Row-major requires LDBB >= N (-6).
- PBSV factors and solves in one call; row-major requires LDAB >= N (-7) and
  LDB >= NRHS (-9). The factor is written back to the row-major band and B is
  converted over the logical N-by-NRHS rectangle.
- All three use the KD+1-row band convention AB(KD+1+i-j, j) shared with
  PBTRF/PBTRS.
- High-level NaN order: PBEQU AB (-5); PBSTF BB (-5); PBSV AB (-6), B (-8).
- Native PBEQU info is returned directly (positive values report a
  non-positive diagonal); PBSTF/PBSV negative info is shifted once for the
  layout argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, KD=1/2, S/SCOND/AMAX equality, split-factor diagonal
  equality, PBSV end-to-end solve residual, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
