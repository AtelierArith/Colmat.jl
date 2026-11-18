# Issue #288: SYTRF/SYTRS/SYSV and HETRF/HETRS/HESV C contracts

Status: the 24 S/D/C/Z high/work entries pass focused factorization/solve,
driver, layout, diagnostic and independent-call checks. The immutable full suite
records 420 sources, 2,180 canonical entries and 8,175 witnesses (2,384 forwards
per integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d}{sytrf,sytrs,sysv}` and `LAPACKE_{c,z}{hetrf,hetrs,hesv}` with
their `_work` forms (24 entries).

## Contracts

- Dense symmetric (SY) or Hermitian (HE) storage with UPLO `U`/`L`; the
  Bunch-Kaufman factor overwrites A and IPIV records the pivots.
- SYTRF/HETRF work takes WORK/LWORK; the high-level entry queries LWORK=-1,
  allocates the requested workspace and calls the work entry. SYTRS/HETRS has
  no workspace argument. SYSV/HESV work takes WORK/LWORK.
- Row-major requires LDA >= N (TRF -5, TRS/SV -6) and LDB >= NRHS (-9). A and B
  are converted over their logical rectangles only; padding is preserved.
- High-level NaN order: TRF A (-4); TRS/SV A (-5), B (-8).
- Negative native INFO is shifted once for the layout argument; positive
  singularity INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, factor and IPIV equivalence across layouts, solve
  residuals, SYSV/HESV end-to-end equivalence, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
