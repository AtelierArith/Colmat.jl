# Issue #288: SPTRF/SPTRS/SPSV C contracts

Status: the 12 S/D high/work entries pass focused packed-layout, Bunch-Kaufman,
solve, driver, diagnostic and independent-call checks. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d}{sptrf,sptrs,spsv}` and `_work` forms (12 entries). The complex
packed variants are outside the original candidate cohort.

## Contracts

- Packed symmetric storage with UPLO `U`/`L`; AP holds N(N+1)/2 values and the
  public C signature has no LDAP argument, so the row-major buffer is tightly
  packed.
- Row-major AP is reordered with the reference `sp_trans` permutation. The
  captured Bunch-Kaufman factor follows the same convention, and SPTRS/SPSV
  convert it back through the same map before solving.
- IPIV is identical across layouts for both the factorization and the driver.
- SPTRS/SPSV take `(layout, uplo, n, nrhs, ap, ipiv, b, ldb)`; row-major
  requires LDB >= NRHS (-8). B is converted over the logical N-by-NRHS
  rectangle only.
- High-level NaN order: SPTRF AP (-4); SPTRS/SPSV AP (-5), B (-7).
- Negative native INFO is shifted once for the layout argument; positive
  singularity INFO is passed through.

## Evidence

- Float32/Float64, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, factor and IPIV equivalence through the sp_trans
  permutation, solve residual, SPSV end-to-end equivalence, padding guards and
  diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
