# Issue #288: POEQU/POEQUB/PPEQU/PPSV/PPTRI C contracts

Status: the 40 S/D/C/Z high/work entries pass focused equilibration, solve,
inverse, layout, diagnostic and independent-call checks. Issue #288 remains
open.

## Connected names

`LAPACKE_{s,d,c,z}{poequ,poequb,ppequ,ppsv,pptri}` and `_work` forms
(40 entries).

## Contracts

- POEQU/POEQUB return the real S (N), SCOND and AMAX from the dense diagonal;
  S is the reciprocal square root of the diagonal and POEQUB bins it into
  powers of two. Row-major requires LDA >= N (-4) and converts A over the
  logical N-by-N rectangle.
- PPEQU does the same from the packed diagonal with UPLO U/L; the row-major
  packed AP is reordered with the reference pp_trans permutation.
- PPSV factors and solves in one call; the factor is written back through the
  same permutation and B is converted over the logical N-by-NRHS rectangle.
  Row-major requires LDB >= NRHS (-7).
- PPTRI takes the packed Cholesky factor and overwrites it with the inverse of
  the original matrix, following the LAPACK contract; the row-major result is
  the pp_trans permutation of the column-major one.
- High-level NaN order: POEQU/POEQUB A (-3, full logical matrix); PPEQU AP
  (-4); PPSV AP (-5), B (-6); PPTRI AP (-4).
- Negative native INFO is shifted once for the layout argument on every path,
  including the column-major native argument diagnostics (for example a
  column-major POEQU LDA < N reports -4 and PPEQU bad UPLO reports -2); this
  batch also aligns the already-published GEEQU/GBEQUB/PBEQU adapters with the
  same reference shift. Positive non-positive-definite INFO is passed through.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, UPLO U/L, N=3/4, S/SCOND/AMAX equality across layouts, the analytic
  reciprocal-square-root diagonal (binned within a factor of two for
  POEQUB/PPTRI-adjacent radix variants), PPSV end-to-end residuals with the
  factor write-back verified through the pp_trans permutation, PPTRI
  factor-to-inverse round trips on both layouts against the mirrored inverse
  identity inv(A) * A = I, padding guards and diagnostics.
- The PPTRI reference factor is built with the native PPTRF entry.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
