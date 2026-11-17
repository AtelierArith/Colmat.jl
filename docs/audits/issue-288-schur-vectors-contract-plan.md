# Issue #288: TREVC C contracts

Status: the 8 S/D/C/Z high/work entries pass focused layout, select, diagnostic
and independent-call checks. The immutable full suite records 369 sources,
1,872 canonical entries and 6,935 witnesses (2,076 forwards per integer ABI).
Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}trevc` and `_work` forms (8 entries).

## Contracts

- SIDE `L`/`R`/`B`, HOWMNY `A`/`S`, SELECT (`lapack_logical`, integer width),
  MM, M and the real conjugate-pair column convention follow the Fortran
  entry.
- Row-major checks are LDT >= N (-7), LDVL >= MM (-9) and LDVR >= MM (-11).
  T, VL and VR are converted over the logical rectangle only; padding is never
  read or written. HOWMNY='S' packs selected vectors into the leading columns
  and leaves the rest unreferenced.
- High-level NaN order: T (-6), VL (-8) when SIDE includes L, VR (-10) when
  SIDE includes R. Real work allocates 3N; complex work allocates 2N complex
  plus N real.
- Invalid layout/flags/dimensions return the reference negative indices; the
  LAPACKE layout shift is applied exactly once.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, SIDE L/R/B, HOWMNY A/S, real 2x2 blocks, SELECT/M propagation,
  padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

TREVC3 and the rest of the 1,616-entry #288 scope are not implied by this
stage.
