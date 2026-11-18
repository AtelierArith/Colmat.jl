# Issue #288: GGHRD/GGHD3 C contracts

Status: the 16 S/D/C/Z high/work entries pass focused layout, option, diagnostic
and independent-call checks. The immutable full suite records 395 sources,
1,992 canonical entries and 7,415 witnesses (2,196 forwards per integer ABI).
Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gghrd,gghd3}` and their `_work` forms (16 entries).

## Contracts

- COMPQ/COMPZ `N`/`V` (V accumulates Q or Z); ILO/IHI follow the reference
  range; A and B are N-by-N.
- Row-major requires LDA >= N (-8), LDB >= N (-10), LDQ >= N (-12) and
  LDZ >= N (-14), matching the reference work entries (the Q/Z dimension
  checks are unconditional there).
- A, B, Q and Z are converted over the logical N-by-N rectangle only; padding
  is preserved and Q/Z are written back only for COMPQ/COMPZ='V'.
- High-level NaN order: A (-7), B (-9), Q (-11, only when V), Z (-13, only when
  V).
- GGHD3 `_work` accepts the leading dimension of WORK and LWORK; the high-level
  entry allocates 3N scalars. GGHRD `_work` has no workspace argument.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, GGHRD/GGHD3, COMPQ/COMPZ N/V, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

HGEQZ and the rest of the 1,616-entry #288 scope are not implied by this stage.
