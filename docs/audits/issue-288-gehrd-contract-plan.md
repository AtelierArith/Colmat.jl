# Issue #288: GEHRD/GEBRD/LARFX C contracts

Status: the 24 S/D/C/Z high/work entries pass focused reduction, reflector,
layout, diagnostic and independent-call checks. The immutable full suite records
428 sources, 2,256 canonical entries and 8,495 witnesses (2,460 forwards per
integer ABI). Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gehrd,gebrd,larfx}` and `_work` forms (24 entries). LARFX
high and work share the same signature, matching the reference headers.

## Contracts

- GEHRD reduces A to upper Hessenberg form over the ILO..IHI range; TAU holds
  the reflectors and the lower part of A stores the reflector vectors. The
  high-level entry queries LWORK=-1, allocates and calls the work entry.
- GEBRD reduces an M-by-N A to bidiagonal form with real D/E and TAUQ/TAUP;
  the work entry takes WORK/LWORK and the high-level entry queries it first.
- LARFX applies an elementary reflector with SIDE L/R using the full vector
  convention shared with LARF; orders <= 10 take the unblocked path and the
  workspace is scratch only.
- Row-major requires LDA >= N (GEHRD -6, GEBRD -5) and LDC >= N (LARFX -8). A
  and C are converted over their logical rectangles only.
- High-level NaN order: GEHRD A (-5); GEBRD A (-4); LARFX C (-7), TAU (-6),
  V (-5, with LV = M or N elements).
- Negative native INFO is shifted once for the layout argument; LARFX reports
  0 because the native routine has no INFO output.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, Hessenberg verification through ORGHR/UNGHR reconstruction
  (Q' A0 Q has zero lower triangle below the subdiagonal), GEBRD square and
  rectangular shapes with D/E/TAUQ/TAUP equality, LARFX reflector identity on
  both blocked and unblocked orders, padding guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
