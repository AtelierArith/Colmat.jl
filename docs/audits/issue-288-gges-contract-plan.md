# Issue #288: GGES/GGES3 C contracts

Status: the 16 S/D/C/Z high/work entries pass focused layout, Schur-form,
eigenvalue/vector, diagnostic and independent-call checks for SORT='N'. Issue
#288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{gges,gges3}` and `_work` forms (16 entries).

## Contracts

- JOBVSL/JOBVSR `N`/`V`, SORT `N`/`S`; the SELECT callback signature follows
  the reference C prototypes (three real arguments for S/D, two complex
  arguments for C/Z).
- Real entries carry ALPHAR/ALPHAI/BETA; the complex entries carry ALPHA/BETA
  and real RWORK in the work form.
- Row-major requires LDA >= N (-8), LDB >= N (-10), LDVSL >= N when vectors are
  requested (-16) and LDVSR >= N (-18). A/B/VSL/VSR are converted over the
  logical N-by-N rectangle; padding is preserved.
- High-level NaN order: A (-7), B (-9). The high-level entries query LWORK=-1,
  allocate the requested workspace (and 8N real RWORK for complex) plus BWORK
  and call the work entry.
- Negative native INFO is shifted once for the layout argument; GGES needs at
  least MAX(8N, 6N+16) workspace and GGES3 at least 6N+16.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, JOBVSL/JOBVSR N and V, alpha/beta and SDIM equality, Schur form
  (S upper Hessenberg, P upper triangular), VSL/VSR orthogonality, padding
  guards and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

GGESX and the rest of the 1,616-entry #288 scope are not implied by this stage.
