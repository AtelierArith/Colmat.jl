# Issue #288: GGEV/GGEV3 C contracts

Status: the 16 S/D/C/Z high/work entries pass focused layout, eigenvalue/vector,
diagnostic and independent-call checks. The immutable full suite is recorded in
`docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{ggev,ggev3}` and `_work` forms (16 entries).

## Contracts

- JOBVL/JOBVR `N`/`V`; real entries return ALPHAR/ALPHAI/BETA and complex
  entries ALPHA/BETA.
- Row-major requires LDA/LDB >= N (-6/-8) and LDVL/LDVR >= N when vectors are
  requested (-13/-15). A/B/VL/VR are converted over the logical N-by-N
  rectangle; padding is preserved.
- High-level NaN checks A (-5), B (-7). Work queries LWORK=-1, allocates the
  requested complex workspace and 8N real RWORK for complex entries.
- GGEV3 uses the blocked native driver; both LAPACKE names are ABI adapters
  over the corresponding native family.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL/LBT
  routes, GGEV/GGEV3, alpha/beta and vector row/column-major equivalence.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

GGEVX and the rest of the 1,616-entry #288 scope are not implied by this stage.
