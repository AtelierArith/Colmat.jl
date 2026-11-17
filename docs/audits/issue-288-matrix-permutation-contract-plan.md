# Issue #288: LAPMR/LAPMT C contracts

Status: the 32 S/D/C/Z high/work entries pass focused numerical, layout,
diagnostic and independent C ABI checks. The immutable full suite is recorded
in `docs/src/assets/lapack-abi-execution.json`. Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}{lapmr,lapmt}` and their `_work` forms (32 entries). They call
the existing native permutation entries.

## Contracts

- `FORWRD` is `lapack_logical` (`lapack_int`) and is passed through unchanged.
- LAPMR permutes rows; LAPMT permutes columns. `K` is used as the reference
  permutation vector and is not modified.
- Row-major storage is converted over the logical M-by-N rectangle only; row
  padding is neither read nor written. Row-major requires `LDA >= N` (-6).
- Invalid layout is -1; negative M/N and a too-small column-major LDA follow
  the Fortran indices shifted by one.

## Evidence

- Four scalar types, both integer ABIs, both layouts, high/work, DLL and LBT
  routes, forward/backward permutations, non-square shapes and padding guards.
- Independent C11 `-Wall -Wextra -Werror` caller for all 32 symbol/ABI variants.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The other LAPACKE families in the 1,616-entry #288 scope are not implied by this
stage.
