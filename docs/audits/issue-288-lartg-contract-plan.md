# Issue #288: LARTGP/LARTGS C contracts

Status: the 8 S/D high/work entries pass focused scalar, diagnostic and
independent-call checks. The immutable full suite records 385 sources, 1,936
canonical entries and 7,191 witnesses (2,140 forwards per integer ABI). Issue
#288 remains open.

## Connected names

`LAPACKE_{s,d}{lartgp,lartgs}` and `_work` forms (8 entries).

## Contracts

- No matrix-layout argument.
- LARTGP takes F/G by value and returns CS/SN/R. High-level NaN order is F (-1),
  G (-2).
- LARTGS takes X/Y/SIGMA by value and returns CS/SN. High-level NaN order is
  SIGMA (-3), X (-1), Y (-2).
- Work entries return 0 after the native Givens routine; invalid numeric inputs
  are diagnosed by the high-level entries when NaN checking is enabled.

## Evidence

- Float32/Float64, both integer ABIs, high/work, DLL and LBT routes, degenerate
  G=0/F=0 cases, Givens orthogonality and diagnostics.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
