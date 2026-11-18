# Issue #288: LANGB/LANHE C contracts

Status: the 12 high/work entries (LANGB for four scalar types, LANHE for the
complex pair) pass focused norm, layout, diagnostic and independent-call checks.
Issue #288 remains open.

## Connected names

`LAPACKE_{s,d,c,z}langb` and `LAPACKE_{c,z}lanhe` with `_work` forms
(12 entries).

## Contracts

- LANGB uses the analysis band convention AB(KU+1+i-j, j) with LDAB >= KL+KU+1;
  row-major requires the same bound (-7). The row-major band is converted over
  the logical band entries and every norm ('1'/'O', 'I', 'M', 'F') returns the
  same value as column-major. The upstream row-major kl/ku swap trick returns
  transposed or garbage norms and is deliberately not reproduced.
- LANHE reads only the selected UPLO triangle; row-major converts that triangle
  over the logical rectangle and requires LDA >= N (-6).
- Both return a real scalar for every scalar type; the work entry takes the real
  WORK vector that the reference allocates for norm 'I'.
- High-level NaN order: LANGB AB (-6); LANHE A (-5).

## Evidence

- LANGB: four scalar types, both integer ABIs, both layouts, high/work, DLL and
  LBT routes, norms 1/I/M/F/O, KL/KU combinations, row==column equivalence and
  dense-norm agreement. LANHE: both complex types, UPLO U/L, the same norm set
  and agreements.
- Diagnostics cover the layout, LDAB/LDA and NaN indices.
- The full-suite artifact records both ABIs and both actual routes.

## Remaining scope

The rest of the 1,616-entry #288 scope is not implied by this stage.
