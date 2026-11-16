# Issue #288: LAUUM C adapters and complex diagonal compatibility

Status: focused numerical, C ABI and completed-call checks pass. Immutable
full-suite proof is pending. Issue #288 remains open.

Eight S/D/C/Z LAUUM high/work names extend the preceding 164-name candidate to
172 original-scope names. The original 1616-name scope retains 1444 names beyond
this candidate; the separate 20 Rook C targets remain additional work.

## Reference and boundary behavior

Reference LAPACK v3.12.1
[LAUUM work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zlauum_work.c)
uses C-int layout, C-char UPLO, lapack_int dimensions/INFO, and identical high/work
signatures without query arguments. Row LDA errors are argument 5 and precede
flag validation; high NaN A diagnostics are argument 4. Layout conversion copies
the selected complex triangle without conjugation and preserves unused storage.

The existing own Fortran boundary's full-complex triangular product differs
from the reference complex LAUUM/LAUU2 algorithms on nonreal diagonals. The
executed `reproducers/lauum_complex_diagonal.jl` gives scalar 2+3i: own result 13,
independent OpenBLAS 4+6i. Reference
[ZLAUU2](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zlauu2.f)
uses the real part of diagonal entries during its updates. The correction saves
independent ILP64 C/Z LAUUM/LAUU2 pointers before forwarding. Real diagonals retain
the native product path; nonreal or NaN complex diagonals use the corresponding
saved implementation. LAUU2 retains its own saved entry rather than substituting
the blocked routine. LP64 INFO conversion is checked. Missing configuration
reports INFO=1; the Julia loader requires saved implementations. Self-recursive
registrations are rejected. No native upstream source is modified.

Real-diagonal tests compare triangular products directly. Nonreal/used-NaN
tests compare saved independent reference outputs, including blocked sizes.
High-level NaN inputs return -4; work calls retain numerical reference behavior.
The independent reference itself propagates scalar lower-triangle diagonal
imaginary NaNs differently at LDA=1 and LDA=3. Row-major reference tests therefore
use the compact LDA=max(1,N) required by Reference LAPACKE, while column-major
tests use caller LDA. Both real and imaginary output components remain checked;
this is not replaced by a whole-complex NaN comparison.

## Executed focused evidence

- Existing Fortran tests pass: solve/inverse 5120, products 768, singular/argument
  752, forwarding/Julia API 88. Products include real/nonreal diagonals,
  LAUUM/LAUU2, both ABIs, selected output and N=0/1/3/80.
- Rust full suite: 452 passed. Two new tests cover the native real-diagonal path,
  wide arguments above int32, hidden character length, saved registration,
  missing configuration, recursive-pointer rejection and guarded INFO narrowing.
- Julia C comparisons: 1,895,200 assertions passed for both ABIs, layouts,
  high/work, DLL/LBT, upper/lower and N=0/1/3/8/80, with real, nonreal and NaN
  diagonals and ignored triangle/padding poison.
- Independent C11 caller passes all eight names with exact C declarations,
  direct Fortran references, explicit saved-driver registration, both layouts,
  high/work, N=0/1/2/70, diagonal variants, output guards and diagnostics.
- Python evidence fixtures: 17 passed. Completed-call tracing validates 172
  names and 688 observations; full_suite_passed remains explicitly false.

Optimized build and repeated independent C/numerical/signature checks pass.
The frozen optimized DLL SHA256 is
`3aeb0dd43cda9f92414e4606727b856df54119964a5bd7018b9ffc11e0f6b95c`.
The completed-call gate passes 3446 checks; registration is 1654 forwards per ABI.
Immutable full package testing, evidence import, documentation and Linux/macOS
artifact provenance checks remain required.
