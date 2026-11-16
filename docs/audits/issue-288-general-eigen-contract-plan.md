# Issue #288: general eigenvalue C adapters

Status: GEEV implementation and focused validation in progress. No full-suite
execution proof has been imported for this candidate. Issue #288 remains open.

The eight S/D/C/Z GEEV high-level and work exports add both LP64 and ILP64
adapters. Together with the preceding range drivers this candidate has 140
numerical LAPACKE names and 1622 forwarding entries per integer ABI. The original
1616-name LAPACKE scope still has 1476 names remaining; the separate 20 Rook C
targets remain additional work. GEEVX is not part of this implementation yet.

## C and numerical contracts

The adapters follow the Reference LAPACK v3.12.1
[real GEEV work wrapper](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dgeev_work.c)
and [complex GEEV work wrapper](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zgeev_work.c).
The layout integer is C int, flags are C char, dimensions and INFO use lapack_int.
Real eigenvalues use separate WR/WI arrays; complex eigenvalues use complex W.
Complex work calls have real RWORK. High-level calls own initialized scratch.

Row-major input and requested output vectors are packed and copied without
conjugation. Unrequested vector arrays and padding remain untouched. Real
conjugate pairs keep the adjacent real/imaginary column representation. Query
calls do not read or transpose A. Negative backend INFO prevents copy-back.
Minimum caller WORK is max(1,2N) for complex, max(1,3N) for real eigenvalues only,
and max(1,4N) for real eigenvectors; a larger optimal query is not a minimum.
The high-level NaN policy scans logical A entries and ignores padding.

The underlying Colmat.jl Fortran boundary owns algorithm scratch; caller complex
RWORK is not used by that boundary. Algorithm identity and partial convergence
diagnostics are not asserted by the present focused tests.

## Evidence and remaining gates

- Rust suite: 438 tests passed, including eight GEEV cases covering both ABIs,
  scalar types, layouts, independently requested left/right vectors, minimum
  workspace, query immutability, N=0/1/3, and padding guards.
- Independent C11 caller: eight names passed strict compiler warnings, real and
  complex prototypes, N=0..3, high/work calls, layouts, eigenvalue/vector
  residuals, guard regions, errors, and a negative dimension outside int32.
- Julia numerical tests: 17,504 assertions passed for N=0/1/3/5, both layouts,
  DLL/LBT routes, both integer ABIs, scalar types and vector jobs. Expected
  spectra come from direct saved OpenBLAS ILP64 GEEV pointers, verified distinct
  from adapter pointers. Matching consumes each reference eigenvalue once.
- Python execution-evidence fixtures: 17 tests passed with real/complex GEEV
  high/work signatures added.
- Traced focused run: 140 names and 560 completed name/ABI/route observations
  passed signature and return-type validation. The evidence explicitly records
  full_suite_passed=false and is not imported as full-suite proof.

Before publication: complete traced focused validation, optimized build checks,
immutable full Pkg.test(), source-bound evidence import, regenerated inventory,
documentation checks, and Linux/macOS CI artifact validation. Focused evidence
must remain explicitly marked full_suite_passed=false.

The next GEEVX stage needs separate balancing/condition-number contracts and
integer output guards. Reference
[DGEEVX](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgeevx.f)
and [ZGEEVX](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zgeevx.f)
make minimum WORK depend on SENSE and requested vectors; optimal query sizes
must not substitute for those minima.
