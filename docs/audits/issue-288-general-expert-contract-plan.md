# Issue #288: GEEVX C adapter plan

Status: immutable full package validation passed at
`6530ac871eb0ac2aabd9a314125f1132e5f8d093`. All 271 source hashes, manifest,
registration and optimized DLL matched the pre-run freeze before evidence import.
This stage adds
eight high/work names beyond the 140-name GEEV candidate. Completion of that
148-name group does not complete the original 1616-name LAPACKE scope or the
separate 20 Rook C targets. Issue #288 remains open.

## Workspace contracts

The authoritative minimum formulas are in Reference LAPACK v3.12.1
[DGEEVX](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/dgeevx.f)
and [ZGEEVX](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zgeevx.f).
For N=0 all modes require one WORK element. For positive N:

| Scalar | SENSE | Requested vectors | Minimum WORK |
| --- | --- | --- | --- |
| real | N | neither | 2N |
| real | N or E | either | 3N |
| real | V or B | any legal job pair | N²+6N |
| complex | N or E | any legal job pair | 2N |
| complex | V or B | any legal job pair | N²+2N |

SENSE E/B requires both left and right vectors. SENSE V permits independently
requested vectors. BALANC accepts N/P/S/B. Lowercase flags must behave equally.
Optimal workspace queries may exceed these legal minima; test minimum and
minimum-minus-one independently. Compute quadratic formulas in a checked wider
integer before conversion. Query calls must preserve A, vector arrays, ILO/IHI,
SCALE, ABNRM, and condition outputs.

## ABI and layout gates

- C layout stays int and flags char, independent of lapack_int width.
- Real WR/WI are separate; complex W has no WI argument.
- ILO/IHI and real work-call IWORK use the requested lapack_int width. Complex
  work calls take real RWORK instead of integer IWORK.
- Row-major LDA is argument 8. Real LDVL/LDVR are arguments 12/14;
  complex LDVL/LDVR are 11/13. Validate row strides before flag errors to match
  Reference LAPACKE. Fortran negative INFO is shifted by one for layout.
- Pack full logical A and requested vectors without conjugation. Preserve
  padding and unrequested vectors. Do not copy row temporaries after negative
  INFO. Real conjugate pairs retain adjacent real/imaginary columns.
- High-level NaN A diagnostics use argument 7. Scratch is initialized and owned.
- Existing Colmat.jl expert Fortran dispatch saves independent implementations
  and bridges LP64 integer outputs through ILP64 with checked conversion.
  C adapters must preserve this behavior rather than estimate condition values.

## Required numerical evidence

Cover S/D/C/Z, both integer ABIs, both layouts, high/work and DLL/LBT, N=0/1/3/5,
all BALANC/SENSE modes and every legal left/right job pair. Use direct saved
OpenBLAS pointers as independent references; compare condition outputs only
when requested. Verify ILO/IHI, SCALE and ABNRM with output guards, balancing
semantics, left/right residuals and unit norms. Include nonnormal matrices,
real conjugate pairs, and complex matrices with nonzero imaginary entries.

An independent C caller must register expert ILP64 fallbacks explicitly before
expert modes because Julia initialization is absent. Guard ILO/IHI with negative
sentinels that reveal partial 32-bit writes into 64-bit values. Check unused
condition arrays and workspaces, bad flags, dimensions outside int32 and each
first-invalid diagnostic. Integrate exact typed execution signatures and
actual returned-call witnesses before the immutable full-suite run.

## Initial execution and empty-output defect

All eight high/work names compile for both integer widths. Four generic Rust
tests pass for real/complex row packing, requested-vector copy-back, negative
INFO output preservation, and all legal/illegal SENSE/job workspace combinations.

The first independent OpenBLAS comparison completed 271,936 assertions with
1,024 failures, all in ILO/IHI for N=0/SENSE=N. The minimal executed
`reproducers/geevx_empty_outputs.jl` showed own Fortran INFO=0/ILO=1/IHI=0/ABNRM=0,
versus independent OpenBLAS INFO=0 and unchanged -777/-777/777 sentinels.
The existing own Fortran SENSE=N path lacked the reference empty quick return.
It now returns after workspace diagnostics/query output and before numerical
output writes. This is a Colmat.jl boundary correction, not a native upstream
defect. The rebuilt minimal example agrees with OpenBLAS, all 275,520 C adapter
checks pass, and all 7,104 existing Fortran argument/workspace boundary checks
pass. The earlier failed run is not counted as successful evidence.

Expanded N=0/1/3/5 and lowercase high-level flags pass 411,648 assertions.
Rust's complete suite passes 442 tests. An independent C11 caller passes strict
compiler warnings and all eight names with both layouts, N=0/1/2, every
BALANC/SENSE mode, high/work calls, integer output guards, legal minimal work,
query preservation, condition outputs and vector residuals. It explicitly
registers separately resolved ILP64 expert fallbacks and verifies pointer
independence. Full source-bound proof remains pending.

The expanded Schur overwrite checks (triangular/quasi-triangular structure,
trace and determinant) and largest-component-real eigenvector convention pass
476,160 assertions. Independent C header diagnostics also pass, including
row-stride precedence, each flag/vector-stride position, NaN A, and a negative
dimension outside int32. Completed-call tracing covers 148 names and 592
name/ABI/route observations with exact signatures; these focused traces remain
explicitly non-full-suite evidence. Optimized build and repeated independent C,
476,160 numerical assertions, and 148-name/592-observation signature checks pass.
The optimized DLL SHA256 is
`db7d68db58c1540a478b4bd52f0dad33b6e5aa74c0fba19b62756c37fc04eb7d`.
Registration is 1,630 forwards per ABI. The full-suite proof validates 1,426
canonical entries: 1,106 Fortran LAPACK, 172 CBLAS and 148 LAPACKE, with 5,071
completed witnesses. Python's full 28-test suite and the 15 inventory checks pass.
Documentation build passes. Linux/macOS CI artifact validation remains a
publication gate.
