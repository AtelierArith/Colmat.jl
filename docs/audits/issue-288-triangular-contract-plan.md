# Issue #288: TRTRI / TRTRS C adapters

Status: initial adapters and focused numerical/C ABI checks pass. No full-suite
proof has been imported for this candidate. Issue #288 remains open.

The sixteen S/D/C/Z TRTRI/TRTRS high/work names add both integer ABIs beyond the
148-name expert candidate. This reaches 164 original-scope names, leaving 1452
of the original 1616 and the separate 20 Rook C targets. It does not complete
the original acceptance conditions.

Reference LAPACK v3.12.1
[TRTRI work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dtrtri_work.c)
and [TRTRS work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dtrtrs_work.c)
use identical high/work numerical signatures; neither work interface has a
workspace query argument. C layout is int, flags char, and dimensions/INFO use
lapack_int. Row-major TRTRI LDA diagnostics are argument 6; TRTRS LDA/LDB are
8/10, with LDB checked against NRHS. Row stride checks precede flag validation.

Only the selected triangle is packed. Unit-diagonal storage is never read,
NaN-scanned or copied back; initialized temporary unit diagonals equal one.
TRTRS leaves A unchanged, preserves B padding, and supports N/T/C without
implicit conjugation during layout conversion. TRTRI writes only the selected
non-unit entries. Negative backend INFO prevents copy-back. High-level NaN
diagnostics are A=-5 for inverse and A=-7/B=-9 for solve. Complex diagonal
entries in non-unit mode retain their full complex meaning.

Required checks include all four types, both integer widths, high/work and
DLL/LBT routes, layouts, upper/lower, unit/non-unit, transpose/conjugate, scalar
and empty shapes, zero right-hand-side count, independently formed factors,
positive singular INFO, input/padding preservation, exact argument diagnostics,
and dimensions outside int32. Poison ignored triangle, diagonal and padding to
detect accidental reads. Compare saved independent OpenBLAS results and residuals.

The initial Rust numerical tests cover N=0/1/3, both layouts and both ABIs,
all scalar types, both high/work calls, unit diagonal NaNs, unused triangle and
padding NaNs, N/T/C, and NRHS=0/2. Inverse products and a known solution used to
construct RHS provide numerical checks. All eight Rust scalar/ABI cases pass,
including singular INFO and unchanged outputs (also with NRHS=0), and exact
header diagnostic precedence.

Independent saved OpenBLAS comparisons pass 105,984 Julia assertions for
N=0/1/3/5, NRHS=0/2, both layouts, jobs, types and integer ABIs, both high/work
and DLL/LBT routes. Reference pointers are verified distinct from adapter
pointers. An independently compiled C11 worker passes all sixteen names,
including singular INFO, unit/unused NaNs, input/padding guards, argument
diagnostics and negative dimensions outside int32. Typed traced signatures,
including all 164 names and 656 completed name/ABI/route observations, pass.
The independent C Julia runner passes two checks. Rust's full suite passes
450 tests and Python's evidence fixtures pass 17 tests. Focused witnesses are
explicitly marked full_suite_passed=false. Optimized repetition and an immutable
full suite remain required.

Optimized build and repeated independent C, numerical and completed-call
signature validation passed. The optimized DLL SHA256 is
`bb3ab591cbd9a35bcfe20a26c01377184f8b438560b70728013a3655b57ef4e5`.
The completed-call gate passes 3,286 checks; registration is 1,646 forwards per
integer ABI. Full-suite evidence remains pending and the prior snapshot remains
stale for this candidate until a successful immutable package run.
