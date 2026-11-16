# Complex SYR LAPACKE auxiliary entries (in progress)

This stage adds exactly four frozen-scope names: LAPACKE_csyr,
LAPACKE_csyr_work, LAPACKE_zsyr and LAPACKE_zsyr_work, each for LP64/ILP64.
Real SYR variants are not added. The prior stages implement 88 canonical
numerical entries; these four bring the implemented subset to 92. The original
1,616-entry scope and the separate 20 Rook coordination targets remain intact.

## ABI and numerical contract

Reference-LAPACK v3.12.1 declarations pass complex ALPHA by value, unlike the
CBLAS coefficient pointer convention. Layout is C int, UPLO is C char and
N/INCX/LDA plus the returned lapack_int follow the selected integer ABI.
Pinned source blobs:

- lapacke_csyr.c: dec181b09f7a2efc6bc24855296c743786db8c97
- lapacke_csyr_work.c: a87fe6c79efe0c8e35a4946cdf6c214232075147
- lapacke_zsyr.c: 3703e60cd306b6b7ef35aa4860e2d4ea26f9f1b5
- lapacke_zsyr_work.c: 118e0a5ba18da2f0884162eaa906ff0c708f6982
- lapacke_c_nancheck.c: d635fd5bfe6b5b6d8e10f2fea814e1c6c2838d0c

The operation is symmetric A += alpha*x*transpose(x), without conjugation;
complex diagonal components remain meaningful. Work entries dispatch to the
existing native Fortran SYR entry. Row-major views transpose the storage and
swap UPLO, without allocating or touching the unselected triangle. An empty
row-major call with LDA=0 uses a legal empty Fortran leading dimension.

The reference's underlying BLAS SYR has no INFO output. Work returns zero for
its numerical-argument errors; the existing native BLAS diagnostic mechanism
records them. The wrapper returns -1 for invalid layout and -8 for invalid
row-major LDA. High-level NaN diagnostics distinguish A (-7), ALPHA (-4) and
X (-5); numeric argument validation remains delegated to the BLAS entry.

The v3.12.1 high wrappers pass literal unit stride to the vector NaN helper.
This adapter deliberately follows the actual INCX instead: scanning unused
padding would reject valid strided inputs and miss active NaNs. Positive and
negative strides are checked, invalid zero-stride scans are deferred to BLAS
validation, and address-span overflow is rejected before scanning. No claim of
matching that reference unit-stride-checking behavior is made.

## Executed checks

- The independent C compiler caller passes float/double complex coefficients
  by value. Both widths, both layouts, upper/lower and lowercase flags, strided
  poisoned padding, empty/null no-ops, alpha zero, integer-width diagnostics,
  invalid dimensions and distinct NaN argument indices pass.
- Actual ILP64 calls use a negative N beyond 32-bit range and a valid N=1 with
  INCX=4,294,967,296, exposing dimension/stride truncation.
- Julia numerical tests pass 13,316 checks through direct DLL and actual LBT
  routes. Scalar outer-product oracles and independently called OpenBLAS
  Fortran SYR agree; input X, matrix padding and the unused triangle are kept.
- The separate C runner passes two checks. Rust's full suite passes 398 tests;
  16 independent evidence fixtures pass, including rejection of a coefficient
  pointer where the LAPACKE declaration requires a by-value complex scalar.

Registration/evidence integration, optimized-release verification and a full
immutable package run remain pending. These focused checks are not a full-suite
artifact and do not establish completion of Issue #288.

The optimized release also passes the numerical and independent C checks.
Focused tracing validates all 92 currently implemented numerical names and
all 368 ABI/route combinations; the completed-call gate passes 1,846 checks.
The actual registered count is 1,574 for each ABI and all 15 inventory checks
pass. The focused artifact explicitly remains a non-full-suite result.

Before the new full run, the prior successful 88-entry snapshot is backed up
and temporarily removed to bootstrap the actual registration table, then
restored as stale evidence. It must be replaced only after a successful full
Pkg.test whose 249 production/test/worker source hashes, manifest, registration
digest and release DLL all match the frozen preflight.
