# Issue 288: SYEV / HEEV candidate contracts

Status: initial implementation and Rust numerical checks only. C callers,
independent reference comparisons, actual LBT routes, optimized release,
immutable full-suite evidence and platform CI remain pending. Issue #288
remains open.

This candidate adds eight canonical names: S/D SYEV and C/Z HEEV, each high
and work, with both integer ABIs. The preceding 108-entry SVD candidate is
still undergoing its frozen full-suite run in a separate worktree; its tracked
sources, manifest, registration and installed DLL must not be modified.

## Pinned contracts

Reference-LAPACK v3.12.1 sources:

| Source | Git blob |
| --- | --- |
| LAPACKE/src/lapacke_dsyev_work.c | dc6a7f156811b86c38003e4eb6194bf3c65c0278 |
| LAPACKE/src/lapacke_dsyev.c | 68a91ff1df288632b74d21774f4df6e1b16f44df |
| LAPACKE/src/lapacke_zheev_work.c | 1f321214babac0c880b90f80d9ff9cf6f936ca0f |
| LAPACKE/src/lapacke_zheev.c | b1fe9c65308e6d945fd08838697596c54bc3c398 |
| LAPACKE/utils/lapacke_zhe_nancheck.c | a8da25f09ff6f92e383a80c2b97bb4f204ad1003 |

Layout is C int; JOBZ and UPLO are C char; N/LDA/LWORK and the INFO return
follow lapack_int. Complex variants return real eigenvalues and accept real
RWORK. Row-major LDA < N returns -6 before the Fortran argument checks. Negative
Fortran INFO shifts down by one. Work queries do not transpose or read A.

Row-major execution packs only the selected logical triangle while retaining
the same UPLO. JOBZ=V copies all eigenvector entries back; JOBZ=N copies only
the selected triangle. Padding and unselected inputs must never be read.
The high-level NaN check returns -5 and follows the reference Hermitian checker,
which delegates to a non-unit triangular check, including the full diagonal.

The existing Fortran boundary reports the native driver's minimum/query WORK
size and owns its numerical scratch; HEEV RWORK is currently unused. Algorithm
identity with Reference-LAPACK SYEV/HEEV is not claimed. Checked adapter
arithmetic prevents workspace overflow before entering the existing Fortran
routine.

## Initial numerical evidence

Eight Rust cases cover four scalar types and both integer widths. Each uses
both layouts, both triangles, JOBZ=N/V, high/work calls and N=0/1/2. Poisoned
unused storage and padding, query immutability, guarded eigenvalue outputs,
analytic real/Hermitian eigenvalues, eigenvector residuals and orthogonality
pass. Complex inputs have nonzero imaginary off-diagonal entries.

A separate executed call to the existing ZHEEV ILP64 entry with imaginary-only
diagonal NaNs returned INFO=0, finite eigenvectors and the expected eigenvalues
2.8074175964327477 and 8.192582403567252. This verifies the initial backend
assumption for that example; it is not a full boundary qualification.
