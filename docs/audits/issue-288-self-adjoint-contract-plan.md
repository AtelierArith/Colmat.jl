# Issue 288: SYEV / HEEV candidate contracts

Status: initial implementation, independent C callers and OpenBLAS comparisons
through both actual DLL/LBT routes pass. Completed-call tracing, optimized
release, immutable full-suite evidence and platform CI remain pending. Issue #288
remains open.

This candidate adds eight canonical names: S/D SYEV and C/Z HEEV, each high
and work, with both integer ABIs. The preceding 108-entry SVD candidate passed its immutable full suite and is
under review in PR #382. The original 1,616-entry scope and separate 20 Rook
coordination targets remain intact.

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


## Hermitian diagonal boundary correction

The initial 2-by-2 ZHEEV example did not expose the full boundary defect. A
3-by-3 example with ignored imaginary diagonal NaNs returned INFO=1, unchanged
sentinel eigenvalues and nonfinite matrix output; independent OpenBLAS returned
INFO=0 and eigenvalues 4.531608866314485, 5.347519465914594 and 8.12087166777092.
This is a Colmat.jl LAPACK boundary requirement; colmat-rs is not modified.

The shared `lapack::selfadj` boundary now detects diagonal components that differ
from their real projection and then copies only the selected lower view into a
normalized native matrix. The diagonal is projected real. Ordinary inputs with
real diagonals retain the borrowed path; normalization adds N-by-N scratch only
when needed, in addition to any row-layout temporary. Upper storage is already
exposed through the existing transpose view and the existing eigenvector
conjugation. No unselected entries are read by normalization.

`reproducers/heev_ignored_diagonal.jl` now reports INFO=0, finite vectors and the
same eigenvalues as OpenBLAS. All 38,600 Julia numerical checks pass, including
imaginary-only diagonal NaNs in complex work entries, both layouts, N=0/1/3,
upper/lower and lowercase flags, and both integer ABIs. Independent C runners
pass two checks; the full Rust suite passes 416 tests; 17 evidence fixtures pass.
