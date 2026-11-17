# Issue #288: LACPY/LASET/LASWP C contracts

Status: the 24 S/D/C/Z high/work entries pass focused numerical, layout,
diagnostic and independent C ABI checks. The immutable full suite is recorded
in `docs/src/assets/lapack-abi-execution.json`; it validates 347 sources,
1,754 canonical entries and 6,491 completed witnesses (1,958 forwards per
integer ABI). Issue #288 remains open.

## Connected names

S/D/C/Z `LAPACKE_{s,d,c,z}{lacpy,laset,laswp}` and their `_work` forms
(24 entries). The Fortran numerical owners are the existing native
`{s,d,c,z}{lacpy,laset,laswp}_` entries.

## Reference contracts

Reference-LAPACK tag `v3.12.1`, inspected sources:

Pinned sources:
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlacpy_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlacpy.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlaset_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlaset.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlaswp_work.c
- https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dlaswp.c

Layout is a C int; `UPLO`/`TYPE` flags are C char; dimensions, pivots and
INFO are `lapack_int`. Row-major leading dimensions are checked before any
numerical work. Negative Fortran INFO shifts by one for the layout argument.

## LACPY

`LAPACKE_xlacpy(layout,uplo,m,n,a,lda,b,ldb)` copies only the logical
M-by-N rectangle. `UPLO='U'` selects `i<=j`, `'L'` selects `i>=j`, and any
other value copies the full rectangle, matching the Fortran adapter. The
destination row count is therefore the selected rectangle, not the caller's
storage pitch; entries outside it are neither read nor written. Row-major
requires `lda>=n` (-6) and `ldb>=n` (-8). The high-level entry checks A for
NaNs in the logical rectangle (-5) before the work entry. Column-major calls
the native Fortran adapter.

## LASET

`LAPACKE_xlaset(layout,uplo,m,n,alpha,beta,a,lda)` fills the selected
rectangle: `beta` on the diagonal and `alpha` off it. The old payload is not
read, so a NaN initial matrix is legal. The high-level entry rejects NaN
`alpha` (-5) and `beta` (-6) when NaN checking is enabled. Row-major requires
`lda>=n` (-8). Column-major calls the native Fortran adapter.

## LASWP

`LAPACKE_xlaswp(layout,n,a,lda,k1,k2,ipiv,incx)` applies the row interchanges
K1..K2. Row-major storage is viewed as an M-by-N matrix with row pitch `lda`;
only the N logical columns are exchanged, so padding beyond column N is
preserved. The reference transposes `max(1,k2,max(IPIV))` rows and swaps
rows; the adapter performs the equivalent in-place swaps. Positive `incx`
processes K1..K2, negative `incx` processes K2..K1 while walking IPIV
backwards, and `incx=0` is a no-op. Row-major requires `lda>=n` (-4).
Column-major calls the native Fortran adapter, which swaps every stored
column. The high-level entry intentionally disables the reference's
documented-buggy NaN check.

## Evidence

- Debug/release Rust builds and `cargo fmt --check` pass.
- Focused Julia tests cover four scalar types, both integer ABIs, both
  layouts, high/work, DLL and LBT routes, empty/rectangular shapes, selected
  triangles, padded storage, NaN policy, negative `incx`, and no-op ranges.
- An independent C11 `-Wall -Wextra -Werror` caller exercises every prototype
  and both integer widths.
- The immutable full `Pkg.test()` artifact records both ABIs and both actual
  routes for every new name. Registration or compilation alone is not
  counted as execution.

## Remaining scope

The original 1,616-entry #288 scope still has 1,140 names. LAPACKE LASCL,
matrix norms and the remaining families are not implied by this stage.
