# Issue #306: bidiagonal SVD and range selection

Status: the 10 canonical Fortran entries are connected for both integer ABIs
and pass focused numerical and diagnostic checks. Issue #306 is complete after
the immutable full-suite artifact is recorded.

## Connected names

- `sbdsqr_`, `dbdsqr_`, `cbdsqr_`, `zbdsqr_`
- `sbdsdc_`, `dbdsdc_`
- `sbdsvdx_`, `dbdsvdx_`
- `slasq1_`, `dlasq1_`

for LP64 and ILP64. The numerical owner is the public native
`bidiagonal_svd` kernel over the caller's `D`/`E`, with no ordinary dense SVD.

## Contracts

### BDSQR

- UPLO U/L, NCVT/NRU/NCC, D/E, VT (`P^T VT`), U (`U Q`) and C (`Q^T C`) are
  applied from the full singular vectors. Partial matrices and zero singular
  values are supported. Work is ignored; the native kernel owns its scratch.
- Invalid arguments follow the reference negative `INFO` order.

### BDSDC

- COMPQ `N` returns singular values. COMPQ `I` returns full `U` and `VT` with
  `VT = V^T`. Both use the native bidiagonal SVD.
- COMPQ `P` is the only mode without a public native compact representation.
  It delegates to the saved original ILP64 LAPACK driver; the compact `Q`/`IQ`
  layout is therefore the reference layout, not a new native encoding. This
  limitation is explicit and covered by a deterministic compact-mode test.

### BDSVDX

- RANGE `A`, `V` (`[VL,VU)`) and `I` (IL..IU, 1 = smallest) select from the
  descending native values and return them ascending, as required.
- JOBZ `V` writes `Z = [U; V]` with the selected left and right vectors in
  columns. No padding is written.

### LASQ1

- Returns the descending singular values of the bidiagonal matrix. The native
  kernel does not claim the dqds algorithm's high relative accuracy; it computes
  the same mathematical SVD.

## Tests

- Four scalar types for BDSQR, two for BDSDC/BDSVDX/LASQ1, both integer ABIs,
  upper and lower bidiagonal storage, empty/1xN cases, zero singular values,
  partial VT/U/C factors, all ranges and modes, reconstruction/orthogonality,
  compact fallback determinism and diagnostics.
- Full-suite evidence records both ABIs and both actual DLL/LBT routes.

## Remaining LAPACKE scope

The LAPACKE BDSQR/BDSDC/BDSVDX high/work entries belong to #288 and are not
claimed complete by this issue.
