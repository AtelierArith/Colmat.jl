# The 0.2 redesign

Version 0.2 changed the design from a Julia–Rust binding to a BLAS backend.
It is not backward compatible with the old version.

| Old usage | Current usage |
|---|---|
| Create `FaerMatrix` / `FaerVector` | Use ordinary `Matrix` / `Vector` |
| Call multiplication or `mul!` on dedicated types | Use the same Base / LinearAlgebra APIs on ordinary arrays |
| Multiply with `FaerTeam` | Use `BLAS.set_num_threads` or `Colmat.set_num_threads` |
| Release Rust handles with `close` / `with_resource` | Ordinary arrays are owned by Julia; no dedicated handle management is needed |
| Generate bindings with RustCall | Build the BLAS ABI shared library with Cargo |
| Use the dedicated factorization, sparse, iterative, and NPY APIs | These Colmat APIs were removed; choose the corresponding Julia API on the use side |

LU, SVD, general/self-adjoint eigenvalues, and Cholesky for ordinary arrays go through
the LAPACK ABI to colmat-rs factorizations. This is not a change that provides a
compatibility layer for the old dedicated-type API.

## Old material

The old feature ledger, the Issue #1 plan, and the reproducers for RustCall and the old
API have been removed from the current documentation. To consult them together with the
old code, use the Git commit before the redesign,
`ced1e13bf765f18b44e92317bc053b8e212fed4e`.

```sh
git show ced1e13bf765f18b44e92317bc053b8e212fed4e:docs/implementation-status.md
git ls-tree -r --name-only ced1e13bf765f18b44e92317bc053b8e212fed4e docs/reproducers
```

The test counts and dependency revisions in the old records describe the history of
verifying that old implementation. Do not reuse them as the current coverage or
verification results.
