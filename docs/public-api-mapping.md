# Public API and BLAS coverage

The API and coverage of Colmat 0.2 are managed on the following pages.

- [Public API on the Julia side](src/reference/api.md)
- [BLAS types, integer ABIs, and symbol list](src/reference/coverage.md)
- [Usage examples with ordinary Julia arrays](src/getting-started.md)

This is not a mapping that exposes every Rust colmat feature (factorizations, sparse
matrices, iterative methods, and so on) to Julia. Matrix operations on the Julia side
use the Base / LinearAlgebra APIs.
