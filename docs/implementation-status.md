# Colmat 0.2 implementation status

Colmat today is a BLAS / LAPACK backend registered with libblastrampoline.

- [Connection candidates and unsupported routines for all LBT routines](src/reference/lbt-inventory.md)
- [Measured mapping for the Julia API](src/reference/support-matrix.md)
- [Supported types, integer ABIs, and routines](src/reference/coverage.md)
- [How forwarding and fallbacks work](src/manual/backend.md)
- [Source layout, dependency updates, and verification](src/development.md)

Coverage is based on the BLAS symbols exposed by `deps/rust/src/abi.rs`, not on the full
feature list of Rust colmat. The forwarding targets of unsupported BLAS and LAPACK
routines are retained.

The old feature ledger, the Issue #1 plan, and the bug reproducers do not apply to the
current design. See [the redesign and how to consult the old material](src/migration.md).
