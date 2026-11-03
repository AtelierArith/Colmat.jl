# Native partial eigenvalue probe

`faer-partial-small.rs` reproduces an upstream limitation at t4a-faer 0.28.1,
commit `d5be1473eed9314fca6f80ad8b4072d86282e890`. The public
`partial_self_adjoint_eigen` call for a 4×4 identity matrix and one requested
pair panics at `operator/self_adjoint_eigen/mod.rs:290`, because the
`max_dim >= n` branch is unimplemented. Default projection dimensions select
that branch for small matrices. Reproduced with the package's existing native
dependencies; this is not a RustCall error.

From the repository root, run this probe separately from package tests:

```sh
mkdir -p deps/rust/examples
cp docs/reproducers/faer-partial-small.rs deps/rust/examples/partial_eigen_probe.rs
cargo run --manifest-path deps/rust/Cargo.toml --example partial_eigen_probe
```

The command currently exits with code 101. Remove the temporary example after
inspection:

```sh
rm deps/rust/examples/partial_eigen_probe.rs
rmdir deps/rust/examples
```

No source under `deps/t4a-faer` is modified. Partial spectrum integration remains
an open Issue #1 requirement; this probe does not replace implementing it.
