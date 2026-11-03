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


## Blocked bidiagonalization without rayon

Reported with full Cargo.toml, Rust MWE, measured outputs and a concrete fix in
[tensor4all/faer#203](https://github.com/tensor4all/faer/issues/203).
`faer-bidiag-no-rayon.rs` contains the standalone Rust example. With
`default-features=false, features=["std","linalg"]` it has relative residual
`6.180742e-1`; adding `rayon` gives `9.783102e-16`. Both calls pass `Par::Seq`.
The scratch bytes are explicitly zeroed to make the omitted Y update's effect
deterministic. The issue includes copy/paste build instructions.

The small partial-eigenvalue panic above is also reported with its standalone
MWE and fallback implementation proposal in
[tensor4all/faer#204](https://github.com/tensor4all/faer/issues/204).


## Other confirmed native API issues

Each report includes its standalone Cargo.toml, full Rust MWE, observed output
and a concrete fix. The locally executed copies are retained here:

| MWE | faer issue | Observed failure |
|---|---|---|
| `faer-rotg-zero.rs` | [#205](https://github.com/tensor4all/faer/issues/205) | Complex zero-q rotg returns r=1 instead of p |
| `faer-jacobi-right.rs` | [#206](https://github.com/tensor4all/faer/issues/206) | Complex right 2×2 application applies conjugated J |
| `faer-lsmr-atol.rs` | [#207](https://github.com/tensor4all/faer/issues/207) | Absolute tolerance is omitted from LSMR threshold |

All three independent examples exit 101 at their expected correctness assertion
on the pinned revision. Colmat already compensates for these cases in its
wrappers; the dependency checkout remains unmodified.


## NPY endianness and unaligned copying

`faer-npy.rs` is a standalone Rust MWE with two modes. Both reports contain
complete Cargo.toml and build instructions and require no NumPy installation.

- [faer#210](https://github.com/tensor4all/faer/issues/210): default mode constructs
  an aligned foreign-endian file containing 1.5; both native reference and copy
  return 3.139837e-319 on the tested little-endian host. Proposed fix: preserve
  byte-order metadata, swap real components on copy and reject foreign-endian
  zero-copy references.
- [faer#211](https://github.com/tensor4all/faer/issues/211): `cargo run -- unaligned`
  uses valid native-endian C-order bytes at offset one; to_mat panics on a
  bytemuck alignment check. Proposed fix: pod_read_unaligned for the copying
  path, keeping alignment requirements for references.

Both modes were executed against d5be1473eed9314fca6f80ad8b4072d86282e890 and
exit 101. Colmat validates and normalizes into aligned native-endian storage
before using faer, without modifying the dependency checkout.

### Small/full-projection partial SVD panic

`faer-partial-svd-small.rs` reproduces a panic for a 4×4 identity and one
requested singular triplet. The public driver caps its minimum projection
space at the matrix dimension, while its callee requires a strict inequality.
Reported with complete executable crate and proposed full-projection correction:
<https://github.com/tensor4all/faer/issues/213>. Colmat reports a native dense
fallback for this boundary; larger partial requests keep the iterative path.
