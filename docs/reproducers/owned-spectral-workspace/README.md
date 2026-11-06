# Owning spectral constructors and caller workspace

This control supplements [faer #268](https://github.com/tensor4all/faer/issues/268#issuecomment-6679048835).
On faer `2b87010cb82efbe0a90eceeb51931a833e1de8e6`, both the owning and
caller-scratch low-level SVD/EVD paths reconstruct the independent inputs.
The owning constructors do not accept a caller stack. Their private factor
fields also prevent installing the low-level outputs through a public parts
constructor. This is an API capability gap, not a numerical failure.

The default executable uses safe Rust only. It checks a 7×5 Float64 SVD and
5×5 real symmetric EVD with an entrywise absolute reconstruction bound of
`1e-10`. Its optional feature probes the current owning-constructor arity;
that feature is deliberately expected to fail compilation with E0061 twice.
This control does not count allocations or measure latency or scaling.

From this directory:

```sh
git init faer
git -C faer fetch --depth=1 https://github.com/tensor4all/faer.git 2b87010cb82efbe0a90eceeb51931a833e1de8e6
git -C faer checkout --detach FETCH_HEAD
cargo run --locked
# Expected compile-only failure for the current owning API:
cargo check --locked --features probe-owned-scratch
```

Executed with Rust 1.99.1, Cargo 1.99.1 on Linux x86_64. The default control
exited 0 and printed both successful reconstruction messages; the optional
probe exited 101 and rejected the extra `&mut MemStack` argument to both
`Svd::new_thin` and `SelfAdjointEigen::new`. Cargo features and the executed
resolution are retained in Cargo.toml and Cargo.lock.

Colmat's `native=true` path retains these actual faer owning types, so its
construction cannot yet consume caller scratch. Output factor allocations
are separate from temporary workspace. The proposed canonical constructor
change, matching scratch queries, and Seq/Team/four-scalar acceptance tests
remain upstream work. The Colmat dependency checkout has no local patch,
and the LLT-only draft PR #269 has not been adopted.
