# Owned allocation capacity getters

Executed control for [faer #284](https://github.com/tensor4all/faer/issues/284),
fixed in [PR #285](https://github.com/tensor4all/faer/pull/285).

The four-scalar default control checks supported reserve/resize/truncate and
value/pointer preservation. The optional feature only probes the proposed getter
signatures. It produces five E0599 errors (exit 101) on historical main
`2b87010cb82efbe0a90eceeb51931a833e1de8e6`. It runs successfully (exit 0) with
the original candidate `e0269f91da8fc147102bd79e4165367deefe35b7`. This is an API capability
gap; the control does not measure allocation counts or timing.

```sh
git init faer
git -C faer fetch --depth=1 https://github.com/tensor4all/faer.git 2b87010cb82efbe0a90eceeb51931a833e1de8e6
git -C faer checkout --detach FETCH_HEAD
cargo run --locked
# Expected current historical API compile-only failure:
cargo check --locked --features probe-capacity
# Merged API control:
git -C faer fetch --depth=1 https://github.com/tensor4all/faer.git a055727febfde5dd6bb78ba37478b0303b97c807
git -C faer checkout --detach FETCH_HEAD
cargo run --locked --features probe-capacity
```

The source contains no unsafe operations. Executed with Rust/Cargo 1.99.1 on
Linux x86_64; exact features and the executed resolution are retained in the
manifest and lockfile. Queried capacity must come from actual native ownership
metadata, including padding, rather than a downstream reservation lower bound.
