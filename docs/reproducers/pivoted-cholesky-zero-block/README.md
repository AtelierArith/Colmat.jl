# Zero block size in faer pivoted Cholesky

Standalone Rust reproduction for [tensor4all/faer #283](https://github.com/tensor4all/faer/issues/283).
No Julia or RustCall is used. The pinned matrix is `[1.0]`, whose Cholesky factor
and rank are independently known to be `[1.0]` and one.

From this directory, fetch the exact tested fork revision into the ignored
`faer/` directory and build before running any timeout:

```sh
git init faer
git -C faer remote add origin https://github.com/tensor4all/faer.git
git -C faer fetch --depth=1 origin 2b87010cb82efbe0a90eceeb51931a833e1de8e6
git -C faer checkout --detach FETCH_HEAD
cargo build --locked
timeout 2s ./target/debug/faer-pivoted-zero-block 1
timeout 2s ./target/debug/faer-pivoted-zero-block 0
timeout 2s ./target/debug/faer-pivoted-zero-block 128
```

The zero-block invocation is deliberately bounded by `timeout`; do not run it
without that bound on the affected revision. On Linux x86_64, rustc 1.99.1 /
cargo 1.99.1, t4a-faer 0.31.0, debug build, the actually executed controls 1 and
128 exit 0 with `completed: rank=1, L=[1]`. Block 0 prints `entering kernel`
and is terminated by the timeout with exit 124. Scratch-query completion
precedes that message. The features match the Colmat consumer.

The panel loop computes `bs=min(n-k,block_size)` and advances by `k+=bs`;
zero skips the pivot work and leaves `k` unchanged. The proposed upstream fix
is to reject zero in query/kernel preconditions and document positive block
sizes. Replace the checkout revision when independently validating a fix;
positive controls must still pass and zero must fail promptly instead of
requiring termination. Colmat validates positive block sizes before query and
compute in both tolerance modes.
