# Zero-sized matrix reservation

Standalone safe control for [faer #286](https://github.com/tensor4all/faer/issues/286).
On `2b87010cb82efbe0a90eceeb51931a833e1de8e6`, the empty `Mat<()>` reservation
and ordinary Float64 reservation succeed; the nonempty-capacity `Mat<()>`
reservation then panics with division by zero in `mat/matown.rs:76` (exit 101).
The control reads no uninitialized element and never invokes `set_dims`.

```sh
git init faer
git -C faer fetch --depth=1 https://github.com/tensor4all/faer.git 2b87010cb82efbe0a90eceeb51931a833e1de8e6
git -C faer checkout --detach FETCH_HEAD
cargo run --locked
```

Executed with Rust/Cargo 1.99.1 on Linux x86_64. The executed lockfile and exact
features are included. This is separate from the missing getter API; PR #285
does not change allocation behavior or fix this case. Colmat's four advertised
native numeric scalars have nonzero size, so no wrapper workaround is needed.
