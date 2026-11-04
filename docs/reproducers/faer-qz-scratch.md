# Real QZ public scratch underallocation

Reported as [tensor4all/faer#252](https://github.com/tensor4all/faer/issues/252).
Verified on clean `19053d8c22993851c032d5373b4ec188998f2b5c`, t4a-faer 0.29.0,
Rust 1.99.1 (58cf8f9c3), Linux x86_64. Julia and RustCall are not needed.

Create sibling `faer` and `mwe` directories. Clone tensor4all/faer into `faer`,
checkout the revision above, and copy [faer-qz-scratch.rs](faer-qz-scratch.rs) to
`mwe/src/main.rs`. Write `mwe/Cargo.toml`:

```toml
[package]
name = "qz-scratch-mwe"
version = "0.0.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

`cargo run --manifest-path mwe/Cargo.toml` exits 101. The source is a deterministic
finite 96x96 upper-Hessenberg / upper-triangular pencil, with identity Q/Z and a
public deflation-window callback returning 32. The exact public query reports
40,448 scratch bytes; executing the same params panics while allocating an AED
window with 24,064 bytes remaining versus 24,576 required.

The query uses maximum window 31 while the executing kernel rounds it upward
to even size 32. The proposed correction is to use the same rounding in the
query, including recursive AED queries. A separate source audit finds one
index-array reservation where the Sylvester solve uses four; reserve all four.
These corrections have not yet been tested as an upstream patch. The report
explicitly distinguishes the executed window failure from the index-array
source finding.

Colmat retains its corrected recursive query bound and takes the maximum with
faer's recommendation. Its dependency checkout remains unmodified. Remove this
workaround only after a fixed upstream revision, the original reproducer and
the n=78/96/258 wrapper regressions pass.
