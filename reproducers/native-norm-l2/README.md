# Native Frobenius norm extreme-range reproduction

This standalone crate pins colmat-rs at `e1c83482f5a61449d30e989808c3ba5f751ec391`
and uses the same `std`, `rayon`, `threadteam` feature set as the adapter.
It contains no Julia, C ABI, unsafe code or local native patches.

Run from the repository root:

```sh
cargo run --release --locked --manifest-path reproducers/native-norm-l2/Cargo.toml
```

On rustc 1.99.1 / x86_64-apple-darwin, the normal controls pass but all eight
extreme cases fail: a 7x7 constant matrix has norm `7*abs(value)`, yet the
smallest positive subnormal gives zero and `MAX/8` gives infinity. All expected
results fit their output scalar type. Exit 101 is the expected diagnostic failure,
not a passing package test or an observed crash in Colmat.jl.

The related [native issue #36](https://github.com/tensor4all/t4a-faer/issues/36)
fixed a less extreme strided-path defect. The native fork's inspected main is the
same revision, but the executed crate fetches the separate colmat-rs repository.
A follow-up comment was attempted on 2026-11-14 UTC and rejected because the
reporting fork is archived/read-only. No report was published; the full
[unpublished draft](report-draft.md) is retained here.

The LAPACK norm adapter normalizes finite components before native norm_l2 and
rescales afterward. That workaround preserves the pinned dependency and does not
resolve the native library defect. Revalidate this crate and the LAPACK regressions
before removing the normalization after a future dependency update.
