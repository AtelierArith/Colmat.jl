# Sparse QR symbolic structure accessor

API extension under investigation; this is not a numerical correctness defect.
At upstream b4fb784488230155e1a21e3505efb688d7932dcf, SymbolicQrRaw is public but
SymbolicQr's raw field is private and no public accessor exists. The standalone
Rust reproducer fails with E0599 when calling `s.raw()`. A proposed read-only
`pub fn raw(&self) -> &SymbolicQrRaw<I>` allows safe inspection of the selected
algorithm and structure, borrowing the existing symbolic owner. No mutable
storage or private layout is exposed.

Run the adjacent .rs file as src/main.rs in a standalone Cargo package with
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false,
features = ["std", "linalg"] }, where ../faer is the pinned upstream clone.
The executable checks both forced simplicial and forced supernodal analyses.
Upstream PR: https://github.com/tensor4all/faer/pull/228 (859d9a2).
Merged upstream on 2026-11-03 as fac26dc2a346f2be41f2011fba2bc8e4d7ff22f5.
The Colmat dependency checkout has not yet adopted this change.
The public integration regression and executable doctest each passed (1 test
each) with std,linalg and no default features. The standalone executable prints
selected simplicial / selected supernodal. Full numerical/performance acceptance
was not run because no numerical path changes; adoption remains with upstream.

A complete portable setup (run from a new empty directory):

```sh
git clone https://github.com/tensor4all/faer.git faer
git -C faer checkout b4fb784488230155e1a21e3505efb688d7932dcf
mkdir -p mwe/src
# Copy the adjacent faer-sparse-qr-symbolic-accessor.rs to mwe/src/main.rs.
cat > mwe/Cargo.toml <<'TOML'
[package]
name = "sparse-accessor-mwe"
version = "0.1.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
TOML
cargo check --manifest-path mwe/Cargo.toml
# Baseline: exit 101, E0599 (no method named raw).
# For the proposed fix, fetch its branch and rerun:
git -C faer fetch origin feat/public-sparse-qr-symbolic-accessor
git -C faer checkout 859d9a2
cargo run --manifest-path mwe/Cargo.toml
# Patched: exit 0, selected simplicial / selected supernodal.
```

The recorded runs used rustc 1.99.1 on Linux, with std/linalg enabled and default
features disabled. The public integration test includes the complete executable
source and is also runnable directly in the upstream PR.
