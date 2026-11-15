# Native dqds relative-accuracy MWE

Reported as [tensor4all/colmat-rs#209](https://github.com/tensor4all/colmat-rs/issues/209).
This standalone crate uses the public API candidate at the exact revision in
Cargo.toml; it does not use Julia or the Colmat.jl adapter.

Run `cargo run --locked`. At the recorded revision it exits 101 because a
successful dqds call loses small-value relative accuracy. Normal-range controls
pass. An accurate success or an error preserving the original diagonal satisfies
the check.

Before removing the Colmat.jl boundary workaround, select the proposed fixed
native revision in Cargo.toml, regenerate Cargo.lock and rerun this MWE plus the
Julia LASQ1 recovery tests. No native implementation patch is included here.
