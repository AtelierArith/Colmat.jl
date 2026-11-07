# Sparse LU `LuRef::factors` builds L/U through triplets

Issue: https://github.com/AtelierArith/Colmat.jl/issues/115 (upstream draft,
not filed).

Baseline: t4a-faer pinned at `d374ec022b76bd99089af396964945fa03d8d29c`
(Colmat main), default features disabled with `std,linalg,rayon`, rustc 1.99.1
on Darwin arm64.

`LuRef::factors` is the only public way to export the L/U factors of an
auto/supernodal sparse LU (`NumericLuRaw` and the supernodal buffers are
private). Both numeric variants collect every entry into a `Vec<Triplet>` and
call `SparseColMat::try_new_from_triplets`, which sorts and deduplicates into
temporary buffers before building the CSC result. Colmat's `F.L`/`F.U` for
auto/supernodal sparse LU therefore cannot be built directly in CSC, unlike
the simplicial and Cholesky factors.

The adjacent `faer-sparse-lu-factors-triplets.rs` counts allocations of one
`factors()` call with a counting global allocator on a deterministic 30×30
matrix (diagonal `4 + i`, otherwise `((17i + 13j) % 19) - 9` where
`(i + j) % 3 == 0` or `|i - j| == 1`, zero-based). Executed output:

```
simplicial: nnz(L)=357 nnz(U)=392 allocations=10 bytes=36448 output_buffers=6 output_bytes=12480
supernodal: nnz(L)=465 nnz(U)=465 allocations=10 bytes=66016 output_buffers=6 output_bytes=15376
```

The two returned CSC matrices need 6 buffers (`12480`/`15376` bytes); the
call makes 10 allocations of `36448`/`66016` bytes, about 3–4× the output.
This is an allocation-efficiency limitation, not a numerical bug.

Run it as a standalone Cargo project whose `Cargo.toml` depends on
`t4a-faer = { path = "<checkout>/faer", default-features = false,
features = ["std", "linalg", "rayon"] }` (package renamed to `faer`), with this
file as `src/main.rs`, then `cargo run --release`.

Proposed upstream change: build the sorted CSC L/U directly in `factors` (for
example a counting pass for the column pointers, then a fill pass), or expose a
borrowed supernodal accessor so callers can do so.
