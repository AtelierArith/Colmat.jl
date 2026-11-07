# Sparse symbolic QR allocates its workspace internally

API request, not a numerical defect. Colmat issue:
[#116](https://github.com/AtelierArith/Colmat.jl/issues/116). Not yet filed
upstream (tensor4all/faer); whether to file is the maintainer's decision.

At t4a-faer `d374ec022b76bd99089af396964945fa03d8d29c` (0.32.0, Colmat's pin),
`faer::sparse::linalg::qr::factorize_symbolic_qr` and
`factorize_symbolic_qr_with_column_ordering` accept no `MemStack` and no `Par`.
They size one workspace as `max(COLAMD, etree/postorder/column counts/symbolic
analysis)` and allocate it with `MemBuffer::try_new(req)`
(`faer/src/sparse/linalg/qr.rs:2450`). No public `*_scratch` query returns that
requirement, and the analysis always runs sequentially. Every step already has
a public stack-taking kernel (`colamd::order`, `col_etree`, `postorder`,
`column_counts_ata`, `factorize_{simplicial,supernodal}_symbolic_qr`).

The crate needs no Julia, RustCall or local faer clone; Cargo pins the executed
git revision and the retained lock records resolved dependencies. A counting
global allocator wraps one `factorize_symbolic_qr_with_column_ordering` call on a
200×100 matrix (nnz 298) with an explicit identity column ordering.

```sh
cargo run --release --locked --manifest-path docs/reproducers/sparse-symbolic-qr-workspace/Cargo.toml
```

Observed (rustc 1.99.1, aarch64-apple-darwin; exit code 101):

```
m=200 n=100 nnz=298 colamd::order_scratch=18128 bytes
threshold=inf len_idx=837 allocations=7 largest=18128 bytes sizes=[18128, 800, 800, 1600, 800, 800, 800]
threshold=0 len_idx=1584 allocations=26 largest=18128 bytes sizes=[18128, 800, 800, ...]
symbolic QR allocated internal workspace (largest, owned bound): [(18128, 6696), (18128, 12672)]
```

The first allocation (18128 B) is the internal workspace; the rest are the
returned symbolic owner (permutation, `min_col`, structure arrays), which are
expected outputs. The assertion bounds each returned array by
`max(m, n + 1, len_idx)` indices. With an explicit ordering COLAMD never runs,
yet its term still sizes the buffer; see
[sparse-symbolic-colamd-workspace](../sparse-symbolic-colamd-workspace/README.md).

Proposed API: `factorize_symbolic_qr_scratch(nrows, ncols, nnz, params,
has_column_ordering) -> StackReq` and a `*_in` variant taking
`stack: &mut MemStack` (optionally `par: Par`), so the workspace can live in
caller scratch and the COLAMD term is only reserved when COLAMD runs. Symbolic
LU and Cholesky follow the same pattern.

Colmat impact: `FaerSparseSymbolic(A; kind=:qr, scratch)` cannot place this
workspace in caller scratch for square/tall auto/supernodal QR. Colmat's own
explicit `algorithm=:simplicial` analysis calls the public kernels on the caller
stack instead.
