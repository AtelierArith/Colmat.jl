# Measurement environment

Recorded with `benchmarks/backends/run.jl` on 2026-11-21.

| Item | Value |
|---|---|
| Host | `x86_64-apple-darwin24.6.0`, Intel iMac 2019, Core i9-9900K (Skylake, AVX2, 8 cores / 16 threads), 64 GiB |
| Julia | 1.13.4, `-t 1` |
| OpenBLAS | 0.3.30 (Julia's bundled `libopenblas64_`), ILP64 |
| MKL | `MKL_jll` 2023.2.0 (`libmkl_rt.2.dylib`), LP64 + ILP64 forwards |
| Colmat | this working tree, `deps/usr/lib/libcolmat_blas.dylib`, sha256 `4c5051aa87afefc9f91bf347d3b7bbf9cacf891a80351b06c082c6688638a745` |
| Colmat.jl base commit | `ae93d38b` plus the changes in this branch |
| colmat-rs | `b376b3ce209184df8b46c734f7882a21e31d3f58` (latest `main`; `deps/upstream.toml` points at it) |
| BenchmarkTools | 1.8.0 |
| Samples | 5 per case, median reported; one sample is about 20 ms of calls |
| Threads | BLAS 1 / 4 / 8 |

The colmat-rs checkout matches `deps/upstream.toml`, so `Pkg.build("Colmat")`
rebuilds the measured library from the pin. The recorded forwarding targets were
checked per backend: the OpenBLAS runs forwarded the measured symbols to
`libopenblas64_.0.3.30.dylib`, the MKL runs to `libmkl_rt.2.dylib`, and the
Colmat runs to `libcolmat_blas.dylib`. Every case has a finite reconstruction
residual; no probe failed.

The measurements were taken while the host was otherwise idle. A concurrent
`colmat-benchmark` campaign run was waiting for completion before this run
started; it is not part of these numbers.

The build was produced with

```sh
julia --project -e 'using Pkg; Pkg.build("Colmat")'
```

or, equivalently, the `cargo build --release --locked` invocation in
`deps/build.jl`.

`benchmarks/backends/results/pre-fix/` keeps the measurements taken before the
adapter changes at colmat-rs `4e6ed0e3`; those are historical context and are not
directly comparable with this pinned revision.
