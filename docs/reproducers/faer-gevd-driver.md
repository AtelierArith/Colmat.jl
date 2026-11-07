# faer generalized EVD driver: tuned scratch, tiny norms and allocations

Executed on t4a-faer `d374ec022b76bd99089af396964945fa03d8d29c` (0.32.0, the
revision Colmat pins), rustc 1.99.1, Darwin arm64, `cargo run --release`.
Julia and RustCall are not involved. Each source is `src/main.rs` of a
standalone crate next to a checkout of that revision in `faer`:

```toml
[package]
name = "gevd-mwe"
version = "0.0.0"
edition = "2021"
[dependencies]
faer = { package = "t4a-faer", path = "../faer/faer", default-features = false, features = ["std", "linalg"] }
```

These findings are recorded for issue #109 and have not been filed upstream.

## `gevd_scratch` ignores `GevdParams::hessenberg`

[faer-gevd-hessenberg-scratch.rs](faer-gevd-hessenberg-scratch.rs) exits 101.
`gevd_scratch` sizes the generalized Hessenberg step with
`generalized_hessenberg_scratch(n, auto!(T))` (block size 32), but
`gevd_real`/`gevd_cplx` run it with `params.hessenberg`:

```
n = 200, hessenberg.block_size = 64: gevd_scratch = 307200 B, generalized_hessenberg_scratch(params.hessenberg) = 614400 B
thread 'main' panicked at faer/src/linalg/gevd/gen_hessenberg/mod.rs:502:5:
buffer is not large enough to allocate an array of type `f64` of the requested length:
 - remaining buffer length (after adjusting for alignment): 4160,
 - requested array length: 9792 (78336 bytes),
```

The fix is to pass `params.hessenberg` to `generalized_hessenberg_scratch`
inside `gevd_scratch`. Colmat forwards `with_schur_tuning`'s generalized
Hessenberg and QZ controls to its GEVD driver, and its query reserves the
larger of faer's query and the tuned Hessenberg requirement.

## Regular pencils of tiny norm return α = β = 0

[faer-gevd-small-norm.rs](faer-gevd-small-norm.rs) exits 101. B is strictly
diagonally dominant, so every eigenvalue is finite and every β must be
nonzero:

```
n = 8, A scaled by 2^-60: zero beta real f64 = 0/8, complex c64 = 0/8
n = 8, A scaled by 1e-200: zero beta real f64 = 0/8, complex c64 = 0/8
n = 8, A scaled by 1e-300: zero beta real f64 = 8/8, complex c64 = 8/8
```

The same happens for Float32 near `1e-30` (29 of 32 pairs at n=32). LAPACK
`ggev` scales A and B into `[sqrt(safmin)/eps, eps/sqrt(safmin)]` before QZ
and scales α and β back afterwards; faer does not. Colmat applies that
power-of-two scaling (the same `schur::scale_in` its QZ path uses) to its A/B
copies and unscales α and β, so the eigenvectors are unchanged.

## Allocations inside the driver

[faer-gevd-allocation.rs](faer-gevd-allocation.rs) counts heap allocations
made inside `gevd_real`/`gevd_cplx` on exactly the queried `gevd_scratch`
buffer, for f32/f64/c32/c64, n = 32 and 96, values/right/left/both, and A
scaled by 1, 1e30 (1e300 for f64/c64) and 1e-20 (1e-200). All 96
configurations exit 0 with three repeated runs at **0 allocations / 0 bytes**.
The first call of a configuration in the process can allocate lazily
initialised library state (six of the 96 first calls, all at n = 96, between
190 B and 4.3 MB); later calls reuse it. Input, output and scratch setup is
outside the count.

## QZ run time jumps at the multishift threshold

[faer-gevd-qz-time.rs](faer-gevd-qz-time.rs) times values-only `gevd_real`
(default params, `Par::Seq`, best of two) on deterministic pencils with a
strongly diagonally dominant B. It exits 0; the times are the finding:

```
n =  64: gevd_real values-only 0.0011 s
n =  74: gevd_real values-only 0.0016 s
n =  75: gevd_real values-only 1.0469 s
n =  80: gevd_real values-only 1.3885 s
n =  96: gevd_real values-only 2.8653 s
n = 120: gevd_real values-only 0.0103 s
n = 149: gevd_real values-only 0.0120 s
n = 150: gevd_real values-only 0.0135 s
n = 160: gevd_real values-only 18.9059 s
```

From n = 75 some pencils take seconds instead of milliseconds; the effect
depends on the pencil, and the cause inside faer's QZ has not been isolated. Through Colmat, random
`randn(n,n)`/`randn(n,n)+2n*I` pairs take 1.2 s (n=75) to 9.1 s (n=120)
against 3–8 ms for LAPACK `ggev3` in Julia, while n=160 takes 30 ms. The
results remain accurate; only the run time is affected. Colmat keeps faer's
QZ and keeps its tuning regressions below n = 75.
