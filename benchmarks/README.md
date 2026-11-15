# General EVD measurements (#318)

Use Julia 1.13. The isolated environment pins BenchmarkTools, JET and AllocCheck;
these are not Colmat runtime dependencies. Build the production library first:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
julia --project=benchmarks/environment -e 'using Pkg; Pkg.instantiate()'
COLMAT_EVD_OUTPUT=/tmp/evd-before.toml julia --startup-file=no --project=benchmarks/environment benchmarks/general_evd.jl --quick
```

Remove `--quick` for four scalar types, sizes 32/96/256, values-only/left/right/both
vectors, all four balance modes, ordinary dense/scaled-isolated inputs, and
configured native participant counts 1/2. The getter verifies the configured
count; individual kernels may choose fewer participants. Validation uses the
original Julia BLAS provider at one thread.

The caller preallocates all arrays. Each destructive native call gets the same
input via BenchmarkTools setup; input-reset time is measured separately. Timing
uses ten samples with one evaluation per sample. Initial per-case calls cross an
invocation boundary so later warm calls cannot precompile the method before the
timer starts. Only the first case of each scalar type includes that method's JIT
compilation; package/tool loading is outside this measurement. The report records
cold compile time separately from steady timing, Julia allocations, residuals,
normalization, input seed, hardware/platform, revisions and binary/tool/script
hashes. Do not run timing while builds or other numerical jobs compete for CPU.

The diagnostic report runs Profile (including C frames), JET and AllocCheck.
AllocCheck cannot prove through the runtime foreign-function pointer and reports
that call as unresolved; this does not establish that Rust is allocation-free.
Julia allocation sampling and BenchmarkTools memory counts only describe the
Julia caller. Native allocation counters and per-stage measurements are still
required before evaluating the optimization in #318. The quick run checks all four scalar types at one size/mode; it is a harness
check, not a general performance conclusion.

## Rust allocation and phase capture

Build the same source separately with `--features profile-general-evd`, then
supply that library with `COLMAT_EVD_LIBRARY` and add `--native-profile` to the
benchmark command. Keep the production and profiling libraries in different
paths; do not replace a library while a Julia process is using it. The feature
is off by default, including its C capture exports and global allocator.

Each case warms the profiler and records five individual captures. The counters
intercept successful Rust GlobalAlloc allocation/zeroed/reallocation requests
and logical frees. Requested byte volume and extra peak live payload bytes are
not allocator bookkeeping, process RSS, direct C malloc or OS thread-stack
measurements. Reallocation counts as replacing the old request with a new one.
Native worker allocations are attributed to the active caller phase. Snapshot
collection waits for in-flight counter updates before checking phase totals.

Phase times are inclusive instrumentation measurements. `total` encloses the
whole native entry and must not be added to its subphases. Before optimization,
`native_evd_combined` is the public high-level EVD call and already includes one
reduction/Schur calculation and requested vectors. The separately tagged
`hessenberg`/`schur` are the extra pass used to return A. Do not label the combined
API's time or shared scratch allocation as eigenvectors alone. Use production
builds for wall-time comparisons; profiling builds add allocator/clock overhead.
