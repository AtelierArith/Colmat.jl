# Issue #318: shared general EVD reduction

GEEV and native GEEVX (`SENSE=N`) now perform one Hessenberg reduction and one
Schur iteration. The returned A, eigenvalues, and requested left/right vectors
come from that same Schur form. Expert condition estimates keep the retained
provider path established in #316.

The implementation composes public colmat APIs at pinned revision
`4e6ed0e34659147a5c744ab6431f150e78db6f82`. It forms the reduction's Q only when
vectors are requested and reuses one scratch buffer sized for the maximum of
non-overlapping reduction, Q, Schur and eigenvector phases. Right vectors are
computed before left vectors. The left calculation uses the reversed transpose
of T and reversed conjugate Q views; real conjugate-pair columns are restored to
LAPACK's real/imaginary order. This avoids a second decomposition and avoids
copying a reversed Schur matrix. Output vector matrices and inverse-balancing
temporaries still allocate.

A componentwise power-of-two scaling protects the public low-level kernels at
extreme magnitudes. Eigenvectors are computed before restoring the Schur matrix
and eigenvalues to the caller's units. ABNRM and balancing metadata retain the
balanced input's original units. Query, minimum workspace and argument checking
remain in the existing exported drivers. Native failures still return INFO=1;
the change does not introduce a new nonconvergence encoding.

An executed 2-by-2 complex fixture with off-diagonals near `1e300` and `1e-300`
exposed an existing inverse-balancing normalization error: dividing a complex
value by a large real norm could square the denominator and produce NaN.
Componentwise real division fixes it. Both left and right scale directions are
covered for c32/c64. The standalone DLL probe now returns finite, unit-norm
vectors for N/S/B balancing and the transposed fixture.

## Measurements

Raw TOML reports, diagnostics, SHA-256 manifest and comparison JSON are in
[`benchmarks/results/issue-318`](../../benchmarks/results/issue-318).
TOML is gzip-compressed without a timestamp; each report retains all cases,
metadata and (for profiling builds) all five native captures per case.

| Measurement | Checked-out source |
| --- | --- |
| Production before | `4f8691fff4cb3e7356eac118bdccab33efadd118` |
| Native counters before | `34b5cc6bb44e05cf4a345d5cbf5abf8b4dc3b658` |
| Production and counters after | `9a80a2bb8d588e4eb8437d77260b11dcf53d3133` |

The original production DLL was retained from the #316 build at `01c98a2`.
`git diff 01c98a2 4f8691f -- deps/rust deps/upstream.toml` is empty: the Rust
sources, Cargo lock/configuration and native pin match the before measurement.
The after DLL was built with default features; a separate copy was built with
`--features profile-general-evd`. Reports record each actual DLL's SHA-256.
No colmat-rs source was changed.

All runs used Julia 1.13.3, rustc 1.99.1 (`58cf8f9c3`, LLVM 23.1.1), Intel
Core i9-9900K @ 3.60 GHz and x86_64 macOS (Julia platform
`x86_64-apple-darwin24.6.0`). BenchmarkTools 1.8.0, JET 0.13.1 and AllocCheck
0.2.8 are pinned in the isolated environment. JET was available and reported no
errors. AllocCheck reports the foreign function pointer dispatch as unresolved;
it cannot establish that Rust allocates nothing. Julia caller measurements
reported zero allocations/bytes in all 768 cases before and after.

Each full run covers four scalar types, n=32/96/256, NN/VN/NV/VV vector modes,
N/P/S/B balancing, deterministic dense and scaled/isolated fixtures, and native
participant settings 1/2. Ten steady-state timing samples use one evaluation
and preallocated inputs, with destructive-input reset outside the call timer.
Initial calls cross an invocation boundary; the first case of each scalar type
includes method compilation, rather than presenting it as steady-state time.
The seed, reset time, cold/compile time and actual configured width are recorded.
A kernel may use fewer participants than configured. Independent validation
uses the original OpenBLAS ILP64 provider at one thread.

The table gives the median of paired before/after production time ratios across
scalar types, balancing modes, fixtures and widths in each group. A ratio above
one indicates a shorter after time. Profiling-build wall times are not used here.

| n | Values only | Left only | Right only | Both |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1.842 | 1.655 | 1.634 | 1.566 |
| 96 | 1.913 | 1.683 | 1.691 | 1.604 |
| 256 | 1.972 | 1.663 | 1.707 | 1.585 |

These are sequential measurements on one host, not a randomized crossover or a
portable speed guarantee. They measure direct ILP64 GEEVX SENSE=N; GEEV uses the
same native core but was not separately timed. All 768 cases returned INFO=0.
The maximum after scaled residual was 2.3781e-7 (f32), 4.9778e-16 (f64),
2.3279e-7 (c32), and 4.7267e-16 (c64), comparable to the recorded before results.
The simple fixture tolerances do not certify every matrix conditioning regime.

Native counters intercept successful Rust GlobalAlloc requests and logical
frees, including zeroed allocations and reallocations. They do not measure
allocator bookkeeping, RSS, OS stacks or direct C malloc. The following are
medians over the same group of cases at n=256; full per-group/per-case data is
preserved in the reports.

| Mode | Allocation count before → after | Requested bytes before → after | Extra peak live bytes before → after |
| --- | ---: | ---: | ---: |
| NN | 21.5 → 12 | 1,845,804 → 210,532 | 1,580,132 → 205,924 |
| VN | 23.5 → 15 | 3,418,668 → 2,700,644 | 2,628,708 → 2,171,748 |
| NV | 23.5 → 15 | 3,418,668 → 2,700,644 | 2,628,708 → 2,171,748 |
| VV | 25.5 → 17 | 4,467,244 → 3,749,220 | 3,152,996 → 2,696,036 |

In the before capture, the additional Hessenberg/Schur pass accounted for about
50% of total measured time in values-only mode and 34–38% with both vectors,
depending on n. Those phase fractions are instrumentation observations, not
speedup estimates. In the after capture, the combined phase encloses the single
reduction, Q formation, Schur and eigenvectors. Nested inclusive times must not
be summed with the enclosing time. Shared scratch allocation is charged to the
enclosing phase. The feature is disabled in production builds.

## Reproduction and contracts

See [`benchmarks/README.md`](../../benchmarks/README.md) for build/run commands.
To recompute paired summaries with Python 3.11 or later:

```sh
python3 benchmarks/compare_general_evd.py benchmarks/results/issue-318/production-before.toml.gz benchmarks/results/issue-318/production-after.toml.gz
python3 benchmarks/compare_general_evd.py benchmarks/results/issue-318/profile-before.toml.gz benchmarks/results/issue-318/profile-after.toml.gz
```

The comparator checks identical case keys, native/tool/platform metadata, all
768 successful cases, ten samples and zero Julia allocations. Profiling JSON
also contains instrumented wall-time ratios; these are not production ratios.

Rust tests cover all four scalar types, all requested vector modes and balance
flags at ordinary, huge and subnormal magnitudes, real-pair residuals, returned
Schur structure, and complex normalization at large balance scales.
`test/general_evd_shared.jl` adds 24,768 checks through DLL and actual LBT stubs
in both integer ABIs. These cover query preservation and minimum work, padded
buffers, Schur blocks/eigenvalue correspondence, balanced norm/trace/ABNRM,
residuals, normalization, unrequested outputs and nonfinite-input INFO.
Subnormal comparisons explicitly account for smallest-component rounding when
restoring eigenvalues. Existing direct GEEV and GEEVX expert/boundary/process
regressions continue to cover retained estimates and integer bridging.


A separate archive audit verifies every compressed/raw SHA-256 and each of the
3,840 captures' phase totals in both profiling runs. Live-byte snapshots and
counted requests have independent capture boundaries: asynchronous worker
allocation/free events can cross the begin/end transition. With participant
setting 2, the before run had five live-vs-request boundary differences (maximum
24 bytes) and the after run had 31 (maximum 48 bytes). Nonzero end-minus-start
live deltas occurred in 21/37 captures respectively (maximum 80/72 bytes).
With participant setting 1, all 1,920 captures in each run had zero live delta
and no boundary discrepancy. These are observed differences, not a universal
error bound; live/peak payload is a window observation and is not a leak test.
The full audit is `capture-audit.json`; reproduce it with:

```sh
python3 benchmarks/audit_general_evd.py benchmarks/results/issue-318
```

The full `Pkg.test(; allow_reresolve=false)` passed at
`98312084c4b449421bb000fb23097c6c7e8c3d4a`, including standard BLAS/LAPACK and
concurrent caller regressions. The all-entry/provenance gate passed 209 checks
and recorded 1,086 LAPACK canonical entries in both integer ABIs against 202
source/build/test hashes. The README example and 293 profiling-feature Rust
tests also passed. Later documentation and execution-snapshot updates reuse
this evidence only when all recorded source hashes still match.

After importing the execution snapshot, all 18 Python regressions, 15 actual-LBT
inventory checks and the documentation build passed. Forwarding remains 1,288
entries per integer ABI; independent-algorithm classifications are unchanged.
