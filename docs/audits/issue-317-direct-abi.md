# Issue #317: direct integer ABI validation (in progress)

The starting point is Colmat.jl `01c98a24f455fa23f37d6e6e28d4f15288c827fc`,
with native colmat `4e6ed0e34659147a5c744ab6431f150e78db6f82`.
This audit is incomplete. It does not certify all 1,086 forwarded LAPACK
canonical symbols, and does not change inventory execution classifications.

The original 76-symbol baseline comprises 48 LU/Cholesky entries, eight SVD
entries, four GEEV entries, four GEEVX entries and twelve selfadjoint EVD entries.
Each count includes applicable scalar variants, before the integer-ABI split.

| Tests | Actual calling route | Contract exercised |
| --- | --- | --- |
| `test/direct_factor_abi.jl` | DLL exports and libblastrampoline's exported LP64/ILP64 stubs | GETRF/GETRF2/GETF2/GETRS/GETRI/GESV and POTRF/POTRF2/POTF2/POTRS/POTRI/POSV; reconstruction, solve, inverse, query/minimum GETRI work, pivots, padded arrays and output guards |
| `test/direct_svd_abi.jl` | DLL exports and libblastrampoline's exported LP64/ILP64 stubs | GESVD/GESDD; full/thin/overwrite/values-only, rectangular and empty arrays, query/minimum/insufficient work, complex RWORK and integer IWORK boundaries |
| `test/direct_geev_abi.jl` | DLL exports and libblastrampoline's exported LP64/ILP64 stubs | GEEV; left/right vectors, eigenvalue comparison, residuals and normalization, query/minimum/insufficient work, invalid options and empty arrays |
| `test/direct_selfadjoint_abi.jl` | DLL exports and libblastrampoline's exported LP64/ILP64 stubs | SYEV/SYEVD/SYEVR or HEEV/HEEVD/HEEVR; query and each workspace minimum, integer M/ISUPPZ/IWORK, all/index/value ranges, UPLO, residuals, orthogonality, guards and empty arrays |
| Existing `test/geevx_expert*.jl` | DLL exports plus isolated LBT process tests | All four types and both ABIs, native/expert modes, integer ILO/IHI/IWORK, query/minimum work, missing implementations and ABI bridging; added in #316 |

The new helpers specialize `ccall` signatures on concrete `Int32`/`Int64` and
scalar types. The LBT cases resolve the trampoline library's own exported
symbols, rather than merely retrieving and calling its backing function pointer.
Independent numerical comparisons call the retained OpenBLAS implementation
with its own concrete ILP64 signature; reconstruction and residual checks use
the original input matrix. Eigenvector phases are not compared.

An ILP64 GETRF dimension of `2^32`, paired with a tiny leading dimension and
zero columns, must return the argument-4 error without accessing large arrays.
Integer and floating outputs have guards; leading dimensions exceed active
matrix dimensions. These cases complement rather than replace isolated
crash detection.

`test/abi_guard_pages.jl` compiles the test-only C caller
`test/abi_guard_page_worker.c`. It first proves a valid ILP64 zero-order POTRF
call can write INFO immediately before a protected page. The companion passes
only four bytes of INFO storage to that same eight-byte callee. The parent
requires the ready marker and a fatal signal, not merely a nonzero exit code.
Core files are disabled. This runs on Linux/macOS; other platforms are skipped
explicitly. A Julia-only caller was unsuitable on macOS because Julia converts
the protected-page fault into `ReadOnlyMemoryError` and exits normally with an
error status, losing evidence of the original terminating signal.

The 70,133 new focused checks pass together with the 80,960 existing GEEVX
checks. Invalid-argument tests read and assert the TLS diagnostic immediately;
the diagnostic is consumed on read, preventing state from leaking to later
tests. Full-suite validation remains pending.

Native sanitizer execution has not been performed for this change. The local
Rust toolchains are stable/beta/versioned releases, with no nightly installed;
The active `rustc 1.99.1` (`58cf8f9c3`, x86_64-apple-darwin, LLVM 23.1.1)
rejects `rustc -Z help` with “the option `Z` is only accepted on the nightly
compiler.” Thus `-Z sanitizer` cannot be claimed as executed. The C guard
page witness covers integer-write ABI mismatch, not general Rust memory safety,
Julia GC interactions or arbitrary numerical workspace errors.

Remaining work includes the execution-evidence catalog for all connected
families, review of contract gaps outside the original baseline, and full-suite
and hosted Linux/macOS validation at the final change. Passing these focused
tests alone does not complete #317.
