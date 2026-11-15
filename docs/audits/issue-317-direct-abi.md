# Issue #317: direct integer ABI validation

The starting point is Colmat.jl `01c98a24f455fa23f37d6e6e28d4f15288c827fc`,
with native colmat `4e6ed0e34659147a5c744ab6431f150e78db6f82`.
The full suite records completed concrete foreign calls for all 1,086 forwarded
LAPACK canonical symbols in each integer ABI. This is entry execution evidence,
not certification of every numerical option or an independent algorithm.

The original 76-symbol baseline comprises 48 LU/Cholesky entries, eight SVD
entries, four GEEV entries, four GEEVX entries and twelve selfadjoint EVD entries.
Each count includes applicable scalar variants, before the integer-ABI split.

| Tests | Actual calling route | Contract exercised |
| --- | --- | --- |
| `test/direct_factor_abi.jl`, `test/direct_factor_boundaries.jl` | DLL exports and libblastrampoline's exported LP64/ILP64 stubs | GETRF/GETRF2/GETF2/GETRS/GETRI/GESV and POTRF/POTRF2/POTF2/POTRS/POTRI/POSV; reconstruction, solve, inverse, query/minimum GETRI work, pivots, padded arrays and output guards |
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

The final suite adds 92,724 checks for direct ABI contracts and recorder
behavior, alongside the 92,567 existing focused GEEVX checks. Invalid-argument
tests read and assert the TLS diagnostic immediately; the diagnostic is
consumed on read, preventing state from leaking to later tests. Rectangular,
empty, singular and ignored-triangle cases have independent numerical and
boundary checks.
The full suite passed with the integrated execution gate. Its exact source
revision is recorded in `docs/src/assets/lapack-abi-execution.json`. A later
metadata commit can reuse the evidence only when every recorded source hash,
dependency manifest, native pin and registration digest still match.

Native sanitizer execution has not been performed for this change. The local
Rust toolchains are stable/beta/versioned releases, with no nightly installed;
The active `rustc 1.99.1` (`58cf8f9c3`, x86_64-apple-darwin, LLVM 23.1.1)
rejects `rustc -Z help` with “the option `Z` is only accepted on the nightly
compiler.” Thus `-Z sanitizer` cannot be claimed as executed. The C guard
page witness covers integer-write ABI mismatch, not general Rust memory safety,
Julia GC interactions or arbitrary numerical workspace errors.

The SVD cases allocate only the documented thin/full outputs (or the dummy
outputs for overwrite/values-only modes), check both complete full bases, and
use LAPACK 3.12.1 minimum complex RWORK for the tested aspect ratios. Selected
EVD allocates Z and ISUPPZ using the independently computed selected count.
Query must preserve result arrays. Reference calls have the same boundary
checks, so an oracle workspace overrun cannot silently validate the adapter.

The collector is part of the ordinary `Pkg.test()` entry point. It maps test
pointer calls to the actual DLL exports or exported LBT stubs, checks concrete
integer signatures, and records calls only after they return. Static
symbol/library ccalls, high-level calls inside the standard library and isolated
workers are not instrumented. The full suite still executes these tests.

All project Julia sources, Rust wrapper sources, test Julia/C files and build
configuration are hashed before testing. The source set and hashes must match
at completion. The importer rechecks them, the dependency manifest, native pin
and registration digest. ABI metadata columns are excluded from the registration
digest to avoid invalidating evidence when only that metadata is rendered.

Eight integer-free Givens entries use the identity of each LBT stub rather than
an inferred integer argument. Sixteen GEQRT2/GEQRT3/LAUUM/LAUU2 entries share
native addresses; numerical calls through their separate LBT stubs identify
all names. Ambiguous DLL addresses remain explicitly unresolved in the raw
artifact. Existing per-family numerical tests and the original-driver additions
supply the contracts; a pointer witness alone does not prove them.

Wide integer output sentinels have nonzero upper and lower halves. The isolated
C test also proves that a narrow INFO write leaves nonzero data in wide storage;
the converse mismatch is the protected-page crash. This catches both directions
without allocating large arrays.

`docs/import_abi_execution.py` validates a successful artifact before publishing
it. `docs/generate_lbt_inventory.py` assigns `direct_call_witness` to those 1,086
LAPACK entries and leaves all other canonical rows `not_catalogued`. The raw
snapshot retains per-call route, concrete signature, source and line, as well as
platform, Julia version, binary hash and exact revision. Linux/macOS CI saves
its own artifacts and must satisfy the same all-entry gate.
