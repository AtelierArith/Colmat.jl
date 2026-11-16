# Issue #288: triangular-pentagonal QR C contracts

Status: all 32 C entries are drafted. Debug numerical, policy, independent C
caller and typed DLL/LBT checks pass in debug and optimized builds. Immutable
full-suite proof passes; publication checks and final CI/main integration remain pending. Issue #288 remains open.

The canonical inventory contains 32 S/D/C/Z high/work TPQRT, TPQRT2, TPMQRT
and TPRFB names. Following the 356-name RZ candidate this would reach 388
original names, leaving 1228 plus the separate 20 Rook C targets. Candidate
counts do not imply merged-main coverage.

Reference LAPACK v3.12.1 primary sources:
- [TPQRT work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztpqrt_work.c)
- [TPQRT2 work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztpqrt2_work.c)
- [TPMQRT work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztpmqrt_work.c)
- [TPRFB work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztprfb_work.c)

TPQRT uses layout,M,N,L,NB,A,LDA,B,LDB,T,LDT; work adds WORK without LWORK
or query. Logical A is upper triangular N-by-N, B is M-by-N pentagonal and
T is NB-by-N. Row dimension errors are LDA -7, LDB -9, LDT -11 in that
order. TPQRT2 omits NB and WORK in both high/work signatures; T is N-by-N.
Its row dimension positions are -5,-7,-9. Structural zero/ignored regions
must be established from numerical contracts before packing or NaN policy.

TPMQRT uses layout,SIDE,TRANS,M,N,K,L,NB,const V,LDV,const T,LDT,A,LDA,B,LDB;
work adds WORK without LWORK/query. Left application uses V M-by-K and A
K-by-N; right uses V N-by-K and A M-by-K. B is M-by-N, T NB-by-K. The row
wrapper checks invalid SIDE before deriving these shapes. Independent strides,
partial blocks and pentagonal active regions need numerical validation.

TPRFB adds DIRECT/STOREV flags and takes layout,SIDE,TRANS,DIRECT,STOREV,
M,N,K,L,const V,LDV,const T,LDT,A,LDA,B,LDB; work adds WORK,LDWORK. Its
Fortran routine has no INFO argument: the reference column-major C wrapper
returns zero after the void call. A native diagnostic must not be converted
into an invented negative C INFO. Row header checks and high NaN returns are
separate C-level behavior. Source observations suggest row conversion uses
caller LDV/LDT and M-derived A dimensions; require executed padded-buffer
probes before choosing any correction. No numerical result is claimed yet.

Required checks include all applicable flags, both integer ABIs, layouts,
high/work and DLL/LBT, L=0/full/partial and block tails, empty/rectangular and
rank-deficient factors, scales, const inputs, structural NaNs/guard pages and
independent factor consumers. Independent C declarations, typed completed
calls, debug/optimized repetition and immutable full proof are required.

This branch inherits temporary RZ diagnostic CI. Restore ordinary repository
CI through parent integration before opening a PR. Native colmat-rs edits
and PRs are not authorized by this stage.

Executed conversion probe: `reproducers/lapacke_tprfb_row_extents.py` compiles
unchanged v3.12.1 S/D/C/Z work wrappers with a numerical no-op and real
transpose helpers. Sixteen scalar/SIDE/STOREV observations all pass. For
M=3,N=5,K=2,LDV=7,LDT=6 it copies V as 7-by-2, T as 6-by-2 and A as
2-by-3 regardless of SIDE/STOREV. All caller buffers are padded for these
extents. This establishes conversion behavior only. It does not establish
numerical accuracy or authorize using these shapes as a numerical oracle.
For SIDE=R, logical A is M-by-K (3-by-2); V shape also depends on SIDE and
STOREV. The adapter must derive logical extents and preserve const/padding
storage. Exact source, reproducer and output hashes are recorded in
issue-288-pentagonal-row-extents.json. full_suite_passed remains false.

The draft adapters provide all 32 high/work exports in both integer ABIs.
Row conversion packs logical arrays and preserves const factor storage,
structural zeros and caller padding. TPRFB keeps the underlying void Fortran
contract: numerical argument diagnostics do not invent a negative C INFO.
High NaN scans and row-wrapper header errors remain C-level behavior.

Debug direct-DLL checks pass 13728 factor assertions, 32768 TPRFB assertions
against elementary reflector products and 26624 TPMQRT assertions using
independent OpenBLAS factors. Tests cover real/complex, both ABIs/layouts,
high/work, both sides/transposes, F/B and C/R storage, pentagonal overlap
and partial blocks. Initial empty-factor oracle calls incorrectly supplied
zero leading dimensions; after passing legal minimum one all factor checks
pass. Numerical source/library/log hashes are in
issue-288-pentagonal-focused-numerical.json. These are focused results only.

A separate unchanged high-wrapper NaN probe passes 32 scalar/layout/SIDE
observations. It confirms A scans use transposed logical dimensions: active
NaNs can be missed and padding NaNs rejected. The adapter uses logical A
rows/columns. Buffers cover both extents and numerical work is stubbed;
this is policy evidence only. See issue-288-pentagonal-nan-extents.json.
The stage still requires expanded empty/scaled/diagnostic tests, independent
C declarations, actual DLL/LBT calls and immutable full-suite evidence.

Debug DLL/LBT tracing passes with all 388 C names and 1552 completed typed
name/ABI/route observations, plus 7766 execution-gate checks. Factor tests
pass 29248 assertions, TPRFB 73728, TPMQRT 57344 and policy 3468, including
forward-pointer equality assertions. The independent C runner passes both
checks with separate prototypes for all 32 entries, scalar reflector products,
const/guard storage and signed ILP64 dimensions beyond int32. Rust passes
460 tests. Exact source/library/log/evidence hashes are recorded in
issue-288-pentagonal-debug-focused.json; full_suite_passed remains false.

Normal parent integration takes RZ full evidence and ordinary repository CI
from fd781ad. It preserves all production/test/Rust source hashes and the
debug DLL. Additional scale/rank/partial-overlap checks are running separately;
optimized repetition and immutable full evidence remain required.

Additional debug scale/rank/partial-overlap checks pass 193536 assertions,
covering powers +/-40 for Float32 and +/-400 for Float64, full/zero/rank-one
factors and partial L. The formatted test is now part of the ordinary suite.
Optimized repetition passes all numerical, policy, independent C and typed
DLL/LBT checks, including this extra test. It records all 388 names and 1552
completed observations, with 7766 gate assertions. Exact 330-source map,
optimized DLL hash and log/evidence digests are recorded in
issue-288-pentagonal-optimized-focused.json. The source is being frozen for
immutable full Pkg.test; no full-suite proof is claimed for this stage.

Immutable full Pkg.test(allow_reresolve=false) passes at
d5a996aed69380855f005ca4b2f6a27ba7ec520c. All 330 source hashes,
manifest, registration and optimized DLL remain unchanged. Imported evidence
validates 1106 Fortran LAPACK, 172 CBLAS and 388 LAPACKE entries with 6123
completed witnesses and 1870 forwards per ABI. Exact source/library/log/TOML
provenance is in issue-288-pentagonal-full-provenance.json. Earlier pending-full
statements are superseded by this result. Publication checks and final PR CI
remain required, after preceding candidates merge.

Normal parent integration at 34f762ab5563a681daf2f798c7def0c007da0601
incorporates the RZ publication branch through 742cbc1. Its ten changed files
are audit documents only. All 330 frozen execution sources, the manifest
and optimized DLL remain identical to the immutable full proof; this is not
a new full-suite run. Python 28 passes after integration. Inventory and docs
checks remain live. Final main integration and publication wait for RZ
PR #398 to merge. Exact parent, changed-file, source-map and library hashes
are recorded in issue-288-pentagonal-rz-integration.json.

The post-integration publication checks have now completed successfully:
Python 28, inventory 15 and the documentation build pass. The final source
and DLL comparison also passes. Earlier pending inventory/docs statements
are superseded by this result; RZ PR #398 and final main integration remain
pending before the pentagonal publication PR.
