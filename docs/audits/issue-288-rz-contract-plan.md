# Issue #288: RZ factorization and Q application C contracts

Status: merged into main in PR #398 (d4b9ecfc224a6824d6cf903d33e37807168c4fe5).
CI 43942939431 (Linux/macOS) and Documentation 43942939256 passed. Both
downloaded artifacts validate 323 source hashes, 1634 canonical entries and
5995 witnesses at test merge 2bc53acc03f6648a4b82ab25a043edfe82694514.
Its parents and tree match the reviewed head and base. The merged tree,
verified optimized DLL and Python 28 post-merge checks pass. Post-merge inventory 15 also passed.

Issue #288 remains open.

The canonical scope contains eight S/D/C/Z TZRZF high/work names and eight
real ORMRZ/complex UNMRZ high/work names. No TZRQF, LATRZ or LARZ C names
exist in the canonical inventory. Following the 340-name pivoted QR candidate,
this stage reaches 356 original names, leaving 1260 plus the separate 20
Rook targets. These are candidate counts, not merged-main coverage.

Primary contracts are Reference LAPACK v3.12.1
[TZRZF work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztzrzf_work.c),
[UNMRZ work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmrz_work.c)
and [UNMRZ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmrz.c).

TZRZF work takes layout,M,N,A,LDA,TAU,WORK,LWORK; high omits WORK/LWORK.
Input A is upper trapezoidal with M<=N; its row reflectors represent
A=(R,0)*Z. Row LDA<N returns -5 before numerical dimension checks. Queries
bypass row packing. Only the upper trapezoid is packed and copied back;
strict lower entries and caller padding remain unchanged. Square factors
use minimum/query work one and publish WORK before insufficient-work errors.

ORMRZ/UNMRZ takes layout,SIDE,TRANS,M,N,K,L,const A,LDA,const TAU,C,LDC;
work adds WORK,LWORK. Q order is M for left application and N for right
application. Only the final L factor columns are reflector tails. Packing
skips zero-TAU reflectors and avoids all A/C reads for identity operations.
Queries and empty operations skip array reads. Successful empty query and
execution publish WORK=1; formal side-dependent minimum validation remains
separate. High-level scratch allocation also satisfies that formal minimum.
Negative numerical INFO shifts Fortran argument positions by one.

Executed reproductions use padded buffers and independent OpenBLAS:

- Run 43904380466 confirms the unchanged high wrapper scans K-by-M factor
  coordinates for SIDE=R: it misses active columns for M<N and reads padding
  for M>N. Numerical work was stubbed, so this proves NaN policy only.
  The adapter uses the actual Q order. See issue-288-rz-reference-nan.json.
- The same run establishes square TZRZF WORK differences on valid triangular
  input. Run 43905749109 confirms the correction accepts LWORK=1 and returns
  WORK=1. See issue-288-rz-square-workspace-reproduction.json.
- Run 43907356886 confirms all 40 own/reference pairs covering square and
  empty application cases, with INFO/WORK and array preservation matching.
  See issue-288-rz-empty-apply-workspace.json.

Focused cloud run 43910282161 passes 33024 factor/consumer and 46176
application assertions, 816 policy checks and 460 Rust tests. Numerical
operators use 100*epsilon tolerance, with a 10x reconstruction allowance.
Run 43912808677 additionally passes two independent C runner assertions;
the runner covers all 16 declarations and both integer ABIs. Artifact ZIP,
revision, formatted-source hashes and captured patches were verified before
local import. See issue-288-rz-focused-numerical.json and
issue-288-rz-c-abi-provenance.json for exact provenance.

Local debug and optimized tracing pass all preceding LAPACKE checks and the RZ tests,
including 7126 execution-gate assertions. It records 1424 completed typed
observations across 356 canonical names, both integer ABIs and DLL/LBT
routes. issue-288-rz-debug-focused.json and issue-288-rz-optimized-focused.json record the 323-source map, installed
library hash and evidence/log digests. This is focused evidence only;
full_suite_passed remains false.

Tests cover independent factors and mutual consumers, reconstruction,
unitarity, both sides, real transpose/complex conjugation, layouts, high/work,
empty/square/wide/deficient/scaled inputs, partial K/L, ignored coordinates,
NaNs, exact headers, queries and minimum work, const inputs, guarded output
storage, sparse identity operations and ILP64 values beyond int32.

The temporary branch workflow also ran exact frozen parent full suites at
618d002 (QL/RQ), 9c2c495 (Q generators) and 09aedc7 (pivoted QR). Those
results prove only their own sources. Restore ordinary repository CI before
an RZ PR, then obtain source-bound immutable full-suite proof for this stage.
Native tensor4all/colmat-rs edits and PRs are not authorized. Upstream reports
must contain an executed MWE and improvement proposal only.

Immutable full Pkg.test(allow_reresolve=false) passes at
2122f249759873f5f1b65d7f67e7a1f0010266dc with all 323 source hashes,
manifest, registration and optimized DLL unchanged. The imported evidence
contains 5995 completed witnesses. Exact TOML/log digests and library/source
provenance are recorded in issue-288-rz-full-provenance.json. Prior statements
about pending full proof are superseded by this result. Publication checks,
preceding-candidate integration and final PR CI remain required.

Post-import Python 28, registration inventory 15 and documentation build pass.
All 323 execution-source hashes and the optimized DLL remain unchanged.
