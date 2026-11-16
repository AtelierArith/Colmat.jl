# Issue 288: SYEVR / HEEVR contract preparation

Status: initial Fortran correction, C adapters and numerical/C ABI checks pass.
Completed-call tracing, optimized release and immutable full-suite evidence
remain pending. The preceding 124-entry candidate passed its immutable full package run and
is under review in PR #384. Issue #288 remains open.

Planned scope: eight S/D SYEVR and C/Z HEEVR high/work names, both integer
widths. The original 1,616-name scope and separate 20 Rook targets remain intact.

## Reference-LAPACK v3.12.1 sources

| Source | Git blob |
| --- | --- |
| LAPACKE/src/lapacke_dsyevr_work.c | 184629f28eea62d0b95f2334828eb0e9b5de5857 |
| LAPACKE/src/lapacke_zheevr_work.c | e5413aef6bf0ce561f9e1faf12073b4ce5125d74 |
| LAPACKE/src/lapacke_dsyevr.c | 9af609447d80ee75a944a54cb0db19d40844f2a2 |
| LAPACKE/src/lapacke_zheevr.c | aecaaca9d13b78190f37c17871730bffe20f6218 |

VL, VU and ABSTOL are real C values, including complex variants; M and ISUPPZ
are integer outputs. Row-major LDA errors return -7 and LDZ errors -16 before
queries. The reference row stride requirement for Z is one for JOBZ=N,
N for RANGE=A/V with vectors, and IU-IL+1 for RANGE=I with vectors. Any work
length of -1 activates a query without reading A. Integer/real/main workspace
minima, partial queries, exact argument diagnostics and output guards need
separate evidence.

The reference row wrapper copies N or IU-IL+1 predicted columns of Z. Only M
columns are meaningful. The candidate must avoid reading uninitialized scratch
or writing unused caller columns when M is smaller, including M=0 and null
unused output pointers. This policy must be explicit in its contract audit.

## Endpoint boundary defect

For the exact Hermitian matrix with diagonal 2 and superdiagonal 1+i, N=3,
the characteristic polynomial is proportional to lambda*(lambda-2)*(lambda-4).
The interval (2,10] must therefore select one eigenvalue. The current Fortran
ZHEEVR ILP64 entry returns INFO=0/M=2 with values 2.0000000000000004 and
3.999999999999999; independent OpenBLAS returns INFO=0/M=1.

`reproducers/heevr_value_endpoint.jl` retains this executed example. The local
boundary currently filters rounded full-spectrum eigenvalues by direct scalar
comparison. The public native full-EVD result alone cannot determine this
interval decision reliably near an endpoint. Investigate the existing saved
backend patterns or a suitable range-aware numerical method before exposing
the C entries. The native EVD approximation itself is not classified as a bug.


## Initial boundary correction

Before writing eigenvalue/vector outputs, the native full-spectrum path now
rejects value intervals whose endpoints lie within 64 scalar epsilons times
the spectral scale of a computed value. The saved ILP64 SYEVR/HEEVR driver
handles those intervals, positive ABSTOL and native failures. The loader saves
all four drivers before forwarding; LP64 bridges M, IWORK and ISUPPZ through
initialized 64-bit buffers with checked conversions and preserved support
output guards. Unsupported conversion or missing saved configuration reports
failure rather than silently truncating an output.

The generator also restores the reference N=0/1 workspace minima of one.
The scalar vector case writes Z=1 and support [1,1] even when a value interval
excludes the scalar eigenvalue. Normal-size queries remain separate from
numerical execution and do not read A or change M.

The minimal endpoint example now agrees with OpenBLAS: INFO=0/M=1 and value
3.999999999999999. The independent-reference regression suite passes 15,752
checks over all four scalar types and both integer widths, N=0/1/3, upper/lower,
N/V jobs, A/I/V ranges and negative/zero/positive ABSTOL. Positive ABSTOL residuals
are checked against the requested absolute tolerance; orthogonality retains
precision-based tolerances. Padding, unused Z columns, eigenvalue/support guards
and scalar minimal workspaces are included. Rust passes 430 tests, including
precision/scale guard checks and a guarded wide-driver conversion test.

This checkpoint does not expose any additional LAPACKE range names and does not
qualify the remaining original acceptance scope as complete.


## Initial LAPACKE range adapter checks

The eight high/work names now export both integer ABIs. VL/VU/ABSTOL follow
real scalar by-value C declarations, with C-int layout, C-char flags and
integer-width-specific M/ISUPPZ outputs and INFO returns. High-level NaN checks
follow reference order: selected A (-6), ABSTOL (-12), and range-V-only VL/VU
(-8/-9). Work queries in any length return without reading A or changing M.

Row-major A conversion reads and writes only the selected triangle. Z copying
uses the returned M columns, with the reference scalar quick-return exception
that writes Z=1 even when its eigenvalue is excluded. Unlike the reference
wrapper's predicted-width copy, unused temporary columns are not read or copied.
This deliberately preserves unused caller columns and permits a null Z when
N>1 selects M=0. N=1/JOBZ=V still requires its scalar output storage.

The independent C11 worker registers separately resolved OpenBLAS wide drivers
through the public saved-backend hooks. It passes both integer widths, both
layouts, upper/lower/lowercase flags, N/V jobs, A/I/V ranges, scalar/empty calls,
M=0/null unused output, exact header and NaN diagnostics, partial workspace
queries, guarded integer outputs and ILP64 arguments above 32-bit range.
Its Julia runner passes two checks.

The direct-DLL and actual-LBT Julia suite passes 148,904 numerical checks,
against independently resolved OpenBLAS range results. It includes endpoints,
positive ABSTOL, poisoned unused storage, scalar output semantics, and ignored
imaginary diagonal NaNs in work calls. The separate Fortran regression suite
passes 15,752 checks. These are focused checks, not a full-suite artifact.


A further executed row-major work call with VL=NaN returned INFO=-9 but
previously overwrote all nine guarded Z entries using an unchanged caller M=99.
The adapter now rejects that invalid bound before packing and does not copy
outputs after a negative backend INFO. The independent C worker checks this
case for all scalar and integer combinations, preserving M and all Z guards.
