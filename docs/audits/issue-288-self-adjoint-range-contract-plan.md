# Issue 288: SYEVR / HEEVR contract preparation

Status: primary contracts and an executed endpoint reproduction collected.
Range adapters, boundary correction, numerical/C ABI checks and full-suite
evidence remain pending. The preceding 124-entry candidate is frozen during
its full package run in a separate worktree. Issue #288 remains open.

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
