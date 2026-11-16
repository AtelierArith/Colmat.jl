# Issue #288: triangular-pentagonal QR C contracts

Status: canonical scope and primary contracts only; implementation and all
execution evidence remain pending. Issue #288 remains open.

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
