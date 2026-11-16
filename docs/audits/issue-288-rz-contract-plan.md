# Issue #288: RZ factorization and Q application C contracts

Status: canonical scope and primary-source investigation only. Adapter,
numerical interoperability and immutable full-suite evidence remain pending.
Issue #288 remains open.

Canonical scope contains eight S/D/C/Z TZRZF high/work names and eight real
ORMRZ/complex UNMRZ high/work names. There are no TZRQF, LATRZ or LARZ C
names in the canonical inventory. This stage follows the 340-name pivoted QR
candidate and would reach 356 original names, leaving 1260 plus the separate
20 Rook C targets. These are candidate counts, not merged-main coverage.

Primary sources are Reference LAPACK v3.12.1
[TZRZF work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_ztzrzf_work.c),
[UNMRZ work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmrz_work.c)
and [UNMRZ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmrz.c).

TZRZF work uses layout,M,N,A,LDA,TAU,WORK,LWORK; high omits WORK/LWORK.
The input is upper trapezoidal, M<=N, and compact row reflectors represent
A=(R,0)*Z. Row-major LDA<N returns -5 before the numerical dimension checks;
queries bypass row packing. Output TAU has M elements. Investigation must
separate query, minimum work and execution WORK(1), including square inputs.

ORMRZ/UNMRZ use layout,SIDE,TRANS,M,N,K,L,const A,LDA,const TAU,C,LDC;
work adds WORK,LWORK. SIDE determines Q order and minimum workspace. The
logical reflector array has K rows and Q-order columns; only its final L
columns represent reflector tails. Negative INFO shifts Fortran arguments
by one. Const factor inputs and caller padding must remain unchanged.

A source observation needs an executed padded-buffer probe: reference UNMRZ
row conversion checks LDA against M, copies K-by-M A and allocates K-by-M
temporary storage even for SIDE=R, whose Q order is N. The high NaN scan also
uses K-by-M before C and TAU scans (-8,-11,-10). When M!=N this may scan
unused factor columns or omit active tails. The proposed
`reproducers/lapacke_rz_nan_extent.py` runs unchanged high wrapper sources
with numerical work stubbed and records source hashes. It is a policy-only
probe, not a numerical oracle. Execution is currently pending executable
startup; no observed result is claimed. Buffers cover both possible extents.

`reproducers/tzrzf_square_workspace.py` is a separate Fortran workspace MWE
with a valid square upper-triangular input and untouched padding. It compares
query, insufficient work and legal execution using caller-specified native
and independent OpenBLAS libraries. Its corrected-input execution remains
pending; source observations are not counted as executed reproduction.

Required evidence includes independent RZ factors and mutual ORMRZ/UNMRZ
consumers, reconstruction, unitarity, left/right N/T/C, all scalar types and
integer ABIs, layouts and DLL/LBT routes. Cover empty/square/wide/deficient
and scaled inputs, partial L and K, ignored coordinates and NaNs, exact
headers/queries/minimal work, output guards, independent C prototypes and
ILP64 values beyond int32. Native colmat-rs changes and PRs are not authorized.

Cloud contract run 43904380466 completed successfully at a172fbb. Its
artifact ZIP digest and reproducer source hashes were verified before
import. Sixteen unchanged-wrapper observations confirm K-by-M NaN scans
for SIDE=R in both layouts: an active final column is missed when M<N and
a padded column is scanned when M>N. Numerical work was stubbed; these
observations establish policy extents only. The own wrapper uses Q order N.

Twelve independent Fortran observations on valid square triangular inputs
show own TZRZF query/execution WORK=3 versus OpenBLAS WORK=1, and unchanged
sentinel WORK on insufficient work versus reference publication of one.
The own boundary now uses minimum/query one for square inputs and publishes
WORK after dimension validation but before the insufficient-work error.
A new LWORK=1 check is prepared. The C adapters are drafted but compilation,
numerical validation and full-suite proof remain pending.
