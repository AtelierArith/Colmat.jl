# Issue #288: Q application C adapters

Status: canonical scope and primary contract investigation; implementation
and full-suite evidence pending. Issue #288 remains open.

The canonical inventory contains 32 high/work names for real ORMQR/ORMLQ/
ORMQL/ORMRQ and complex UNMQR/UNMLQ/UNMQL/UNMRQ. These follow the 260-name
Q-generation candidate and would bring the candidate to 292 names, leaving
1324 original-scope names plus the separate 20 Rook C targets. Candidate
counts are not merged-main counts.

Reference LAPACK v3.12.1
[ORMQR work](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dormqr_work.c)
uses layout, SIDE, TRANS, M,N,K, const A,LDA,const TAU,C,LDC,WORK,LWORK.
High omits WORK/LWORK. Row-major LDA and LDC checks (-8/-11) precede flag
and dimension checks; query calls use temporary column-major strides without
reading A or C. QR/QL store A as order-by-K; LQ/RQ store K-by-order, where
order is M for SIDE=L and N for SIDE=R. SIDE/TRANS are C chars and numerical
integers follow lapack_int. Real transpose and complex adjoint options need
separate checks, as do query and minimal WORK.

High-level NaN scans visit A, then C, then TAU and return -7/-10/-9.
Primary source investigation also found a reference asymmetry requiring an
explicit independent contract probe before choosing behavior: real ORMLQ uses
K-by-order, but complex
[UNMLQ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_zunmlq.c)
uses K-by-M; real and complex
[ORMRQ high](https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/LAPACKE/src/lapacke_dormrq.c)
also use K-by-M. These differ for SIDE=R. Tests must distinguish addressable
padded inputs, skipped NaNs, precedence and numerical reflector dimensions;
a source observation alone does not establish a safe complete contract.

Required evidence includes independent OpenBLAS applications and explicit Q
products, all orientations, scalar types, ABIs/layouts, DLL/LBT and high/work,
SIDE=L/R and legal transpose flags, empty/rectangular inputs and K=0/partial/
full, complex phase TAU, const A/TAU, C padding, query immutability, diagnostics,
active/inactive NaNs and signed ILP64 values outside int32. Independent C,
typed completed-call traces, optimized repetition and immutable full package
proof are required before publishing an implementation PR. No native source
changes or upstream PRs are part of this stage.

Executed `reproducers/lapacke_q_apply_nan_extent.py`: unchanged v3.12.1 high
wrappers for all 16 scalar/family combinations, both layouts and two right-side
rectangles produce 64 verified observations. Numerical work is explicitly
stubbed; this is a policy-only probe, not a numerical reproduction. Every
primary source SHA-256 and observation is stored in the adjacent reference
policy JSON. The buffers are padded for either convention, so the probe itself
performs no out-of-bounds access.

For SIDE=R,M=3,N=5,K=2, real ORMLQ scans 2x5 and finds a NaN in column 5;
complex UNMLQ and all ORMRQ/UNMRQ scan 2x3 and miss it. At M=5,N=3 the latter
scan 2x5 and reject a NaN beyond the logical 2x3 reflector matrix. The planned
Colmat.jl high boundary uses the actual Q order for all orientations. This
intentional compatibility correction follows Issue #288's explicit requirement
not to read unused storage and checks all active reflectors. Tests must retain
both observed reference cases and the logical-storage expectation; no family,
SIDE or scalar case is excluded. NaN diagnostic precedence remains A, C, TAU.
