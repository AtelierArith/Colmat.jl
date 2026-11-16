# Issue #372: real-symmetric/Hermitian ROOK adapters

The adapter uses public `LbltParams.pivoting=PivotingStrategy::Rook` at native
revision `4e6ed0e34659147a5c744ab6431f150e78db6f82`. It adds the 20 Fortran scalar
entries for real SY and complex Hermitian HE: TRF, TF2, TRS, TRI and SV, each in
LP64/ILP64. Complex transpose-symmetric SY is a separate unsupported contract.
The corresponding LAPACKE candidates remain part of #288.

Native factors store a final permutation, separate block subdiagonal and global
unit L. Each ROOK two-column pivot requires two independently encoded negative
IPIV entries, unlike classic BK's one interchange. Packing applies both swaps
before mapping global L rows into the stepwise compact columns; unpacking applies
both to the accumulated earlier columns. Upper storage uses a reversed basis,
including the order of the two swaps. Tests pass native factors to retained
OpenBLAS solve/inverse and its factors back to native solve/inverse. An independent
product of elementary permutation/triangular matrices reconstructs the original
matrix; it does not merely round-trip the adapter's pack/unpack helpers.

The IPIV reference is [LAPACK 3.12.1 DSYTRF_ROOK](https://netlib.org/lapack/explore-html/db/df4/dsytrf__rook_8f_source.html).
[ZHETRI_ROOK](https://netlib.org/lapack/explore-html/d5/d87/group__hetri__rook_ga7d0ffc1f972e620bee37e3eed6188c19.html)
defines the selected-triangle inverse and N-element caller work. TRF/SV accept
the legal minimum caller workspace of one and return one from their queries;
the native operation owns its scratch. No identical pivot sequence, dedicated
Fortran implementation, blocking strategy or performance is claimed.

## Subnormal scaling

A nonzero subnormal D block can make the native reciprocal overflow before it
cancels with the matching RHS, producing NaNs for an exact identity solution.
A tiny pivot preceding normal trailing rows can also generate nonfinite factors.
The standalone safe public-API MWE is retained at
[`docs/reproducers/issue-212-lblt.rs`](../reproducers/issue-212-lblt.rs) and reported
with the pinned manifest and improvement proposal in
[tensor4all/colmat-rs#212](https://github.com/tensor4all/colmat-rs/issues/212).
Both dev and release executions fail the desired invariant. Normal controls pass.
Main `a5de9b3` was only inspected; the owning factor/solve blobs match the executed
revision. No native source, push or PR is part of this work.

The original posted Rust example used `powi(-1040)`, which returned zero in the
local dev build; that example did not establish a nonsingular-system defect.
The report was corrected and re-executed with exact nonzero subnormal bits and
explicit positivity/subnormality assertions. The retained source is the corrected
executed MWE. Julia probes use nonzero `exp2` values.

The shared dense/packed BK adapter now applies a symmetric power-of-two
congruence only to input rows whose entire magnitude is subnormal. After native
factorization, with R equal to the permuted row scale, it restores
`L = R^-1 L' R` and `D = R^-1 D' R^-1` componentwise. Normal input units/pivot
selection are preserved. Sequential component operations avoid overflow from
forming the product of two large row scales.

For subnormal D, public unit-triangular solves surround a public native solve on
block-diagonal factors with D and forward-transformed RHS rows scaled by the
same safe powers of two. This handles mixed normal/subnormal blocks without
writing a replacement private block solver. Normal D uses the original native
solve. The rare path allocates an additional owned matrix; there is no
allocation-free or throughput-parity claim. These are adapter-side protections;
they do not resolve native #212 or promise accuracy for arbitrary conditioning.

OpenBLAS 0.3.30's classic TRF also produced NaN factors (INFO=3 for the 4x4
all-subnormal block fixture) in all four scalar types locally. Its raw classic
multi-block factors are therefore not an oracle for that test. The native
adapter must instead reconstruct every original element and recover the known
identity solution. ROOK subnormal cross-provider factors/solves remain tested;
classic/ROOK ordinary-value factor interchange remains tested. The reference's
1x1 reciprocal path is not used as an oracle for an exact finite solve either.
No report was sent to a separate upstream tracker.

## Validation so far

On Julia 1.13.3, rustc 1.99.1, x86_64 macOS, native pin 4e6ed0e:

| Focused test | Checks |
| --- | ---: |
| Factor interchange, elementary reconstruction, DLL and named LBT stubs | 57,568 |
| Chain search requiring both nontrivial swaps | 1,536 |
| Uniform extreme 2x2 units (f32 ±100, f64 ±900 binary exponents) | 2,208 |
| Singular factors, inverse INFO and unchanged driver RHS | 11,520 |
| Invalid arguments, minimum work, wide integer/address boundaries and zero RHS | 1,864 |
| Nonzero subnormal blocks, mixed scales, classic/ROOK, exact finite solutions | 13,312 |
| Total | 88,008 |

All four scalar types, both integer ABIs and both triangle directions are
included. Concrete DLL/LBT calls verify the backing pointer, rather than treating
fallback success as native execution. Integer/output padding guards, query
preservation, unused triangles and Hermitian imaginary diagonals are checked.
Subnormal inverse entries in these fixtures exceed the scalar range; those
cases test representable solves, not an impossible finite-inverse requirement.
The Rust suite passed 295 tests. Full package/standard regression and refreshed
all-entry execution evidence are still required before merging.
