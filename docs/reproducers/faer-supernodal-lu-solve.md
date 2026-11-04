# Forced supernodal LU solve regression

Report: https://github.com/tensor4all/faer/issues/236

The adjacent executable was run against unmodified upstream main
`07a09e629de8ef4d027331e0464f8c94d34f2cbf` with default features disabled and
`std,linalg` enabled, rustc 1.99.1 on Linux x86_64.
For n=40, max residual is 300.07386715030725 for forced supernodal,
versus 2.318145675417327e-13 for auto and forced simplicial.
The complete portable Cargo project instructions are in the report.
The executable prints residuals for all cases and exits successfully; it is a
diagnostic comparison, not a passing numerical regression test.

Root cause verified: `lu_in_place_recursion` in dense partial-pivot LU gates the
panel triangular solve on the `rayon` feature. Removing that guard in a separate
upstream worktree fixes this probe without enabling Rayon: reconstruction error
7.105427357601002e-15, solve residual 1.5631940186722204e-13. Unmodified main with
Rayon enabled also passes. Colmat enables Rayon; its focused sparse LU component
tests pass 274 assertions after rebuilding against 07a09e6.
The upstream patch and regression are at `/tmp/colmat-faer-lu-baseline`, branch
`fix/lu-triangular-solve-without-rayon`; regression compilation is in progress.

Fix PR: https://github.com/tensor4all/faer/pull/237
Four scalar regression tests pass with and without Rayon. The CI sequential
regression lane now includes this test target. Hosted CI and merge remain pending.
