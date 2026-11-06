# faer triangular-copy adoption control

This standalone program validates the numerical contract of the copy path
adopted in [faer PR #282](https://github.com/tensor4all/faer/pull/282), pinned to
`2b87010cb82efbe0a90eceeb51931a833e1de8e6` (faer 0.31.0). It is a regression
control, not a bug reproduction or benchmark; the elementwise implementation
before that PR is also expected to pass.

From this directory, run:

```sh
cargo run --locked --release
```

Four native scalars are checked against an independent elementwise oracle for
full/lower copies, empty/small/tall/wide/tile-boundary shapes, reversed rows and
columns, conjugation, distinct NaN payloads and signed zeros. Lower copies must
leave the upper sentinel bits intact. Cholesky and partial-pivoted LU from the
same row-major/column-major matrix must have identical active factors and LU
permutations at 65, 129 and 257, separately with Seq and Team2. Timings are not
measured, and no performance claim is made.
