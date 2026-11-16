# Issue #288: LAPACKE connection audit (in progress)

The full acceptance scope remains the 1,616 canonical entries in
`issue-288-lapacke-scope.json`. That file is derived from the original #262
inventory, not the shrinking current set of unconnected entries. This issue
must not be closed upon completing only the initial numerical families.

## Reference contract

Reference-LAPACK tag `v3.12.1`:

- `LAPACKE/include/lapacke.h`, blob `82cc4e6c1e4d29ae81b88342fa07ece9339124bb`.
- `LAPACKE/src/lapacke_dgetrf_work.c`, blob `62e6fa502734cebb5363462c842e839792816d33`.
- `LAPACKE/src/lapacke_dgetrf.c`, blob `6232721b4c595495494b978e4148e00af4de16e8`.

`matrix_layout` remains C `int` in both ABIs. Dimensions, pivot outputs and
returned INFO follow `lapack_int`. Negative Fortran INFO shifts by one for
the layout argument. The row-major GETRF work entry checks LDA before the
underlying Fortran parameter checks; row-major empty matrices permit LDA=0
when N=0. Positive numerical INFO is preserved.

## Initial implementation

`deps/rust/src/lapacke.rs` introduces GETRF work entry adapters for the four
scalar types and both integer ABIs. Column-major dispatches to this adapter's
existing native-backed Fortran entry. Row-major copies logical entries to a
column-major temporary, factors there and copies back without touching row
padding. Transpose allocation failure/address overflow returns -1011.

These new exports are **not yet registered in LBT**. They therefore do not
change the advertised connection count or establish completion of any frozen
scope entry. High-level GETRF still requires the separate optional NaN-check
contract. Other families require their own workspace, layout and callback
contracts; GETRF conversion is not a blanket implementation for them.

## Remaining gates

- Independent C ABI calls and actual LBT calls for every new registered entry.
- Source-bound execution evidence for both integer ABIs and dispatch routes.
- High-level versus work entry tests, NaN policy and argument precedence.
- Layout/stride/padding, complex data and external Fortran interoperability.
- Full Rust and Julia regressions, documentation and inventory regeneration.
- Remaining full frozen scope, including numerical dependencies #306/#313.

Native numerical revision: `4e6ed0e34659147a5c744ab6431f150e78db6f82`.

## Local verification

The targeted Rust run passed all nine GETRF work tests, including nonzero imaginary matrix entries for both complex precisions and
ABIs. Tests reconstruct P*A=L*U with scalar arithmetic, compare padded row/column
results and pivot indices, preserve padding, propagate singular INFO and reject
invalid/overflowing dimensions before dereferencing null buffers. This is
adapter-level evidence, not C compiler or LBT execution evidence.

The full local Rust suite passed: 398 unit tests, zero failures; formatting
checks passed. Julia/LBT validation remains pending until registration exists.
