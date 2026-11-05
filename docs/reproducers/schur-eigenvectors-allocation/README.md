# Internal Schur range-scaling allocation

Executed against latest faer main `2d4ea4eab8d8f724269508e16b4aaafaf5c67ae7`
(0.29.0). [Upstream report #270](https://github.com/tensor4all/faer/issues/270)
contains complete source, toolchain/platform and all 36 observed counter rows.
This crate needs no Julia, RustCall or local faer clone; Cargo pins the executed
git revision and the retained lock records resolved dependencies.

```sh
cargo run --release --locked --manifest-path docs/reproducers/schur-eigenvectors-allocation/Cargo.toml
```

For n=32, Seq, all four scalars, three repetitions after warmup:

| Input | Allocations | Requested bytes f32/f64/c32/c64 |
|---|---:|---|
| ordinary | 0 | 0/0/0/0 |
| large or small | 1 | 4096/8192/8192/16384 |

Setup, output/Q/query workspace allocation and independent normalized residual
and unit-column checks are outside the counter. Those numerical checks pass.
The public scratch query covers only V; `scaled_into_range` creates a separate
owning matrix in private real/complex Schur back substitution. Proposed canonical
correction is to put that optional copy in queried stack space and propagate
its lifetime through all consumers, preserving the original public T for final
real-pair classification/normalization. That correction is not implemented here.
Colmat connects declared kernel scratch and packed real output, while retaining
the existing numerically correct kernel and its extra extreme-scale allocation.
This is not a timing/scaling or whole-API allocation claim.
