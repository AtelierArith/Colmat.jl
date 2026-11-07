# faer stats reverse results for a negative output-axis stride

Reported as [tensor4all/faer#317](https://github.com/tensor4all/faer/issues/317).

`stats::col_mean`/`col_varm` normalize a negative row stride with `mat.reverse_rows()` without
reversing `out` (or `col_mean`), so `row_mean`/`row_varm` of `reverse_cols(A)` come back in reversed
order. This happens for both `NanHandling` modes.

Executed on t4a-faer `d374ec0` and `57e0f37`, rustc 1.99.1, Darwin x86_64, `--release`, with
[`faer-stats-negative-stride.rs`](faer-stats-negative-stride.rs) as `src/main.rs`. Output:

```
Propagate: mean view [1.0, 2.0, 3.0, 4.0] owned [4.0, 3.0, 2.0, 1.0]
Propagate: varm view [20.0, 10.0, 20.0, 50.0] owned [32.0, 18.0, 8.0, 2.0]
Ignore: mean view [1.0, 2.0, 3.0, 4.0] owned [4.0, 3.0, 2.0, 1.0]
Ignore: varm view [20.0, 10.0, 20.0, 50.0] owned [32.0, 18.0, 8.0, 2.0]
```

The owned results are correct. Colmat flips input, output and means together before calling faer
(`deps/rust/src/statistics.rs`).
