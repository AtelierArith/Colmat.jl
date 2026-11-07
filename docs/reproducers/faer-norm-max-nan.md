# faer norm_max ignores NaN

Reported as [tensor4all/faer#316](https://github.com/tensor4all/faer/issues/316).

`norm_max` returns the largest finite magnitude, and `0` when every entry is NaN. `norm_l1` and
`norm_l2` of the same input return NaN. The NaN is lost on contiguous, strided, column-major and
complex inputs.

Executed on t4a-faer `d374ec0` and `57e0f37`, rustc 1.99.1, Darwin x86_64, `--release`. The
standalone crate depends on the checkout (`features = ["std", "linalg"]`) and uses
[`faer-norm-max-nan.rs`](faer-norm-max-nan.rs) as `src/main.rs`. Output:

```
1 1 NaN NaN
1 1
1 1
0 99
```

Expected: NaN on every `norm_max` value. Colmat checks for NaN natively before using `norm_max`
for `norm(A, Inf)` (`deps/rust/src/typed.rs`).
