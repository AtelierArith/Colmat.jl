# faer symbolic LU/QR keep the COLAMD workspace with an explicit ordering

Reported as [tensor4all/faer#322](https://github.com/tensor4all/faer/issues/322).

`factorize_symbolic_lu_with_column_ordering` and `factorize_symbolic_qr_with_column_ordering` size
their internal `MemBuffer` as `StackReq::or(colamd::order_scratch(m, n, nnz), ...)` even when a
column ordering is supplied, and take no caller `MemStack`. Colmat runs COLAMD itself on caller
scratch and passes the ordering, so the faer-internal COLAMD-sized buffer remains (row 56 of the
API mapping, issue #118).

Executed on t4a-faer `d374ec0` (latest `main`, 0.32.0), rustc 1.99.1, Darwin x86_64, `--release`.
The standalone crate depends on the checkout (`features = ["std", "linalg"]`) and uses
[`faer-symbolic-colamd-scratch.rs`](faer-symbolic-colamd-scratch.rs) as `src/main.rs`. A counting
global allocator records the largest single allocation:

```
n = 2000, nnz = 7990, colamd::order_scratch = 332712 B
LU simplicial: explicit ordering -> largest allocation 332712 B, total 380712 B
QR simplicial: explicit ordering -> largest allocation 332712 B, total 428712 B
LU supernodal: explicit ordering -> largest allocation 468080 B, total 1836584 B
QR supernodal: explicit ordering -> largest allocation 468080 B, total 1885168 B
```

With an explicit ordering and a simplicial factorization, the largest allocation is exactly the
COLAMD workspace that is never used. With a supernodal factorization, the supernodal symbolic
workspace is larger and dominates.
