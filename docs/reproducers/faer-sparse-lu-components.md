# Native sparse LU component access

Issue: https://github.com/AtelierArith/Colmat.jl/issues/22

Baseline: t4a-faer 0.28.2, revision
`0bf88db6e70c29488df91a128f5e2562f8859730`.

The adjacent Rust compile probe calls `LuRef::l_factor` and `LuRef::u_factor`.
`cargo check` with default features disabled and `std,linalg` enabled exits 101:
both methods are absent (E0599). This establishes an API gap, not a numerical bug.

`NumericLuRaw` is private; its supernodal storage is also private. The high-level
`solvers::Lu` does not expose a borrowed `LuRef` or its permutations. Existing
simplicial component support uses a separately owned low-level factorization;
repeating that approach for auto/supernodal would not expose their saved factors.

Proposed upstream extension: a borrowed accessor on `solvers::Lu`, plus owned
CSC L/U extraction on `LuRef` that dispatches over the saved numeric variant.
The supernodal extraction must apply internal pivoting consistently with the
public row permutation, preserve unit diagonal L, and satisfy
`L * U = A[p, q]`. Validate real and complex scalar types, both numeric variants,
empty factors, pivoting across supernodes, and solve/reconstruction consistency.
Do not refactor the source snapshot or change the dependency checkout as a workaround.

The probe was executed in a standalone Cargo project using the pinned dependency
path. Implementation and numerical validation of the proposed API remain pending.

Upstream request: https://github.com/tensor4all/faer/issues/235

Current upstream main `07a09e629de8ef4d027331e0464f8c94d34f2cbf` was fetched;
its sparse LU implementation is unchanged from the verified baseline.
A separate prototype at `/tmp/colmat-faer-lu-work` adds supernodal CSC extraction
and passes `cargo check --no-default-features --features std,linalg`.
Numerical tests and high-level access remain pending; this is not adopted code.

## Numerical prototype investigation

The standalone reconstruction probe passed dimensions 0, 1, 3, and 12 for
all three selection thresholds using Float64. At dimension 40, forced
supernodal reconstruction failed (max error 11.897907859181734). The saved
factor's existing solve also produced max residual 300.07386715030725 for
`b[i] = i + 1`; auto gave reconstruction error 5.33e-15 and solve residual
2.32e-13. This does not yet establish an upstream defect: rerun the solve-only
probe on an unmodified current upstream before reporting. The deterministic
matrix is `A[i,j] = 0.1` on the diagonal, otherwise
`((17i + 13j) % 19) - 9` when `(i+j) % 3 == 0` or `abs(i-j) == 1`, and zero
elsewhere (zero-based indices). The executable prototype is currently at
`/tmp/colmat-lu-accessor-mwe/src/main.rs`.

After the separate dense LU feature-guard fix, the component extraction
prototype reconstructs all cases (n=0,1,3,12,40, three thresholds) for all four
scalar types. Complex fixtures currently have real-valued coefficients.
Public API documentation, genuinely complex fixtures, and upstream adoption
remain pending.
