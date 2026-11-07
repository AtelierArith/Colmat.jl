# 作業領域とスレッドチーム

caller-owned scratch と team の共通の使い方を説明します。各分解の詳細は該当ページを参照してください。

## Caller-owned parallel teams

```julia
using Colmat, LinearAlgebra
FaerTeam(2) do team
    A = FaerMatrix([1.0 2; 3 4])
    B = team_mul(team, A, A)
    @assert Matrix(B) == [7.0 10; 15 22]
    mul!(A, team, A, A, 2.0, 0.0)
    @assert Matrix(A) == 2Matrix(B)
    C = zeros(2, 2)
    mul!(C, A, A; team)             # writes C in place on the team
    F = lu(A; team)
    @assert Matrix(F; team) ≈ Matrix(A)
end # releases the reservation, including when the body throws
```

`FaerTeam(width)` retains a reusable native pool/team. Width includes the caller
and `width-1` reserved workers; width one is sequential. Each multiplication
borrows a `Par::Team` only during the native call. Four scalar types, promotion,
views, transposes, coefficients and aliased destinations are supported. Copies
reserve independent teams. Close teams promptly; GC also owns their handles.

`team_mul`, `mul!(dest, team, A, B[, alpha, beta])`, `pfaffian(A; team=team)`
and supported factorization, solve, operator and reduction entry points with `team=` use this explicit team. Consult each operation’s page for sequential kernels and restrictions. Calls are serialized
by Colmat's owner lock, and faer may choose a narrower region for small work.
The API controls worker reservations; it does not promise a speedup for every
matrix size.

Dense `mul!(C, A, B[, alpha, beta]; team)` and the positional
`mul!(C, team, A, B[, alpha, beta])` run faer's gemm directly in `C` (native
owners and views, `faerview` and Julia strided destinations; at least one
operand must be native or `faerview`) without a temporary result; only mixed scalar types stage the promoted operands. faer's
`gemm_team` cost model decides whether a product is large enough to split, so
Colmat adds no threshold. Dense LU/QR/Cholesky/LDLT/Bunch–Kaufman factors,
retained or computed in place, accept `team` in `reconstruct`, `FaerMatrix(F)`
and `Matrix(F)`. Operators such as `A * B` and `F \ B`, and property access
such as `F.Q`, cannot take keywords and stay sequential; use `mul!`, `ldiv!` or
`team_mul` instead. Component copies (`F.L`, `F.U`, `F.R`, ...), `kron`,
permutations, statistics, elementwise operations and sparse symbolic analysis
are sequential because faer exposes no parallel form for them.

A team cannot adopt a threadteam pool owned by another library: threadteam
0.5.1 has no API to attach to an existing pool (its `Pool` is consumed by
`into_team`), and a `Team` from a separately built crate is not ABI-compatible.
Reuse one `FaerTeam` across operations instead. The owner lock is kept: one team
runs one execution at a time anyway (threadteam's own gate), and running different
teams concurrently would need per-owner locking across every native call.

Pool options are keywords of the constructor, and `team_diagnostics` reads the
reservation and wait counters of the pool:

```julia
using Colmat

FaerTeam(2; spin_polls=0, qos=:utility, pin=false, acquire_timeout=5,
         dispatch_cost=[2 => 2e-6, 4 => 4e-6]) do team
    A = FaerMatrix(rand(200, 200))
    team_mul(team, A, A)
    d = team_diagnostics(team)
    @assert d.width <= 2 && d.spin_polls == 0 && !d.pin && d.dispatch_cost > 0
end
```

- `acquire_timeout` (seconds, default 30): how long reserving the workers may
  wait for idle threads.
- `spin_polls`: spin budget of a team wait (reserved members waiting for work,
  joins and barriers) before yielding and sleeping; `0` sleeps after the yields.
- `qos`: macOS worker class (`:default`, `:utility`, `:user_initiated`,
  `:user_interactive`); the calling Julia thread is never changed.
- `pin` (default `true`): place participants on distinct CPUs of the caller's
  CPU set during team sections on Linux; pass `false` when the CPUs are shared
  with other work.
- `dispatch_cost`: cost of one team section that faer weighs against the
  estimated work when it decides whether to split (gemm, LU, Householder
  updates, ...): seconds for every width, or `width => seconds` pairs. Larger
  values split less often; `nothing` keeps threadteam's measured host default.
  On a 1000×1000 `Float64` matrix with width 4 (aarch64 macOS), the default
  gave `mul!` 8.6 ms and `lu` 11.3 ms, and `dispatch_cost=1.0` gave the
  sequential 34.7 ms and 19.6 ms.

`copy(team)` reserves a new pool with the same options. A team cannot borrow an
external (for example Julia's) thread pool: threadteam reserves its own worker
threads, and faer's symbolic sparse analyses take no `Par`, so they stay
sequential.

## Parallel-region thresholds

`bidiagonalize[!]` and `tridiagonalize[!]` accept `par_threshold=FaerParThreshold()` and forward the
same policy through `scratch_requirements`, `FaerScratch` and `reserve!`.
`FaerParThreshold(:derived)` keeps faer's default cost model.
`FaerParThreshold(n)` sets a fixed floor in the reduction's work units:
updated trailing-block elements for bidiagonalization, or `rem*rem÷2` for the
remaining triangle in tridiagonalization. Native Hermitian/Symmetric
`hessenberg[!]` forwards the same policy. For either family,
zero and one select the same policy for positive work. The integer must satisfy
`0 <= n < typemax(UInt)`. This fixed floor controls the reduction's trailing
updates; explicit U/V or Q applications retain their own parallel rules.

`FaerParThreshold(:never)` keeps both reduction and explicit U/V or Q formation
sequential. A supplied `FaerTeam` must still be open. These policies create no
team; reuse an explicit team when parallel work is desired. Query and execute
with the same policy, and reserve again when changing it. See the
[bidiagonal](reductions.md#Bidiagonal-parallel-policy) and
[tridiagonal](reductions.md#Tridiagonal-parallel-policy) examples.

## Reusable native scratch

```julia
using Colmat
A = FaerMatrix([0.0 2; -2 0])
requirements = scratch_requirements(pfaffian, A)
workspace = FaerScratch(pfaffian, A)
@assert scratch_capacity(workspace) == requirements
@assert pfaffian(A; scratch=workspace) == 2.0
FaerTeam(2) do team
    @assert pfaffian(A; scratch=workspace, team=team) == 2.0
end
reserve!(workspace, requirements.bytes; alignment=requirements.alignment)
close(workspace)
```

`FaerScratch` owns an aligned native MemBuffer. Capacity/alignment are explicit,
and insufficient workspace raises an error before kernel execution. Grow it
using `reserve!(workspace, pfaffian, A)` for a new operation size, or supply byte
capacity/alignment directly. Calls never grow the workspace implicitly; copy
and deepcopy allocate independent storage. Contents are scratch, not user data.
Pfaffian validates shape, finiteness and skew-symmetry and leaves A unchanged.
`team=` works with supplied scratch or with a temporary workspace released after
the call. Supported consumers include dense and sparse factorizations/solves, spectral
methods, operators, iterative solvers and reductions, as documented on their pages.
Julia/FFI bookkeeping may still allocate even
when the native workspace is reused.

## 操作別のリソース対応

必要量の照会・構築・実行では同じ操作、入力、数値型、`team`、許容誤差を使います。
入力や team を変えた場合は必要量を再照会し、必要なら `reserve!` で拡張してください。

| 操作 | 詳細 |
| :--- | :--- |
| 密分解・保存因子の求解・SVD、`lu!/qr!/cholesky!(A; scratch)` の直接 in-place 分解 | [密行列の分解と求解](dense.md) |
| 固有値・Schur・QZ・並べ替え | [固有値問題](spectral.md) |
| 疎 LU・Cholesky の構築・更新・求解、ordering | [疎直接法](sparse.md) |
| 疎 QR の構築・更新・最小ノルム求解 | [疎 QR](sparse-qr.md) |
| 作用素・前処理・CG・BiCGSTAB・LSMR | [線形作用素・反復法](operators.md) |
| Householder 適用・三重対角化・帯行列固有値 | [反射・回転・縮約](reductions.md) |
