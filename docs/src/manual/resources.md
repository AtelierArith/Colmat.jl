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
matrix size. Inputs needing conversion and destination writes use temporaries.

## Parallel-region thresholds

`bidiagonalize[!]` accepts `par_threshold=FaerParThreshold()` and forwards the
same policy through `scratch_requirements`, `FaerScratch` and `reserve!`.
`FaerParThreshold(:derived)` keeps faer's default cost model.
`FaerParThreshold(n)` sets a fixed floor in updated trailing-block elements;
zero and one select the same policy for positive work. The integer must satisfy
`0 <= n < typemax(UInt)`. This fixed floor controls the reduction's trailing
updates; explicit U/V applications retain their own parallel rules.

`FaerParThreshold(:never)` keeps both reduction and explicit U/V formation
sequential. A supplied `FaerTeam` must still be open. These policies create no
team; reuse an explicit team when parallel work is desired. Query and execute
with the same policy, and reserve again when changing it. See the
[bidiagonal example](reductions.md#Bidiagonal-parallel-policy).

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
| 密分解・保存因子の求解・SVD | [密行列の分解と求解](dense.md) |
| 固有値・Schur・QZ・並べ替え | [固有値問題](spectral.md) |
| 疎 LU・Cholesky の構築・更新・求解、ordering | [疎直接法](sparse.md) |
| 疎 QR の構築・更新・最小ノルム求解 | [疎 QR](sparse-qr.md) |
| 作用素・前処理・CG・BiCGSTAB・LSMR | [線形作用素・反復法](operators.md) |
| Householder 適用・三重対角化・帯行列固有値 | [反射・回転・縮約](reductions.md) |
