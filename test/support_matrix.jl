# Run with `julia --project=test test/support_matrix.jl --output=PATH`.
# Included by runtests.jl without --output, this only verifies the probes.
module ColmatSupportMatrix
using Colmat, LinearAlgebra, Random, Test, Libdl

const TYPES = (Float32, Float64, ComplexF32, ComplexF64)
const INTERFACE = sizeof(LinearAlgebra.BlasInt) == 8 ? :ilp64 : :lp64
const ROOT = normpath(joinpath(@__DIR__, ".."))

function scalar_product(A, B)
    return [
        sum(
            (A[i, k] * B[k, j] for k in axes(A, 2));
            init=zero(promote_type(eltype(A), eltype(B))),
        ) for i in axes(A, 1), j in axes(B, 2)
    ]
end
scalar_norm(A) = sqrt(sum(abs2, A))
function residual_ok(A, B)
    return scalar_norm(A - B) <=
           300eps(real(eltype(A))) *
           max(one(real(eltype(A))), scalar_norm(A), scalar_norm(B))
end
prefix(::Type{Float32}) = "s"
prefix(::Type{Float64}) = "d"
prefix(::Type{ComplexF32}) = "c"
prefix(::Type{ComplexF64}) = "z"

function entry(T, operation)
    p = prefix(T)
    operation == :dot && return T <: Real ? "cblas_$(p)dot" : "cblas_$(p)dotc_sub"
    operation == :norm &&
        return T <: Real ? "$(p)nrm2_" : (T == ComplexF32 ? "scnrm2_" : "dznrm2_")
    operation == :hermitian && return "$(p)$(T <: Real ? "syevr" : "heevr")_"
    return "$(p)$(operation)_"
end

function backing(symbol)
    lib = BLAS.lbt_find_backing_library(symbol, INTERFACE)
    isnothing(lib) && return ("転送先を取得できず", "不明")
    native = realpath(lib.libname) == realpath(Colmat.backend_library())
    return (basename(lib.libname), native ? "Colmat" : "既存バックエンド")
end

function fixture(T)
    rng = MersenneTwister(391)
    A = randn(rng, T, 6, 6) + 8I
    B = randn(rng, T, 6, 4)
    x = randn(rng, T, 6)
    return A, B, x
end

function check_case(id, T)
    A, B, x = fixture(T)
    if id == :matmul
        return residual_ok(A * B, scalar_product(A, B))
    elseif id == :mul
        C = similar(B)
        mul!(C, A, B)
        return residual_ok(C, scalar_product(A, B))
    elseif id == :matvec
        return residual_ok(A * x, vec(scalar_product(A, reshape(x, :, 1))))
    elseif id == :dot
        return isapprox(dot(x, x), sum(conj(v) * v for v in x); rtol=300eps(real(T)))
    elseif id == :norm
        return isapprox(norm(x), scalar_norm(x); rtol=300eps(real(T)))
    elseif id == :axpy
        y = copy(x)
        BLAS.axpy!(T(2), x, y)
        return residual_ok(y, 3x)
    elseif id == :trsv
        U = UpperTriangular(A)
        y = U \ x
        return residual_ok(vec(scalar_product(Matrix(U), reshape(y, :, 1))), x)
    elseif id == :trsm
        U = UpperTriangular(A)
        Y = U \ B
        return residual_ok(scalar_product(Matrix(U), Y), B)
    elseif id == :lu
        F = lu(A)
        return residual_ok(scalar_product(Matrix(F.L), Matrix(F.U)), A[F.p, :])
    elseif id == :solve
        F = lu(A)
        y = F \ x
        return residual_ok(vec(scalar_product(A, reshape(y, :, 1))), x)
    elseif id == :qr
        F = qr(A)
        return residual_ok(scalar_product(Matrix(F.Q), Matrix(F.R)), A)
    elseif id == :pivoted_qr
        F = qr(A, ColumnNorm())
        return residual_ok(scalar_product(Matrix(F.Q), Matrix(F.R)), A[:, F.p])
    elseif id in (:svd, :svd_qr)
        F = id == :svd ? svd(B) : svd(B; alg=LinearAlgebra.QRIteration())
        scaled = F.U .* reshape(F.S, 1, :)
        return residual_ok(scalar_product(scaled, F.Vt), B)
    elseif id == :svdvals
        D = zeros(T, 5, 3)
        for (i, v) in enumerate((5, 3, 1))
            D[i, i] = T(v)
        end
        return residual_ok(svdvals(D), real(T)[5, 3, 1])
    elseif id == :eigen
        F = eigen(A)
        return residual_ok(
            scalar_product(A, F.vectors), F.vectors .* reshape(F.values, 1, :)
        )
    elseif id == :hermitian
        H = Hermitian(A + A')
        F = eigen(H)
        return residual_ok(
            scalar_product(Matrix(H), F.vectors), F.vectors .* reshape(F.values, 1, :)
        )
    elseif id == :cholesky
        H = Hermitian(scalar_product(A, A'))
        F = cholesky(H)
        return residual_ok(scalar_product(Matrix(F.L), Matrix(F.L)'), Matrix(H))
    end
    return error("Unknown probe $id")
end

const CASES = [
    (id=:matmul, api="A * B", op=:gemm, rust="行列積"),
    (id=:mul, api="mul!(C, A, B)", op=:gemm, rust="行列積"),
    (id=:matvec, api="A * x", op=:gemv, rust="行列積"),
    (id=:dot, api="dot(x, x)", op=:dot, rust="Rust 互換層"),
    (id=:norm, api="norm(x)", op=:norm, rust="Rust 互換層"),
    (id=:axpy, api="BLAS.axpy!", op=:axpy, rust="Rust 互換層"),
    (id=:trsv, api="UpperTriangular(A) \\ x", op=:trsv, rust="Rust 互換層"),
    (id=:trsm, api="UpperTriangular(A) \\ B", op=:trsm, rust="Rust 互換層"),
    (id=:lu, api="lu(A)", op=:getrf, rust="partial_piv_lu（未接続）"),
    (id=:solve, api="lu(A) \\ x", op=:getrs, rust="LU solve（未接続）"),
    (id=:qr, api="qr(A)", op=:geqrt, rust="qr（未接続）"),
    (id=:pivoted_qr, api="qr(A, ColumnNorm())", op=:geqp3, rust="col_piv_qr（未接続）"),
    (id=:svd, api="svd(A)", op=:gesdd, rust="svd（未接続）"),
    (
        id=:svd_qr,
        api="svd(A; alg=LinearAlgebra.QRIteration())",
        op=:gesvd,
        rust="svd（未接続）",
    ),
    (id=:svdvals, api="svdvals(A)", op=:gesdd, rust="SVD（未接続）"),
    (id=:eigen, api="eigen(A)", op=:geevx, rust="eigen（未接続）"),
    (
        id=:hermitian,
        api="eigen(Hermitian(A))",
        op=:hermitian,
        rust="self_adjoint_eigen（未接続）",
    ),
    (id=:cholesky, api="cholesky(Hermitian(A))", op=:potrf, rust="llt（未接続）"),
]

function probe()
    rows = []
    @testset "Julia API support matrix" begin
        for case in CASES
            results = String[]
            libraries = String[]
            routes = String[]
            symbols = String[]
            @testset "$(case.api)" begin
                for T in TYPES
                    @test check_case(case.id, T)
                    push!(results, "成功")
                    symbol = entry(T, case.op)
                    lib, route = backing(symbol)
                    push!(libraries, lib)
                    push!(routes, route)
                    push!(symbols, symbol)
                    @test route != "不明"
                end
            end
            push!(rows, (; case..., results, libraries, routes, symbols))
        end
    end
    return rows
end

function report(path, rows)
    revision = strip(read(`git -C $ROOT rev-parse HEAD`, String))
    rust_revision = strip(
        read(`git -C $(joinpath(ROOT,"deps","colmat-rs")) rev-parse HEAD`, String)
    )
    open(path, "w") do io
        println(io, "# Julia API の実測対応表\n")
        println(
            io,
            "通常配列の呼び出しが成功するかと、その入口を Colmat が提供するかを分けて記録します。",
        )
        println(
            io,
            "入口は Julia 1.13 のラッパー実装から選び、LBT の登録先を照合します。動的な呼び出し回数の計測やライブラリ内部を追跡したトレースではありません。\n",
        )
        println(io, "## 検証条件\n")
        println(io, "- Julia: `$(VERSION)` / `$(Sys.MACHINE)`。")
        println(io, "- Colmat ソース: `$(revision)`。")
        println(io, "- Rust colmat ソース: `$(rust_revision)`。")
        println(io, "- Julia の BLAS 整数 ABI: `$(INTERFACE)`。")
        println(io, "- Colmat 実行幅: `$(Colmat.get_num_threads())`。")
        println(io, "- 要素型は Float32 / Float64 / ComplexF32 / ComplexF64 の順。\n")
        println(io, "## 実行結果\n")
        println(
            io,
            "| Julia API | Float32 | Float64 | ComplexF32 | ComplexF64 | 主な入口の登録先 | Rust 側の実装・接続 |",
        )
        println(io, "|---|---|---|---|---|---|---|")
        for row in rows
            println(
                io,
                "| `$(row.api)` | $(join(row.results," | ")) | $(join(unique(row.routes)," / ")) | $(all(==("Colmat"), row.routes) ? replace(row.rust, "（未接続）" => "（接続済み）") : row.rust) |",
            )
        end
        println(io, "\n## 入口シンボルと実際のライブラリ\n")
        println(io, "各行は上表と同じ API を確認しています。シンボルは上記4型の順です。\n")
        println(io, "| Julia API | 確認した入口 | LBT の登録先 |")
        println(io, "|---|---|---|")
        for row in rows
            println(
                io,
                "| `$(row.api)` | $(join(("`$s`" for s in row.symbols),", ")) | $(join(("`$s`" for s in unique(row.libraries)),", ")) |",
            )
        end
        println(io, "\n## 判定の範囲\n")
        println(
            io,
            "- 「成功」は、このプローブの小さな有限値入力で参照計算または分解の残差を満たしたことを表します。",
        )
        println(
            io,
            "- 行列積・求解・分解再構成の残差は、BLAS を呼ばないスカラーの積和で確認します。",
        )
        println(
            io,
            "- 「既存バックエンド」での成功は、Rust colmat の該当分解が接続済みであるという意味ではありません。",
        )
        println(
            io,
            "- Rust 側の機能名は固定 revision の `faer/src/linalg/solvers.rs` で確認した対応候補です。未接続の Rust 関数をこのプローブが直接実行したとは主張しません。",
        )
        println(
            io,
            "- この表は全 BLAS / LAPACK の網羅検証ではありません。特異行列、非有限値、一般化固有値、他 OS の結果も含みません。",
        )
        println(
            io,
            "- LP64 の直接 ABI テストや負ストライド等の既存回帰テストとは別の、Julia 高水準 API の確認です。\n",
        )
        println(io, "## 再生成\n")
        println(
            io, "リポジトリのルートで、ビルド後の新しい Julia プロセスから実行します。\n"
        )
        println(
            io,
            "```sh\njulia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md\n```\n",
        )
        return println(
            io,
            "検証が失敗したときは表を上書きしません。呼び出しケースと検証方法は `test/support_matrix.jl` にあります。",
        )
    end
end

function main()
    rows = probe()
    for arg in ARGS
        startswith(arg, "--output=") || continue
        report(split(arg, "="; limit=2)[2], rows)
    end
end
main()
end # module
