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
    isnothing(lib) && return ("forwarding target unavailable", "unknown")
    native = realpath(lib.libname) == realpath(Colmat.backend_library())
    return (basename(lib.libname), native ? "Colmat" : "existing backend")
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
    (id=:matmul, api="A * B", op=:gemm, rust="matrix product"),
    (id=:mul, api="mul!(C, A, B)", op=:gemm, rust="matrix product"),
    (id=:matvec, api="A * x", op=:gemv, rust="matrix product"),
    (id=:dot, api="dot(x, x)", op=:dot, rust="Rust compatibility layer"),
    (id=:norm, api="norm(x)", op=:norm, rust="Rust compatibility layer"),
    (id=:axpy, api="BLAS.axpy!", op=:axpy, rust="Rust compatibility layer"),
    (id=:trsv, api="UpperTriangular(A) \\ x", op=:trsv, rust="Rust compatibility layer"),
    (id=:trsm, api="UpperTriangular(A) \\ B", op=:trsm, rust="Rust compatibility layer"),
    (id=:lu, api="lu(A)", op=:getrf, rust="partial_piv_lu (not connected)"),
    (id=:solve, api="lu(A) \\ x", op=:getrs, rust="LU solve (not connected)"),
    (id=:qr, api="qr(A)", op=:geqrt, rust="qr (not connected)"),
    (id=:pivoted_qr, api="qr(A, ColumnNorm())", op=:geqp3, rust="col_piv_qr (not connected)"),
    (id=:svd, api="svd(A)", op=:gesdd, rust="svd (not connected)"),
    (
        id=:svd_qr,
        api="svd(A; alg=LinearAlgebra.QRIteration())",
        op=:gesvd,
        rust="svd (not connected)",
    ),
    (id=:svdvals, api="svdvals(A)", op=:gesdd, rust="SVD (not connected)"),
    (id=:eigen, api="eigen(A)", op=:geevx, rust="eigen (not connected)"),
    (
        id=:hermitian,
        api="eigen(Hermitian(A))",
        op=:hermitian,
        rust="self_adjoint_eigen (not connected)",
    ),
    (id=:cholesky, api="cholesky(Hermitian(A))", op=:potrf, rust="llt (not connected)"),
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
                    push!(results, "success")
                    symbol = entry(T, case.op)
                    lib, route = backing(symbol)
                    push!(libraries, lib)
                    push!(routes, route)
                    push!(symbols, symbol)
                    @test route != "unknown"
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
        println(io, "# Measured Julia API mapping\n")
        println(
            io,
            "Records, separately, whether calls on ordinary arrays succeed and whether Colmat provides their entry points.",
        )
        println(
            io,
            "Entry points are chosen from the Julia 1.13 wrapper implementations and cross-checked against the LBT registration targets. This is not a dynamic call-count measurement or a trace of library internals.\n",
        )
        println(io, "## Verification conditions\n")
        println(io, "- Julia: `$(VERSION)` / `$(Sys.MACHINE)`.")
        println(io, "- Colmat source: `$(revision)`.")
        println(io, "- Rust colmat source: `$(rust_revision)`.")
        println(io, "- Julia BLAS integer ABI: `$(INTERFACE)`.")
        println(io, "- Colmat execution width: `$(Colmat.get_num_threads())`.")
        println(io, "- Element types in order: Float32 / Float64 / ComplexF32 / ComplexF64.\n")
        println(io, "## Results\n")
        println(
            io,
            "| Julia API | Float32 | Float64 | ComplexF32 | ComplexF64 | Entry-point registration | Rust-side implementation / connection |",
        )
        println(io, "|---|---|---|---|---|---|---|")
        for row in rows
            println(
                io,
                "| `$(row.api)` | $(join(row.results," | ")) | $(join(unique(row.routes)," / ")) | $(all(==("Colmat"), row.routes) ? replace(row.rust, " (not connected)" => " (connected)") : row.rust) |",
            )
        end
        println(io, "\n## Entry symbols and actual libraries\n")
        println(io, "Each row verifies the same API as the table above. Symbols follow the four types in order.\n")
        println(io, "| Julia API | Verified entry points | LBT registration |")
        println(io, "|---|---|---|")
        for row in rows
            println(
                io,
                "| `$(row.api)` | $(join(("`$s`" for s in row.symbols),", ")) | $(join(("`$s`" for s in unique(row.libraries)),", ")) |",
            )
        end
        println(io, "\n## What the verdicts mean\n")
        println(
            io,
            "- \"Success\" means the probe met the reference computation or factorization residual on its small finite inputs.",
        )
        println(
            io,
            "- Residuals of matrix products, solves, and factorization reconstruction are checked with scalar multiply-adds that do not call BLAS.",
        )
        println(
            io,
            "- A success on the \"existing backend\" does not mean the corresponding Rust colmat factorization is connected.",
        )
        println(
            io,
            "- The Rust-side feature names are correspondence candidates confirmed in `faer/src/linalg/solvers.rs` at the pinned revision. This probe does not claim to have executed unconnected Rust functions directly.",
        )
        println(
            io,
            "- This table is not an exhaustive BLAS / LAPACK verification. It does not cover singular matrices, non-finite values, generalized eigenvalues, or other operating systems.",
        )
        println(
            io,
            "- It verifies high-level Julia APIs, separate from the existing regression tests such as the direct LP64 ABI tests and negative strides.\n",
        )
        println(io, "## Regeneration\n")
        println(
            io, "Run from the repository root in a fresh Julia process after building.\n"
        )
        println(
            io,
            "```sh\njulia --project=test test/support_matrix.jl --output=docs/src/reference/support-matrix.md\n```\n",
        )
        return println(
            io,
            "The table is not overwritten when verification fails. The call cases and the verification method are in `test/support_matrix.jl`.",
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
