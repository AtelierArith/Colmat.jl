using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_ggev_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, jl, jr, n,
        a, lda, b, ldb, ar, ai, be, vl, ldvl, vr, ldvr, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                if high
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                        layout, jl, jr, n, a, lda, b, ldb, ar, be, vl, ldvl, vr, ldvr)
                else
                    work = fill($T(NaN), max(1, 2n)); rwork = fill($R(NaN), max(1, 8n))
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T},
                         $J, Ptr{$R}),
                        layout, jl, jr, n, a, lda, b, ldb, ar, be, vl, ldvl, vr,
                        ldvr, work, length(work), rwork)
                end
            elseif high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, jl, jr, n, a, lda, b, ldb, ar, ai, be, vl, ldvl, vr, ldvr)
            else
                work = fill($T(NaN), max(1, 8n))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J),
                    layout, jl, jr, n, a, lda, b, ldb, ar, ai, be, vl, ldvl, vr,
                    ldvr, work, length(work))
            end
        end
    end
end
function row_major(matrix, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), m * ld)
    for j in 1:n, i in 1:m
        out[(i - 1) * ld + j] = matrix[i, j]
    end
    return out
end
function from_row_major(vector, m, n, ld)
    out = fill(eltype(vector)(NaN), m, n)
    for j in 1:n, i in 1:m
        out[i, j] = vector[(i - 1) * ld + j]
    end
    return out
end

@testset "LAPACKE GGEV/GGEV3 layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), family in ("ggev", "ggev3"),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(1400 + n)
        a = randn(rng, T, n, n); b = randn(rng, T, n, n)
        acol = vec(copy(a)); bcol = vec(copy(b))
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        vlref = zeros(T, n, n); vrref = zeros(T, n, n)
        @test lapacke_ggev_call(
            T, R, J, pointer, 102, 'V', 'V', n, acol, n, bcol, n, ar, ai, be,
            vec(vlref), n, vec(vrref), n, name, route, high,
        ) == 0
        ld = n + 2
        arow = row_major(a, ld); brow = row_major(b, ld)
        row = randn(rng, T, n * ld); vrrow = randn(rng, T, n * ld)
        ar2 = zeros(T, n); ai2 = zeros(T, n); be2 = zeros(T, n)
        info = lapacke_ggev_call(
            T, R, J, pointer, 101, 'V', 'V', n, arow, ld, brow, ld, ar2, ai2,
            be2, row, ld, vrrow, ld, name, route, high,
        )
        @test info == 0
        tol = 500 * eps(R)
        @test isapprox(ar2, ar; rtol=tol, atol=tol)
        @test isapprox(ai2, ai; rtol=tol, atol=tol)
        @test isapprox(be2, be; rtol=tol, atol=tol)
        @test isapprox(from_row_major(row, n, n, ld), vlref; rtol=tol, atol=tol)
        @test isapprox(from_row_major(vrrow, n, n, ld), vrref; rtol=tol, atol=tol)
    end
end

@testset "LAPACKE GGEV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "ggev"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3; a = Matrix{T}(I, n, n); b = Matrix{T}(I, n, n)
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        @test lapacke_ggev_call(
            T, R, J, pointer, 301, 'N', 'N', n, vec(a), n, vec(b), n, ar, ai,
            be, zeros(T, n * n), n, zeros(T, n * n), n, name, :dll, true,
        ) == -1
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
