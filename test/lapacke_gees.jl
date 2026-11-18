using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gees_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, jobvs, sort, n,
        a, lda, sdim, wr, wi, vs, ldvs, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                if high
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                         Ptr{$J}, Ptr{$T}, Ptr{$T}, $J),
                        layout, jobvs, sort, C_NULL, n, a, lda, sdim, wr, vs, ldvs)
                else
                    work = fill($T(NaN), max(1, 2n))
                    rwork = fill($R(NaN), max(1, n))
                    bwork = zeros($J, max(1, n))
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                         Ptr{$J}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R},
                         Ptr{$J}),
                        layout, jobvs, sort, C_NULL, n, a, lda, sdim, wr, vs,
                        ldvs, work, length(work), rwork, bwork)
                end
            elseif high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                     Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J),
                    layout, jobvs, sort, C_NULL, n, a, lda, sdim, wr, wi, vs, ldvs)
            else
                work = fill($T(NaN), max(1, 4n))
                bwork = zeros($J, max(1, n))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                     Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$J}),
                    layout, jobvs, sort, C_NULL, n, a, lda, sdim, wr, wi, vs,
                    ldvs, work, length(work), bwork)
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

@testset "LAPACKE GEES layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), jobvs in ('N', 'V'),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "gees" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(1200 + n)
        a = randn(rng, T, n, n)
        acol = vec(copy(a))
        w = zeros(T, n); wi = zeros(R, n)
        mref = Ref{J}(-1)
        ref = lapacke_gees_call(
            T, R, J, pointer, 102, jobvs, 'N', n, acol, n, mref,
            w, wi, zeros(T, n * n), n, name, route, high,
        )
        @test ref == 0
        ld = n + 2
        arow = row_major(a, ld)
        mrow = Ref{J}(-1)
        vsrow = fill(T(NaN), n * ld)
        wrow = zeros(T, n); wirow = zeros(R, n)
        info = lapacke_gees_call(
            T, R, J, pointer, 101, jobvs, 'N', n, arow, ld, mrow,
            wrow, wirow, vsrow, ld, name, route, high,
        )
        @test info == 0
        @test mrow[] == mref[]
        tol = 500 * eps(R)
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        if jobvs == 'V'
            vsref = zeros(T, n, n)
            acol2 = vec(copy(a)); m2 = Ref{J}(-1)
            @test lapacke_gees_call(
                T, R, J, pointer, 102, jobvs, 'N', n, acol2, n, m2,
                w, wi, vec(vsref), n, name, route, high,
            ) == 0
            @test isapprox(from_row_major(vsrow, n, n, ld), vsref; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE GEES diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "gees"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3; a = Matrix{T}(I, n, n); w = zeros(T, n); wi = zeros(R, n)
        m = Ref{J}(-1)
        @test lapacke_gees_call(
            T, R, J, pointer, 301, 'V', 'N', n, vec(a), n, m, w, wi,
            zeros(T, n * n), n, name, :dll, true,
        ) == -1
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
