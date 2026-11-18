using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_pttrf_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, n, d::Vector{$R}, e::Vector{$T},
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($J, Ptr{$R}, Ptr{$T}), n, d, e)
        end
    end
    @eval function lapacke_pttrs_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, uplo, n, nrhs,
        d::Vector{$R}, e::Vector{$T}, b::Vector{$T}, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                ccall(pointer, $J,
                    (Cint, Cchar, $J, $J, Ptr{$R}, Ptr{$T}, Ptr{$T}, $J),
                    layout, uplo, n, nrhs, d, e, b, ldb)
            else
                ccall(pointer, $J,
                    (Cint, $J, $J, Ptr{$R}, Ptr{$T}, Ptr{$T}, $J),
                    layout, n, nrhs, d, e, b, ldb)
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

@testset "LAPACKE PTTRF/PTTRS layouts, factorizations and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), nrhs in (1, 3), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "pttrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "pttrs" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(fname, interface) == Libdl.dlsym(native, fname * suffix)
        @test BLAS.lbt_get_forward(sname, interface) == Libdl.dlsym(native, sname * suffix)
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        rng = MersenneTwister(3400 + n + nrhs)
        d0 = R[2 + i for i in 1:n]
        e0 = T <: Complex ? T[complex(0.25i, -0.1i) for i in 1:max(0, n - 1)] :
             T[0.25i for i in 1:max(0, n - 1)]
        d, e = copy(d0), copy(e0)
        @test lapacke_pttrf_call(T, R, J, pfac, J(n), d, e, fname, route) == 0
        # Factor buffers from the work entry must match.
        dwork, ework = copy(d0), copy(e0)
        workname = "LAPACKE_" * prefix * "pttrf_work"
        @test lapacke_pttrf_call(
            T, R, J, Libdl.dlsym(native, workname * suffix), J(n), dwork, ework,
            workname, :dll,
        ) == 0
        @test isequal(d, dwork)
        @test isequal(e, ework)
        # Solve against an explicit tridiagonal factor and compare layouts.
        rngb = MersenneTwister(3500 + n * nrhs)
        b0 = randn(rngb, T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_pttrs_call(
            T, R, J, psol, 102, 'U', J(n), J(nrhs), d, e, bcol, max(1, n),
            sname, route,
        ) == 0
        ld = nrhs + 2
        brow = row_major(b0, ld)
        @test lapacke_pttrs_call(
            T, R, J, psol, 101, 'U', J(n), J(nrhs), d, e, brow, ld,
            sname, route,
        ) == 0
        tol = 2000 * eps(R)
        @test isapprox(
            from_row_major(brow, n, nrhs, ld), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        # Residual against the original tridiagonal matrix.
        A = Tridiagonal(
            T <: Complex ? conj.(e0) : e0, d0,
            T <: Complex ? e0 : e0,
        )
        x = reshape(bcol, n, nrhs)
        @test isapprox(A * x, b0; rtol=500 * max(1, n) * eps(R) * norm(b0), atol=500 * eps(R) * norm(b0))
    end
end

@testset "LAPACKE PTTRF/PTTRS diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "pttrf"
        sname = "LAPACKE_" * prefix * "pttrs"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        n, nrhs = 3, 2
        d = zeros(R, n); e = zeros(T, n - 1)
        b = zeros(T, n * nrhs)
        badd = copy(b); badd[1] = T(NaN)
        @test lapacke_pttrf_call(
            T, R, J, pfac, J(n), R[NaN, 1, 1], e, fname, :dll,
        ) == -2
        @test lapacke_pttrf_call(
            T, R, J, pfac, J(n), d, T[T(NaN), T(NaN)], fname, :dll,
        ) == -3
        @test lapacke_pttrs_call(
            T, R, J, psol, 301, 'U', J(n), J(nrhs), d, e, b, max(1, n),
            sname, :dll,
        ) == -1
        @test lapacke_pttrs_call(
            T, R, J, psol, 101, 'U', J(n), J(nrhs), d, e, b, nrhs - 1,
            sname, :dll,
        ) == (T <: Complex ? -8 : -7)
        @test lapacke_pttrs_call(
            T, R, J, psol, 102, 'U', J(n), J(nrhs), d, e, badd, n,
            sname, :dll,
        ) == (T <: Complex ? -7 : -6)
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
