using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_larft_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, direct, storev, n, k, v, ldv,
        tau, t, ldt, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                layout, direct, storev, n, k, v, ldv, tau, t, ldt)
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

@testset "LAPACKE LARFT layouts, DIRECT and STOREV" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), (n, k) in ((4, 2), (5, 3)), direct in ('F', 'B'),
        storev in ('C', 'R'), high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "larft" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(900 + n)
        nrv, ncv = storev == 'C' ? (n, k) : (k, n)
        v = randn(rng, T, nrv, ncv)
        tau = randn(rng, T, k)
        t = zeros(T, k, k)
        tcol = vec(copy(t)); vcol = vec(copy(v))
        @test lapacke_larft_call(
            T, J, pointer, 102, direct, storev, n, k, vcol, nrv, tau, tcol, max(1, k),
            name, route, high,
        ) == 0
        ld = ncv + 2
        trow = vec(copy(t)); vrow = row_major(v, ld)
        @test lapacke_larft_call(
            T, J, pointer, 101, direct, storev, n, k, vrow, ld, tau, trow, k,
            name, route, high,
        ) == 0
        tol = 500 * eps(real(T))
        @test isapprox(reshape(trow, k, k), transpose(reshape(tcol, k, k)); rtol=tol, atol=tol)
    end
end

@testset "LAPACKE LARFT diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "larft"
        pointer = Libdl.dlsym(native, name * suffix)
        n, k = 3, 2
        v = zeros(T, n, k); tau = zeros(T, k); t = zeros(T, k, k)
        @test lapacke_larft_call(
            T, J, pointer, 301, 'F', 'C', n, k, vec(v), n, tau, vec(t), k,
            name, :dll, true,
        ) == -1
        @test lapacke_larft_call(
            T, J, pointer, 101, 'F', 'C', n, k, vec(v), n, tau, vec(t), k - 1,
            name, :dll, true,
        ) == -10
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
