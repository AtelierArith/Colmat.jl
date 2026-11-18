using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_larfb_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, trans, direct, storev,
        m, n, k, v, ldv, t, ldt, c, ldc, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T},
                     $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, side, trans, direct, storev, m, n, k, v, ldv, t,
                    ldt, c, ldc)
            else
                work = fill($T(NaN), max(1, k))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T},
                     $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, side, trans, direct, storev, m, n, k, v, ldv, t,
                    ldt, c, ldc, work, k)
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

@testset "LAPACKE LARFB layouts and options" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), (m, n, k) in ((4, 3, 2), (5, 4, 3)),
        side in ('L', 'R'), trans in (T <: Complex ? ('N', 'C') : ('N', 'T')),
        direct in ('F', 'B'), storev in ('C', 'R'), high in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "larfb" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(1000 + m + n)
        left = side == 'L'
        col = storev == 'C'
        nrv, ncv = (col && left) ? (m, k) : (col ? (n, k) : (left ? (k, m) : (k, n)))
        v = randn(rng, T, nrv, ncv)
        t = randn(rng, T, k, k)
        c = randn(rng, T, m, n)
        ldv = ncv + 2
        ldt = k + 2
        ldc = n + 2
        vcol = vec(copy(v)); ccol = vec(copy(c))
        @test lapacke_larfb_call(
            T, J, pointer, 102, side, trans, direct, storev, m, n, k, vcol, nrv,
            vec(t), k, ccol, m, name, route, high,
        ) == 0
        vrow = row_major(v, ldv); trow = row_major(t, ldt); crow = row_major(c, ldc)
        @test lapacke_larfb_call(
            T, J, pointer, 101, side, trans, direct, storev, m, n, k, vrow, ldv,
            trow, ldt, crow, ldc, name, route, high,
        ) == 0
        tol = 500 * eps(real(T))
        @test isapprox(from_row_major(crow, m, n, ldc), reshape(ccol, m, n); rtol=tol, atol=tol)
    end
end

@testset "LAPACKE LARFB diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "larfb"
        pointer = Libdl.dlsym(native, name * suffix)
        m, n, k = 3, 3, 2
        v = zeros(T, m * m); t = zeros(T, k * k); c = zeros(T, m * m)
        @test lapacke_larfb_call(
            T, J, pointer, 301, 'L', 'N', 'F', 'C', m, n, k, v, m, t, k, c, m,
            name, :dll, true,
        ) == -1
        @test lapacke_larfb_call(
            T, J, pointer, 101, 'L', 'N', 'F', 'C', m, n, k, v, m, t, k, c, n - 1,
            name, :dll, true,
        ) == -14
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
