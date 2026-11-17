using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_orgbr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, vect, m, n, k, a, lda, tau,
        name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(pointer, $J,
                    (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}),
                    layout, vect, m, n, k, a, lda, tau)
            else
                work = fill($T(NaN), max(1, 4max(m, n)))
                ccall(pointer, $J,
                    (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout, vect, m, n, k, a, lda, tau, work, length(work))
            end
        end
    end
    @eval function lapacke_orghr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, n, ilo, ihi, a, lda, tau,
        name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(pointer, $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}),
                    layout, n, ilo, ihi, a, lda, tau)
            else
                work = fill($T(NaN), max(1, 4n))
                ccall(pointer, $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout, n, ilo, ihi, a, lda, tau, work, length(work))
            end
        end
    end
    @eval function lapacke_ormbr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, vect, side, trans, m, n, k,
        a, lda, tau, c, ldc, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J),
                    layout, vect, side, trans, m, n, k, a, lda, tau, c, ldc)
            else
                work = fill($T(NaN), max(1, 4max(m, n)))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, vect, side, trans, m, n, k, a, lda, tau, c, ldc,
                    work, length(work))
            end
        end
    end
    @eval function lapacke_ormhr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, trans, m, n, ilo, ihi,
        a, lda, tau, c, ldc, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J),
                    layout, side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc)
            else
                work = fill($T(NaN), max(1, 4max(m, n)))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc,
                    work, length(work))
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

@testset "LAPACKE ORGBR/ORMBR and ORGHR/ORMHR layouts" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        complex = T <: Complex
        names = (
            gen=(complex ? "ungbr" : "orgbr"),
            app=(complex ? "unmbr" : "ormbr"),
            hgen=(complex ? "unghr" : "orghr"),
            happ=(complex ? "unmhr" : "ormhr"),
        )
        pointers = Dict{Symbol,Ptr{Nothing}}()
        for (key, family) in pairs(names)
            name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
            @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
            pointers[key] = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        end
        rng = MersenneTwister(300 + (J === Int32 ? 1 : 2))
        # ORGBR
        m, n, k = 5, 5, 3
        a = randn(rng, T, m, n)
        tau = randn(rng, T, min(m, k))
        for vect in ('Q', 'P')
            acol = vec(copy(a))
            ld = n + 2
            arow = row_major(a, ld)
            name = "LAPACKE_" * prefix * names.gen * (high ? "" : "_work")
            @test lapacke_orgbr_call(
                T, J, pointers[:gen], 102, vect, m, n, k, acol, m, tau, name, route, high,
            ) == 0
            @test lapacke_orgbr_call(
                T, J, pointers[:gen], 101, vect, m, n, k, arow, ld, tau, name, route, high,
            ) == 0
            @test isapprox(from_row_major(arow, m, n, ld), reshape(acol, m, n); rtol=100eps(real(T)), atol=100eps(real(T)))
        end
        # ORGHR
        nn = 4
        ah = randn(rng, T, nn, nn)
        tau_h = randn(rng, T, nn - 1)
        acol = vec(copy(ah)); ld = nn + 2; arow = row_major(ah, ld)
        name = "LAPACKE_" * prefix * names.hgen * (high ? "" : "_work")
        @test lapacke_orghr_call(T, J, pointers[:hgen], 102, nn, 1, nn, acol, nn, tau_h, name, route, high) == 0
        @test lapacke_orghr_call(T, J, pointers[:hgen], 101, nn, 1, nn, arow, ld, tau_h, name, route, high) == 0
        @test isapprox(from_row_major(arow, nn, nn, ld), reshape(acol, nn, nn); rtol=100eps(real(T)), atol=100eps(real(T)))
        # ORMBR and ORMHR
        for side in ('L', 'R'), trans in (complex ? ('N', 'C') : ('N', 'T'))
            nq = side == 'L' ? nn : nn
            for vect in ('Q', 'P')
                ar, ac = vect == 'Q' ? (nq, min(nq, 3)) : (min(nq, 3), nq)
                ar = vect == 'Q' ? ar : ar
                ac = vect == 'Q' ? ac : ac
                ab = randn(rng, T, ar, ac)
                taub = randn(rng, T, min(nq, 3))
                c = randn(rng, T, nn, nn)
                acol = vec(copy(ab)); ccol = vec(copy(c))
                ld_a = ac + 2; ld_c = nn + 2
                arow = row_major(ab, ld_a); crow = row_major(c, ld_c)
                name = "LAPACKE_" * prefix * names.app * (high ? "" : "_work")
                @test lapacke_ormbr_call(
                    T, J, pointers[:app], 102, vect, side, trans, nn, nn, 3,
                    acol, ar, taub, ccol, nn, name, route, high,
                ) == 0
                @test lapacke_ormbr_call(
                    T, J, pointers[:app], 101, vect, side, trans, nn, nn, 3,
                    arow, ld_a, taub, crow, ld_c, name, route, high,
                ) == 0
                @test isapprox(from_row_major(crow, nn, nn, ld_c), reshape(ccol, nn, nn); rtol=200eps(real(T)), atol=200eps(real(T)))
            end
            ah = randn(rng, T, nn, nn)
            tau_h = randn(rng, T, nn - 1)
            c = randn(rng, T, nn, nn)
            acol = vec(copy(ah)); ccol = vec(copy(c))
            ld_a = nn + 2; ld_c = nn + 2
            arow = row_major(ah, ld_a); crow = row_major(c, ld_c)
            name = "LAPACKE_" * prefix * names.happ * (high ? "" : "_work")
            @test lapacke_ormhr_call(
                T, J, pointers[:happ], 102, side, trans, nn, nn, 1, nn,
                acol, nn, tau_h, ccol, nn, name, route, high,
            ) == 0
            @test lapacke_ormhr_call(
                T, J, pointers[:happ], 101, side, trans, nn, nn, 1, nn,
                arow, ld_a, tau_h, crow, ld_c, name, route, high,
            ) == 0
            @test isapprox(from_row_major(crow, nn, nn, ld_c), reshape(ccol, nn, nn); rtol=200eps(real(T)), atol=200eps(real(T)))
        end
    end
end

@testset "LAPACKE reduction vector diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        n = 4
        a = randn(MersenneTwister(7), T, n, n)
        tau = zeros(T, n - 1)
        c = Matrix{T}(I, n, n)
        gen = "LAPACKE_" * prefix * (T <: Complex ? "ungbr" : "orgbr")
        hgen = "LAPACKE_" * prefix * (T <: Complex ? "unghr" : "orghr")
        app = "LAPACKE_" * prefix * (T <: Complex ? "unmbr" : "ormbr")
        happ = "LAPACKE_" * prefix * (T <: Complex ? "unmhr" : "ormhr")
        @test lapacke_orgbr_call(
            T, J, Libdl.dlsym(native, gen * suffix), 301, 'Q', n, n, 2,
            vec(a), n, tau, gen, :dll, true,
        ) == -1
        @test lapacke_orgbr_call(
            T, J, Libdl.dlsym(native, gen * suffix), 102, '?', n, n, 2,
            vec(a), n, tau, gen, :dll, true,
        ) == -2
        @test lapacke_orgbr_call(
            T, J, Libdl.dlsym(native, gen * suffix), 101, 'Q', n, n, 2,
            vec(a), n - 1, tau, gen, :dll, true,
        ) == -7
        @test lapacke_orghr_call(
            T, J, Libdl.dlsym(native, hgen * suffix), 102, -1, 1, n,
            vec(a), n, tau, hgen, :dll, true,
        ) == -2
        @test lapacke_orghr_call(
            T, J, Libdl.dlsym(native, hgen * suffix), 101, n, 1, n,
            vec(a), n - 1, tau, hgen, :dll, true,
        ) == -6
        @test lapacke_ormbr_call(
            T, J, Libdl.dlsym(native, app * suffix), 102, '?', 'L', 'N', n, n, 2,
            vec(a), n, tau, vec(c), n, app, :dll, true,
        ) == -2
        @test lapacke_ormbr_call(
            T, J, Libdl.dlsym(native, app * suffix), 102, 'Q', '?', 'N', n, n, 2,
            vec(a), n, tau, vec(c), n, app, :dll, true,
        ) == -3
        @test lapacke_ormhr_call(
            T, J, Libdl.dlsym(native, happ * suffix), 101, 'L', 'N', n, n, 1, n,
            vec(a), n, tau, vec(c), n - 1, happ, :dll, true,
        ) == -12
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
