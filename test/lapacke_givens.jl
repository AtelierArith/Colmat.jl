using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64))
    @eval function lapacke_lartgp_call(
        ::Type{$T}, ::Type{$J}, pointer, f::$T, g::$T, cs, sn, r, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($T, $T, Ptr{$T}, Ptr{$T}, Ptr{$T}), f, g, cs, sn, r)
        end
    end
    @eval function lapacke_lartgs_call(
        ::Type{$T}, ::Type{$J}, pointer, x::$T, y::$T, sigma::$T, cs, sn,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($T, $T, $T, Ptr{$T}, Ptr{$T}), x, y, sigma, cs, sn)
        end
    end
end

@testset "LAPACKE LARTGP/LARTGS high/work and ABIs" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        gp = "LAPACKE_" * prefix * "lartgp" * (high ? "" : "_work")
        gs = "LAPACKE_" * prefix * "lartgs" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(gp, interface) == Libdl.dlsym(native, gp * suffix)
        @test BLAS.lbt_get_forward(gs, interface) == Libdl.dlsym(native, gs * suffix)
        pgp = Libdl.dlsym(route === :dll ? native : trampoline, gp * suffix)
        pgs = Libdl.dlsym(route === :dll ? native : trampoline, gs * suffix)
        rng = MersenneTwister(1100)
        for (f, g) in ((T(0), T(0)), (T(0), T(1)), (T(2), T(-3)), (T(-4), T(0)))
            cs = Ref(T(NaN)); sn = Ref(T(NaN)); r = Ref(T(NaN))
            @test lapacke_lartgp_call(T, J, pgp, f, g, cs, sn, r, gp, route) == 0
            @test isapprox(cs[]^2 + sn[]^2, one(T); rtol=50eps(T), atol=50eps(T))
            @test isapprox(cs[] * f + sn[] * g, r[]; rtol=100eps(T), atol=100eps(T))
            # Work entry gives the same result.
            cs2 = Ref(T(NaN)); sn2 = Ref(T(NaN)); r2 = Ref(T(NaN))
            work = "LAPACKE_" * prefix * "lartgp_work"
            pwork = Libdl.dlsym(native, work * suffix)
            @test lapacke_lartgp_call(T, J, pwork, f, g, cs2, sn2, r2, work, :dll) == 0
            @test isequal((cs2[], sn2[], r2[]), (cs[], sn[], r[]))
        end
        for (x, y, sigma) in ((T(0), T(0), T(0)), (T(1), T(2), T(3)), (T(-2), T(5), T(1)))
            cs = Ref(T(NaN)); sn = Ref(T(NaN))
            @test lapacke_lartgs_call(T, J, pgs, x, y, sigma, cs, sn, gs, route) == 0
            @test isapprox(cs[]^2 + sn[]^2, one(T); rtol=50eps(T), atol=50eps(T))
            @test isfinite(cs[]) && isfinite(sn[])
        end
    end
end

@testset "LAPACKE LARTGP/LARTGS diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"),), J in (Int32, Int64), high in (false, true)
        suffix = J === Int64 ? "64_" : ""
        gp = "LAPACKE_" * prefix * "lartgp" * (high ? "" : "_work")
        gs = "LAPACKE_" * prefix * "lartgs" * (high ? "" : "_work")
        pgp = Libdl.dlsym(native, gp * suffix)
        pgs = Libdl.dlsym(native, gs * suffix)
        if high
            cs = Ref(T(0)); sn = Ref(T(0)); r = Ref(T(0))
            @test lapacke_lartgp_call(T, J, pgp, T(NaN), T(1), cs, sn, r, gp, :dll) == -1
            @test lapacke_lartgp_call(T, J, pgp, T(1), T(NaN), cs, sn, r, gp, :dll) == -2
            cs2 = Ref(T(0)); sn2 = Ref(T(0))
            @test lapacke_lartgs_call(T, J, pgs, T(1), T(2), T(NaN), cs2, sn2, gs, :dll) == -3
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
