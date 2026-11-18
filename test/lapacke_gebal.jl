using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gebal_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, job, n, a, lda,
        ilo, ihi, scale::Vector{$R}, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, Ptr{$R}),
                layout, job, n, a, lda, ilo, ihi, scale)
        end
    end
    @eval function lapacke_gebak_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, job, side, n, ilo,
        ihi, scale::Vector{$R}, m, v, ldv, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$R}, $J, Ptr{$T}, $J),
                layout, job, side, n, ilo, ihi, scale, m, v, ldv)
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

@testset "LAPACKE GEBAL/GEBAK layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 5), job in ('N', 'P', 'S', 'B'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        bname = "LAPACKE_" * prefix * "gebal" * (save ? "_work" : "")
        kname = "LAPACKE_" * prefix * "gebak" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(bname, interface) == Libdl.dlsym(native, bname * suffix)
        @test BLAS.lbt_get_forward(kname, interface) == Libdl.dlsym(native, kname * suffix)
        pbal = Libdl.dlsym(route === :dll ? native : trampoline, bname * suffix)
        pbak = Libdl.dlsym(route === :dll ? native : trampoline, kname * suffix)
        rng = MersenneTwister(5300 + n)
        # A matrix with an isolated leading row/column to exercise permutation.
        a0 = zeros(T, n, n)
        for j in 1:n, i in 1:n
            a0[i, j] = randn(rng, T)
        end
        a0[1, :] .= zero(T)
        a0[:, 1] .= zero(T)
        a0[1, 1] = one(T)
        # GEBAL on both layouts.
        acol = vec(copy(a0))
        iloc = Ref{J}(-1); ihic = Ref{J}(-1)
        scalec = zeros(R, n)
        @test lapacke_gebal_call(
            T, R, J, pbal, 102, job, n, acol, n, iloc, ihic, scalec, bname, route,
        ) == 0
        ld = n + 2
        arow = row_major(a0, ld)
        ilor = Ref{J}(-1); ihir = Ref{J}(-1)
        scaler = zeros(R, n)
        @test lapacke_gebal_call(
            T, R, J, pbal, 101, job, n, arow, ld, ilor, ihir, scaler, bname, route,
        ) == 0
        @test isequal(ilor[], iloc[])
        @test isequal(ihir[], ihic[])
        @test isequal(scaler, scalec)
        tol = 100 * eps(R)
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        # GEBAK on both layouts and both sides.
        m = 2
        v0 = randn(MersenneTwister(5400 + n), T, n, m)
        for side in ('L', 'R')
            vcol = vec(copy(v0))
            @test lapacke_gebak_call(
                T, R, J, pbak, 102, job, side, n, iloc[], ihic[], scalec, m,
                vcol, n, kname, route,
            ) == 0
            lbs = m + 2
            vrow = row_major(v0, lbs)
            @test lapacke_gebak_call(
                T, R, J, pbak, 101, job, side, n, ilor[], ihir[], scaler, m,
                vrow, lbs, kname, route,
            ) == 0
            @test isapprox(
                from_row_major(vrow, n, m, lbs), reshape(vcol, n, m);
                rtol=100 * eps(R), atol=100 * eps(R),
            )
        end
    end
end

@testset "LAPACKE GEBAL/GEBAK diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        bname = "LAPACKE_" * prefix * "gebal"
        kname = "LAPACKE_" * prefix * "gebak"
        pbal = Libdl.dlsym(native, bname * suffix)
        pbak = Libdl.dlsym(native, kname * suffix)
        n, m = 3, 2
        a = zeros(T, n * n)
        ilo = Ref{J}(0); ihi = Ref{J}(0)
        scale = ones(R, n)
        @test lapacke_gebal_call(
            T, R, J, pbal, 301, 'B', n, a, n, ilo, ihi, scale, bname, :dll,
        ) == -1
        @test lapacke_gebal_call(
            T, R, J, pbal, 101, 'B', n, a, n - 1, ilo, ihi, scale, bname, :dll,
        ) == -5
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_gebal_call(
            T, R, J, pbal, 102, 'B', n, bad, n, ilo, ihi, scale, bname, :dll,
        ) == -4
        v = zeros(T, n * m)
        @test lapacke_gebak_call(
            T, R, J, pbak, 301, 'B', 'L', n, 1, n, scale, m, v, n, kname, :dll,
        ) == -1
        @test lapacke_gebak_call(
            T, R, J, pbak, 101, 'B', 'L', n, 1, n, scale, m, v, m - 1,
            kname, :dll,
        ) == -10
        badscale = copy(scale); badscale[1] = R(NaN)
        @test lapacke_gebak_call(
            T, R, J, pbak, 102, 'B', 'L', n, 1, n, badscale, m, v, n,
            kname, :dll,
        ) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
