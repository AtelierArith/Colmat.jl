using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gehrd_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, n, ilo, ihi, a, lda, tau,
        work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout, n, ilo, ihi, a, lda, tau, work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}),
                    layout, n, ilo, ihi, a, lda, tau)
            end
        end
    end
    @eval function lapacke_gebrd_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, m, n, a, lda,
        d::Vector{$R}, e::Vector{$R}, tauq, taup, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$T},
                     Ptr{$T}, Ptr{$T}, $J),
                    layout, m, n, a, lda, d, e, tauq, taup, work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$T},
                     Ptr{$T}),
                    layout, m, n, a, lda, d, e, tauq, taup)
            end
        end
    end
    @eval function lapacke_orghr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, n, ilo, ihi, q, ldq, tau,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}),
                layout, n, ilo, ihi, q, ldq, tau)
        end
    end
    @eval function lapacke_larfx_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, m, n, v, tau::$T, c,
        ldc, work, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, $T, Ptr{$T}, $J, Ptr{$T}),
                layout, side, m, n, v, tau, c, ldc, work)
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

@testset "LAPACKE GEHRD/GEBRD/LARFX layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 5), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        hname = "LAPACKE_" * prefix * "gehrd" * (save ? "_work" : "")
        bname = "LAPACKE_" * prefix * "gebrd" * (save ? "_work" : "")
        xname = "LAPACKE_" * prefix * "larfx" * (save ? "_work" : "")
        for nm in (hname, bname, xname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        phd = Libdl.dlsym(route === :dll ? native : trampoline, hname * suffix)
        pbd = Libdl.dlsym(route === :dll ? native : trampoline, bname * suffix)
        pfx = Libdl.dlsym(route === :dll ? native : trampoline, xname * suffix)
        rng = MersenneTwister(5900 + n)
        # GEHRD on both layouts.
        a0 = randn(rng, T, n, n)
        acol = vec(copy(a0)); taucol = zeros(T, max(0, n - 1))
        @test lapacke_gehrd_call(
            T, J, phd, 102, n, 1, n, acol, n, taucol, fill(T(NaN), 2n), save,
            hname, route,
        ) == 0
        ld = n + 2
        arow = row_major(a0, ld); taurow = zeros(T, max(0, n - 1))
        @test lapacke_gehrd_call(
            T, J, phd, 101, n, 1, n, arow, ld, taurow, fill(T(NaN), 2n), save,
            hname, route,
        ) == 0
        tol = 500 * eps(R)
        @test isapprox(taurow, taucol; rtol=tol, atol=tol)
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        # Reconstructing Q with ORGHR shows Q' * A0 * Q is upper Hessenberg.
        hgen = "LAPACKE_" * prefix * (T <: Complex ? "unghr" : "orghr")
        phd2 = Libdl.dlsym(route === :dll ? native : trampoline, hgen * suffix)
        qbuf = reshape(acol, n, n)[:]
        qi = lapacke_orghr_call(
            T, J, phd2, 102, n, 1, n, qbuf, n, taucol, hgen, route,
        )
        @test qi == 0
        Q = reshape(qbuf, n, n)
        Hfull = Q' * a0 * Q
        @test all(isapprox(Hfull[i, j], 0; atol=1000 * eps(R)) for i in 3:n for j in 1:i-2)
        # GEBRD on both layouts (square and rectangular).
        for (m, nn) in ((n, n), (n + 1, n - 1))
            m < 1 && continue
            b0 = randn(MersenneTwister(6000 + m + nn), T, m, nn)
            k = min(m, nn)
            bcol = vec(copy(b0))
            dcol = zeros(R, k); ecol = zeros(R, max(0, k - 1))
            tqcol = zeros(T, k); tpcol = zeros(T, k)
            @test lapacke_gebrd_call(
                T, R, J, pbd, 102, m, nn, bcol, m, dcol, ecol, tqcol, tpcol,
                fill(T(NaN), 2m + 2nn), save, bname, route,
            ) == 0
            lds = nn + 2
            brow = row_major(b0, lds)
            drow = zeros(R, k); erow = zeros(R, max(0, k - 1))
            tqrow = zeros(T, k); tprow = zeros(T, k)
            @test lapacke_gebrd_call(
                T, R, J, pbd, 101, m, nn, brow, lds, drow, erow, tqrow, tprow,
                fill(T(NaN), 2m + 2nn), save, bname, route,
            ) == 0
            @test isapprox(drow, dcol; rtol=tol, atol=tol)
            @test isapprox(erow, ecol; rtol=tol, atol=tol)
            @test isapprox(tqrow, tqcol; rtol=tol, atol=tol)
            @test isapprox(tprow, tpcol; rtol=tol, atol=tol)
            @test isapprox(
                from_row_major(brow, m, nn, lds), reshape(bcol, m, nn);
                rtol=tol, atol=tol,
            )
        end
        # LARFX on both layouts, both sides, with a large order to force the
        # workspace path and a small order to force the unblocked path.
        for side in ('L', 'R'), (m, nn) in ((n, 3), (12, 3))
            order = side == 'L' ? m : nn
            v = randn(MersenneTwister(6100 + order), T, order)
            tau = T(0.7)
            c0 = randn(MersenneTwister(6200 + order), T, m, nn)
            ccol = vec(copy(c0))
            work = fill(T(NaN), max(1, m + nn))
            @test lapacke_larfx_call(
                T, J, pfx, 102, side, m, nn, v, tau, ccol, m, work, xname,
                route,
            ) == 0
            lds = nn + 2
            crow = row_major(c0, lds)
            @test lapacke_larfx_call(
                T, J, pfx, 101, side, m, nn, v, tau, crow, lds, work, xname,
                route,
            ) == 0
            @test isapprox(
                from_row_major(crow, m, nn, lds), reshape(ccol, m, nn);
                rtol=500 * eps(R), atol=500 * eps(R),
            )
            # The reflector identity holds on the column-major result with the
            # full vector convention shared by LARF and LARFX.
            if side == 'L'
                expect = c0 - tau .* v .* (v' * c0)
            else
                expect = c0 - tau .* (c0 * v) .* v'
            end
            @test isapprox(
                reshape(ccol, m, nn), expect;
                rtol=1000 * eps(R), atol=1000 * eps(R),
            )
        end
    end
end

@testset "LAPACKE GEHRD/GEBRD/LARFX diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        hname = "LAPACKE_" * prefix * "gehrd"
        bname = "LAPACKE_" * prefix * "gebrd"
        xname = "LAPACKE_" * prefix * "larfx"
        phd = Libdl.dlsym(native, hname * suffix)
        pbd = Libdl.dlsym(native, bname * suffix)
        pfx = Libdl.dlsym(native, xname * suffix)
        n = 4
        a = zeros(T, n * n)
        tau = zeros(T, n - 1)
        @test lapacke_gehrd_call(
            T, J, phd, 301, n, 1, n, a, n, tau, T[], false, hname, :dll,
        ) == -1
        @test lapacke_gehrd_call(
            T, J, phd, 101, n, 1, n, a, n - 1, tau, T[], false, hname, :dll,
        ) == -6
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_gehrd_call(
            T, J, phd, 102, n, 1, n, bad, n, tau, T[], false, hname, :dll,
        ) == -5
        d = zeros(R, n); e = zeros(R, n - 1)
        tq = zeros(T, n); tp = zeros(T, n)
        @test lapacke_gebrd_call(
            T, R, J, pbd, 301, n, n, a, n, d, e, tq, tp, T[], false,
            bname, :dll,
        ) == -1
        @test lapacke_gebrd_call(
            T, R, J, pbd, 101, n, n, a, n - 1, d, e, tq, tp, T[], false,
            bname, :dll,
        ) == -5
        badb = copy(a); badb[1] = T(NaN)
        @test lapacke_gebrd_call(
            T, R, J, pbd, 102, n, n, badb, n, d, e, tq, tp, T[], false,
            bname, :dll,
        ) == -4
        m = 4
        v = zeros(T, m)
        c = zeros(T, m * m)
        work = zeros(T, m)
        @test lapacke_larfx_call(
            T, J, pfx, 301, 'L', m, m, v, T(0), c, m, work, xname, :dll,
        ) == -1
        @test lapacke_larfx_call(
            T, J, pfx, 101, 'L', m, m, v, T(0), c, m - 1, work, xname, :dll,
        ) == -8
        badc = copy(c); badc[1] = T(NaN)
        @test lapacke_larfx_call(
            T, J, pfx, 102, 'L', m, m, v, T(0), badc, m, work, xname, :dll,
        ) == -7
        @test lapacke_larfx_call(
            T, J, pfx, 102, 'L', m, m, v, T(NaN), c, m, work, xname, :dll,
        ) == -6
        badv = copy(v); badv[1] = T(NaN)
        @test lapacke_larfx_call(
            T, J, pfx, 102, 'L', m, m, badv, T(0), c, m, work, xname, :dll,
        ) == -5
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
