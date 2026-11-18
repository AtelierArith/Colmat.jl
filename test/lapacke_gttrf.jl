using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_gttrf_call(
        ::Type{$T}, ::Type{$J}, pointer, n, dl, d, du, du2, ipiv, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$J}),
                  n, dl, d, du, du2, ipiv)
        end
    end
    @eval function lapacke_gttrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, trans, n, nrhs, dl, d, du, du2,
        ipiv, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                 Ptr{$J}, Ptr{$T}, $J),
                layout, trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb)
        end
    end
    @eval function lapacke_gtsv_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, n, nrhs, dl, d, du, b, ldb,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J),
                layout, n, nrhs, dl, d, du, b, ldb)
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

@testset "LAPACKE GTTRF/GTTRS/GTSV layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (1, 4), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "gttrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "gttrs" * (save ? "_work" : "")
        vname = "LAPACKE_" * prefix * "gtsv" * (save ? "_work" : "")
        for nm in (fname, sname, vname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix)
        rng = MersenneTwister(4600 + n)
        d0 = T[2 + i for i in 1:n]
        dl0 = T[0.3i - 0.1 for i in 1:max(0, n - 1)]
        du0 = T[0.2i + 0.05 for i in 1:max(0, n - 1)]
        # GTTRF: layout-free, high matches work.
        dl, d, du = copy(dl0), copy(d0), copy(du0)
        du2 = zeros(T, max(0, n - 2))
        ipiv = zeros(J, n)
        @test lapacke_gttrf_call(T, J, pfac, J(n), dl, d, du, du2, ipiv, fname, route) == 0
        dlw, dw, duw, du2w = copy(dl0), copy(d0), copy(du0), zeros(T, max(0, n - 2))
        ipivw = zeros(J, n)
        workname = "LAPACKE_" * prefix * "gttrf_work"
        @test lapacke_gttrf_call(
            T, J, Libdl.dlsym(native, workname * suffix), J(n), dlw, dw, duw,
            du2w, ipivw, workname, :dll,
        ) == 0
        @test isequal(dl, dlw) && isequal(d, dw) && isequal(du, duw)
        @test isequal(du2, du2w) && isequal(ipiv, ipivw)
        # GTTRS on both layouts with the shared factors.
        nrhs = 2
        A = Tridiagonal(dl0, d0, du0)
        for trans in (T <: Complex ? ('N', 'C') : ('N', 'T'))
            op = trans == 'N' ? A : (T <: Complex ? A' : A')
            b0 = trans == 'N' ?
                randn(MersenneTwister(4700 + n), T, n, nrhs) :
                randn(MersenneTwister(4700 + n), T, n, nrhs)
            bcol = vec(copy(b0))
            @test lapacke_gttrs_call(
                T, J, psol, 102, trans, n, nrhs, dl, d, du, du2, ipiv, bcol, n,
                sname, route,
            ) == 0
            lbs = nrhs + 2
            brow = row_major(b0, lbs)
            @test lapacke_gttrs_call(
                T, J, psol, 101, trans, n, nrhs, dl, d, du, du2, ipiv, brow,
                lbs, sname, route,
            ) == 0
            tol = 4000 * eps(real(T))
            @test isapprox(
                from_row_major(brow, n, nrhs, lbs), reshape(bcol, n, nrhs);
                rtol=tol, atol=tol,
            )
            scale = 4000 * n * eps(real(T)) * norm(b0)
            @test isapprox(op * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        end
        # GTSV end to end.
        dls, ds, dus = copy(dl0), copy(d0), copy(du0)
        b0 = randn(MersenneTwister(4800 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_gtsv_call(
            T, J, psv, 102, n, nrhs, dls, ds, dus, bcol, n, vname, route,
        ) == 0
        dlr, dr, dur = copy(dl0), copy(d0), copy(du0)
        brow = row_major(b0, nrhs + 2)
        @test lapacke_gtsv_call(
            T, J, psv, 101, n, nrhs, dlr, dr, dur, brow, nrhs + 2, vname, route,
        ) == 0
        tol = 4000 * eps(real(T))
        @test isapprox(
            from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        @test isequal(dlr, dls) && isequal(dr, ds) && isequal(dur, dus)
        scale = 4000 * n * eps(real(T)) * norm(b0)
        @test isapprox(A * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE GTTRF/GTTRS/GTSV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "gttrf"
        sname = "LAPACKE_" * prefix * "gttrs"
        vname = "LAPACKE_" * prefix * "gtsv"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        psv = Libdl.dlsym(native, vname * suffix)
        n, nrhs = 4, 2
        dl = zeros(T, n - 1); d = zeros(T, n); du = zeros(T, n - 1)
        du2 = zeros(T, n - 2); ipiv = zeros(J, n)
        b = zeros(T, n * nrhs)
        @test lapacke_gttrf_call(
            T, J, pfac, J(n), dl, T[NaN, 0, 0, 0], du, du2, ipiv, fname, :dll,
        ) == -3
        @test lapacke_gttrf_call(
            T, J, pfac, J(n), T[NaN, 0, 0], d, du, du2, ipiv, fname, :dll,
        ) == -2
        @test lapacke_gttrf_call(
            T, J, pfac, J(n), dl, d, T[NaN, 0, 0], du2, ipiv, fname, :dll,
        ) == -4
        @test lapacke_gttrs_call(
            T, J, psol, 301, 'N', n, nrhs, dl, d, du, du2, ipiv, b, n,
            sname, :dll,
        ) == -1
        @test lapacke_gttrs_call(
            T, J, psol, 101, 'N', n, nrhs, dl, d, du, du2, ipiv, b, nrhs - 1,
            sname, :dll,
        ) == -11
        @test lapacke_gttrs_call(
            T, J, psol, 102, 'N', n, nrhs, dl, T[NaN, 0, 0, 0], du, du2, ipiv,
            b, n, sname, :dll,
        ) == -6
        @test lapacke_gtsv_call(
            T, J, psv, 301, n, nrhs, dl, d, du, b, n, vname, :dll,
        ) == -1
        @test lapacke_gtsv_call(
            T, J, psv, 101, n, nrhs, dl, d, du, b, nrhs - 1, vname, :dll,
        ) == -8
        @test lapacke_gtsv_call(
            T, J, psv, 102, n, nrhs, T[NaN, 0, 0], d, du, b, n, vname, :dll,
        ) == -4
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
