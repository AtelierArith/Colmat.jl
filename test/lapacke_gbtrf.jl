using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_gbtrf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, m, n, kl, ku, ab, ldab, ipiv,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$J}),
                layout, m, n, kl, ku, ab, ldab, ipiv)
        end
    end
    @eval function lapacke_gbtrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, trans, n, kl, ku, nrhs, ab,
        ldab, ipiv, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                layout, trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb)
        end
    end
    @eval function lapacke_gbsv_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, n, kl, ku, nrhs, ab, ldab,
        ipiv, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                layout, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb)
        end
    end
end

const BR(kl, ku) = 2 * kl + ku + 1
bandrow(i, j, kl, ku) = kl + ku + i - j
function band_col(matrix, kl, ku)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), BR(kl, ku) * n)
    for j in 1:n, i in 1:m
        abs(i - j) > kl + ku && continue
        i + ku < j && continue
        j + kl < i && continue
        out[bandrow(i, j, kl, ku) + (j - 1) * BR(kl, ku) + 1] = matrix[i, j]
    end
    return out
end
function band_row(matrix, kl, ku, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), BR(kl, ku) * ld)
    for j in 1:n, i in 1:m
        abs(i - j) > kl + ku && continue
        i + ku < j && continue
        j + kl < i && continue
        out[bandrow(i, j, kl, ku) * ld + j] = matrix[i, j]
    end
    return out
end
function band_matrix_col(ab, m, n, kl, ku)
    out = zeros(eltype(ab), m, n)
    for j in 1:n, i in 1:m
        abs(i - j) > kl + ku && continue
        i + ku < j && continue
        j + kl < i && continue
        out[i, j] = ab[bandrow(i, j, kl, ku) + (j - 1) * BR(kl, ku) + 1]
    end
    return out
end
function band_matrix_row(ab, m, n, kl, ku, ld)
    out = zeros(eltype(ab), m, n)
    for j in 1:n, i in 1:m
        abs(i - j) > kl + ku && continue
        i + ku < j && continue
        j + kl < i && continue
        out[i, j] = ab[bandrow(i, j, kl, ku) * ld + j]
    end
    return out
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

@testset "LAPACKE GBTRF/GBTRS/GBSV band layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), (m, n, kl, ku) in ((4, 4, 1, 2), (5, 4, 2, 1)),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "gbtrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "gbtrs" * (save ? "_work" : "")
        vname = "LAPACKE_" * prefix * "gbsv" * (save ? "_work" : "")
        for nm in (fname, sname, vname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix)
        rng = MersenneTwister(4200 + m + n)
        full = randn(rng, T, m, n)
        band = zeros(T, m, n)
        for j in 1:n, i in 1:m
            (i + ku >= j && j + kl >= i) || continue
            band[i, j] = full[i, j]
        end
        if m == n
            band = band + T(0.1) * Matrix{T}(I, m, n)
        end
        ld = n + 2
        # GBTRF.
        abcol = band_col(band, kl, ku)
        ipivcol = zeros(J, min(m, n))
        @test lapacke_gbtrf_call(
            T, J, pfac, 102, m, n, kl, ku, abcol, BR(kl, ku), ipivcol,
            fname, route,
        ) == 0
        abrow = band_row(band, kl, ku, ld)
        ipivrow = zeros(J, min(m, n))
        @test lapacke_gbtrf_call(
            T, J, pfac, 101, m, n, kl, ku, abrow, ld, ipivrow, fname, route,
        ) == 0
        tol = 2000 * eps(real(T))
        @test isequal(ipivrow, ipivcol)
        @test isapprox(
            band_matrix_row(abrow, m, n, kl, ku, ld),
            band_matrix_col(abcol, m, n, kl, ku);
            rtol=tol, atol=tol,
        )
        # GBTRS with the shared factors (square case only).
        if m == n
            nrhs = 2
            b0 = randn(MersenneTwister(4300 + n), T, n, nrhs)
            for trans in (T <: Complex ? ('N', 'C', 'T') : ('N', 'T'))
                bcol = vec(copy(b0))
                @test lapacke_gbtrs_call(
                    T, J, psol, 102, trans, n, kl, ku, nrhs, abcol,
                    BR(kl, ku), ipivcol, bcol, n, sname, route,
                ) == 0
                brow = row_major(b0, nrhs + 2)
                @test lapacke_gbtrs_call(
                    T, J, psol, 101, trans, n, kl, ku, nrhs, abrow, ld,
                    ipivrow, brow, nrhs + 2, sname, route,
                ) == 0
                @test isapprox(
                    from_row_major(brow, n, nrhs, nrhs + 2),
                    reshape(bcol, n, nrhs); rtol=tol, atol=tol,
                )
                if trans == 'N'
                    scale = 2000 * n * eps(real(T)) * norm(b0)
                    @test isapprox(band * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
                end
            end
        end
        # GBSV end to end.
        if m == n
            nrhs = 2
            b0 = randn(MersenneTwister(4400 + n), T, n, nrhs)
            abcol2 = band_col(band, kl, ku)
            abrow2 = band_row(band, kl, ku, ld)
            ipivc = zeros(J, n); ipivr = zeros(J, n)
            bcol = vec(copy(b0)); brow = row_major(b0, nrhs + 2)
            @test lapacke_gbsv_call(
                T, J, psv, 102, n, kl, ku, nrhs, abcol2, BR(kl, ku), ipivc,
                bcol, n, vname, route,
            ) == 0
            @test lapacke_gbsv_call(
                T, J, psv, 101, n, kl, ku, nrhs, abrow2, ld, ipivr, brow,
                nrhs + 2, vname, route,
            ) == 0
            @test isequal(ipivr, ipivc)
            @test isapprox(
                from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
                rtol=tol, atol=tol,
            )
            scale = 2000 * n * eps(real(T)) * norm(b0)
            @test isapprox(band * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        end
    end
end

@testset "LAPACKE GBTRF/GBTRS/GBSV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "gbtrf"
        sname = "LAPACKE_" * prefix * "gbtrs"
        vname = "LAPACKE_" * prefix * "gbsv"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        psv = Libdl.dlsym(native, vname * suffix)
        m, n, kl, ku, nrhs = 4, 4, 1, 2, 2
        ab = zeros(T, BR(kl, ku) * n)
        ipiv = zeros(J, min(m, n))
        b = zeros(T, n * nrhs)
        @test lapacke_gbtrf_call(
            T, J, pfac, 301, m, n, kl, ku, ab, BR(kl, ku), ipiv, fname, :dll,
        ) == -1
        @test lapacke_gbtrf_call(
            T, J, pfac, 101, m, n, kl, ku, ab, n - 1, ipiv, fname, :dll,
        ) == -7
        bad = copy(ab); bad[bandrow(1, 1, kl, ku) + 1] = T(NaN)
        @test lapacke_gbtrf_call(
            T, J, pfac, 102, m, n, kl, ku, bad, BR(kl, ku), ipiv, fname, :dll,
        ) == -6
        @test lapacke_gbtrs_call(
            T, J, psol, 301, 'N', n, kl, ku, nrhs, ab, BR(kl, ku), ipiv, b, n,
            sname, :dll,
        ) == -1
        @test lapacke_gbtrs_call(
            T, J, psol, 101, 'N', n, kl, ku, nrhs, ab, n - 1, ipiv, b, nrhs,
            sname, :dll,
        ) == -8
        @test lapacke_gbtrs_call(
            T, J, psol, 101, 'N', n, kl, ku, nrhs, ab, n, ipiv, b, nrhs - 1,
            sname, :dll,
        ) == -11
        @test lapacke_gbsv_call(
            T, J, psv, 301, n, kl, ku, nrhs, ab, BR(kl, ku), ipiv, b, n,
            vname, :dll,
        ) == -1
        @test lapacke_gbsv_call(
            T, J, psv, 101, n, kl, ku, nrhs, ab, n - 1, ipiv, b, nrhs,
            vname, :dll,
        ) == -7
        @test lapacke_gbsv_call(
            T, J, psv, 101, n, kl, ku, nrhs, ab, n, ipiv, b, nrhs - 1,
            vname, :dll,
        ) == -10
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
