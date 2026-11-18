using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_pbtrf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, kd, ab, ldab,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, $J, Ptr{$T}, $J),
                  layout, uplo, n, kd, ab, ldab)
        end
    end
    @eval function lapacke_pbtrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, kd, nrhs, ab, ldab,
        b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                  layout, uplo, n, kd, nrhs, ab, ldab, b, ldb)
        end
    end
end

function band_col(matrix, uplo, kd)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[(kd + i - j + 1) + (j - 1) * (kd + 1)] = matrix[i, j]
        else
            -kd <= d <= 0 || continue
            out[(i - j + 1) + (j - 1) * (kd + 1)] = matrix[i, j]
        end
    end
    return out
end
function band_row(matrix, uplo, kd, ld)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), (kd + 1) * ld)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[(kd + i - j) * ld + j] = matrix[i, j]
        else
            -kd <= d <= 0 || continue
            out[(i - j) * ld + j] = matrix[i, j]
        end
    end
    return out
end
function band_matrix_col(ab, uplo, n, kd)
    out = zeros(eltype(ab), n, n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[i, j] = ab[(kd + i - j + 1) + (j - 1) * (kd + 1)]
        else
            -kd <= d <= 0 || continue
            out[i, j] = ab[(i - j + 1) + (j - 1) * (kd + 1)]
        end
    end
    return out
end
function band_matrix_row(ab, uplo, n, kd, ld)
    out = zeros(eltype(ab), n, n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[i, j] = ab[(kd + i - j) * ld + j]
        else
            -kd <= d <= 0 || continue
            out[i, j] = ab[(i - j) * ld + j]
        end
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

@testset "LAPACKE PBTRF/PBTRS band layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (n, kd) in ((3, 1), (5, 2)), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "pbtrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "pbtrs" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(fname, interface) == Libdl.dlsym(native, fname * suffix)
        @test BLAS.lbt_get_forward(sname, interface) == Libdl.dlsym(native, sname * suffix)
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        rng = MersenneTwister(3600 + n)
        bmaker = zeros(T, n, n)
        for j in 1:n, i in j:min(n, j + kd)
            bmaker[i, j] = randn(rng, T)
        end
        m = bmaker * bmaker' + T(n) * Matrix{T}(I, n, n)
        ab0 = band_col(m, uplo, kd)
        abcol = copy(ab0)
        @test lapacke_pbtrf_call(
            T, J, pfac, 102, uplo, n, kd, abcol, kd + 1, fname, route,
        ) == 0
        ld = n + 2
        abrow = band_row(m, uplo, kd, ld)
        @test lapacke_pbtrf_call(
            T, J, pfac, 101, uplo, n, kd, abrow, ld, fname, route,
        ) == 0
        tol = 3000 * eps(R)
        @test isapprox(
            band_matrix_row(abrow, uplo, n, kd, ld),
            band_matrix_col(abcol, uplo, n, kd);
            rtol=tol, atol=tol,
        )
        # Factor work entry on the column-major copy must agree.
        abwork = copy(ab0)
        workname = "LAPACKE_" * prefix * "pbtrf_work"
        @test lapacke_pbtrf_call(
            T, J, Libdl.dlsym(native, workname * suffix), 102, uplo, n, kd,
            abwork, kd + 1, workname, :dll,
        ) == 0
        @test isequal(abwork, abcol)
        # Solve and check layouts plus residual.
        nrhs = 2
        b0 = randn(MersenneTwister(3700 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_pbtrs_call(
            T, J, psol, 102, uplo, n, kd, nrhs, abcol, kd + 1, bcol, n,
            sname, route,
        ) == 0
        lbs = nrhs + 2
        brow = row_major(b0, lbs)
        @test lapacke_pbtrs_call(
            T, J, psol, 101, uplo, n, kd, nrhs, abrow, ld, brow, lbs,
            sname, route,
        ) == 0
        @test isapprox(
            from_row_major(brow, n, nrhs, lbs), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 1000 * max(1, n) * eps(R) * norm(b0)
        # The residual check exercises the native band solve; the conversion
        # contract itself is covered by the layout and factor comparisons.
        scale = 1000 * n * eps(R) * norm(b0)
        @test isapprox(m * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE PBTRF/PBTRS diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "pbtrf"
        sname = "LAPACKE_" * prefix * "pbtrs"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        n, kd, nrhs = 4, 1, 2
        ab = zeros(T, (kd + 1) * n)
        b = zeros(T, n * nrhs)
        @test lapacke_pbtrf_call(
            T, J, pfac, 301, 'U', n, kd, ab, kd + 1, fname, :dll,
        ) == -1
        @test lapacke_pbtrf_call(
            T, J, pfac, 101, 'U', n, kd, ab, n - 1, fname, :dll,
        ) == -6
        bad = copy(ab); bad[kd + 1] = T(NaN)
        @test lapacke_pbtrf_call(
            T, J, pfac, 102, 'U', n, kd, bad, kd + 1, fname, :dll,
        ) == -5
        @test lapacke_pbtrs_call(
            T, J, psol, 301, 'U', n, kd, nrhs, ab, kd + 1, b, n,
            sname, :dll,
        ) == -1
        @test lapacke_pbtrs_call(
            T, J, psol, 101, 'U', n, kd, nrhs, ab, n - 1, b, nrhs,
            sname, :dll,
        ) == -7
        @test lapacke_pbtrs_call(
            T, J, psol, 101, 'U', n, kd, nrhs, ab, n, b, nrhs - 1,
            sname, :dll,
        ) == -9
        @test lapacke_pbtrs_call(
            T, J, psol, 102, 'U', n, kd, nrhs, bad, kd + 1, b, n,
            sname, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
