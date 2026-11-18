using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64))
    @eval function lapacke_sptrf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, ap, ipiv, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$J}),
                  layout, uplo, n, ap, ipiv)
        end
    end
    @eval function lapacke_sptrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, ap, ipiv, b,
        ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$J}, Ptr{$T}, $J),
                layout, uplo, n, nrhs, ap, ipiv, b, ldb)
        end
    end
    @eval function lapacke_spsv_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, ap, ipiv, b,
        ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$J}, Ptr{$T}, $J),
                layout, uplo, n, nrhs, ap, ipiv, b, ldb)
        end
    end
end

function packed_col(matrix, uplo)
    n = size(matrix, 1)
    out = eltype(matrix)[]
    if uplo == 'U'
        for j in 1:n, i in 1:j
            push!(out, matrix[i, j])
        end
    else
        for j in 1:n, i in j:n
            push!(out, matrix[i, j])
        end
    end
    return out
end
function permute_row_from_col(ap, uplo, n)
    upper = uplo == 'U'
    out = fill(eltype(ap)(NaN), length(ap))
    for j in 1:n, i in 1:n
        (upper ? i <= j : i >= j) || continue
        rowidx = upper ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j
        colidx = upper ? j * (j - 1) ÷ 2 + i : (j - 1) * n - (j - 1) * j ÷ 2 + i
        out[rowidx] = ap[colidx]
    end
    return out
end
function packed_row(matrix, uplo)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), n * (n + 1) ÷ 2)
    if uplo == 'U'
        for i in 1:n, j in i:n
            out[(i - 1) * n - (i - 1) * i ÷ 2 + j] = matrix[i, j]
        end
    else
        for i in 1:n, j in 1:i
            out[(i - 1) * i ÷ 2 + j] = matrix[i, j]
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

@testset "LAPACKE SPTRF/SPTRS/SPSV packed layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "sptrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "sptrs" * (save ? "_work" : "")
        vname = "LAPACKE_" * prefix * "spsv" * (save ? "_work" : "")
        for nm in (fname, sname, vname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix)
        rng = MersenneTwister(4900 + n)
        full = randn(rng, T, n, n)
        sym = Matrix(Symmetric(full, uplo == 'L' ? :L : :U))
        # SPTRF on both layouts.
        apcol = packed_col(sym, uplo)
        ipivcol = zeros(J, n)
        @test lapacke_sptrf_call(
            T, J, pfac, 102, uplo, n, apcol, ipivcol, fname, route,
        ) == 0
        aprow = packed_row(sym, uplo)
        ipivrow = zeros(J, n)
        @test lapacke_sptrf_call(
            T, J, pfac, 101, uplo, n, aprow, ipivrow, fname, route,
        ) == 0
        @test isequal(ipivrow, ipivcol)
        # The row-major factor buffer is the sp_trans permutation of the
        # column-major one, exactly as in the reference.
        @test isequal(aprow, permute_row_from_col(apcol, uplo, n))
        # SPTRS with the shared factors.
        nrhs = 2
        b0 = randn(MersenneTwister(5000 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_sptrs_call(
            T, J, psol, 102, uplo, n, nrhs, apcol, ipivcol, bcol, n,
            sname, route,
        ) == 0
        lbs = nrhs + 2
        brow = row_major(b0, lbs)
        @test lapacke_sptrs_call(
            T, J, psol, 101, uplo, n, nrhs, aprow, ipivrow, brow, lbs,
            sname, route,
        ) == 0
        tol = 4000 * eps(T)
        @test isapprox(
            from_row_major(brow, n, nrhs, lbs), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 4000 * n * eps(T) * norm(b0)
        @test isapprox(sym * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        # SPSV end to end.
        apcol2 = packed_col(sym, uplo)
        aprow2 = packed_row(sym, uplo)
        ipivc = zeros(J, n); ipivr = zeros(J, n)
        bcol2 = vec(copy(b0)); brow2 = row_major(b0, nrhs + 2)
        @test lapacke_spsv_call(
            T, J, psv, 102, uplo, n, nrhs, apcol2, ipivc, bcol2, n,
            vname, route,
        ) == 0
        @test lapacke_spsv_call(
            T, J, psv, 101, uplo, n, nrhs, aprow2, ipivr, brow2, nrhs + 2,
            vname, route,
        ) == 0
        @test isequal(ipivr, ipivc)
        @test isequal(aprow2, permute_row_from_col(apcol2, uplo, n))
        @test isapprox(
            from_row_major(brow2, n, nrhs, nrhs + 2), reshape(bcol2, n, nrhs);
            rtol=tol, atol=tol,
        )
        @test isapprox(sym * reshape(bcol2, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE SPTRF/SPTRS/SPSV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"),), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "sptrf"
        sname = "LAPACKE_" * prefix * "sptrs"
        vname = "LAPACKE_" * prefix * "spsv"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        psv = Libdl.dlsym(native, vname * suffix)
        n, nrhs = 3, 2
        ap = zeros(T, n * (n + 1) ÷ 2)
        ipiv = zeros(J, n)
        b = zeros(T, n * nrhs)
        @test lapacke_sptrf_call(
            T, J, pfac, 301, 'U', n, ap, ipiv, fname, :dll,
        ) == -1
        bad = copy(ap); bad[1] = T(NaN)
        @test lapacke_sptrf_call(
            T, J, pfac, 102, 'U', n, bad, ipiv, fname, :dll,
        ) == -4
        @test lapacke_sptrs_call(
            T, J, psol, 301, 'U', n, nrhs, ap, ipiv, b, n, sname, :dll,
        ) == -1
        @test lapacke_sptrs_call(
            T, J, psol, 101, 'U', n, nrhs, ap, ipiv, b, nrhs - 1, sname, :dll,
        ) == -8
        @test lapacke_sptrs_call(
            T, J, psol, 102, 'U', n, nrhs, bad, ipiv, b, n, sname, :dll,
        ) == -5
        @test lapacke_spsv_call(
            T, J, psv, 301, 'U', n, nrhs, ap, ipiv, b, n, vname, :dll,
        ) == -1
        @test lapacke_spsv_call(
            T, J, psv, 101, 'U', n, nrhs, ap, ipiv, b, nrhs - 1, vname, :dll,
        ) == -8
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
