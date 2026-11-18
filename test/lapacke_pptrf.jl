using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_pptrf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, ap, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}), layout, uplo, n, ap)
        end
    end
    @eval function lapacke_pptrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, ap, b, ldb,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$T}, $J),
                  layout, uplo, n, nrhs, ap, b, ldb)
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
function unpack(matrix, uplo)
    n = size(matrix, 1)
    out = zeros(eltype(matrix), n, n)
    if uplo == 'U'
        for j in 1:n, i in 1:j
            out[i, j] = matrix[i, j]
            out[j, i] = conj(matrix[i, j])
        end
    else
        for j in 1:n, i in j:n
            out[i, j] = matrix[i, j]
            out[j, i] = conj(matrix[i, j])
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
function packed_matrix(vector, uplo, n)
    out = zeros(eltype(vector), n, n)
    if uplo == 'U'
        k = 1
        for j in 1:n, i in 1:j
            out[i, j] = vector[k]
            k += 1
        end
    else
        k = 1
        for j in 1:n, i in j:n
            out[i, j] = vector[k]
            k += 1
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

@testset "LAPACKE PPTRF/PPTRS packed layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "pptrf" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "pptrs" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(fname, interface) == Libdl.dlsym(native, fname * suffix)
        @test BLAS.lbt_get_forward(sname, interface) == Libdl.dlsym(native, sname * suffix)
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        rng = MersenneTwister(4000 + n)
        full = randn(rng, T, n, n)
        herm = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        apcol = packed_col(herm, uplo)
        @test lapacke_pptrf_call(T, J, pfac, 102, uplo, n, apcol, fname, route) == 0
        aprow = packed_row(herm, uplo)
        @test lapacke_pptrf_call(T, J, pfac, 101, uplo, n, aprow, fname, route) == 0
        tol = 4000 * eps(R)
        # LAPACKE's row-major packed factor is the pp_trans permutation of the
        # column-major factor, and PPTRS consumes it through the same map.
        @test isapprox(aprow, permute_row_from_col(apcol, uplo, n); rtol=tol, atol=tol)
        nrhs = 2
        b0 = randn(MersenneTwister(4100 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_pptrs_call(
            T, J, psol, 102, uplo, n, nrhs, apcol, bcol, n, sname, route,
        ) == 0
        lbs = nrhs + 2
        brow = row_major(b0, lbs)
        @test lapacke_pptrs_call(
            T, J, psol, 101, uplo, n, nrhs, aprow, brow, lbs, sname, route,
        ) == 0
        @test isapprox(
            from_row_major(brow, n, nrhs, lbs), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 2000 * n * eps(R) * norm(b0)
        @test isapprox(herm * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE PPTRF/PPTRS diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "pptrf"
        sname = "LAPACKE_" * prefix * "pptrs"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        n, nrhs = 3, 2
        ap = zeros(T, n * (n + 1) ÷ 2)
        b = zeros(T, n * nrhs)
        @test lapacke_pptrf_call(T, J, pfac, 301, 'U', n, ap, fname, :dll) == -1
        bad = copy(ap); bad[1] = T(NaN)
        @test lapacke_pptrf_call(T, J, pfac, 102, 'U', n, bad, fname, :dll) == -4
        @test lapacke_pptrs_call(
            T, J, psol, 301, 'U', n, nrhs, ap, b, n, sname, :dll,
        ) == -1
        @test lapacke_pptrs_call(
            T, J, psol, 101, 'U', n, nrhs, ap, b, nrhs - 1, sname, :dll,
        ) == -7
        @test lapacke_pptrs_call(
            T, J, psol, 102, 'U', n, nrhs, bad, b, n, sname, :dll,
        ) == -5
        badd = copy(b); badd[1] = T(NaN)
        @test lapacke_pptrs_call(
            T, J, psol, 102, 'U', n, nrhs, ap, badd, n, sname, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
