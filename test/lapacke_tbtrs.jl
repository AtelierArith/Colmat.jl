using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_tbtrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, trans, diag, n, kd, nrhs,
        ab, ldab, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                layout, uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb)
        end
    end
    @eval function lapacke_tptrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, trans, diag, n, nrhs, ap,
        b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, Ptr{$T}, $J),
                layout, uplo, trans, diag, n, nrhs, ap, b, ldb)
        end
    end
    @eval function lapacke_tptri_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, diag, n, ap, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}),
                  layout, uplo, diag, n, ap)
        end
    end
end

const BR(kd) = kd + 1
function band_col(matrix, uplo, kd)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), BR(kd) * n)
    for j in 1:n, i in 1:n
        if uplo == 'U'
            0 <= j - i <= kd || continue
            out[kd + i - j + (j - 1) * BR(kd) + 1] = matrix[i, j]
        else
            0 <= i - j <= kd || continue
            out[i - j + (j - 1) * BR(kd) + 1] = matrix[i, j]
        end
    end
    return out
end
function band_row(matrix, uplo, kd, ld)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), BR(kd) * ld)
    for j in 1:n, i in 1:n
        if uplo == 'U'
            0 <= j - i <= kd || continue
            out[(kd + i - j) * ld + j] = matrix[i, j]
        else
            0 <= i - j <= kd || continue
            out[(i - j) * ld + j] = matrix[i, j]
        end
    end
    return out
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
            out[i, j] = vector[k]; k += 1
        end
    else
        k = 1
        for j in 1:n, i in j:n
            out[i, j] = vector[k]; k += 1
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

@testset "LAPACKE TBTRS/TPTRS/TPRI layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), (n, kd) in ((3, 1), (5, 2)), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        bname = "LAPACKE_" * prefix * "tbtrs" * (save ? "_work" : "")
        pname = "LAPACKE_" * prefix * "tptrs" * (save ? "_work" : "")
        iname = "LAPACKE_" * prefix * "tptri" * (save ? "_work" : "")
        for nm in (bname, pname, iname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        ptb = Libdl.dlsym(route === :dll ? native : trampoline, bname * suffix)
        ptp = Libdl.dlsym(route === :dll ? native : trampoline, pname * suffix)
        pti = Libdl.dlsym(route === :dll ? native : trampoline, iname * suffix)
        rng = MersenneTwister(5500 + n)
        # Triangular band factor with unit diagonal shifted to be well conditioned.
        band = zeros(T, n, n)
        for j in 1:n, i in 1:n
            if uplo == 'U'
                0 <= j - i <= kd || continue
                band[i, j] = i == j ? T(2) : T(0.3) * randn(rng, T)
            else
                0 <= i - j <= kd || continue
                band[i, j] = i == j ? T(2) : T(0.3) * randn(rng, T)
            end
        end
        nrhs = 2
        b0 = randn(MersenneTwister(5600 + n), T, n, nrhs)
        for trans in (T <: Complex ? ('N', 'C') : ('N', 'T')), diagc in ('N', 'U')
            # The effective operator replaces the stored diagonal with units
            # when DIAG='U'; TRANS only selects the solve side.
            unit = diagc == 'U'
            aeff = unit ? band + Matrix{T}(I, n, n) - Diagonal(LinearAlgebra.diag(band)) : band
            op = trans == 'N' ? aeff : (T <: Complex ? aeff' : aeff')
            # TBTRS on both layouts; AB always stores the selected triangle.
            abcol = band_col(band, uplo, kd)
            bcol = vec(copy(b0))
            @test lapacke_tbtrs_call(
                T, J, ptb, 102, uplo, trans, diagc, n, kd, nrhs, abcol, BR(kd),
                bcol, n, bname, route,
            ) == 0
            ld = n + 2
            abrow = band_row(band, uplo, kd, ld)
            brow = row_major(b0, nrhs + 2)
            @test lapacke_tbtrs_call(
                T, J, ptb, 101, uplo, trans, diagc, n, kd, nrhs, abrow, ld,
                brow, nrhs + 2, bname, route,
            ) == 0
            tol = 4000 * eps(real(T))
            @test isapprox(
                from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
                rtol=tol, atol=tol,
            )
            scale = 4000 * n * eps(real(T)) * norm(b0)
            @test isapprox(op * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        end
        # TPTRS and TPTRI use the full triangular factor.
        full = Matrix(UpperTriangular(zeros(T, n, n)))
        if uplo == 'U'
            for j in 1:n, i in 1:j
                full[i, j] = i == j ? T(2) : T(0.3) * randn(rng, T)
            end
        else
            for j in 1:n, i in j:n
                full[i, j] = i == j ? T(2) : T(0.3) * randn(rng, T)
            end
        end
        tri = uplo == 'U' ? UpperTriangular(full) : LowerTriangular(full)
        for trans in (T <: Complex ? ('N', 'C') : ('N', 'T'))
            op = trans == 'N' ? tri : (T <: Complex ? tri' : tri')
            apcol = packed_col(Matrix(tri), uplo)
            bcol = vec(copy(b0))
            @test lapacke_tptrs_call(
                T, J, ptp, 102, uplo, trans, 'N', n, nrhs, apcol, bcol, n,
                pname, route,
            ) == 0
            aprow = packed_row(Matrix(tri), uplo)
            brow = row_major(b0, nrhs + 2)
            @test lapacke_tptrs_call(
                T, J, ptp, 101, uplo, trans, 'N', n, nrhs, aprow, brow,
                nrhs + 2, pname, route,
            ) == 0
            tol = 4000 * eps(real(T))
            @test isapprox(
                from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
                rtol=tol, atol=tol,
            )
            scale = 4000 * n * eps(real(T)) * norm(b0)
            @test isapprox(op * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        end
        # TPTRI on both layouts.
        apcol = packed_col(Matrix(tri), uplo)
        @test lapacke_tptri_call(T, J, pti, 102, uplo, 'N', n, apcol, iname, route) == 0
        invcol = packed_matrix(apcol, uplo, n)
        aprow = packed_row(Matrix(tri), uplo)
        @test lapacke_tptri_call(T, J, pti, 101, uplo, 'N', n, aprow, iname, route) == 0
        # The row-major packed inverse is the tp_trans permutation of the
        # column-major one, matching the reference convention.
        tol = 8000 * eps(real(T))
        @test isapprox(aprow, permute_row_from_col(apcol, uplo, n); rtol=tol, atol=tol)
        @test isapprox(invcol * Matrix(tri), Matrix{T}(I, n, n); rtol=tol, atol=tol)
    end
end

@testset "LAPACKE TBTRS/TPTRS/TPTRI diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        bname = "LAPACKE_" * prefix * "tbtrs"
        pname = "LAPACKE_" * prefix * "tptrs"
        iname = "LAPACKE_" * prefix * "tptri"
        ptb = Libdl.dlsym(native, bname * suffix)
        ptp = Libdl.dlsym(native, pname * suffix)
        pti = Libdl.dlsym(native, iname * suffix)
        n, kd, nrhs = 4, 1, 2
        ab = zeros(T, BR(kd) * n)
        b = zeros(T, n * nrhs)
        @test lapacke_tbtrs_call(
            T, J, ptb, 301, 'U', 'N', 'N', n, kd, nrhs, ab, BR(kd), b, n,
            bname, :dll,
        ) == -1
        @test lapacke_tbtrs_call(
            T, J, ptb, 101, 'U', 'N', 'N', n, kd, nrhs, ab, n - 1, b, nrhs,
            bname, :dll,
        ) == -9
        @test lapacke_tbtrs_call(
            T, J, ptb, 101, 'U', 'N', 'N', n, kd, nrhs, ab, n, b, nrhs - 1,
            bname, :dll,
        ) == -11
        bad = copy(ab); bad[kd + 1] = T(NaN)
        @test lapacke_tbtrs_call(
            T, J, ptb, 102, 'U', 'N', 'N', n, kd, nrhs, bad, BR(kd), b, n,
            bname, :dll,
        ) == -8
        ap = zeros(T, n * (n + 1) ÷ 2)
        @test lapacke_tptrs_call(
            T, J, ptp, 301, 'U', 'N', 'N', n, nrhs, ap, b, n, pname, :dll,
        ) == -1
        @test lapacke_tptrs_call(
            T, J, ptp, 101, 'U', 'N', 'N', n, nrhs, ap, b, nrhs - 1, pname, :dll,
        ) == -9
        badp = copy(ap); badp[1] = T(NaN)
        @test lapacke_tptrs_call(
            T, J, ptp, 102, 'U', 'N', 'N', n, nrhs, badp, b, n, pname, :dll,
        ) == -7
        @test lapacke_tptri_call(
            T, J, pti, 301, 'U', 'N', n, ap, iname, :dll,
        ) == -1
        @test lapacke_tptri_call(
            T, J, pti, 102, 'U', 'N', n, badp, iname, :dll,
        ) == -5
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
