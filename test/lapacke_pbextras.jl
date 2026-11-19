using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_pbequ_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, uplo, n, kd, ab,
        ldab, s::Vector{$R}, cond, amax, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}),
                layout, uplo, n, kd, ab, ldab, s, cond, amax)
        end
    end
    @eval function lapacke_pbstf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, kb, bb, ldbb,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, $J, Ptr{$T}, $J),
                  layout, uplo, n, kb, bb, ldbb)
        end
    end
    @eval function lapacke_pbsv_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, kd, nrhs, ab, ldab,
        b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                layout, uplo, n, kd, nrhs, ab, ldab, b, ldb)
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

@testset "LAPACKE PBEQU/PBSTF/PBSV band layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (n, kd) in ((3, 1), (5, 2)), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        ename = "LAPACKE_" * prefix * "pbequ" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "pbstf" * (save ? "_work" : "")
        vname = "LAPACKE_" * prefix * "pbsv" * (save ? "_work" : "")
        for nm in (ename, sname, vname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        peq = Libdl.dlsym(route === :dll ? native : trampoline, ename * suffix)
        pst = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix)
        rng = MersenneTwister(6800 + n + kd)
        band = zeros(T, n, n)
        for j in 1:n, i in 1:n
            if uplo == 'U'
                0 <= j - i <= kd || continue
                band[i, j] = i == j ? T(2 + i) : T(0.3) * randn(rng, T)
            else
                0 <= i - j <= kd || continue
                band[i, j] = i == j ? T(2 + i) : T(0.3) * randn(rng, T)
            end
        end
        m = band * band' + T(n) * Matrix{T}(I, n, n)  # well conditioned SPD
        # PBEQU on both layouts.
        abcol = band_col(m, uplo, kd)
        scol = zeros(R, n); ccol = Ref(R(-1)); acol = Ref(R(-1))
        @test lapacke_pbequ_call(
            T, R, J, peq, 102, uplo, n, kd, abcol, BR(kd), scol, ccol, acol,
            ename, route,
        ) == 0
        ld = n + 2
        abrow = band_row(m, uplo, kd, ld)
        srow = zeros(R, n); crow = Ref(R(-1)); arow = Ref(R(-1))
        @test lapacke_pbequ_call(
            T, R, J, peq, 101, uplo, n, kd, abrow, ld, srow, crow, arow,
            ename, route,
        ) == 0
        @test isapprox(srow, scol; rtol=1e-5, atol=1e-5)
        @test isapprox(crow[], ccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(arow[], acol[]; rtol=1e-5, atol=1e-5)
        # PBSTF on the SPD band: the split factor matches across layouts.
        bbcol = band_col(m, uplo, kd)
        @test lapacke_pbstf_call(
            T, J, pst, 102, uplo, n, kd, bbcol, BR(kd), sname, route,
        ) == 0
        bbrow = band_row(m, uplo, kd, ld)
        @test lapacke_pbstf_call(
            T, J, pst, 101, uplo, n, kd, bbrow, ld, sname, route,
        ) == 0
        # The split factor squares back to the band diagonal block.
        dcol = uplo == 'U' ?
            [bbcol[kd + (j - 1) * BR(kd) + 1] for j in 1:n] :
            [bbcol[(j - 1) * BR(kd) + 1] for j in 1:n]
        drow = [bbrow[kd * ld + j] for j in 1:n]
        if uplo == 'U'
            @test isapprox(drow, dcol; rtol=1e-5, atol=1e-5)
        else
            drow0 = [bbrow[j] for j in 1:n]
            @test isapprox(drow0, dcol; rtol=1e-5, atol=1e-5)
        end
        # PBSV end to end.
        nrhs = 2
        b0 = randn(MersenneTwister(6900 + n), T, n, nrhs)
        abcol2 = band_col(m, uplo, kd)
        bcol = vec(copy(b0))
        @test lapacke_pbsv_call(
            T, J, psv, 102, uplo, n, kd, nrhs, abcol2, BR(kd), bcol, n,
            vname, route,
        ) == 0
        abrow2 = band_row(m, uplo, kd, ld)
        brow = row_major(b0, nrhs + 2)
        @test lapacke_pbsv_call(
            T, J, psv, 101, uplo, n, kd, nrhs, abrow2, ld, brow, nrhs + 2,
            vname, route,
        ) == 0
        tol = 4000 * eps(R)
        @test isapprox(
            from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 4000 * n * eps(R) * norm(b0)
        @test isapprox(m * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE PBEQU/PBSTF/PBSV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        ename = "LAPACKE_" * prefix * "pbequ"
        sname = "LAPACKE_" * prefix * "pbstf"
        vname = "LAPACKE_" * prefix * "pbsv"
        peq = Libdl.dlsym(native, ename * suffix)
        pst = Libdl.dlsym(native, sname * suffix)
        psv = Libdl.dlsym(native, vname * suffix)
        n, kd, nrhs = 4, 1, 2
        ab = zeros(T, BR(kd) * n)
        s = zeros(R, n); cond = Ref(R(0)); amax = Ref(R(0))
        @test lapacke_pbequ_call(
            T, R, J, peq, 301, 'U', n, kd, ab, BR(kd), s, cond, amax,
            ename, :dll,
        ) == -1
        @test lapacke_pbequ_call(
            T, R, J, peq, 101, 'U', n, kd, ab, n - 1, s, cond, amax,
            ename, :dll,
        ) == -6
        @test lapacke_pbequ_call(
            T, R, J, peq, 102, 'U', n, kd, ab, kd, s, cond, amax,
            ename, :dll,
        ) == -6
        bad = copy(ab); bad[kd + 1] = T(NaN)
        @test lapacke_pbequ_call(
            T, R, J, peq, 102, 'U', n, kd, bad, BR(kd), s, cond, amax,
            ename, :dll,
        ) == -5
        bb = zeros(T, BR(kd) * n)
        @test lapacke_pbstf_call(
            T, J, pst, 301, 'U', n, kd, bb, BR(kd), sname, :dll,
        ) == -1
        @test lapacke_pbstf_call(
            T, J, pst, 101, 'U', n, kd, bb, n - 1, sname, :dll,
        ) == -6
        badb = copy(bb); badb[kd + 1] = T(NaN)
        @test lapacke_pbstf_call(
            T, J, pst, 102, 'U', n, kd, badb, BR(kd), sname, :dll,
        ) == -5
        b = zeros(T, n * nrhs)
        @test lapacke_pbsv_call(
            T, J, psv, 301, 'U', n, kd, nrhs, ab, BR(kd), b, n, vname, :dll,
        ) == -1
        @test lapacke_pbsv_call(
            T, J, psv, 101, 'U', n, kd, nrhs, ab, n - 1, b, nrhs, vname, :dll,
        ) == -7
        @test lapacke_pbsv_call(
            T, J, psv, 101, 'U', n, kd, nrhs, ab, n, b, nrhs - 1, vname, :dll,
        ) == -9
        @test lapacke_pbsv_call(
            T, J, psv, 102, 'U', n, kd, nrhs, bad, BR(kd), b, n, vname, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
