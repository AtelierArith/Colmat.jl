using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_3arg_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, n, first, second,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}),
                  layout, transr, uplo, n, first, second)
        end
    end
    @eval function lapacke_trttf_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, n, a, lda, arf,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}),
                layout, transr, uplo, n, a, lda, arf)
        end
    end
    @eval function lapacke_tfttr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, n, arf, a, lda,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $J),
                layout, transr, uplo, n, arf, a, lda)
        end
    end
    @eval function lapacke_tpttr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, ap, a, lda, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$T}, $J),
                  layout, uplo, n, ap, a, lda)
        end
    end
    @eval function lapacke_trttp_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, a, lda, ap, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}),
                  layout, uplo, n, a, lda, ap)
        end
    end
end

function rfp_dims(transr, n)
    if uppercase(transr) == 'N'
        iseven(n) ? (n + 1, n ÷ 2) : (n, (n + 1) ÷ 2)
    else
        iseven(n) ? (n ÷ 2, n + 1) : ((n + 1) ÷ 2, n)
    end
end
function native_trttf(native, ::Type{T}, ::Type{J}, dense, transr, uplo, n) where {T,J}
    suffix = J === Int64 ? "64_" : ""
    p = Libdl.dlsym(native, (T <: Complex ? (T === ComplexF32 ? "c" : "z") :
                             (T === Float32 ? "s" : "d")) * "trttf_" * suffix)
    arf = zeros(T, n * (n + 1) ÷ 2)
    info = Ref{J}(-9)
    ccall(p, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}),
        transr, uplo, J(n), vec(copy(dense)), J(n), arf, info)
    @assert info[] == 0
    return arf
end
function rfp_row_from_column(acol, transr, n)
    (row, col) = rfp_dims(transr, n)
    m = reshape(acol, row, col)
    out = fill(eltype(acol)(NaN), row * col)
    for j in 1:col, i in 1:row
        out[(i - 1) * col + j] = m[i, j]
    end
    return out
end
function packed_col(dense, uplo)
    n = size(dense, 1)
    out = eltype(dense)[]
    if uplo == 'U'
        for j in 1:n, i in 1:j
            push!(out, dense[i, j])
        end
    else
        for j in 1:n, i in j:n
            push!(out, dense[i, j])
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
function triangle_approx(m1, m2, uplo; rtol, atol)
    n = size(m1, 1)
    for j in 1:n, i in 1:n
        (uplo == 'U' ? i <= j : i >= j) || continue
        isapprox(m1[i, j], m2[i, j]; rtol=rtol, atol=atol) || return false
    end
    return true
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

@testset "LAPACKE TRTTF/TFTTR/TFTTP/TPTTF/TPTTR/TRTTP layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 4),
        transr in ('N', T <: Complex ? 'C' : 'T'), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        names = Dict(
            "trttf" => "LAPACKE_" * prefix * "trttf" * (save ? "_work" : ""),
            "tfttr" => "LAPACKE_" * prefix * "tfttr" * (save ? "_work" : ""),
            "tfttp" => "LAPACKE_" * prefix * "tfttp" * (save ? "_work" : ""),
            "tpttf" => "LAPACKE_" * prefix * "tpttf" * (save ? "_work" : ""),
            "tpttr" => "LAPACKE_" * prefix * "tpttr" * (save ? "_work" : ""),
            "trttp" => "LAPACKE_" * prefix * "trttp" * (save ? "_work" : ""),
        )
        for (_, nm) in names
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pick(f) = Libdl.dlsym(route === :dll ? native : trampoline, names[f] * suffix)
        rng = MersenneTwister(7200 + n)
        full = randn(rng, T, n, n)
        herm = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        arfref = native_trttf(native, T, J, herm, transr, uplo, n)
        apcolref = packed_col(herm, uplo)
        # TRTTF on both layouts.
        acol = vec(copy(herm))
        arfcol = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_trttf_call(
            T, J, pick("trttf"), 102, transr, uplo, n, acol, n, arfcol,
            names["trttf"], route,
        ) == 0
        @test isequal(arfcol, arfref)
        ld = n + 2
        arow = row_major(herm, ld)
        arfrow = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_trttf_call(
            T, J, pick("trttf"), 101, transr, uplo, n, arow, ld, arfrow,
            names["trttf"], route,
        ) == 0
        @test isequal(arfrow, rfp_row_from_column(arfref, transr, n))
        # TFTTR: RFP back to full on both layouts.
        tcol = fill(T(NaN), n * n)
        @test lapacke_tfttr_call(
            T, J, pick("tfttr"), 102, transr, uplo, n, arfcol, tcol, n,
            names["tfttr"], route,
        ) == 0
        trow = fill(T(NaN), n * ld)
        @test lapacke_tfttr_call(
            T, J, pick("tfttr"), 101, transr, uplo, n, arfrow, trow, ld,
            names["tfttr"], route,
        ) == 0
        # TFTTR and TPTTR write only the selected triangle.
        tol = 10 * eps(R)
        @test triangle_approx(from_row_major(trow, n, n, ld), reshape(tcol, n, n), uplo; rtol=tol, atol=tol)
        @test triangle_approx(reshape(tcol, n, n), herm, uplo; rtol=tol, atol=tol)
        # TRTTP: full to packed on both layouts.
        apcol = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_trttp_call(
            T, J, pick("trttp"), 102, uplo, n, acol, n, apcol, names["trttp"],
            route,
        ) == 0
        @test isequal(apcol, apcolref)
        aprow = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_trttp_call(
            T, J, pick("trttp"), 101, uplo, n, arow, ld, aprow, names["trttp"],
            route,
        ) == 0
        @test isequal(aprow, permute_row_from_col(apcolref, uplo, n))
        # TPTTR: packed to full on both layouts.
        ucol = fill(T(NaN), n * n)
        @test lapacke_tpttr_call(
            T, J, pick("tpttr"), 102, uplo, n, apcol, ucol, n, names["tpttr"],
            route,
        ) == 0
        urow = fill(T(NaN), n * ld)
        @test lapacke_tpttr_call(
            T, J, pick("tpttr"), 101, uplo, n, aprow, urow, ld, names["tpttr"],
            route,
        ) == 0
        @test triangle_approx(from_row_major(urow, n, n, ld), reshape(ucol, n, n), uplo; rtol=tol, atol=tol)
        @test triangle_approx(reshape(ucol, n, n), herm, uplo; rtol=tol, atol=tol)
        # TPTTF: packed to RFP on both layouts.
        arfcol2 = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_3arg_call(
            T, J, pick("tpttf"), 102, transr, uplo, n, apcol, arfcol2,
            names["tpttf"], route,
        ) == 0
        @test isequal(arfcol2, arfref)
        arfrow2 = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_3arg_call(
            T, J, pick("tpttf"), 101, transr, uplo, n, aprow, arfrow2,
            names["tpttf"], route,
        ) == 0
        @test isequal(arfrow2, rfp_row_from_column(arfref, transr, n))
        # TFTTP: RFP to packed on both layouts.
        apcol2 = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_3arg_call(
            T, J, pick("tfttp"), 102, transr, uplo, n, arfcol, apcol2,
            names["tfttp"], route,
        ) == 0
        @test isequal(apcol2, apcolref)
        aprow2 = fill(T(NaN), n * (n + 1) ÷ 2)
        @test lapacke_3arg_call(
            T, J, pick("tfttp"), 101, transr, uplo, n, arfrow, aprow2,
            names["tfttp"], route,
        ) == 0
        @test isequal(aprow2, permute_row_from_col(apcolref, uplo, n))
    end
end

@testset "LAPACKE TRTTF/TFTTR/TFTTP/TPTTF/TPTTR/TRTTP diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        trttf = "LAPACKE_" * prefix * "trttf"
        tfttr = "LAPACKE_" * prefix * "tfttr"
        tfttp = "LAPACKE_" * prefix * "tfttp"
        tpttf = "LAPACKE_" * prefix * "tpttf"
        tpttr = "LAPACKE_" * prefix * "tpttr"
        trttp = "LAPACKE_" * prefix * "trttp"
        ptrf = Libdl.dlsym(native, trttf * suffix)
        pttr = Libdl.dlsym(native, tfttr * suffix)
        pttp = Libdl.dlsym(native, tfttp * suffix)
        pptf = Libdl.dlsym(native, tpttf * suffix)
        pptr = Libdl.dlsym(native, tpttr * suffix)
        pttp2 = Libdl.dlsym(native, trttp * suffix)
        n = 3
        a = zeros(T, n * n)
        arf = zeros(T, n * (n + 1) ÷ 2)
        ap = zeros(T, n * (n + 1) ÷ 2)
        # Layout errors.
        @test lapacke_trttf_call(T, J, ptrf, 301, 'N', 'U', n, a, n, arf, trttf, :dll) == -1
        @test lapacke_tfttr_call(T, J, pttr, 301, 'N', 'U', n, arf, a, n, tfttr, :dll) == -1
        @test lapacke_3arg_call(T, J, pttp, 301, 'N', 'U', n, arf, ap, tfttp, :dll) == -1
        @test lapacke_3arg_call(T, J, pptf, 301, 'N', 'U', n, ap, arf, tpttf, :dll) == -1
        @test lapacke_tpttr_call(T, J, pptr, 301, 'U', n, ap, a, n, tpttr, :dll) == -1
        @test lapacke_trttp_call(T, J, pttp2, 301, 'U', n, a, n, ap, trttp, :dll) == -1
        # Leading-dimension errors.
        @test lapacke_trttf_call(T, J, ptrf, 101, 'N', 'U', n, a, n - 1, arf, trttf, :dll) == -6
        @test lapacke_tfttr_call(T, J, pttr, 101, 'N', 'U', n, arf, a, n - 1, tfttr, :dll) == -7
        @test lapacke_tpttr_call(T, J, pptr, 101, 'U', n, ap, a, n - 1, tpttr, :dll) == -6
        @test lapacke_trttp_call(T, J, pttp2, 101, 'U', n, a, n - 1, ap, trttp, :dll) == -5
        # NaN errors.
        bada = copy(a); bada[1] = T(NaN)
        @test lapacke_trttf_call(T, J, ptrf, 102, 'N', 'U', n, bada, n, arf, trttf, :dll) == -5
        @test lapacke_trttp_call(T, J, pttp2, 102, 'U', n, bada, n, ap, trttp, :dll) == -4
        badarf = copy(arf); badarf[1] = T(NaN)
        @test lapacke_tfttr_call(T, J, pttr, 102, 'N', 'U', n, badarf, a, n, tfttr, :dll) == -5
        @test lapacke_3arg_call(T, J, pttp, 102, 'N', 'U', n, badarf, ap, tfttp, :dll) == -5
        badap = copy(ap); badap[1] = T(NaN)
        @test lapacke_3arg_call(T, J, pptf, 102, 'N', 'U', n, badap, arf, tpttf, :dll) == -5
        @test lapacke_tpttr_call(T, J, pptr, 102, 'U', n, badap, a, n, tpttr, :dll) == -4
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
