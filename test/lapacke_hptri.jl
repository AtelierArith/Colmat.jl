using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_hp_trf(::Type{$T}, ::Type{$J}, p, layout, uplo, n, ap, ipiv, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$J}), layout, uplo, n, ap, ipiv)
        end
    end
    @eval function lapacke_hp_tri(::Type{$T}, ::Type{$J}, p, work, layout, uplo, n, ap, ipiv, w, name, route)
        lapacke288_selected_entry(name, $J, route) do
            work ? ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$J}, Ptr{$T}),
                         layout, uplo, n, ap, ipiv, w) :
                   ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$J}),
                         layout, uplo, n, ap, ipiv)
        end
    end
    @eval function lapacke_hp_trs(::Type{$T}, ::Type{$J}, p, layout, uplo, n, nrhs, ap, ipiv, b, ldb, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$J}, Ptr{$T}, $J),
                  layout, uplo, n, nrhs, ap, ipiv, b, ldb)
        end
    end
    @eval function lapacke_hp_sv(::Type{$T}, ::Type{$J}, p, layout, uplo, n, nrhs, ap, ipiv, b, ldb, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$J}, Ptr{$T}, $J),
                  layout, uplo, n, nrhs, ap, ipiv, b, ldb)
        end
    end
    @eval function lapacke_sy_trf(::Type{$T}, ::Type{$J}, p, layout, uplo, n, a, lda, ipiv, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}), layout, uplo, n, a, lda, ipiv)
        end
    end
    @eval function lapacke_sy_trs2(::Type{$T}, ::Type{$J}, p, work, layout, uplo, n, nrhs, a, lda, ipiv, b, ldb, w, name, route)
        lapacke288_selected_entry(name, $J, route) do
            work ? ccall(p, $J, (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J, Ptr{$T}),
                         layout, uplo, n, nrhs, a, lda, ipiv, b, ldb, w) :
                   ccall(p, $J, (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                         layout, uplo, n, nrhs, a, lda, ipiv, b, ldb)
        end
    end
end

function hp_pack_col(m, uplo)
    n = size(m, 1); out = eltype(m)[]
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); push!(out, m[i, j]); end
    out
end
function hp_pack_row(m, uplo)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * (n + 1) ÷ 2)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        out[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j] = m[i, j]
    end
    out
end
function hp_unpack_col(ap, n, uplo)
    m = zeros(eltype(ap), n, n); k = 0
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); k += 1; m[i, j] = ap[k]; end
    m
end
function hp_unpack_row(ap, n, uplo)
    m = zeros(eltype(ap), n, n)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        m[i, j] = ap[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j]
    end
    m
end
hp_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
hp_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
hp_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
hp_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]
hp_matrix(T, n, uplo, seed) = begin
    f = randn(MersenneTwister(seed), T, n, n)
    h = f * f' + T(n) * Matrix{T}(I, n, n)
    Matrix(T <: Complex ? Hermitian(h, uplo == 'L' ? :L : :U) : Symmetric(h, uplo == 'L' ? :L : :U))
end

@testset "LAPACKE packed SPTRI/HPTRF/HPTRS/HPSV/HPTRI" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix, fam) in ((Float32, "s", "sp"), (Float64, "d", "sp"),
                             (ComplexF32, "c", "hp"), (ComplexF64, "z", "hp")),
        J in (Int32, Int64), n in (0, 1, 4), uplo in ('U', 'L'),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        w = work ? "_work" : ""
        names = Dict(k => "LAPACKE_" * prefix * fam * k * w for k in ("trf", "trs", "sv", "tri"))
        for k in ("trf", "trs", "sv", "tri")
            @test BLAS.lbt_get_forward(names[k], interface) == Libdl.dlsym(native, names[k] * suffix)
        end
        lib = route === :dll ? native : trampoline
        ptr(k) = Libdl.dlsym(lib, names[k] * suffix)
        herm = hp_matrix(T, n, uplo, 9100 + n)
        tol = 20000 * max(n, 1) * eps(real(T))
        ap = Dict(102 => hp_pack_col(herm, uplo), 101 => hp_pack_row(herm, uplo))
        ipiv = Dict(102 => zeros(J, max(n, 1)), 101 => zeros(J, max(n, 1)))
        for layout in (102, 101)
            @test lapacke_hp_trf(T, J, ptr("trf"), layout, uplo, n, ap[layout], ipiv[layout],
                                 names["trf"], route) == 0
        end
        @test isequal(ipiv[101], ipiv[102])
        fac_col = hp_unpack_col(ap[102], n, uplo); fac_row = hp_unpack_row(ap[101], n, uplo)
        @test isapprox(fac_col, fac_row; rtol=tol, atol=tol)
        # solve with the shared factors
        nrhs = 2
        b0 = randn(MersenneTwister(9200 + n), T, n, nrhs)
        for layout in (102, 101)
            ld = layout == 102 ? max(n, 1) : nrhs + 2
            b = layout == 102 ? hp_colvec(b0, ld) : hp_rowmajor(b0, ld)
            @test lapacke_hp_trs(T, J, ptr("trs"), layout, uplo, n, nrhs, copy(ap[layout]),
                                 ipiv[layout], b, ld, names["trs"], route) == 0
            x = layout == 102 ? hp_fromcol(b, n, nrhs, ld) : hp_fromrow(b, n, nrhs, ld)
            @test isapprox(herm * x, b0; rtol=tol * norm(b0), atol=tol * norm(b0))
            # driver
            ap2 = layout == 102 ? hp_pack_col(herm, uplo) : hp_pack_row(herm, uplo)
            ip2 = zeros(J, max(n, 1))
            b2 = layout == 102 ? hp_colvec(b0, ld) : hp_rowmajor(b0, ld)
            @test lapacke_hp_sv(T, J, ptr("sv"), layout, uplo, n, nrhs, ap2, ip2, b2, ld,
                                names["sv"], route) == 0
            x2 = layout == 102 ? hp_fromcol(b2, n, nrhs, ld) : hp_fromrow(b2, n, nrhs, ld)
            @test isapprox(herm * x2, b0; rtol=tol * norm(b0), atol=tol * norm(b0))
        end
        # inverse from the shared factors
        expected = n == 0 ? herm : inv(herm)
        itol = 20000 * max(n, 1) * eps(real(T)) * max(1, opnorm(expected, 1))
        wk = fill(T(NaN), max(n, 1) + 2)
        for layout in (102, 101)
            a = copy(ap[layout])
            @test lapacke_hp_tri(T, J, ptr("tri"), work, layout, uplo, n, a, ipiv[layout], wk,
                                 names["tri"], route) == 0
            inv_m = layout == 102 ? hp_unpack_col(a, n, uplo) : hp_unpack_row(a, n, uplo)
            for j in 1:n, i in 1:n
                (uplo == 'U' ? i <= j : i >= j) || continue
                @test isapprox(inv_m[i, j], expected[i, j]; atol=itol, rtol=itol)
            end
        end
    end
end

@testset "LAPACKE SYTRS2/HETRS2" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix, fam) in ((Float32, "s", "sy"), (Float64, "d", "sy"),
                             (ComplexF32, "c", "he"), (ComplexF64, "z", "he")),
        J in (Int32, Int64), n in (0, 1, 4), uplo in ('U', 'L'),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * fam * "trf"
        sname = "LAPACKE_" * prefix * fam * "trs2" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(sname, interface) == Libdl.dlsym(native, sname * suffix)
        lib = route === :dll ? native : trampoline
        herm = hp_matrix(T, n, uplo, 9300 + n)
        nrhs = 2
        b0 = randn(MersenneTwister(9400 + n), T, n, nrhs)
        tol = 20000 * max(n, 1) * eps(real(T)) * max(norm(b0), 1)
        for layout in (102, 101)
            lda = layout == 102 ? max(n, 1) : n + 2
            ldb = layout == 102 ? max(n, 1) : nrhs + 2
            a = layout == 102 ? vcat(vec(copy(herm)), fill(T(NaN), n == 0 ? 1 : 0)) : hp_rowmajor(herm, lda)
            ipiv = zeros(J, max(n, 1))
            @test lapacke_sy_trf(T, J, Libdl.dlsym(lib, fname * suffix), layout, uplo, n, a, lda, ipiv,
                                 fname, route) == 0
            before = copy(a)
            b = layout == 102 ? hp_colvec(b0, ldb) : hp_rowmajor(b0, ldb)
            wk = fill(T(NaN), max(n, 1) + 2)
            @test lapacke_sy_trs2(T, J, Libdl.dlsym(lib, sname * suffix), work, layout, uplo, n, nrhs,
                                  a, lda, ipiv, b, ldb, wk, sname, route) == 0
            @test isequal(a, before)  # the factor is read only
            x = layout == 102 ? hp_fromcol(b, n, nrhs, ldb) : hp_fromrow(b, n, nrhs, ldb)
            @test isapprox(herm * x, b0; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE HPTRI/SYTRS2 diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix, fam) in ((Float64, "d", "sp"), (ComplexF64, "z", "hp")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * fam * "tri"
        p = Libdl.dlsym(native, name * suffix)
        ap = zeros(T, 6); ipiv = ones(J, 3)
        @test lapacke_hp_tri(T, J, p, false, 301, 'U', 3, ap, ipiv, T[], name, :dll) == -1
        bad = copy(ap); bad[2] = T(NaN)
        @test lapacke_hp_tri(T, J, p, false, 102, 'U', 3, bad, ipiv, T[], name, :dll) == -4
        if T <: Complex
            diag = copy(ap); diag[1] = T(1, NaN)   # imaginary part of a Hermitian diagonal
            @test lapacke_hp_tri(T, J, p, false, 102, 'U', 3, diag, ipiv, T[], name, :dll) != -4
        end
    end
    for (T, prefix, fam) in ((Float64, "d", "sy"), (ComplexF64, "z", "he")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * fam * "trs2"
        p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 9); ipiv = ones(J, 3); b = zeros(T, 6)
        @test lapacke_sy_trs2(T, J, p, false, 301, 'U', 3, 2, a, 3, ipiv, b, 2, T[], name, :dll) == -1
        @test lapacke_sy_trs2(T, J, p, false, 101, 'U', 3, 2, a, 2, ipiv, b, 2, T[], name, :dll) == -6
        @test lapacke_sy_trs2(T, J, p, false, 101, 'U', 3, 2, a, 3, ipiv, b, 1, T[], name, :dll) == -9
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
