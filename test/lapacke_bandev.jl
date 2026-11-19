using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # shape: :band or :packed. work: false=high, true=_work. dc: divide-and-conquer workspace family.
    @eval function lapacke_bev_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, shape, dc, work, query,
                                   layout, jobz, uplo, n, kd, ab, ldab, w::Vector{$R}, z::Vector{$T}, ldz,
                                   wk::Vector{$T}, rw::Vector{$R}, iw::Vector{$J}, name, route)
        lapacke288_selected_entry(name, $J, route) do
            lq = query ? $J(-1) : $J(length(wk)); lrq = query ? $J(-1) : $J(length(rw))
            liq = query ? $J(-1) : $J(length(iw))
            if shape === :band
                if !work
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J),
                          layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz)
                elseif !dc && !($T <: Complex)
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}),
                          layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz, wk)
                elseif !dc
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$R}),
                          layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz, wk, rw)
                elseif !($T <: Complex)
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, $J),
                          layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz, wk, lq, iw, liq)
                else
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$J}, $J),
                          layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz, wk, lq, rw, lrq, iw, liq)
                end
            else
                if !work
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$T}, $J),
                          layout, jobz, uplo, n, ab, w, z, ldz)
                elseif !dc && !($T <: Complex)
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}),
                          layout, jobz, uplo, n, ab, w, z, ldz, wk)
                elseif !dc
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$R}),
                          layout, jobz, uplo, n, ab, w, z, ldz, wk, rw)
                elseif !($T <: Complex)
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, $J),
                          layout, jobz, uplo, n, ab, w, z, ldz, wk, lq, iw, liq)
                else
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$J}, $J),
                          layout, jobz, uplo, n, ab, w, z, ldz, wk, lq, rw, lrq, iw, liq)
                end
            end
        end
    end
end

be_band_col(m, uplo, kd) = begin
    n = size(m, 1); out = fill(eltype(m)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[kd + i - j + (j - 1) * (kd + 1) + 1] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[i - j + (j - 1) * (kd + 1) + 1] = m[i, j]; end
    end
    out
end
be_band_row(m, uplo, kd, ld) = begin
    n = size(m, 1); out = fill(eltype(m)(NaN), n * ld)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[(kd + i - j) * ld + j] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[(i - j) * ld + j] = m[i, j]; end
    end
    out
end
be_pack_col(m, uplo) = (n = size(m, 1); out = eltype(m)[];
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); push!(out, m[i, j]); end; out)
be_pack_row(m, uplo) = (n = size(m, 1); out = fill(eltype(m)(NaN), n * (n + 1) ÷ 2);
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        out[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j] = m[i, j]
    end; out)
be_zcol(n, ld) = fill(NaN, 0)
be_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]
be_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]

@testset "LAPACKE SBEV/HBEV/SPEV/HPEV and BEVD/PEVD" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, bf, pf) in ((Float32, Float32, "s", "sb", "sp"), (Float64, Float64, "d", "sb", "sp"),
                                   (ComplexF32, Float32, "c", "hb", "hp"), (ComplexF64, Float64, "z", "hb", "hp")),
        J in (Int32, Int64), n in (0, 1, 5), uplo in ('U', 'L'), shape in (:band, :packed),
        dc in (false, true), work in (false, true), jobz in ('N', 'V'), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        base = shape === :band ? bf : pf
        name = "LAPACKE_" * prefix * base * (dc ? "evd" : "ev") * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        kd = min(2, max(n - 1, 0))
        rng = MersenneTwister(9700 + n)
        full = randn(rng, T, n, n)
        h = full + full'
        h = shape === :band ? T[abs(i - j) <= kd ? h[i, j] : zero(T) for i in 1:n, j in 1:n] : h
        herm = Matrix(T <: Complex ? Hermitian(h, uplo == 'L' ? :L : :U) : Symmetric(h, uplo == 'L' ? :L : :U))
        expected = n == 0 ? R[] : eigvals(Hermitian(herm))
        tol = 2000 * max(n, 1) * eps(R) * max(1, maximum(abs, expected; init=R(0)))
        for layout in (102, 101)
            ld = layout == 102 ? max(n, 1) : n + 2
            lda = shape === :band ? (layout == 102 ? kd + 1 : n + 2) : 1
            ab = shape === :band ? (layout == 102 ? be_band_col(herm, uplo, kd) : be_band_row(herm, uplo, kd, lda)) :
                                    (layout == 102 ? be_pack_col(herm, uplo) : be_pack_row(herm, uplo))
            n == 0 && (ab = fill(T(NaN), 1))
            w = fill(R(NaN), max(n, 1))
            z = fill(T(NaN), max(n * ld, 1))
            wk = fill(T(NaN), max(3n, 1) + 4); rw = fill(R(NaN), max(3n, 1) + 4); iw = zeros(J, max(3 * n, 1) + 4)
            if dc && work
                qwk = [T(NaN)]; qrw = [R(NaN)]; qiw = zeros(J, 1)
                @test lapacke_bev_call(T, R, J, p, shape, dc, work, true, layout, jobz, uplo, n, kd, copy(ab), lda,
                                       copy(w), copy(z), ld, qwk, qrw, qiw, name, route) == 0
                wk = fill(T(NaN), max(Int(real(qwk[1])), 1)); iw = zeros(J, max(Int(qiw[1]), 1))
                rw = T <: Complex ? fill(R(NaN), max(Int(qrw[1]), 1)) : rw
            end
            @test lapacke_bev_call(T, R, J, p, shape, dc, work, false, layout, jobz, uplo, n, kd, ab, lda,
                                   w, z, ld, wk, rw, iw, name, route) == 0
            n == 0 && continue
            @test isapprox(w, expected; rtol=tol, atol=tol)
            if jobz == 'V'
                zm = layout == 102 ? be_fromcol(z, n, ld) : be_fromrow(z, n, ld)
                @test isapprox(Matrix(Hermitian(herm)) * zm, zm * Diagonal(T.(w)); rtol=tol, atol=tol)
                @test isapprox(zm' * zm, Matrix{T}(I, n, n); rtol=tol, atol=tol)
                if layout == 101
                    for i in 1:n, j in (n + 1):ld
                        @test isnan(z[(i - 1) * ld + j])
                    end
                end
            end
        end
    end
end

@testset "LAPACKE band/packed EVD diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, bf, pf) in ((Float64, Float64, "d", "sb", "sp"), (ComplexF64, Float64, "z", "hb", "hp")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for (shape, base) in ((:band, bf), (:packed, pf)), dc in (false, true)
            name = "LAPACKE_" * prefix * base * (dc ? "evd" : "ev")
            p = Libdl.dlsym(native, name * suffix)
            ab = zeros(T, 20); w = zeros(R, 4); z = zeros(T, 16)
            call(layout, ldab, ldz, mat) = lapacke_bev_call(T, R, J, p, shape, dc, false, false, layout, 'V', 'U', 4, 1,
                                                             mat, ldab, w, z, ldz, T[], R[], J[], name, :dll)
            @test call(301, 2, 4, ab) == -1
            if shape === :band
                @test call(101, 3, 4, ab) == -7
                @test call(101, 4, 3, ab) == -10
                bad = copy(ab); bad[2] = T(NaN)
                @test call(102, 2, 4, bad) == -6
            else
                @test call(101, 1, 3, ab) == -8
                bad = copy(ab); bad[2] = T(NaN)
                @test call(102, 1, 4, bad) == -5
            end
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
