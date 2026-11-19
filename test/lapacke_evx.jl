using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function evx_band_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, kd,
                               ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                              Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}),
                      layout, jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
        function evx_packed_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n,
                                 ap, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$J}),
                      layout, jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
        function evx_sptrd(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, ap, d, e, tau)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$R}, Ptr{$T}), layout, uplo, n, ap, d, e, tau)
            end
        end
    end
    if T <: Complex
        @eval begin
            function evx_band_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, kd,
                                   ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                                  Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz,
                          wk, rw, iw, ifail)
                end
            end
            function evx_packed_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n,
                                     ap, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                end
            end
        end
    else
        @eval begin
            function evx_band_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, kd,
                                   ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                                  Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz,
                          wk, iw, ifail)
                end
            end
            function evx_packed_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n,
                                     ap, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, m, w, z, ldz, wk, iw, ifail)
                end
            end
        end
    end
end

function evx_band_col(m, uplo, kd)
    n = size(m, 1); out = fill(eltype(m)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[kd + i - j + (j - 1) * (kd + 1) + 1] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[i - j + (j - 1) * (kd + 1) + 1] = m[i, j]; end
    end
    out
end
function evx_band_row(m, uplo, kd, ld)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * ld)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[(kd + i - j) * ld + j] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[(i - j) * ld + j] = m[i, j]; end
    end
    out
end
function evx_pack_col(m, uplo)
    n = size(m, 1); out = eltype(m)[]
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); push!(out, m[i, j]); end
    out
end
function evx_pack_row(m, uplo)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * (n + 1) ÷ 2)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        out[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j] = m[i, j]
    end
    out
end
function evx_unpack_col(ap, n, uplo)
    m = zeros(eltype(ap), n, n); k = 0
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); k += 1; m[i, j] = ap[k]; end
    m
end
function evx_unpack_row(ap, n, uplo)
    m = zeros(eltype(ap), n, n)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        m[i, j] = ap[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j]
    end
    m
end
evx_fromcol(v, n, ld, cols) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:cols]
evx_fromrow(v, n, ld, cols) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:cols]

@testset "LAPACKE SBEVX/HBEVX/SPEVX/HPEVX" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, bf, pf) in ((Float32, Float32, "s", "sb", "sp"), (Float64, Float64, "d", "sb", "sp"),
                                   (ComplexF32, Float32, "c", "hb", "hp"), (ComplexF64, Float64, "z", "hb", "hp")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), shape in (:band, :packed), range in ('A', 'V', 'I'),
        jobz in ('N', 'V'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        base = (shape === :band ? bf : pf) * "evx"
        name = "LAPACKE_" * prefix * base * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        kd = min(2, n - 1)
        rng = MersenneTwister(9600 + n)
        f = randn(rng, T, n, n); h = f + f'
        h = T[abs(i - j) <= kd ? h[i, j] : zero(T) for i in 1:n, j in 1:n]
        herm = Matrix(T <: Complex ? Hermitian(h, uplo == 'L' ? :L : :U) : Symmetric(h, uplo == 'L' ? :L : :U))
        ev = sort(real.(eigvals(herm)))
        il, iu = n == 1 ? (1, 1) : (2, 4)
        vl, vu = n == 1 ? (ev[1] - 1, ev[1] + 1) : ((ev[1] + ev[2]) / 2, (ev[4] + ev[5]) / 2)
        idx = range == 'A' ? (1:n) : range == 'I' ? (il:iu) : (n == 1 ? (1:1) : (2:4))
        expected = ev[idx]
        tol = 5000 * n * eps(R) * max(1, maximum(abs, ev))
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            if shape === :band
                lda = layout == 102 ? kd + 1 : n + 2
                ab = layout == 102 ? evx_band_col(herm, uplo, kd) : evx_band_row(herm, uplo, kd, lda)
            else
                lda = 1
                ab = layout == 102 ? evx_pack_col(herm, uplo) : evx_pack_row(herm, uplo)
            end
            q = fill(T(NaN), max(n * ld, 1)); z = fill(T(NaN), max(n * ld, 1))
            w = fill(R(NaN), n); mm = J[-1]; ifail = zeros(J, n)
            wk = fill(T(NaN), 8n + 8); rw = fill(R(NaN), 7n + 8); iw = zeros(J, 5n + 8)
            args = (layout, jobz, range, uplo, J(n))
            if shape === :band
                info = work ?
                    evx_band_work(T, R, J, p, name, route, args..., J(kd), ab, J(lda), q, J(ld), R(vl), R(vu), J(il), J(iu),
                                  R(0), mm, w, z, J(ld), wk, rw, iw, ifail) :
                    evx_band_high(T, R, J, p, name, route, args..., J(kd), ab, J(lda), q, J(ld), R(vl), R(vu), J(il), J(iu),
                                  R(0), mm, w, z, J(ld), ifail)
            else
                info = work ?
                    evx_packed_work(T, R, J, p, name, route, args..., ab, R(vl), R(vu), J(il), J(iu), R(0), mm, w, z,
                                    J(ld), wk, rw, iw, ifail) :
                    evx_packed_high(T, R, J, p, name, route, args..., ab, R(vl), R(vu), J(il), J(iu), R(0), mm, w, z,
                                    J(ld), ifail)
            end
            @test info == 0
            @test mm[1] == length(expected)
            @test isapprox(w[1:mm[1]], expected; rtol=tol, atol=tol)
            if jobz == 'V' && mm[1] > 0
                x = layout == 102 ? evx_fromcol(z, n, ld, mm[1]) : evx_fromrow(z, n, ld, mm[1])
                @test isapprox(herm * x, x * Diagonal(T.(w[1:mm[1]])); rtol=tol, atol=tol)
                @test isapprox(x' * x, Matrix{T}(I, mm[1], mm[1]); rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE SPTRD/HPTRD" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam) in ((Float32, Float32, "s", "sptrd"), (Float64, Float64, "d", "sptrd"),
                                (ComplexF32, Float32, "c", "hptrd"), (ComplexF64, Float64, "z", "hptrd")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9650 + n)
        f = randn(rng, T, n, n)
        herm = Matrix(T <: Complex ? Hermitian(f + f', uplo == 'L' ? :L : :U) : Symmetric(f + f', uplo == 'L' ? :L : :U))
        ev = sort(real.(eigvals(herm)))
        tol = 5000 * n * eps(R) * max(1, maximum(abs, ev))
        res = Dict()
        for layout in (102, 101)
            ap = layout == 102 ? evx_pack_col(herm, uplo) : evx_pack_row(herm, uplo)
            d = fill(R(NaN), n); e = fill(R(NaN), max(n - 1, 1)); tau = fill(T(NaN), max(n - 1, 1))
            @test evx_sptrd(T, R, J, p, name, route, layout, uplo, J(n), ap, d, e, tau) == 0
            tri = n == 1 ? reshape([d[1]], 1, 1) : Matrix(SymTridiagonal(d, e[1:n-1]))
            @test isapprox(sort(real.(eigvals(tri))), ev; rtol=tol, atol=tol)
            res[layout] = (d, e, tau)
        end
        @test isapprox(res[101][1], res[102][1]; rtol=tol, atol=tol)
    end
end

@testset "LAPACKE band/packed EVX diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, bf, pf) in ((Float64, Float64, "d", "sb", "sp"), (ComplexF64, Float64, "z", "hb", "hp")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        nb = "LAPACKE_" * prefix * bf * "evx"; npk = "LAPACKE_" * prefix * pf * "evx"
        pb = Libdl.dlsym(native, nb * suffix); pp = Libdl.dlsym(native, npk * suffix)
        ab = zeros(T, 24); q = zeros(T, 16); z = zeros(T, 16); w = zeros(R, 4); mm = J[0]; ifail = zeros(J, 4)
        cb(layout, ldab, ldq, ldz, aa; vl=R(0), vu=R(1), abstol=R(0), range='A') =
            evx_band_high(T, R, J, pb, nb, :dll, layout, 'V', range, 'U', J(4), J(1), aa, J(ldab), q, J(ldq), vl, vu,
                          J(1), J(4), abstol, mm, w, z, J(ldz), ifail)
        @test cb(301, 2, 4, 4, ab) == -1
        @test cb(101, 3, 4, 4, ab) == -8
        @test cb(101, 4, 3, 4, ab) == -10
        @test cb(101, 4, 4, 3, ab) == -19
        bad = copy(ab); bad[2] = T(NaN)
        @test cb(102, 2, 4, 4, bad) == -7
        @test cb(102, 2, 4, 4, ab; abstol=R(NaN)) == -15
        @test cb(102, 2, 4, 4, ab; range='V', vl=R(NaN)) == -11
        cpk(layout, ldz, aa; vl=R(0), vu=R(1), abstol=R(0), range='A') =
            evx_packed_high(T, R, J, pp, npk, :dll, layout, 'V', range, 'U', J(4), aa, vl, vu, J(1), J(4), abstol, mm, w, z,
                            J(ldz), ifail)
        @test cpk(301, 4, ab) == -1
        @test cpk(101, 3, ab) == -15
        @test cpk(102, 4, bad) == -6
        @test cpk(102, 4, ab; abstol=R(NaN)) == -11
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
