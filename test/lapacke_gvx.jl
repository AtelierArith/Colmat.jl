using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function gvx_evx_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, a, lda,
                              vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$J}),
                      layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
        function gvx_dense_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                              Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}),
                      layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
        function gvx_band_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, ka, kb,
                               ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R,
                              $J, $J, $R, Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}),
                      layout, jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
        function gvx_packed_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                 ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$J}),
                      layout, itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
            end
        end
    end
    if T <: Complex
        @eval begin
            function gvx_evx_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, a, lda,
                                  vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, rw, iw, ifail)
                end
            end
            function gvx_dense_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                    a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                                  Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz,
                          wk, lw, rw, iw, ifail)
                end
            end
            function gvx_band_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, ka, kb,
                                   ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R,
                                  $J, $J, $R, Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z,
                          ldz, wk, rw, iw, ifail)
                end
            end
            function gvx_packed_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                     ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, Ptr{$R}, Ptr{$J}, Ptr{$J}),
                          layout, itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                end
            end
        end
    else
        @eval begin
            function gvx_evx_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, a, lda,
                                  vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, iw, ifail)
                end
            end
            function gvx_dense_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                    a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, wk, lw, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R, $J, $J, $R,
                                  Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}),
                          layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz,
                          wk, lw, iw, ifail)
                end
            end
            function gvx_band_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, uplo, n, ka, kb,
                                   ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R,
                                  $J, $J, $R, Ptr{$J}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, m, w, z,
                          ldz, wk, iw, ifail)
                end
            end
            function gvx_packed_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, itype, jobz, range, uplo, n,
                                     ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, wk, rw, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, Cchar, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                                  Ptr{$T}, $J, Ptr{$T}, Ptr{$J}, Ptr{$J}),
                          layout, itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, m, w, z, ldz, wk, iw, ifail)
                end
            end
        end
    end
end

function gx_band_col(m, uplo, kd)
    n = size(m, 1); out = fill(eltype(m)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[kd + i - j + (j - 1) * (kd + 1) + 1] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[i - j + (j - 1) * (kd + 1) + 1] = m[i, j]; end
    end
    out
end
function gx_band_row(m, uplo, kd, ld)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * ld)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[(kd + i - j) * ld + j] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[(i - j) * ld + j] = m[i, j]; end
    end
    out
end
function gx_pack_col(m, uplo)
    n = size(m, 1); out = eltype(m)[]
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); push!(out, m[i, j]); end
    out
end
function gx_pack_row(m, uplo)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * (n + 1) ÷ 2)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        out[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j] = m[i, j]
    end
    out
end
gx_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
gx_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
gx_fromcol(v, n, ld, cols) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:cols]
gx_fromrow(v, n, ld, cols) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:cols]

@testset "LAPACKE SYEVX/HEEVX and SYGVX/HEGVX/SBGVX/HBGVX/SPGVX/HPGVX" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, hf) in ((Float32, Float32, "s", "sy"), (Float64, Float64, "d", "sy"),
                               (ComplexF32, Float32, "c", "he"), (ComplexF64, Float64, "z", "he")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), shape in (:evx, :dense, :band, :packed),
        range in ('A', 'V', 'I'), jobz in ('N', 'V'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fam = shape === :evx ? hf * "evx" : shape === :dense ? hf * "gvx" :
              shape === :band ? (T <: Complex ? "hbgvx" : "sbgvx") : (T <: Complex ? "hpgvx" : "spgvx")
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        ka, kb = min(2, n - 1), min(1, n - 1)
        rng = MersenneTwister(9700 + n)
        f = randn(rng, T, n, n); h = f + f'
        g = randn(rng, T, n, n); gm = g + g'
        if shape in (:band,)
            h = T[abs(i - j) <= ka ? h[i, j] : zero(T) for i in 1:n, j in 1:n]
            gm = T[abs(i - j) <= kb ? gm[i, j] : zero(T) for i in 1:n, j in 1:n]
        end
        A = Matrix(T <: Complex ? Hermitian(h, uplo == 'L' ? :L : :U) : Symmetric(h, uplo == 'L' ? :L : :U))
        B = shape === :evx ? Matrix{T}(I, n, n) :
            Matrix(T <: Complex ? Hermitian(gm + T(2 * n + 4) * I, uplo == 'L' ? :L : :U) :
                   Symmetric(gm + T(2 * n + 4) * I, uplo == 'L' ? :L : :U))
        ev = sort(real.(eigvals(B \ A)))
        il, iu = n == 1 ? (1, 1) : (2, 4)
        vl, vu = n == 1 ? (ev[1] - 1, ev[1] + 1) : ((ev[1] + ev[2]) / 2, (ev[4] + ev[5]) / 2)
        idx = range == 'A' ? (1:n) : range == 'I' ? (il:iu) : (n == 1 ? (1:1) : (2:4))
        expected = ev[idx]
        tol = 50000 * n * eps(R) * max(1, maximum(abs, ev))
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            if shape === :band
                lda = layout == 102 ? ka + 1 : n + 2; ldb = layout == 102 ? kb + 1 : n + 2
                a = layout == 102 ? gx_band_col(A, uplo, ka) : gx_band_row(A, uplo, ka, lda)
                b = layout == 102 ? gx_band_col(B, uplo, kb) : gx_band_row(B, uplo, kb, ldb)
            elseif shape === :packed
                lda = 1; ldb = 1
                a = layout == 102 ? gx_pack_col(A, uplo) : gx_pack_row(A, uplo)
                b = layout == 102 ? gx_pack_col(B, uplo) : gx_pack_row(B, uplo)
            else
                lda = ld; ldb = ld
                a = layout == 102 ? gx_colvec(A, ld) : gx_rowmajor(A, ld)
                b = layout == 102 ? gx_colvec(B, ld) : gx_rowmajor(B, ld)
            end
            q = fill(T(NaN), max(n * ld, 1)); z = fill(T(NaN), max(n * ld, 1))
            w = fill(R(NaN), n); mm = J[-1]; ifail = zeros(J, n)
            wk = fill(T(NaN), 8n + 8); rw = fill(R(NaN), 7n + 8); iw = zeros(J, 5n + 8)
            lw = J(length(wk))
            qwk = [T(NaN)]
            info = if shape === :evx
                work ? (begin
                    gvx_evx_work(T, R, J, p, name, route, layout, jobz, range, uplo, J(n), a, J(lda), R(vl), R(vu), J(il), J(iu),
                                 R(0), copy(mm), copy(w), copy(z), J(ld), qwk, J(-1), rw, iw, ifail)
                    gvx_evx_work(T, R, J, p, name, route, layout, jobz, range, uplo, J(n), a, J(lda), R(vl), R(vu), J(il), J(iu),
                                 R(0), mm, w, z, J(ld), wk, lw, rw, iw, ifail)
                end) : gvx_evx_high(T, R, J, p, name, route, layout, jobz, range, uplo, J(n), a, J(lda), R(vl), R(vu),
                                     J(il), J(iu), R(0), mm, w, z, J(ld), ifail)
            elseif shape === :dense
                work ? (begin
                    gvx_dense_work(T, R, J, p, name, route, layout, J(1), jobz, range, uplo, J(n), copy(a), J(lda), copy(b), J(ldb),
                                   R(vl), R(vu), J(il), J(iu), R(0), copy(mm), copy(w), copy(z), J(ld), qwk, J(-1), rw, iw, ifail)
                    gvx_dense_work(T, R, J, p, name, route, layout, J(1), jobz, range, uplo, J(n), a, J(lda), b, J(ldb),
                                   R(vl), R(vu), J(il), J(iu), R(0), mm, w, z, J(ld), wk, lw, rw, iw, ifail)
                end) : gvx_dense_high(T, R, J, p, name, route, layout, J(1), jobz, range, uplo, J(n), a, J(lda), b, J(ldb),
                                       R(vl), R(vu), J(il), J(iu), R(0), mm, w, z, J(ld), ifail)
            elseif shape === :band
                work ? gvx_band_work(T, R, J, p, name, route, layout, jobz, range, uplo, J(n), J(ka), J(kb), a, J(lda), b, J(ldb),
                                     q, J(ld), R(vl), R(vu), J(il), J(iu), R(0), mm, w, z, J(ld), wk, rw, iw, ifail) :
                       gvx_band_high(T, R, J, p, name, route, layout, jobz, range, uplo, J(n), J(ka), J(kb), a, J(lda), b, J(ldb),
                                     q, J(ld), R(vl), R(vu), J(il), J(iu), R(0), mm, w, z, J(ld), ifail)
            else
                work ? gvx_packed_work(T, R, J, p, name, route, layout, J(1), jobz, range, uplo, J(n), a, b, R(vl), R(vu),
                                       J(il), J(iu), R(0), mm, w, z, J(ld), wk, rw, iw, ifail) :
                       gvx_packed_high(T, R, J, p, name, route, layout, J(1), jobz, range, uplo, J(n), a, b, R(vl), R(vu),
                                       J(il), J(iu), R(0), mm, w, z, J(ld), ifail)
            end
            @test info == 0
            @test mm[1] == length(expected)
            @test isapprox(w[1:mm[1]], expected; rtol=tol, atol=tol)
            if jobz == 'V' && mm[1] > 0
                x = layout == 102 ? gx_fromcol(z, n, ld, mm[1]) : gx_fromrow(z, n, ld, mm[1])
                @test isapprox(A * x, B * x * Diagonal(T.(w[1:mm[1]])); rtol=tol, atol=tol)
                @test isapprox(x' * B * x, Matrix{T}(I, mm[1], mm[1]); rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE selected generalized diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        cx = T <: Complex
        n_evx = "LAPACKE_" * prefix * (cx ? "heevx" : "syevx"); n_dns = "LAPACKE_" * prefix * (cx ? "hegvx" : "sygvx")
        n_bnd = "LAPACKE_" * prefix * (cx ? "hbgvx" : "sbgvx"); n_pk = "LAPACKE_" * prefix * (cx ? "hpgvx" : "spgvx")
        p_evx = Libdl.dlsym(native, n_evx * suffix); p_dns = Libdl.dlsym(native, n_dns * suffix)
        p_bnd = Libdl.dlsym(native, n_bnd * suffix); p_pk = Libdl.dlsym(native, n_pk * suffix)
        a = zeros(T, 24); b = zeros(T, 24); q = zeros(T, 16); z = zeros(T, 16); w = zeros(R, 4); mm = J[0]; ifail = zeros(J, 4)
        bad = copy(a); bad[5] = T(NaN)      # (1,2) lies in the upper triangle
        nan = R(NaN); zr = R(0); one = R(1)
        ev(layout, lda, ldz, aa; vl=zr, abstol=zr, rg='A') = gvx_evx_high(T, R, J, p_evx, n_evx, :dll, layout, 'V', rg, 'U', J(4),
            aa, J(lda), vl, one, J(1), J(4), abstol, mm, w, z, J(ldz), ifail)
        @test ev(301, 4, 4, a) == -1
        @test ev(101, 3, 4, a) == -7
        @test ev(101, 4, 3, a) == -16
        @test ev(102, 4, 4, bad) == -6
        @test ev(102, 4, 4, a; abstol=nan) == -12
        @test ev(102, 4, 4, a; rg='V', vl=nan) == -8
        dn(layout, lda, ldb, ldz, aa, bb) = gvx_dense_high(T, R, J, p_dns, n_dns, :dll, layout, J(1), 'V', 'A', 'U', J(4), aa,
            J(lda), bb, J(ldb), zr, one, J(1), J(4), zr, mm, w, z, J(ldz), ifail)
        @test dn(301, 4, 4, 4, a, b) == -1
        @test dn(101, 3, 4, 4, a, b) == -8
        @test dn(101, 4, 3, 4, a, b) == -10
        @test dn(101, 4, 4, 3, a, b) == -19
        @test dn(102, 4, 4, 4, bad, b) == -7
        @test dn(102, 4, 4, 4, a, bad) == -9
        bn(layout, ldab, ldbb, ldq, ldz, aa, bb) = gvx_band_high(T, R, J, p_bnd, n_bnd, :dll, layout, 'V', 'A', 'U', J(4), J(1), J(1),
            aa, J(ldab), bb, J(ldbb), q, J(ldq), zr, one, J(1), J(4), zr, mm, w, z, J(ldz), ifail)
        @test bn(301, 2, 2, 4, 4, a, b) == -1
        @test bn(101, 3, 4, 4, 4, a, b) == -9
        @test bn(101, 4, 3, 4, 4, a, b) == -11
        @test bn(101, 4, 4, 3, 4, a, b) == -13
        @test bn(101, 4, 4, 4, 3, a, b) == -22
        @test bn(102, 2, 2, 4, 4, bad, b) == -8
        @test bn(102, 2, 2, 4, 4, a, bad) == -10
        pk(layout, ldz, aa, bb) = gvx_packed_high(T, R, J, p_pk, n_pk, :dll, layout, J(1), 'V', 'A', 'U', J(4), aa, bb, zr, one,
            J(1), J(4), zr, mm, w, z, J(ldz), ifail)
        @test pk(301, 4, a, b) == -1
        @test pk(101, 3, a, b) == -17
        @test pk(102, 4, bad, b) == -7
        @test pk(102, 4, a, bad) == -8
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
