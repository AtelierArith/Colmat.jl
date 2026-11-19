using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for J in (Int32, Int64)
    # (source type, destination type, entry) -- literal ccalls keep the ABI recorder able to trace them.
    @eval begin
        function lag2_call(::Type{Float64}, ::Type{Float32}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{Float64}, $J, Ptr{Float32}, $J), layout, m, n, a, lda, b, ldb)
            end
        end
        function lag2_call(::Type{Float32}, ::Type{Float64}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{Float32}, $J, Ptr{Float64}, $J), layout, m, n, a, lda, b, ldb)
            end
        end
        function lag2_call(::Type{ComplexF64}, ::Type{ComplexF32}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{ComplexF64}, $J, Ptr{ComplexF32}, $J), layout, m, n, a, lda, b, ldb)
            end
        end
        function lag2_call(::Type{ComplexF32}, ::Type{ComplexF64}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{ComplexF32}, $J, Ptr{ComplexF64}, $J), layout, m, n, a, lda, b, ldb)
            end
        end
    end
end
for (T, R) in ((ComplexF32, Float32), (ComplexF64, Float64)), J in (Int32, Int64)
    @eval begin
        function cr_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb, c, ldc)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$T}, $J), layout, m, n, a, lda, b, ldb, c, ldc)
            end
        end
        function cr_call_w(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb, c, ldc, wk)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$T}, $J, Ptr{$R}), layout, m, n, a, lda, b, ldb, c, ldc, wk)
            end
        end
        # LARCM: real A, complex B
        function rc_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb, c, ldc)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{$R}, $J, Ptr{$T}, $J, Ptr{$T}, $J), layout, m, n, a, lda, b, ldb, c, ldc)
            end
        end
        function rc_call_w(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, m, n, a, lda, b, ldb, c, ldc, wk)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, Ptr{$R}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}), layout, m, n, a, lda, b, ldb, c, ldc, wk)
            end
        end
    end
end
for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)), J in (Int32, Int64)
    @eval begin
        function pstrf_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, piv, rank, tol)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, $R), layout, uplo, n, a, lda, piv, rank, tol)
            end
        end
        function pstrf_call_w(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, piv, rank, tol, wk)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, $R, Ptr{$R}),
                      layout, uplo, n, a, lda, piv, rank, tol, wk)
            end
        end
    end
end

l2_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
l2_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
l2_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
l2_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]

@testset "LAPACKE LAG2C/LAG2D/LAG2S/LAG2Z" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (S, D, fam) in ((Float64, Float32, "dlag2s"), (Float32, Float64, "slag2d"),
                        (ComplexF64, ComplexF32, "zlag2c"), (ComplexF32, ComplexF64, "clag2z")),
        J in (Int32, Int64), (m, n) in ((3, 4), (4, 3), (1, 1)), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        a0 = randn(MersenneTwister(9900 + m), S, m, n)
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2; ldb = layout == 102 ? m : n + 2
            a = layout == 102 ? l2_colvec(a0, lda) : l2_rowmajor(a0, lda)
            b = fill(D(NaN), layout == 102 ? ldb * n : m * ldb)
            @test lag2_call(S, D, J, p, name, route, layout, J(m), J(n), a, J(lda), b, J(ldb)) == 0
            res = layout == 102 ? l2_fromcol(b, m, n, ldb) : l2_fromrow(b, m, n, ldb)
            @test res == D.(a0)
            if layout == 101
                for i in 1:m, j in (n + 1):ldb
                    @test isnan(b[(i - 1) * ldb + j])
                end
            end
        end
    end
    # overflow reports a positive INFO
    big = [Float64(floatmax(Float32)) * 4]
    out = [Float32(0)]
    name = "LAPACKE_dlag2s"
    p = Libdl.dlsym(native, name)
    @test lag2_call(Float64, Float32, Int32, p, name, :dll, 102, Int32(1), Int32(1), big, Int32(1), out, Int32(1)) > 0
end

@testset "LAPACKE LACRM/LARCM" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n) in ((3, 4), (4, 2), (1, 1)), fam in ("lacrm", "larcm"), work in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9950 + m + n)
        if fam == "lacrm"
            a0 = randn(rng, T, m, n); b0 = randn(rng, R, n, n); want = a0 * b0
            adims, bdims = (m, n), (n, n)
        else
            a0 = randn(rng, R, m, m); b0 = randn(rng, T, m, n); want = a0 * b0
            adims, bdims = (m, m), (m, n)
        end
        tol = 100 * eps(R) * max(1, norm(want))
        for layout in (102, 101)
            lda = layout == 102 ? adims[1] : adims[2] + 2
            ldb = layout == 102 ? bdims[1] : bdims[2] + 2
            ldc = layout == 102 ? m : n + 2
            a = layout == 102 ? l2_colvec(a0, lda) : l2_rowmajor(a0, lda)
            b = layout == 102 ? l2_colvec(b0, ldb) : l2_rowmajor(b0, ldb)
            c = fill(T(NaN), layout == 102 ? ldc * n : m * ldc)
            wk = fill(R(NaN), 2m * n + 2)
            args = (layout, J(m), J(n), a, J(lda), b, J(ldb), c, J(ldc))
            info = fam == "lacrm" ?
                (work ? cr_call_w(T, R, J, p, name, route, args..., wk) : cr_call(T, R, J, p, name, route, args...)) :
                (work ? rc_call_w(T, R, J, p, name, route, args..., wk) : rc_call(T, R, J, p, name, route, args...))
            @test info == 0
            res = layout == 102 ? l2_fromcol(c, m, n, ldc) : l2_fromrow(c, m, n, ldc)
            @test isapprox(res, want; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE PSTRF" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "pstrf" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9970 + n)
        f = randn(rng, T, n, n)
        spd = f * f' + T(0.1) * Matrix{T}(I, n, n)
        sym = Matrix(T <: Complex ? Hermitian(spd, uplo == 'L' ? :L : :U) : Symmetric(spd, uplo == 'L' ? :L : :U))
        tol = 100 * n * eps(R) * max(1, norm(spd))
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? l2_colvec(sym, ld) : l2_rowmajor(sym, ld)
            piv = zeros(J, n); rank = J[-1]; wk = fill(R(NaN), 2n + 2)
            info = work ? pstrf_call_w(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), piv, rank, R(-1), wk) :
                          pstrf_call(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), piv, rank, R(-1))
            @test info == 0
            @test rank[1] == n
            @test sort(piv) == collect(1:n)
            fac = layout == 102 ? l2_fromcol(a, n, n, ld) : l2_fromrow(a, n, n, ld)
            P = Matrix{T}(I, n, n)[:, piv]
            if uplo == 'U'
                U = triu(fac); @test isapprox(U' * U, P' * sym * P; rtol=tol, atol=tol)
            else
                L = tril(fac); @test isapprox(L * L', P' * sym * P; rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE aux diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_dlag2s"; p = Libdl.dlsym(native, name * suffix)
        a = zeros(Float64, 12); b = zeros(Float32, 12)
        @test lag2_call(Float64, Float32, J, p, name, :dll, 301, J(3), J(4), a, J(3), b, J(3)) == -1
        @test lag2_call(Float64, Float32, J, p, name, :dll, 101, J(3), J(4), a, J(3), b, J(4)) == -5
        @test lag2_call(Float64, Float32, J, p, name, :dll, 101, J(3), J(4), a, J(4), b, J(3)) == -7
        bad = copy(a); bad[1] = NaN
        @test lag2_call(Float64, Float32, J, p, name, :dll, 102, J(3), J(4), bad, J(3), b, J(3)) == -4
        for (T, R, prefix) in ((ComplexF64, Float64, "z"),)
            nc = "LAPACKE_" * prefix * "lacrm"; pc = Libdl.dlsym(native, nc * suffix)
            ca = zeros(T, 12); cb = zeros(R, 16); cc = zeros(T, 12)
            @test cr_call(T, R, J, pc, nc, :dll, 301, J(3), J(4), ca, J(3), cb, J(4), cc, J(3)) == -1
            @test cr_call(T, R, J, pc, nc, :dll, 101, J(3), J(4), ca, J(3), cb, J(4), cc, J(4)) == -5
            @test cr_call(T, R, J, pc, nc, :dll, 101, J(3), J(4), ca, J(4), cb, J(3), cc, J(4)) == -7
            @test cr_call(T, R, J, pc, nc, :dll, 101, J(3), J(4), ca, J(4), cb, J(4), cc, J(3)) == -9
            badc = copy(ca); badc[1] = T(NaN)
            @test cr_call(T, R, J, pc, nc, :dll, 102, J(3), J(4), badc, J(3), cb, J(4), cc, J(3)) == -4
            ns = "LAPACKE_zpstrf"; ps = Libdl.dlsym(native, ns * suffix)
            sa = zeros(T, 16); piv = zeros(J, 4); rk = J[0]
            @test pstrf_call(T, R, J, ps, ns, :dll, 301, 'U', J(4), sa, J(4), piv, rk, R(-1)) == -1
            @test pstrf_call(T, R, J, ps, ns, :dll, 101, 'U', J(4), sa, J(3), piv, rk, R(-1)) == -5
            sbad = copy(sa); sbad[5] = T(NaN)
            @test pstrf_call(T, R, J, ps, ns, :dll, 102, 'U', J(4), sbad, J(4), piv, rk, R(-1)) == -4
            @test pstrf_call(T, R, J, ps, ns, :dll, 102, 'U', J(4), sa, J(4), piv, rk, R(NaN)) == -8
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
