using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    if T <: Complex
        @eval begin
            function tsn_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, ijob, wq, wz, sel, n, a, lda, b, ldb,
                              alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                                  Ptr{$T}, $J, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}),
                          layout, ijob, wq, wz, sel, n, a, lda, b, ldb, alphar, beta, q, ldq, z, ldz, m, pl, pr, dif)
                end
            end
            function tsn_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, ijob, wq, wz, sel, n, a, lda, b, ldb,
                              alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif, wk, lw, iw, liw)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                                  Ptr{$T}, $J, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}, $J),
                          layout, ijob, wq, wz, sel, n, a, lda, b, ldb, alphar, beta, q, ldq, z, ldz, m, pl, pr, dif,
                          wk, lw, iw, liw)
                end
            end
        end
    else
        @eval begin
            function tsn_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, ijob, wq, wz, sel, n, a, lda, b, ldb,
                              alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T},
                                  Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}),
                          layout, ijob, wq, wz, sel, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif)
                end
            end
            function tsn_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, ijob, wq, wz, sel, n, a, lda, b, ldb,
                              alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif, wk, lw, iw, liw)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T},
                                  Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}, $J),
                          layout, ijob, wq, wz, sel, n, a, lda, b, ldb, alphar, alphai, beta, q, ldq, z, ldz, m, pl, pr, dif,
                          wk, lw, iw, liw)
                end
            end
        end
    end
end

tsn_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
tsn_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
tsn_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]
tsn_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]

@testset "LAPACKE TGSEN" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), ijob in (0, 1), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "tgsen" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9200 + n)
        a0 = triu(randn(rng, T, n, n)); b0 = triu(randn(rng, T, n, n))
        for i in 1:n
            a0[i, i] = T(i); b0[i, i] = T(1 + 0.25 * i)
        end
        selected = n == 1 ? [true] : [false, false, true, true]
        sel0 = J.(selected)
        m_expected = count(selected)
        ratios0 = [a0[i, i] / b0[i, i] for i in 1:n]
        order = vcat(findall(selected), findall(!, selected))
        tol = 5000 * eps(R) * max(n, 1)
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            pack(m) = layout == 102 ? tsn_colvec(m, ld) : tsn_rowmajor(m, ld)
            unpack(v) = layout == 102 ? tsn_fromcol(v, n, ld) : tsn_fromrow(v, n, ld)
            a = pack(a0); b = pack(b0); q = pack(Matrix{T}(I, n, n)); z = pack(Matrix{T}(I, n, n))
            alphar = fill(T(NaN), n); alphai = fill(T(NaN), n); beta = fill(T(NaN), n)
            sel = copy(sel0); m = J[-1]; pl = R[NaN]; pr = R[NaN]; dif = R[NaN, NaN]
            if work
                wk = fill(T(NaN), 1); iw = zeros(J, 1)
                @test tsn_work(T, R, J, p, name, route, layout, J(ijob), J(1), J(1), copy(sel), J(n), copy(a), J(ld), copy(b), J(ld),
                               copy(alphar), copy(alphai), copy(beta), copy(q), J(ld), copy(z), J(ld), copy(m), copy(pl), copy(pr),
                               copy(dif), wk, J(-1), iw, J(-1)) == 0
                # LAPACK's TGSYL step needs one more element than the reported minimum for the estimate jobs.
                wk = fill(T(NaN), max(Int(real(wk[1])), 1) + 2n * n + 1); iw = zeros(J, max(Int(iw[1]), 1))
                @test tsn_work(T, R, J, p, name, route, layout, J(ijob), J(1), J(1), sel, J(n), a, J(ld), b, J(ld),
                               alphar, alphai, beta, q, J(ld), z, J(ld), m, pl, pr, dif, wk, J(length(wk)), iw, J(length(iw))) == 0
            else
                @test tsn_high(T, R, J, p, name, route, layout, J(ijob), J(1), J(1), sel, J(n), a, J(ld), b, J(ld),
                               alphar, alphai, beta, q, J(ld), z, J(ld), m, pl, pr, dif) == 0
            end
            @test m[1] == m_expected
            an, bn, qn, zn = unpack(a), unpack(b), unpack(q), unpack(z)
            ratios = [an[i, i] / bn[i, i] for i in 1:n]
            @test isapprox(ratios, ratios0[order]; rtol=tol)
            @test isapprox(qn * an * zn', a0; rtol=tol * 10, atol=tol * 10)
            @test isapprox(qn * bn * zn', b0; rtol=tol * 10, atol=tol * 10)
            @test isapprox(qn' * qn, Matrix{T}(I, n, n); rtol=tol, atol=tol)
            if ijob == 1 && n > 1
                @test isfinite(pl[1]) && isfinite(pr[1])
            end
        end
    end
end

@testset "LAPACKE TGSEN diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "tgsen"; p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 16); b = zeros(T, 16); q = zeros(T, 16); z = zeros(T, 16); sel = J[1, 0, 0, 0]
        al = zeros(T, 4); ai = zeros(T, 4); be = zeros(T, 4); m = J[0]; pl = R[0]; pr = R[0]; dif = R[0, 0]
        g(layout, lda, ldb, ldq, ldz, aa, bb, qq=q, zz=z) = tsn_high(T, R, J, p, name, :dll, layout, J(0), J(1), J(1), sel, J(4), aa, J(lda),
            bb, J(ldb), al, ai, be, qq, J(ldq), zz, J(ldz), m, pl, pr, dif)
        @test g(301, 4, 4, 4, 4, a, b) == -1
        @test g(101, 3, 4, 4, 4, a, b) == -8
        @test g(101, 4, 3, 4, 4, a, b) == -10
        @test g(101, 4, 4, 3, 4, a, b) == -(T <: Complex ? 14 : 15)
        @test g(101, 4, 4, 4, 3, a, b) == -(T <: Complex ? 16 : 17)
        bad = copy(a); bad[1] = T(NaN)
        @test g(102, 4, 4, 4, 4, bad, b) == -7
        @test g(102, 4, 4, 4, 4, a, bad) == -9
        @test g(102, 4, 4, 4, 4, a, b, bad) == -(T <: Complex ? 13 : 14)
        @test g(102, 4, 4, 4, 4, a, b, q, bad) == -(T <: Complex ? 15 : 16)
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
