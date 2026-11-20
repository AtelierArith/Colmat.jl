using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function sx_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobu, jobvt, range, m, n, a, lda,
                         vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, superb)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, $R, $R, $J, $J, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}),
                      layout, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, superb)
            end
        end
    end
    if T <: Complex
        @eval function sx_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobu, jobvt, range, m, n, a, lda,
                               vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, wk, lw, rw, iw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, $R, $R, $J, $J, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$J}),
                      layout, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, wk, lw, rw, iw)
            end
        end
    else
        @eval function sx_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobu, jobvt, range, m, n, a, lda,
                               vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, wk, lw, rw, iw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, $R, $R, $J, $J, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}),
                      layout, jobu, jobvt, range, m, n, a, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, wk, lw, iw)
            end
        end
    end
end

sx_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
sx_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
sx_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
sx_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]

@testset "LAPACKE GESVDX" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n) in ((6, 4), (4, 6), (1, 1)), range in ('A', 'V', 'I'), vec in (false, true),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "gesvdx" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        k = min(m, n)
        A0 = randn(MersenneTwister(9300 + m + n), T, m, n)
        sv = svdvals(A0)   # descending
        il, iu = k == 1 ? (1, 1) : (2, 3)
        vl, vu = k == 1 ? (sv[1] - 1, sv[1] + 1) : ((sv[3] + sv[4 > k ? 3 : 4]) / 2 - (k >= 4 ? 0 : 0), sv[1] + 1)
        expected = range == 'A' ? sv : range == 'I' ? sv[il:iu] : sv[sv .> vl .&& sv .<= vu]
        jobu = vec ? 'V' : 'N'
        tol = 5000 * max(m, n) * eps(R) * max(1, sv[1])
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2
            ldu = layout == 102 ? m : k + 2
            ldvt = layout == 102 ? k : n + 2
            a = layout == 102 ? sx_colvec(A0, lda) : sx_rowmajor(A0, lda)
            u = fill(T(NaN), max(layout == 102 ? ldu * k : m * ldu, 1))
            vt = fill(T(NaN), max(layout == 102 ? ldvt * n : k * ldvt, 1))
            ns = J[-1]; s = fill(R(NaN), k); superb = zeros(J, 12k + 12)
            wk = fill(T(NaN), 1)
            if work
                @test sx_work(T, R, J, p, name, route, layout, jobu, jobu, range, J(m), J(n), copy(a), J(lda), R(vl), R(vu), J(il), J(iu),
                              copy(ns), copy(s), copy(u), J(ldu), copy(vt), J(ldvt), wk, J(-1), fill(R(NaN), 17k * k + 17), superb) == 0
                wk = fill(T(NaN), Int(real(wk[1])))
                rw = fill(R(NaN), 17k * k + 17)
                @test sx_work(T, R, J, p, name, route, layout, jobu, jobu, range, J(m), J(n), a, J(lda), R(vl), R(vu), J(il), J(iu),
                              ns, s, u, J(ldu), vt, J(ldvt), wk, J(length(wk)), rw, superb) == 0
            else
                @test sx_high(T, R, J, p, name, route, layout, jobu, jobu, range, J(m), J(n), a, J(lda), R(vl), R(vu), J(il), J(iu),
                              ns, s, u, J(ldu), vt, J(ldvt), superb) == 0
            end
            @test ns[1] == length(expected)
            @test isapprox(s[1:ns[1]], expected; rtol=tol, atol=tol)
            if vec && ns[1] > 0
                U = layout == 102 ? sx_fromcol(u, m, ns[1], ldu) : sx_fromrow(u, m, ns[1], ldu)
                V = layout == 102 ? sx_fromcol(vt, ns[1], n, ldvt) : sx_fromrow(vt, ns[1], n, ldvt)
                @test isapprox(A0 * V', U * Diagonal(T.(s[1:ns[1]])); rtol=tol, atol=tol)
                @test isapprox(U' * U, Matrix{T}(I, ns[1], ns[1]); rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE GESVDX diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "gesvdx"; p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 24); u = zeros(T, 24); vt = zeros(T, 24); ns = J[0]; s = zeros(R, 4); superb = zeros(J, 48)
        g(layout, lda, ldu, ldvt, aa; vl=R(0), rg='A') = sx_high(T, R, J, p, name, :dll, layout, 'V', 'V', rg, J(6), J(4), aa, J(lda),
            vl, R(1), J(1), J(4), ns, s, u, J(ldu), vt, J(ldvt), superb)
        @test g(301, 6, 6, 4, a) == -1
        @test g(101, 3, 4, 4, a) == -8
        @test g(101, 4, 3, 4, a) == -16
        @test g(101, 4, 4, 3, a) == -18
        bad = copy(a); bad[1] = T(NaN)
        @test g(102, 6, 6, 4, bad) == -7
        @test g(102, 6, 6, 4, a; rg='V', vl=R(NaN)) == -9
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
