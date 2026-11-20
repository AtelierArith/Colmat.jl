using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function sq_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, joba, jobp, jobr, jobu, jobv, m, n, a, lda,
                         s, u, ldu, v, ldv, numrank)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J,
                              Ptr{$J}),
                      layout, joba, jobp, jobr, jobu, jobv, m, n, a, lda, s, u, ldu, v, ldv, numrank)
            end
        end
        function sq_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, joba, jobp, jobr, jobu, jobv, m, n, a, lda,
                         s, u, ldu, v, ldv, numrank, iw, liw, wk, lw, rw, lrw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J,
                              Ptr{$J}, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$R}, $J),
                      layout, joba, jobp, jobr, jobu, jobv, m, n, a, lda, s, u, ldu, v, ldv, numrank, iw, liw, wk, lw, rw, lrw)
            end
        end
    end
end

sq_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
sq_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
sq_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
sq_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]

@testset "LAPACKE GESVDQ" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n) in ((6, 4), (4, 4)), (jobu, jobv) in (('N', 'N'), ('S', 'V'), ('A', 'V')), work in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "gesvdq" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        A0 = randn(MersenneTwister(9100 + m + n), T, m, n)
        sv = svdvals(A0)
        tol = 5000 * max(m, n) * eps(R) * max(1, sv[1])
        res = Dict()
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2
            ucols = jobu == 'A' ? m : n
            ldu = layout == 102 ? m : ucols + 2
            ldv = layout == 102 ? n : n + 2
            a = layout == 102 ? sq_colvec(A0, lda) : sq_rowmajor(A0, lda)
            s = fill(R(NaN), n)
            u = fill(T(NaN), max(layout == 102 ? ldu * max(m, n) : m * ldu, 1))
            v = fill(T(NaN), max(layout == 102 ? ldv * n : n * ldv, 1))
            nr = J[-1]
            if work
                qi = J[0]; qw = fill(T(NaN), 2); qr = fill(R(NaN), 2)
                @test sq_work(T, R, J, p, name, route, layout, 'A', 'N', 'N', jobu, jobv, J(m), J(n), copy(a), J(lda), copy(s),
                              copy(u), J(ldu), copy(v), J(ldv), copy(nr), qi, J(-1), qw, J(-1), qr, J(-1)) == 0
                iw = zeros(J, max(Int(qi[1]), 1)); wk = fill(T(NaN), max(Int(real(qw[1])), 2)); rw = fill(R(NaN), max(Int(qr[1]), 2))
                @test sq_work(T, R, J, p, name, route, layout, 'A', 'N', 'N', jobu, jobv, J(m), J(n), a, J(lda), s, u, J(ldu), v, J(ldv), nr,
                              iw, J(length(iw)), wk, J(length(wk)), rw, J(length(rw))) == 0
            else
                @test sq_high(T, R, J, p, name, route, layout, 'A', 'N', 'N', jobu, jobv, J(m), J(n), a, J(lda), s, u, J(ldu), v, J(ldv), nr) == 0
            end
            @test nr[1] == min(m, n)
            @test isapprox(s, sv; rtol=tol, atol=tol)
            if jobu != 'N'
                ncol = jobu == 'A' ? m : n
                U = layout == 102 ? sq_fromcol(u, m, ncol, ldu) : sq_fromrow(u, m, ncol, ldu)
                V = layout == 102 ? sq_fromcol(v, n, n, ldv) : sq_fromrow(v, n, n, ldv)
                res[layout] = (U[:, 1:n], V)
                @test isapprox(U[:, 1:n]' * U[:, 1:n], Matrix{T}(I, n, n); rtol=tol, atol=tol)
                # V is returned as the n-by-n logical matrix: A = U * diag(s) * V or * V'.
                @test isapprox(U[:, 1:n] * Diagonal(T.(s)) * V, A0; rtol=tol * 10, atol=tol * 10) ||
                      isapprox(U[:, 1:n] * Diagonal(T.(s)) * V', A0; rtol=tol * 10, atol=tol * 10)
            end
        end
        if jobu != 'N'
            @test isapprox(res[101][1], res[102][1]; rtol=tol * 10, atol=tol * 10)
            @test isapprox(res[101][2], res[102][2]; rtol=tol * 10, atol=tol * 10)
        end
    end
end

@testset "LAPACKE GESVDQ diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "gesvdq"; p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 24); u = zeros(T, 36); v = zeros(T, 36); s = zeros(R, 4); nr = J[0]
        g(layout, lda, ldu, ldv, aa) = sq_high(T, R, J, p, name, :dll, layout, 'A', 'N', 'N', 'S', 'V', J(6), J(4), aa, J(lda), s, u, J(ldu), v, J(ldv), nr)
        @test g(301, 6, 6, 4, a) == -1
        @test g(101, 3, 4, 4, a) == -10
        @test g(101, 4, 3, 4, a) == -13
        @test g(101, 4, 4, 3, a) == -15
        bad = copy(a); bad[1] = T(NaN)
        @test g(102, 6, 6, 4, bad) == -9
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
