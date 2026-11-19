using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function ts_getsls(::Type{$T}, ::Type{$J}, p, name, route, work, layout, trans, m, n, nrhs, a, lda, b, ldb, wk, lw)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, trans, m, n, nrhs, a, lda, b, ldb, wk, lw) :
                       ccall(p, $J, (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, trans, m, n, nrhs, a, lda, b, ldb)
            end
        end
        function ts_getsqrhrt(::Type{$T}, ::Type{$J}, p, name, route, work, layout, m, n, mb1, nb1, nb2, a, lda, t, ldt, wk, lw)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, $J, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, m, n, mb1, nb1, nb2, a, lda, t, ldt, wk, lw) :
                       ccall(p, $J, (Cint, $J, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, m, n, mb1, nb1, nb2, a, lda, t, ldt)
            end
        end
        function ts_orgtsqr(::Type{$T}, ::Type{$J}, p, name, route, work, layout, m, n, mb, nb, a, lda, t, ldt, wk, lw)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, m, n, mb, nb, a, lda, t, ldt, wk, lw) :
                       ccall(p, $J, (Cint, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, m, n, mb, nb, a, lda, t, ldt)
            end
        end
        function ts_orhr(::Type{$T}, ::Type{$J}, p, name, route, layout, m, n, nb, a, lda, t, ldt, d)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}),
                      layout, m, n, nb, a, lda, t, ldt, d)
            end
        end
    end
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function ts_ref_orhr(::Type{$T}, ptr, name_unused, m, n, nb, a, t, d, info)
        ccall(ptr, Cvoid, (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ref{Int64}),
              m, n, nb, a, m, t, nb, d, info)
    end
    @eval function ts_ref_latsqr(::Type{$T}, ptr, m, n, mb, nb, a, t, wk, info)
        ccall(ptr, Cvoid, (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ref{Int64},
                           Ptr{$T}, Ref{Int64}, Ref{Int64}),
              m, n, mb, nb, a, m, t, nb, wk, length(wk), info)
    end
end

ts_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
ts_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
ts_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
ts_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]
ts_pack(m, layout, ld) = layout == 102 ? ts_colvec(m, ld) : ts_rowmajor(m, ld)
ts_unpack(v, m, n, layout, ld) = layout == 102 ? ts_fromcol(v, m, n, ld) : ts_fromrow(v, m, n, ld)

function ts_reference()
    lib = only(filter(l -> l.interface == :ilp64 && occursin("openblas", lowercase(l.libname)), BLAS.get_config().loaded_libs))
    return Libdl.dlopen(lib.libname)
end

@testset "LAPACKE GETSLS" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n) in ((8, 3), (3, 8), (5, 5)), nrhs in (1, 2), trans in ('N', T <: Complex ? 'C' : 'T'),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "getsls" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9500 + m + n)
        A = randn(rng, T, m, n)
        op = trans == 'N' ? A : A'
        rows_b = max(m, n)
        B0 = zeros(T, rows_b, nrhs)
        rhs = randn(rng, T, size(op, 1), nrhs)
        B0[1:size(op, 1), :] = rhs
        want = op \ rhs
        tol = 5000 * max(m, n) * eps(R) * max(1, norm(want))
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2; ldb = layout == 102 ? rows_b : nrhs + 2
            a = ts_pack(A, layout, lda); b = ts_pack(B0, layout, ldb)
            wk = fill(T(NaN), 1)
            if work
                @test ts_getsls(T, J, p, name, route, true, layout, trans, J(m), J(n), J(nrhs), copy(a), J(lda), copy(b), J(ldb), wk, J(-1)) == 0
                wk = fill(T(NaN), Int(real(wk[1])))
            end
            @test ts_getsls(T, J, p, name, route, work, layout, trans, J(m), J(n), J(nrhs), a, J(lda), b, J(ldb), wk, J(length(wk))) == 0
            x = ts_unpack(b, rows_b, nrhs, layout, ldb)[1:size(op, 2), :]
            @test isapprox(x, want; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE GETSQRHRT / ORGTSQR_ROW / ORHR_COL against the reference LAPACK" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    ref = ts_reference()
    for (T, R, prefix, org) in ((Float32, Float32, "s", "orgtsqr_row"), (Float64, Float64, "d", "orgtsqr_row"),
                                (ComplexF32, Float32, "c", "ungtsqr_row"), (ComplexF64, Float64, "z", "ungtsqr_row")),
        J in (Int32, Int64), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        lib = route === :dll ? native : trampoline
        m, n, mb, nb = 12, 3, 6, 2
        rng = MersenneTwister(9600)
        A0 = randn(rng, T, m, n)
        ncols = n * (m - n + (mb - n) - 1) ÷ (mb - n)   # n * ceil((m-n)/(mb-n))
        tol = 5000 * m * eps(R) * max(1, norm(A0))
        # --- GETSQRHRT ---
        name = "LAPACKE_" * prefix * "getsqrhrt" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(lib, name * suffix)
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2; ldt = layout == 102 ? 2 : n + 2
            a = ts_pack(A0, layout, lda); t = fill(T(NaN), layout == 102 ? ldt * n : 2 * ldt)
            wk = fill(T(NaN), 1)
            if work
                @test ts_getsqrhrt(T, J, p, name, route, true, layout, J(m), J(n), J(mb), J(nb), J(2), copy(a), J(lda), copy(t), J(ldt), wk, J(-1)) == 0
                wk = fill(T(NaN), Int(real(wk[1])))
            end
            @test ts_getsqrhrt(T, J, p, name, route, work, layout, J(m), J(n), J(mb), J(nb), J(2), a, J(lda), t, J(ldt), wk, J(length(wk))) == 0
            Rm = triu(ts_unpack(a, m, n, layout, lda)[1:n, 1:n])
            @test isapprox(Rm' * Rm, A0' * A0; rtol=tol, atol=tol)
        end
        # --- ORHR_COL: reconstruct Householder vectors from an orthonormal Q ---
        Q0 = Matrix(qr(randn(rng, T, m, n)).Q)[:, 1:n]
        name = "LAPACKE_" * prefix * (T <: Complex ? "unhr_col" : "orhr_col") * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(lib, name * suffix)
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2; ldt = layout == 102 ? nb : n + 2
            a = ts_pack(Q0, layout, lda); t = fill(T(NaN), layout == 102 ? ldt * n : nb * ldt); d = fill(T(NaN), n)
            @test ts_orhr(T, J, p, name, route, layout, J(m), J(n), J(nb), a, J(lda), t, J(ldt), d) == 0
            V = ts_unpack(a, m, n, layout, lda)
            Tm = ts_unpack(t, nb, n, layout, ldt)
            # Independent reference result.
            a_r = copy(Q0); t_r = zeros(T, nb, n); d_r = zeros(T, n); info = Ref{Int64}(-9)
            ts_ref_orhr(T, Libdl.dlsym(ref, prefix * (T <: Complex ? "unhr_col" : "orhr_col") * "_64_"), nothing,
                        m, n, nb, a_r, t_r, d_r, info)
            @test info[] == 0
            @test isapprox(V, a_r; rtol=tol, atol=tol)
            @test isapprox(d, d_r; rtol=tol, atol=tol)
            # T holds an upper-triangular nb-by-nb block per NB columns; the rest is unreferenced.
            for jb in 1:nb:n, j in jb:min(jb + nb - 1, n), i in 1:min(j - jb + 1, nb)
                @test isapprox(Tm[i, j], t_r[i, j]; rtol=tol, atol=tol)
            end
        end
        # --- ORGTSQR_ROW: explicit Q from the TSQR factors (computed by the reference LATSQR) ---
        latsqr = Libdl.dlsym(ref, prefix * "latsqr_64_")
        a_ref = copy(A0); t_ref = zeros(T, nb, ncols); wk = zeros(T, n * nb); info = Ref{Int64}(-9)
        ts_ref_latsqr(T, latsqr, m, n, mb, nb, a_ref, t_ref, wk, info)
        @test info[] == 0
        name = "LAPACKE_" * prefix * org * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(lib, name * suffix)
        for layout in (102, 101)
            lda = layout == 102 ? m : n + 2; ldt = layout == 102 ? nb : ncols + 2
            a = ts_pack(a_ref, layout, lda); t = ts_pack(t_ref, layout, ldt)
            wk2 = fill(T(NaN), 1)
            if work
                @test ts_orgtsqr(T, J, p, name, route, true, layout, J(m), J(n), J(mb), J(nb), copy(a), J(lda), copy(t), J(ldt), wk2, J(-1)) == 0
                wk2 = fill(T(NaN), max(Int(real(wk2[1])), 1))
            end
            @test ts_orgtsqr(T, J, p, name, route, work, layout, J(m), J(n), J(mb), J(nb), a, J(lda), t, J(ldt), wk2, J(length(wk2))) == 0
            Q = ts_unpack(a, m, n, layout, lda)
            @test isapprox(Q' * Q, Matrix{T}(I, n, n); rtol=tol, atol=tol)
            Rf = triu(a_ref[1:n, 1:n])
            @test isapprox(Q * Rf, A0; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE tall-skinny diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        a = zeros(T, 48); b = zeros(T, 48); t = zeros(T, 48); d = zeros(T, 8); wk = zeros(T, 8)
        ng = "LAPACKE_" * prefix * "getsls"; pg = Libdl.dlsym(native, ng * suffix)
        @test ts_getsls(T, J, pg, ng, :dll, false, 301, 'N', J(6), J(3), J(2), a, J(6), b, J(6), wk, J(0)) == -1
        @test ts_getsls(T, J, pg, ng, :dll, false, 101, 'N', J(6), J(3), J(2), a, J(2), b, J(2), wk, J(0)) == -7
        @test ts_getsls(T, J, pg, ng, :dll, false, 101, 'N', J(6), J(3), J(2), a, J(3), b, J(1), wk, J(0)) == -9
        bad = copy(a); bad[1] = T(NaN)
        @test ts_getsls(T, J, pg, ng, :dll, false, 102, 'N', J(6), J(3), J(2), bad, J(6), b, J(6), wk, J(0)) == -6
        nh = "LAPACKE_" * prefix * (T <: Complex ? "unhr_col" : "orhr_col"); ph = Libdl.dlsym(native, nh * suffix)
        @test ts_orhr(T, J, ph, nh, :dll, 301, J(6), J(3), J(2), a, J(6), t, J(2), d) == -1
        @test ts_orhr(T, J, ph, nh, :dll, 101, J(6), J(3), J(2), a, J(2), t, J(3), d) == -6
        @test ts_orhr(T, J, ph, nh, :dll, 101, J(6), J(3), J(2), a, J(3), t, J(2), d) == -8
        @test ts_orhr(T, J, ph, nh, :dll, 102, J(6), J(3), J(2), bad, J(6), t, J(2), d) == -5
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
