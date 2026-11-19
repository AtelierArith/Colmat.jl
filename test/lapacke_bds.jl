using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function bds_bdsqr(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, work, layout, uplo, n, ncvt, nru, ncc,
                           d, e, vt, ldvt, u, ldu, c, ldc, wk)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, Cchar, $J, $J, $J, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}),
                             layout, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, wk) :
                       ccall(p, $J, (Cint, Cchar, $J, $J, $J, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                             layout, uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc)
            end
        end
    end
end
for (R, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64))
    @eval begin
        function bds_bdsdc(::Type{$R}, ::Type{$J}, p, name, route, work, layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, wk, iw)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}, $J, Ptr{$R}, $J, Ptr{$R}, Ptr{$J},
                                     Ptr{$R}, Ptr{$J}),
                             layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, wk, iw) :
                       ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}, $J, Ptr{$R}, $J, Ptr{$R}, Ptr{$J}),
                             layout, uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq)
            end
        end
        function bds_bdsvdx(::Type{$R}, ::Type{$J}, p, name, route, work, layout, uplo, jobz, range, n, d, e, vl, vu, il, iu,
                            ns, s, z, ldz, superb, wk, iw)
            lapacke288_selected_entry(name, $J, route) do
                work ? ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, Ptr{$J}, Ptr{$R},
                                     Ptr{$R}, $J, Ptr{$R}, Ptr{$J}),
                             layout, uplo, jobz, range, n, d, e, vl, vu, il, iu, ns, s, z, ldz, wk, iw) :
                       ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, Ptr{$J}, Ptr{$R},
                                     Ptr{$R}, $J, Ptr{$J}),
                             layout, uplo, jobz, range, n, d, e, vl, vu, il, iu, ns, s, z, ldz, superb)
            end
        end
    end
end

bd_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
bd_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
bd_fromcol(v, m, n, ld) = [v[i + (j - 1) * ld] for i in 1:m, j in 1:n]
bd_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]
bd_pack(m, layout, ld) = layout == 102 ? bd_colvec(m, ld) : bd_rowmajor(m, ld)
bd_unpack(v, m, n, layout, ld) = layout == 102 ? bd_fromcol(v, m, n, ld) : bd_fromrow(v, m, n, ld)
bd_matrix(d, e, uplo) = Matrix(Bidiagonal(d, e, uplo == 'U' ? :U : :L))

@testset "LAPACKE BDSQR" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), vectors in (false, true), work in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "bdsqr" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9400 + n)
        d0 = randn(rng, R, n) .+ R(2); e0 = randn(rng, R, max(n - 1, 0))
        Bm = bd_matrix(d0, e0, uplo)
        ev = sort(svdvals(Bm); rev=true)
        tol = 5000 * n * eps(R) * max(1, ev[1])
        ncvt = vectors ? n : 0; nru = vectors ? n : 0; ncc = vectors ? 2 : 0
        C0 = randn(rng, T, n, 2)
        for layout in (102, 101)
            ldvt = layout == 102 ? max(n, 1) : max(ncvt, 1) + 2
            ldu = layout == 102 ? max(nru, 1) : n + 2
            ldc = layout == 102 ? max(n, 1) : max(ncc, 1) + 2
            I0 = Matrix{T}(I, n, n)
            vt = ts = vectors ? bd_pack(I0, layout, ldvt) : fill(T(NaN), 1)
            u = vectors ? bd_pack(I0, layout, ldu) : fill(T(NaN), 1)
            c = vectors ? bd_pack(C0, layout, ldc) : fill(T(NaN), 1)
            d = copy(d0); e = vcat(copy(e0), zeros(R, 1)); wk = fill(R(NaN), 4n + 8)
            @test bds_bdsqr(T, R, J, p, name, route, work, layout, uplo, J(n), J(ncvt), J(nru), J(ncc), d, e, vt, J(ldvt), u, J(ldu), c, J(ldc), wk) == 0
            @test isapprox(d, ev; rtol=tol, atol=tol)
            if vectors
                U = bd_unpack(u, n, n, layout, ldu); VT = bd_unpack(vt, n, n, layout, ldvt); Cm = bd_unpack(c, n, 2, layout, ldc)
                @test isapprox(U * Diagonal(T.(d)) * VT, Bm; rtol=tol, atol=tol)
                @test isapprox(U' * C0, Cm; rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE BDSDC and BDSVDX" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (R, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        lib = route === :dll ? native : trampoline
        rng = MersenneTwister(9450 + n)
        d0 = randn(rng, R, n) .+ R(2); e0 = randn(rng, R, max(n - 1, 0))
        Bm = bd_matrix(d0, e0, uplo)
        ev = sort(svdvals(Bm); rev=true)
        tol = 5000 * n * eps(R) * max(1, ev[1])
        # --- BDSDC (COMPQ = N and I) ---
        name = "LAPACKE_" * prefix * "bdsdc" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(lib, name * suffix)
        for compq in ('N', 'I'), layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            u = fill(R(NaN), max(n * ld, 1)); vt = fill(R(NaN), max(n * ld, 1))
            d = copy(d0); e = vcat(copy(e0), zeros(R, 1)); q = fill(R(NaN), 1); iq = zeros(J, 1)
            wk = fill(R(NaN), 3n * n + 4n + 8); iw = zeros(J, 8n + 8)
            @test bds_bdsdc(R, J, p, name, route, work, layout, uplo, compq, J(n), d, e, u, J(ld), vt, J(ld), q, iq, wk, iw) == 0
            @test isapprox(d, ev; rtol=tol, atol=tol)
            if compq == 'I'
                U = bd_unpack(u, n, n, layout, ld); VT = bd_unpack(vt, n, n, layout, ld)
                @test isapprox(U * Diagonal(d) * VT, Bm; rtol=tol, atol=tol)
            end
        end
        # --- BDSVDX ---
        name = "LAPACKE_" * prefix * "bdsvdx" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(lib, name * suffix)
        for range in ('A', 'V', 'I'), jobz in ('N', 'V'), layout in (102, 101)
            il, iu = n == 1 ? (1, 1) : (2, 3)
            asc = sort(ev)
            vl, vu = n == 1 ? (asc[1] - 1, asc[1] + 1) : ((asc[1] + asc[2]) / 2, (asc[3] + asc[4]) / 2)
            expected = range == 'A' ? asc : range == 'I' ? asc[il:iu] : (n == 1 ? asc : asc[2:3])
            ld = layout == 102 ? 2n : n + 2
            z = fill(R(NaN), max(2n * (layout == 102 ? n : ld), 1))
            d = copy(d0); e = vcat(copy(e0), zeros(R, 1)); ns = J[-1]; s = fill(R(NaN), n); superb = zeros(J, 2n + 2)
            wk = fill(R(NaN), 14n + 8); iw = zeros(J, 12n + 8)
            @test bds_bdsvdx(R, J, p, name, route, work, layout, uplo, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu),
                             ns, s, z, J(ld), superb, wk, iw) == 0
            @test ns[1] == length(expected)
            @test isapprox(sort(s[1:ns[1]]), expected; rtol=tol, atol=tol)
            if jobz == 'V' && ns[1] > 0
                Z = layout == 102 ? bd_fromcol(z, 2n, ns[1], ld) : bd_fromrow(z, 2n, ns[1], ld)
                for j in 1:ns[1]
                    # Z holds [U; V]: B * v = s * u (equal scaling for both halves).
                    @test isapprox(Bm * Z[n+1:2n, j], s[j] * Z[1:n, j]; rtol=tol * 10, atol=tol * 10)
                end
            end
        end
    end
end

@testset "LAPACKE bidiagonal SVD diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        nq = "LAPACKE_" * prefix * "bdsqr"; pq = Libdl.dlsym(native, nq * suffix)
        d = ones(R, 4); e = ones(R, 4); vt = zeros(T, 16); u = zeros(T, 16); c = zeros(T, 16); wk = zeros(R, 16)
        q(layout, ldvt, ldu, ldc, dd, ee; vv=vt, uu=u, cc=c) = bds_bdsqr(T, R, J, pq, nq, :dll, false, layout, 'U', J(4), J(4), J(4), J(4),
            dd, ee, vv, J(ldvt), uu, J(ldu), cc, J(ldc), wk)
        @test q(301, 4, 4, 4, d, e) == -1
        @test q(101, 3, 4, 4, d, e) == -10
        @test q(101, 4, 3, 4, d, e) == -12
        @test q(101, 4, 4, 3, d, e) == -14
        bad = copy(d); bad[2] = R(NaN)
        @test q(102, 4, 4, 4, bad, e) == -7
        @test q(102, 4, 4, 4, d, bad) == -8
        badm = copy(vt); badm[1] = T(NaN)
        @test q(102, 4, 4, 4, d, e; vv=badm) == -9
        @test q(102, 4, 4, 4, d, e; uu=badm) == -11
        @test q(102, 4, 4, 4, d, e; cc=badm) == -13
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
