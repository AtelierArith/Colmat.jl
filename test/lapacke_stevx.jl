using Test, Colmat, LinearAlgebra, Libdl, Random

isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function tsel_stegr_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                 vl, vu, il, iu, abstol, m, w, z, ldz, isuppz)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$J}),
                      layout, jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz)
            end
        end
        function tsel_stegr_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                 vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, wk, lw, iw, liw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, $R, Ptr{$J}, Ptr{$R},
                              Ptr{$T}, $J, Ptr{$J}, Ptr{$R}, $J, Ptr{$J}, $J),
                      layout, jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz, wk, lw, iw, liw)
            end
        end
        function tsel_stemr_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                 vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, Ptr{$J}, Ptr{$R}, Ptr{$T},
                              $J, $J, Ptr{$J}, Ptr{$J}),
                      layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac)
            end
        end
        function tsel_stemr_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                 vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac, wk, lw, iw, liw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$R}, Ptr{$R}, $R, $R, $J, $J, Ptr{$J}, Ptr{$R}, Ptr{$T},
                              $J, $J, Ptr{$J}, Ptr{$J}, Ptr{$R}, $J, Ptr{$J}, $J),
                      layout, jobz, range, n, d, e, vl, vu, il, iu, m, w, z, ldz, nzc, isuppz, tryrac, wk, lw, iw, liw)
            end
        end
    end
    if !(T <: Complex)
        @eval begin
            function tsel_stevx_high(::Type{$T}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                     vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $T, $T, $J, $J, $T, Ptr{$J}, Ptr{$T},
                                  Ptr{$T}, $J, Ptr{$J}),
                          layout, jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, ifail)
                end
            end
            function tsel_stevx_work(::Type{$T}, ::Type{$J}, p, name, route, layout, jobz, range, n, d, e,
                                     vl, vu, il, iu, abstol, m, w, z, ldz, wk, iw, ifail)
                lapacke288_selected_entry(name, $J, route) do
                    ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $T, $T, $J, $J, $T, Ptr{$J}, Ptr{$T},
                                  Ptr{$T}, $J, Ptr{$T}, Ptr{$J}, Ptr{$J}),
                          layout, jobz, range, n, d, e, vl, vu, il, iu, abstol, m, w, z, ldz, wk, iw, ifail)
                end
            end
        end
    end
end

tsel_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
tsel_fromcol(v, n, ld, cols) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:cols]
tsel_fromrow(v, n, ld, cols) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:cols]

@testset "LAPACKE STEVX/STEVR/STEGR/STEMR" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 5), fam in (:stevx, :stevr, :stegr, :stemr), range in ('A', 'V', 'I'),
        jobz in ('N', 'V'), work in (false, true), route in (:dll, :lbt_trampoline)

        (fam in (:stevx, :stevr) && T <: Complex) && continue
        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * String(fam) * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9800 + n)
        d0 = randn(rng, R, n) .+ R(2); e0 = randn(rng, R, max(n - 1, 0))
        ref = n == 1 ? d0 : eigvals(SymTridiagonal(d0, e0))
        ev = sort(ref)
        il, iu = n == 1 ? (1, 1) : (2, 4)
        vl, vu = n == 1 ? (ev[1] - 1, ev[1] + 1) : ((ev[1] + ev[2]) / 2, (ev[4] + ev[5]) / 2)
        idx = range == 'A' ? (1:n) : range == 'I' ? (il:iu) : (n == 1 ? (1:1) : (2:4))
        expected = ev[idx]
        tol = 5000 * n * eps(R) * max(1, maximum(abs, ev))
        tri = n == 1 ? reshape(d0, 1, 1) : Matrix(SymTridiagonal(d0, e0))
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            d = copy(d0); e = vcat(copy(e0), zeros(R, 2))
            z = fill(T(NaN), max(n * ld, 1)); w = fill(R(NaN), n); mm = J[-1]
            ifail = zeros(J, n); isuppz = zeros(J, 2n + 2); tryrac = J[1]
            wk = fill(R(NaN), 20n + 64); iw = zeros(J, 10n + 64); lw = J(length(wk)); liw = J(length(iw))
            info = if fam === :stevx
                work ? tsel_stevx_work(T, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu), R(0),
                                       mm, w, z, J(ld), wk, iw, ifail) :
                       tsel_stevx_high(T, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu), R(0),
                                       mm, w, z, J(ld), ifail)
            elseif fam === :stemr
                if work
                    qw = [R(NaN)]; qi = zeros(J, 1)
                    tsel_stemr_work(T, R, J, p, name, route, layout, jobz, range, J(n), copy(d), copy(e), R(vl), R(vu), J(il), J(iu),
                                    copy(mm), copy(w), copy(z), J(ld), J(n), copy(isuppz), copy(tryrac), qw, J(-1), qi, J(-1))
                    wk = fill(R(NaN), Int(qw[1])); iw = zeros(J, Int(qi[1]))
                    tsel_stemr_work(T, R, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu),
                                    mm, w, z, J(ld), J(n), isuppz, tryrac, wk, J(length(wk)), iw, J(length(iw)))
                else
                    tsel_stemr_high(T, R, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu),
                                    mm, w, z, J(ld), J(n), isuppz, tryrac)
                end
            else
                # STEVR (real only) and STEGR share this signature; they use the same ccall shape.
                if work
                    qw = [R(NaN)]; qi = zeros(J, 1)
                    tsel_stegr_work(T, R, J, p, name, route, layout, jobz, range, J(n), copy(d), copy(e), R(vl), R(vu), J(il), J(iu), R(0),
                                    copy(mm), copy(w), copy(z), J(ld), copy(isuppz), qw, J(-1), qi, J(-1))
                    wk = fill(R(NaN), Int(qw[1])); iw = zeros(J, Int(qi[1]))
                    tsel_stegr_work(T, R, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu), R(0),
                                    mm, w, z, J(ld), isuppz, wk, J(length(wk)), iw, J(length(iw)))
                else
                    tsel_stegr_high(T, R, J, p, name, route, layout, jobz, range, J(n), d, e, R(vl), R(vu), J(il), J(iu), R(0),
                                    mm, w, z, J(ld), isuppz)
                end
            end
            @test info == 0
            @test mm[1] == length(expected)
            @test isapprox(w[1:mm[1]], expected; rtol=tol, atol=tol)
            if jobz == 'V' && mm[1] > 0
                x = layout == 102 ? tsel_fromcol(z, n, ld, mm[1]) : tsel_fromrow(z, n, ld, mm[1])
                @test isapprox(tri * x, x * Diagonal(T.(w[1:mm[1]])); rtol=tol, atol=tol)
                @test isapprox(x' * x, Matrix{T}(I, mm[1], mm[1]); rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE tridiagonal selected diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        ng = "LAPACKE_" * prefix * "stegr"; nm = "LAPACKE_" * prefix * "stemr"
        pg = Libdl.dlsym(native, ng * suffix); pm = Libdl.dlsym(native, nm * suffix)
        d = ones(R, 4); e = ones(R, 4); z = zeros(T, 16); w = zeros(R, 4); mm = J[0]; isuppz = zeros(J, 8); tryrac = J[1]
        nan = R(NaN)
        g(layout, ldz, dd, ee; abstol=R(0), rg='A', vl=R(0)) =
            tsel_stegr_high(T, R, J, pg, ng, :dll, layout, 'V', rg, J(4), dd, ee, vl, R(1), J(1), J(4), abstol, mm, w, z, J(ldz), isuppz)
        @test g(301, 4, d, e) == -1
        @test g(101, 3, d, e) == -15
        bad = copy(d); bad[2] = nan
        @test g(102, 4, bad, e) == -5
        @test g(102, 4, d, bad) == -6
        @test g(102, 4, d, e; abstol=nan) == -11
        @test g(102, 4, d, e; rg='V', vl=nan) == -7
        s(layout, ldz, dd, ee) = tsel_stemr_high(T, R, J, pm, nm, :dll, layout, 'V', 'A', J(4), dd, ee, R(0), R(1), J(1), J(4), mm, w, z,
                                                  J(ldz), J(4), isuppz, tryrac)
        @test s(301, 4, d, e) == -1
        @test s(101, 3, d, e) == -14
        @test s(102, 4, bad, e) == -5
        @test s(102, 4, d, bad) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
