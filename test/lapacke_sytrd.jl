using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    @eval begin
        function trd_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, d, e, tau)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}), layout, uplo, n, a, lda, d, e, tau)
            end
        end
        function trd_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, d, e, tau, wk, lw)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$T}, $J),
                      layout, uplo, n, a, lda, d, e, tau, wk, lw)
            end
        end
        function equb_high(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, s, scond, amax)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}), layout, uplo, n, a, lda, s, scond, amax)
            end
        end
        function equb_work(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, layout, uplo, n, a, lda, s, scond, amax, wk)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$T}),
                      layout, uplo, n, a, lda, s, scond, amax, wk)
            end
        end
    end
end

function rd_rowmajor(m, ld)
    out = fill(eltype(m)(NaN), size(m, 1) * ld)
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[(i - 1) * ld + j] = m[i, j]
    end
    out
end
function rd_colvec(m, ld)
    out = fill(eltype(m)(NaN), ld * size(m, 2))
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[i + (j - 1) * ld] = m[i, j]
    end
    out
end

@testset "LAPACKE SYTRD/HETRD" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam) in ((Float32, Float32, "s", "sytrd"), (Float64, Float64, "d", "sytrd"),
                                (ComplexF32, Float32, "c", "hetrd"), (ComplexF64, Float64, "z", "hetrd")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9810 + n)
        f = randn(rng, T, n, n)
        herm = Matrix(T <: Complex ? Hermitian(f + f', uplo == 'L' ? :L : :U) : Symmetric(f + f', uplo == 'L' ? :L : :U))
        ev = sort(real.(eigvals(herm)))
        tol = 5000 * n * eps(R) * max(1, maximum(abs, ev))
        outs = Dict()
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? rd_colvec(herm, ld) : rd_rowmajor(herm, ld)
            d = fill(R(NaN), n); e = fill(R(NaN), max(n - 1, 1)); tau = fill(T(NaN), max(n - 1, 1))
            if work
                qw = [T(NaN)]
                @test trd_work(T, R, J, p, name, route, layout, uplo, J(n), copy(a), J(ld), copy(d), copy(e), copy(tau), qw, J(-1)) == 0
                wk = fill(T(NaN), max(Int(real(qw[1])), 1))
                @test trd_work(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), d, e, tau, wk, J(length(wk))) == 0
            else
                @test trd_high(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), d, e, tau) == 0
            end
            tri = n == 1 ? reshape([d[1]], 1, 1) : Matrix(SymTridiagonal(d, e[1:n-1]))
            @test isapprox(sort(real.(eigvals(tri))), ev; rtol=tol, atol=tol)
            outs[layout] = (copy(d), copy(e), copy(tau))
        end
        @test isapprox(outs[101][1], outs[102][1]; rtol=tol, atol=tol)
    end
end

@testset "LAPACKE SYEQUB/HEEQUB" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam) in ((Float32, Float32, "s", "syequb"), (Float64, Float64, "d", "syequb"),
                                (ComplexF32, Float32, "c", "syequb"), (ComplexF64, Float64, "z", "syequb"),
                                (ComplexF32, Float32, "c", "heequb"), (ComplexF64, Float64, "z", "heequb")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9830 + n)
        f = randn(rng, T, n, n) .* T[R(10)^(i % 3) for i in 1:n, _ in 1:n]
        full = fam == "heequb" ? f + f' : f + transpose(f)
        sym = Matrix(fam == "heequb" ? Hermitian(full, uplo == 'L' ? :L : :U) : Symmetric(full, uplo == 'L' ? :L : :U))
        res = Dict()
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? rd_colvec(sym, ld) : rd_rowmajor(sym, ld)
            before = copy(a)
            s = fill(R(NaN), n); sc = R[NaN]; am = R[NaN]; wk = fill(T(NaN), 3n + 3)
            @test (work ? equb_work(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), s, sc, am, wk) :
                          equb_high(T, R, J, p, name, route, layout, uplo, J(n), a, J(ld), s, sc, am)) == 0
            @test isequal(a, before)   # A is read only
            @test all(s .> 0) && 0 < sc[1] <= 1
            @test isapprox(am[1], maximum(x -> abs(real(x)) + abs(imag(x)), sym); rtol=10 * eps(R))   # CABS1 for complex
            res[layout] = (copy(s), sc[1], am[1])
        end
        @test isapprox(res[101][1], res[102][1]; rtol=10 * eps(R))
        @test isapprox(res[101][2], res[102][2]; rtol=10 * eps(R))
    end
end

@testset "LAPACKE SYTRD/SYEQUB diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, tf, ef) in ((Float64, Float64, "d", "sytrd", "syequb"), (ComplexF64, Float64, "z", "hetrd", "heequb")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        nt = "LAPACKE_" * prefix * tf; ne = "LAPACKE_" * prefix * ef
        pt = Libdl.dlsym(native, nt * suffix); pe = Libdl.dlsym(native, ne * suffix)
        a = zeros(T, 16); d = zeros(R, 4); e = zeros(R, 3); tau = zeros(T, 3); s = zeros(R, 4); sc = R[0]; am = R[0]
        bad = copy(a); bad[5] = T(NaN)      # (1,2) lies in the upper triangle
        low = copy(a); low[2] = T(NaN)      # (2,1) is outside UPLO='U'
        tr(layout, lda, aa) = trd_high(T, R, J, pt, nt, :dll, layout, 'U', J(4), aa, J(lda), d, e, tau)
        @test tr(301, 4, a) == -1
        @test tr(101, 3, a) == -5
        @test tr(102, 4, bad) == -4
        @test tr(102, 4, low) != -4
        eq(layout, lda, aa) = equb_high(T, R, J, pe, ne, :dll, layout, 'U', J(4), aa, J(lda), s, sc, am)
        @test eq(301, 4, a) == -1
        @test eq(101, 3, a) == -5
        @test eq(102, 4, bad) == -4
        @test eq(102, 4, low) != -4
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
