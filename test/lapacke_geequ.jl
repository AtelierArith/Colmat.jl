using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_geequ_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, m, n, a, lda,
        r::Vector{$R}, c::Vector{$R}, rc, cc, amax, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$R},
                 Ptr{$R}, Ptr{$R}),
                layout, m, n, a, lda, r, c, rc, cc, amax)
        end
    end
end

function row_major(matrix, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), m * ld)
    for j in 1:n, i in 1:m
        out[(i - 1) * ld + j] = matrix[i, j]
    end
    return out
end

@testset "LAPACKE GEEQU/GEEQUB layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n) in ((4, 4), (5, 3)),
        family in ("geequ", "geequb"), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(6300 + m + n)
        # Scale rows and columns so the equilibration has real work to do,
        # keeping the element type exactly T.
        a0 = randn(rng, T, m, n)
        a0 .*= one(T) .+ T(0.25) .* T.(1:m)
        a0 .*= one(T) .+ T(0.1) .* transpose(T.(1:n))
        acol = vec(copy(a0))
        rcol = zeros(R, m); ccol = zeros(R, n)
        rccol = Ref(R(-1)); cccol = Ref(R(-1)); amaxcol = Ref(R(-1))
        @test lapacke_geequ_call(
            T, R, J, pointer, 102, m, n, acol, m, rcol, ccol, rccol, cccol,
            amaxcol, name, route,
        ) == 0
        ld = n + 2
        arow = row_major(a0, ld)
        rrow = zeros(R, m); crow = zeros(R, n)
        rcrow = Ref(R(-1)); ccrow = Ref(R(-1)); amaxrow = Ref(R(-1))
        @test lapacke_geequ_call(
            T, R, J, pointer, 101, m, n, arow, ld, rrow, crow, rcrow, ccrow,
            amaxrow, name, route,
        ) == 0
        @test isapprox(rrow, rcol; rtol=1e-5, atol=1e-5)
        @test isapprox(crow, ccol; rtol=1e-5, atol=1e-5)
        @test isapprox(rcrow[], rccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(ccrow[], cccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(amaxrow[], amaxcol[]; rtol=1e-5, atol=1e-5)
        # Equilibrated rows reach a near-unity maximum measured with the
        # magnitude the native kernel uses (modulus for real, component
        # L1 for complex); GEEQUB bins the scaling into powers of two.
        mag(x) = T <: Complex ? abs(real(x)) + abs(imag(x)) : abs(x)
        scaled = maximum(maximum(mag, (Diagonal(rcol) * a0)[i, :]) for i in 1:m)
        if family == "geequ"
            @test isapprox(scaled, 1.0; rtol=1e-3, atol=1e-3)
        else
            @test 0.5 <= scaled <= 2.0
        end
    end
end

@testset "LAPACKE GEEQU/GEEQUB diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64),
        family in ("geequ", "geequb")

        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * family
        pointer = Libdl.dlsym(native, name * suffix)
        m, n = 4, 4
        a = zeros(T, m * n)
        r = zeros(R, m); c = zeros(R, n)
        rc = Ref(R(0)); cc = Ref(R(0)); amax = Ref(R(0))
        @test lapacke_geequ_call(
            T, R, J, pointer, 301, m, n, a, m, r, c, rc, cc, amax, name, :dll,
        ) == -1
        @test lapacke_geequ_call(
            T, R, J, pointer, 101, m, n, a, n - 1, r, c, rc, cc, amax,
            name, :dll,
        ) == -5
        @test lapacke_geequ_call(
            T, R, J, pointer, 102, m, n, a, m - 1, r, c, rc, cc, amax,
            name, :dll,
        ) == -5
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_geequ_call(
            T, R, J, pointer, 102, m, n, bad, m, r, c, rc, cc, amax,
            name, :dll,
        ) == -4
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
