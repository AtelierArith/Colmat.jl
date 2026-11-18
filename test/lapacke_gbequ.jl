using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gbequ_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, m, n, kl, ku, ab,
        ldab, r::Vector{$R}, c::Vector{$R}, rc, cc, amax, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, $J, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R},
                 Ptr{$R}, Ptr{$R}, Ptr{$R}),
                layout, m, n, kl, ku, ab, ldab, r, c, rc, cc, amax)
        end
    end
end

const AR(kl, ku) = kl + ku + 1
function band_col(matrix, kl, ku)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), AR(kl, ku) * n)
    for j in 1:n, i in 1:m
        (i + ku >= j && j + kl >= i) || continue
        out[ku + i - j + (j - 1) * AR(kl, ku) + 1] = matrix[i, j]
    end
    return out
end
function band_row(matrix, kl, ku, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), AR(kl, ku) * ld)
    for j in 1:n, i in 1:m
        (i + ku >= j && j + kl >= i) || continue
        out[(ku + i - j) * ld + j] = matrix[i, j]
    end
    return out
end

@testset "LAPACKE GBEQU/GBEQUB band layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (m, n, kl, ku) in ((4, 4, 1, 2), (5, 4, 2, 1)),
        family in ("gbequ", "gbequb"), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(4500 + m + n)
        full = randn(rng, T, m, n)
        band = zeros(T, m, n)
        for j in 1:n, i in 1:m
            (i + ku >= j && j + kl >= i) || continue
            band[i, j] = full[i, j]
        end
        # Column-major reference.
        abcol = band_col(band, kl, ku)
        rcol = zeros(R, m); ccol = zeros(R, n)
        rccol = Ref(R(-1)); cccol = Ref(R(-1)); amaxcol = Ref(R(-1))
        @test lapacke_gbequ_call(
            T, R, J, pointer, 102, m, n, kl, ku, abcol, AR(kl, ku), rcol, ccol,
            rccol, cccol, amaxcol, name, route,
        ) == 0
        # Row-major with padding.
        ld = n + 2
        abrow = band_row(band, kl, ku, ld)
        rrow = zeros(R, m); crow = zeros(R, n)
        rcrow = Ref(R(-1)); ccrow = Ref(R(-1)); amaxrow = Ref(R(-1))
        @test lapacke_gbequ_call(
            T, R, J, pointer, 101, m, n, kl, ku, abrow, ld, rrow, crow,
            rcrow, ccrow, amaxrow, name, route,
        ) == 0
        @test isapprox(rrow, rcol; rtol=1e-5, atol=1e-5)
        @test isapprox(crow, ccol; rtol=1e-5, atol=1e-5)
        @test isapprox(rcrow[], rccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(ccrow[], cccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(amaxrow[], amaxcol[]; rtol=1e-5, atol=1e-5)
        # Row scaling matches the column scale contract.
        # The equilibrated band stays within a small factor of unity for both
        # the direct and the radix-binned variants.
        scaled = maximum(abs.(Diagonal(rcol) * (band * Diagonal(ccol))))
        @test 0.05 < scaled < 20.0
    end
end

@testset "LAPACKE GBEQU/GBEQUB diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64),
        family in ("gbequ", "gbequb")

        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * family
        pointer = Libdl.dlsym(native, name * suffix)
        m, n, kl, ku = 4, 4, 1, 2
        ab = zeros(T, AR(kl, ku) * n)
        r = zeros(R, m); c = zeros(R, n)
        rc = Ref(R(0)); cc = Ref(R(0)); amax = Ref(R(0))
        @test lapacke_gbequ_call(
            T, R, J, pointer, 301, m, n, kl, ku, ab, AR(kl, ku), r, c, rc, cc,
            amax, name, :dll,
        ) == -1
        @test lapacke_gbequ_call(
            T, R, J, pointer, 101, m, n, kl, ku, ab, n - 1, r, c, rc, cc,
            amax, name, :dll,
        ) == -7
        bad = copy(ab); bad[ku + 1 + 1] = T(NaN)
        @test lapacke_gbequ_call(
            T, R, J, pointer, 102, m, n, kl, ku, bad, AR(kl, ku), r, c, rc, cc,
            amax, name, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
