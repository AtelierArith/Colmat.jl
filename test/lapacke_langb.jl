using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_langb_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, norm, n, kl, ku,
        ab, ldab, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $R,
                    (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$R}),
                    layout, norm, n, kl, ku, ab, ldab, work)
            else
                ccall(pointer, $R,
                    (Cint, Cchar, $J, $J, $J, Ptr{$T}, $J),
                    layout, norm, n, kl, ku, ab, ldab)
            end
        end
    end
end
for (T, R) in ((ComplexF32, Float32), (ComplexF64, Float64)), J in (Int32, Int64)
    @eval function lapacke_lanhe_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, norm, uplo, n, a,
        lda, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $R,
                    (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$R}),
                    layout, norm, uplo, n, a, lda, work)
            else
                ccall(pointer, $R,
                    (Cint, Cchar, Cchar, $J, Ptr{$T}, $J),
                    layout, norm, uplo, n, a, lda)
            end
        end
    end
end

const AR(kl, ku) = kl + ku + 1
function band_col(matrix, kl, ku)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), AR(kl, ku) * n)
    for j in 1:n, i in 1:n
        j - ku <= i <= j + kl || continue
        out[ku + i - j + (j - 1) * AR(kl, ku) + 1] = matrix[i, j]
    end
    return out
end
function band_row(matrix, kl, ku, ld)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), AR(kl, ku) * ld)
    for j in 1:n, i in 1:n
        j - ku <= i <= j + kl || continue
        out[(ku + i - j) * ld + j] = matrix[i, j]
    end
    return out
end
function row_major(matrix, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), m * ld)
    for j in 1:n, i in 1:m
        out[(i - 1) * ld + j] = matrix[i, j]
    end
    return out
end

@testset "LAPACKE LANGB/LANHE layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (n, kl, ku) in ((4, 1, 2), (5, 2, 1)),
        norm in ('1', 'I', 'M', 'F', 'O'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "langb" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(5700 + n + kl)
        band = zeros(T, n, n)
        for j in 1:n, i in 1:n
            j - ku <= i <= j + kl || continue
            band[i, j] = randn(rng, T)
        end
        abcol = band_col(band, kl, ku)
        ref = lapacke_langb_call(
            T, R, J, pointer, 102, norm, n, kl, ku, abcol, AR(kl, ku),
            fill(R(NaN), n), save, name, route,
        )
        @test ref >= 0
        # Row-major matches the column-major result for every norm.
        ld = n + 2
        abrow = band_row(band, kl, ku, ld)
        got = lapacke_langb_call(
            T, R, J, pointer, 101, norm, n, kl, ku, abrow, ld,
            fill(R(NaN), n), save, name, route,
        )
        @test isapprox(got, ref; rtol=2000 * eps(R), atol=2000 * eps(R))
        # And both agree with the dense norm of the logical band matrix.
        dense = norm in ('1', 'O') ? opnorm(band, 1) :
                norm == 'I' ? opnorm(band, Inf) :
                norm == 'M' ? maximum(abs.(band)) :
                LinearAlgebra.norm(band)
        @test isapprox(ref, R(dense); rtol=2000 * eps(R), atol=2000 * eps(R))
    end
    # LANHE (Hermitian, complex only).
    for (T, R, prefix) in ((ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'),
        norm in ('1', 'I', 'M', 'F'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "lanhe" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(5800 + n)
        full = randn(rng, T, n, n)
        herm = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        herm = Matrix(Hermitian(herm, uplo == 'L' ? :L : :U))
        acol = vec(copy(herm))
        ref = lapacke_lanhe_call(
            T, R, J, pointer, 102, norm, uplo, n, acol, n,
            fill(R(NaN), n), save, name, route,
        )
        @test ref >= 0
        ld = n + 2
        arow = row_major(herm, ld)
        got = lapacke_lanhe_call(
            T, R, J, pointer, 101, norm, uplo, n, arow, ld,
            fill(R(NaN), n), save, name, route,
        )
        @test isapprox(got, ref; rtol=2000 * eps(R), atol=2000 * eps(R))
        dense = norm in ('1', 'O') ? opnorm(herm, 1) :
                norm == 'I' ? opnorm(herm, Inf) :
                norm == 'M' ? maximum(abs.(herm)) :
                LinearAlgebra.norm(herm)
        @test isapprox(ref, R(dense); rtol=2000 * eps(R), atol=2000 * eps(R))
    end
end

@testset "LAPACKE LANGB/LANHE diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"),), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "langb"
        pointer = Libdl.dlsym(native, name * suffix)
        n, kl, ku = 4, 1, 2
        ab = zeros(T, AR(kl, ku) * n)
        @test lapacke_langb_call(
            T, R, J, pointer, 301, '1', n, kl, ku, ab, AR(kl, ku),
            R[], false, name, :dll,
        ) == -1
        @test lapacke_langb_call(
            T, R, J, pointer, 101, '1', n, kl, ku, ab, kl + ku,
            R[], false, name, :dll,
        ) == -7
        bad = copy(ab); bad[ku + 1] = T(NaN)
        @test lapacke_langb_call(
            T, R, J, pointer, 102, '1', n, kl, ku, bad, AR(kl, ku),
            R[], false, name, :dll,
        ) == -6
    end
    for (T, R, prefix) in ((ComplexF64, Float64, "z"),), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "lanhe"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        a = zeros(T, n * n)
        @test lapacke_lanhe_call(
            T, R, J, pointer, 301, '1', 'U', n, a, n, R[], false, name, :dll,
        ) == -1
        @test lapacke_lanhe_call(
            T, R, J, pointer, 101, '1', 'U', n, a, n - 1, R[], false,
            name, :dll,
        ) == -6
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_lanhe_call(
            T, R, J, pointer, 102, '1', 'U', n, bad, n, R[], false,
            name, :dll,
        ) == -5
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
