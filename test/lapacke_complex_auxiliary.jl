using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((ComplexF32, Float32), (ComplexF64, Float64)), J in (Int32, Int64)
    @eval function lapacke_lacgv_call(
        ::Type{$T}, ::Type{$J}, pointer, n, x, incx, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($J, Ptr{$T}, $J), n, x, incx)
        end
    end
    @eval function lapacke_lacp2_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, uplo, m, n,
        a::Vector{$R}, lda, b::Vector{$T}, ldb, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$R}, $J, Ptr{$T}, $J),
                layout, uplo, m, n, a, lda, b, ldb)
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
function from_row_major(vector, m, n, ld)
    out = fill(eltype(vector)(NaN), m, n)
    for j in 1:n, i in 1:m
        out[i, j] = vector[(i - 1) * ld + j]
    end
    return out
end

@testset "LAPACKE LACGV conjugation and LACP2 layouts" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        cgv = "LAPACKE_" * prefix * "lacgv" * (high ? "" : "_work")
        cp2 = "LAPACKE_" * prefix * "lacp2" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(cgv, interface) == Libdl.dlsym(native, cgv * suffix)
        @test BLAS.lbt_get_forward(cp2, interface) == Libdl.dlsym(native, cp2 * suffix)
        pcgv = Libdl.dlsym(route === :dll ? native : trampoline, cgv * suffix)
        pcp2 = Libdl.dlsym(route === :dll ? native : trampoline, cp2 * suffix)
        # LACGV with positive and negative strides.
        for n in (0, 1, 5), incx in (-2, -1, 1, 3)
            logical = [T(complex(i + 0.5, -i / 3)) for i in 1:n]
            x = fill(T(NaN), n == 0 ? 0 : 1 + (n - 1) * abs(incx))
            first = incx < 0 ? 1 + (n - 1) * (-incx) : 1
            for i in 1:n
                x[first + (i - 1) * incx] = logical[i]
            end
            before = copy(x)
            @test lapacke_lacgv_call(T, J, pcgv, n, x, incx, cgv, route, high) == 0
            for i in 1:n
                @test x[first + (i - 1) * incx] == conj(logical[i])
            end
            @test isequal(x[[q for q in eachindex(x) if q ∉ Set(first + (i - 1) * incx for i in 1:n)]],
                          before[[q for q in eachindex(x) if q ∉ Set(first + (i - 1) * incx for i in 1:n)]])
        end
        # LACP2 row/column-major equivalence.
        for (m, n) in ((1, 1), (4, 3)), uplo in ('L', 'U')
            rng = MersenneTwister(600 + m + n)
            a = randn(rng, R, m, n)
            b = randn(rng, T, m, n)
            ld = n + 2
            acol = vec(copy(a)); bcol = vec(copy(b))
            @test lapacke_lacp2_call(
                T, R, J, pcp2, 102, uplo, m, n, acol, max(1, m), bcol,
                max(1, m), cp2, route, high,
            ) == 0
            arow = row_major(a, ld); brow = row_major(b, ld)
            @test lapacke_lacp2_call(
                T, R, J, pcp2, 101, uplo, m, n, arow, ld, brow, ld,
                cp2, route, high,
            ) == 0
            @test isequal(from_row_major(brow, m, n, ld), reshape(bcol, m, n))
        end
    end
end

@testset "LAPACKE LACGV/LACP2 diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((ComplexF64, Float64, "z"),), J in (Int32, Int64), high in (false, true)
        suffix = J === Int64 ? "64_" : ""
        cgv = "LAPACKE_" * prefix * "lacgv" * (high ? "" : "_work")
        cp2 = "LAPACKE_" * prefix * "lacp2" * (high ? "" : "_work")
        pcgv = Libdl.dlsym(native, cgv * suffix)
        pcp2 = Libdl.dlsym(native, cp2 * suffix)
        if high
            x = T[T(NaN)]
            @test lapacke_lacgv_call(T, J, pcgv, 1, x, 1, cgv, :dll, high) == -2
        end
        a = zeros(R, 9); b = zeros(T, 9)
        @test lapacke_lacp2_call(
            T, R, J, pcp2, 301, 'U', 3, 3, a, 3, b, 3, cp2, :dll, high,
        ) == -1
        @test lapacke_lacp2_call(
            T, R, J, pcp2, 101, 'U', 3, 3, a, 2, b, 3, cp2, :dll, high,
        ) == -6
        @test lapacke_lacp2_call(
            T, R, J, pcp2, 101, 'U', 3, 3, a, 3, b, 2, cp2, :dll, high,
        ) == -8
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
