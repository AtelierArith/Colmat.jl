using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_lascl_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, kind, kl, ku,
        cfrom::$R, cto::$R, m, n, a, lda, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, $R, $R, $J, $J, Ptr{$T}, $J),
                layout, kind, kl, ku, cfrom, cto, m, n, a, lda)
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

@testset "LAPACKE LASCL layouts and types" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    cases = (
        ('G', 4, 3, 0, 0, 4),
        ('L', 4, 4, 0, 0, 4),
        ('U', 4, 4, 0, 0, 4),
        ('H', 4, 4, 0, 0, 4),
        ('B', 4, 4, 2, 2, 3),
        ('Q', 4, 4, 2, 2, 3),
        ('Z', 4, 4, 1, 2, 5),
    )
    for (T, R, prefix) in
        ((Float32, Float32, "s"), (Float64, Float64, "d"),
         (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (kind, m, n, kl, ku, nr) in cases,
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "lascl" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(500 + Int(kind) + m + n)
        block = randn(rng, T, nr, n)
        ld = n + 2
        acol = vec(copy(block))
        arow = row_major(block, ld)
        cfrom, cto = R(2), R(1)
        @test lapacke_lascl_call(
            T, R, J, pointer, 102, kind, J(kl), J(ku), cfrom, cto, J(m), J(n),
            acol, max(1, nr), name, route, high,
        ) == 0
        @test lapacke_lascl_call(
            T, R, J, pointer, 101, kind, J(kl), J(ku), cfrom, cto, J(m), J(n),
            arow, ld, name, route, high,
        ) == 0
        tol = 200 * eps(R)
        @test isapprox(from_row_major(arow, nr, n, ld), reshape(acol, nr, n); rtol=tol, atol=tol)
        saved = row_major(block, ld)
        for i in 1:nr, j in n+1:ld
            @test isequal(arow[(i - 1) * ld + j], saved[(i - 1) * ld + j])
        end
    end
end

@testset "LAPACKE LASCL diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), high in (false, true)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "lascl" * (high ? "" : "_work")
        pointer = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 9)
        @test lapacke_lascl_call(
            T, R, J, pointer, 301, 'G', J(0), J(0), R(2), R(1), J(3), J(3),
            a, 3, name, :dll, high,
        ) == -1
        @test lapacke_lascl_call(
            T, R, J, pointer, 101, 'G', J(0), J(0), R(2), R(1), J(3), J(3),
            a, 2, name, :dll, high,
        ) == -9
        if high
            bad = copy(a); bad[1] = T(NaN)
            @test lapacke_lascl_call(
                T, R, J, pointer, 102, 'G', J(0), J(0), R(2), R(1), J(3), J(3),
                bad, 3, name, :dll, high,
            ) == -9
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
