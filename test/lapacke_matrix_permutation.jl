using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke_perm_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, forward, m, n, a, lda, k,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer, $J,
                (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$J}),
                layout, forward, m, n, a, lda, k,
            )
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

@testset "LAPACKE LAPMR/LAPMT permutation and row-major storage" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), m in (1, 4), n in (1, 3), forward in (0, 1),
        family in ("lapmr", "lapmt"), high in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(10m + n + forward)
        matrix = randn(rng, T, m, n)
        count = family == "lapmr" ? m : n
        base = collect(J, 1:count)
        randperm(rng, count)
        perm = J[count - base[i] + 1 for i in 1:count]
        # Column-major.
        a_col = vec(copy(matrix))
        k_col = copy(perm)
        ref = lapacke_perm_call(
            T, J, pointer, 102, J(forward), m, n, a_col, max(1, m), k_col,
            name, route,
        )
        @test ref == 0
        # Row-major with padding.
        ld = n + 2
        a_row = row_major(matrix, ld)
        k_row = copy(perm)
        info = lapacke_perm_call(
            T, J, pointer, 101, J(forward), m, n, a_row, ld, k_row,
            name, route,
        )
        @test info == 0
        @test isequal(k_row, k_col)
        @test isapprox(
            from_row_major(a_row, m, n, ld),
            reshape(a_col, m, n);
            rtol=0,
            atol=0,
        )
        # Padding is untouched.
        saved = row_major(matrix, ld)
        for i in 1:m, j in n+1:ld
            @test isequal(a_row[(i - 1) * ld + j], saved[(i - 1) * ld + j])
        end
    end
end

@testset "LAPACKE LAPMR/LAPMT diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for family in ("lapmr", "lapmt"), high in (false, true)
            name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
            pointer = Libdl.dlsym(native, name * suffix)
            a = zeros(T, 9)
            k = J[1, 2, 3]
            @test lapacke_perm_call(
                T, J, pointer, 301, J(1), 3, 3, a, 3, k, name, :dll
            ) == -1
            @test lapacke_perm_call(
                T, J, pointer, 101, J(1), 3, 3, a, 2, k, name, :dll
            ) == -6
        end
    end
end
