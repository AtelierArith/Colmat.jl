using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_gghrd_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, compq, compz, n, ilo, ihi,
        a, lda, b, ldb, q, ldq, z, ldz, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                 Ptr{$T}, $J, Ptr{$T}, $J),
                layout, compq, compz, n, ilo, ihi, a, lda, b, ldb, q, ldq, z, ldz)
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

@testset "LAPACKE GGHRD/GGHD3 layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (1, 4), family in ("gghrd", "gghd3"),
        compq in ('N', 'V'), compz in ('N', 'V'), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(2000 + n)
        a = randn(rng, T, n, n)
        b = randn(rng, T, n, n)
        q0 = randn(rng, T, n, n)
        z0 = randn(rng, T, n, n)
        ilo, ihi = 1, n
        # Column-major reference.
        acol = vec(copy(a)); bcol = vec(copy(b))
        qcol = vec(copy(q0)); zcol = vec(copy(z0))
        @test lapacke_gghrd_call(
            T, J, pointer, 102, compq, compz, n, ilo, ihi, acol, n, bcol, n,
            qcol, n, zcol, n, name, route,
        ) == 0
        # Row-major with padding.
        ld = n + 2
        arow = row_major(a, ld); brow = row_major(b, ld)
        qrow = row_major(q0, ld); zrow = row_major(z0, ld)
        @test lapacke_gghrd_call(
            T, J, pointer, 101, compq, compz, n, ilo, ihi, arow, ld, brow, ld,
            qrow, ld, zrow, ld, name, route,
        ) == 0
        tol = 500 * eps(real(T))
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        @test isapprox(from_row_major(brow, n, n, ld), reshape(bcol, n, n); rtol=tol, atol=tol)
        if compq == 'V'
            @test isapprox(from_row_major(qrow, n, n, ld), reshape(qcol, n, n); rtol=tol, atol=tol)
        end
        if compz == 'V'
            @test isapprox(from_row_major(zrow, n, n, ld), reshape(zcol, n, n); rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE GGHRD/GGHD3 diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64),
        family in ("gghrd", "gghd3")

        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * family
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        a = Matrix{T}(I, n, n); b = Matrix{T}(I, n, n)
        q = Matrix{T}(I, n, n); z = Matrix{T}(I, n, n)
        @test lapacke_gghrd_call(
            T, J, pointer, 301, 'V', 'V', n, 1, n, vec(a), n, vec(b), n,
            vec(q), n, vec(z), n, name, :dll,
        ) == -1
        @test lapacke_gghrd_call(
            T, J, pointer, 101, 'V', 'V', n, 1, n, vec(a), n - 1, vec(b), n,
            vec(q), n, vec(z), n, name, :dll,
        ) == -8
        @test lapacke_gghrd_call(
            T, J, pointer, 101, 'V', 'V', n, 1, n, vec(a), n, vec(b), n,
            vec(q), n - 1, vec(z), n, name, :dll,
        ) == -12
        @test lapacke_gghrd_call(
            T, J, pointer, 101, 'V', 'V', n, 1, n, vec(a), n, vec(b), n,
            vec(q), n, vec(z), n - 1, name, :dll,
        ) == -14
        bad = copy(vec(a)); bad[1] = T(NaN)
        @test lapacke_gghrd_call(
            T, J, pointer, 102, 'V', 'V', n, 1, n, bad, n, vec(b), n,
            vec(q), n, vec(z), n, name, :dll,
        ) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
