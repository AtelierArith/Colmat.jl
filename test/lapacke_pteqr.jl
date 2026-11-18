using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_pteqr_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, compz, n,
        d::Vector{$R}, e::Vector{$R}, z, ldz, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$R}),
                    layout, compz, n, d, e, z, ldz, work)
            else
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J),
                    layout, compz, n, d, e, z, ldz)
            end
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

@testset "LAPACKE PTEQR layouts, COMPZ and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 5), compz in ('N', 'V', 'I'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "pteqr" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(3900 + n)
        d0 = R[1 + i for i in 1:n]
        e0 = R[0.5i - 0.2 for i in 1:max(0, n - 1)]
        dcol, ecol = copy(d0), copy(e0)
        zcol = vec(Matrix{T}(I, n, n))
        work = fill(R(NaN), max(1, 4n - 4))
        @test lapacke_pteqr_call(
            T, R, J, pointer, 102, compz, n, dcol, ecol, zcol, n, work, save,
            name, route,
        ) == 0
        ld = n + 2
        drow, erow = copy(d0), copy(e0)
        zrow = row_major(Matrix{T}(I, n, n), ld)
        @test lapacke_pteqr_call(
            T, R, J, pointer, 101, compz, n, drow, erow, zrow, ld, work, save,
            name, route,
        ) == 0
        tol = 3000 * eps(R)
        @test isapprox(drow, dcol; rtol=tol, atol=tol)
        @test isapprox(erow, ecol; rtol=tol, atol=tol)
        if compz != 'N'
            Z = from_row_major(zrow, n, n, ld)
            Zc = reshape(zcol, n, n)
            @test isapprox(Z, Zc; rtol=tol, atol=tol)
            @test isapprox(Z' * Z, Matrix{T}(I, n, n); rtol=tol, atol=tol)
            A = Tridiagonal(R.(e0), R.(d0), R.(e0))
            @test isapprox(A * Z, Z * Diagonal(dcol); rtol=100tol, atol=100tol)
        end
    end
end

@testset "LAPACKE PTEQR diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "pteqr"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        d = zeros(R, n); e = zeros(R, n - 1)
        z = vec(Matrix{T}(I, n, n))
        @test lapacke_pteqr_call(
            T, R, J, pointer, 301, 'V', n, d, e, z, n, R[], false, name, :dll,
        ) == -1
        @test lapacke_pteqr_call(
            T, R, J, pointer, 101, 'V', n, d, e, z, n - 1, R[], false, name, :dll,
        ) == -7
        @test lapacke_pteqr_call(
            T, R, J, pointer, 102, 'V', n, R[NaN, 0, 0], e, z, n, R[], false, name, :dll,
        ) == -4
        @test lapacke_pteqr_call(
            T, R, J, pointer, 102, 'V', n, d, R[NaN, 0], z, n, R[], false, name, :dll,
        ) == -5
        badz = copy(z); badz[1] = T(NaN)
        @test lapacke_pteqr_call(
            T, R, J, pointer, 102, 'V', n, d, e, badz, n, R[], false, name, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
