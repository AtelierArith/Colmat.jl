using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_hseqr_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, job, compz, n,
        ilo, ihi, h, ldh, wr::Vector{$T}, wi, z, ldz, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                w = wr
                if save
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                        layout, job, compz, n, ilo, ihi, h, ldh, w, z, ldz,
                        work, length(work))
                else
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$T}, $J),
                        layout, job, compz, n, ilo, ihi, h, ldh, w, z, ldz)
                end
            elseif save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T},
                     Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz,
                    work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T},
                     Ptr{$R}, Ptr{$T}, $J),
                    layout, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz)
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

@testset "LAPACKE HSEQR layouts, COMPZ and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 4), compz in ('N', 'I', 'V'),
        job in ('E', 'S'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "hseqr" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(6500 + n)
        # Upper Hessenberg input with nonzero subdiagonal.
        h0 = zeros(T, n, n)
        for j in 1:n, i in 1:n
            i <= j + 1 || continue
            h0[i, j] = i == j ? T(2 + i) : T(0.3) * randn(rng, T)
        end
        z0 = Matrix{T}(I, n, n)
        # Column-major reference.
        hcol = vec(copy(h0))
        wrcol = zeros(T, n); wicol = zeros(R, n)
        zcol = compz == 'N' ? zeros(T, 0) : vec(copy(z0))
        ldz_ref = compz == 'N' ? 1 : n
        ref = lapacke_hseqr_call(
            T, R, J, pointer, 102, job, compz, n, 1, n, hcol, n, wrcol, wicol,
            zcol, ldz_ref, fill(T(NaN), 2n), save, name, route,
        )
        @test ref == 0
        # Row-major with padding.
        ld = n + 2
        hrow = row_major(h0, ld)
        wrrow = zeros(T, n); wirow = zeros(R, n)
        zrow = compz == 'N' ? zeros(T, 0) : row_major(z0, ld)
        info = lapacke_hseqr_call(
            T, R, J, pointer, 101, job, compz, n, 1, n, hrow, ld, wrrow,
            wirow, zrow, ld, fill(T(NaN), 2n), save, name, route,
        )
        @test info == 0
        tol = 1000 * eps(R)
        @test isapprox(wrrow, wrcol; rtol=tol, atol=tol)
        if !(T <: Complex)
            @test isapprox(wirow, wicol; rtol=tol, atol=tol)
        end
        @test isapprox(from_row_major(hrow, n, n, ld), reshape(hcol, n, n); rtol=tol, atol=tol)
        if compz != 'N'
            @test isapprox(from_row_major(zrow, n, n, ld), reshape(zcol, n, n); rtol=tol, atol=tol)
        end
        # The Schur form: H is quasi-upper-triangular and Z is orthogonal.
        H = reshape(hcol, n, n)
        if T <: Complex
            @test all(isapprox(H[i, j], 0; atol=2000 * eps(R)) for i in 2:n for j in 1:i-1)
        else
            @test all(isapprox(H[i, j], 0; atol=2000 * eps(R)) for i in 3:n for j in 1:i-2)
        end
        if compz != 'N'
            Z = reshape(zcol, n, n)
            @test isapprox(Z' * Z, Matrix{T}(I, n, n); rtol=2000 * eps(R), atol=2000 * eps(R))
            # COMPZ='I' starts from the identity, so Z*H*Z' reproduces H0 when
            # COMPZ='I'; with COMPZ='V' the input Z is the identity here too.
            @test isapprox(Z * H * Z', h0; rtol=5000 * eps(R), atol=5000 * eps(R))
        end
    end
end

@testset "LAPACKE HSEQR diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "hseqr"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        h = zeros(T, n * n)
        wr = zeros(T, n); wi = zeros(R, n)
        z = zeros(T, n * n)
        @test lapacke_hseqr_call(
            T, R, J, pointer, 301, 'E', 'I', n, 1, n, h, n, wr, wi, z, n,
            T[], false, name, :dll,
        ) == -1
        @test lapacke_hseqr_call(
            T, R, J, pointer, 101, 'E', 'I', n, 1, n, h, n - 1, wr, wi, z, n,
            T[], false, name, :dll,
        ) == -8
        @test lapacke_hseqr_call(
            T, R, J, pointer, 101, 'E', 'I', n, 1, n, h, n, wr, wi, z, n - 1,
            T[], false, name, :dll,
        ) == -12
        bad = copy(h); bad[1] = T(NaN)
        @test lapacke_hseqr_call(
            T, R, J, pointer, 102, 'E', 'I', n, 1, n, bad, n, wr, wi, z, n,
            T[], false, name, :dll,
        ) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
