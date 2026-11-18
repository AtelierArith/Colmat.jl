using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gges_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, jobvsl, jobvsr,
        sort, n, a, lda, b, ldb, sdim, ar::Vector{$T}, ai,
        be::Vector{$T}, vsl, ldvsl, vsr, ldvsr, work, rwork, bwork, save,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                alpha = ar
                if save
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T},
                         $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                         $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$J}),
                        layout, jobvsl, jobvsr, sort, C_NULL, n, a, lda, b,
                        ldb, sdim, alpha, be, vsl, ldvsl, vsr, ldvsr, work,
                        length(work), rwork, bwork)
                else
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T},
                         $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                         $J, Ptr{$T}, $J),
                        layout, jobvsl, jobvsr, sort, C_NULL, n, a, lda, b,
                        ldb, sdim, alpha, be, vsl, ldvsl, vsr, ldvsr)
                end
            elseif save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                     $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}),
                    layout, jobvsl, jobvsr, sort, C_NULL, n, a, lda, b, ldb,
                    sdim, ar, ai, be, vsl, ldvsl, vsr, ldvsr, work,
                    length(work), bwork)
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                     $J, Ptr{$T}, $J),
                    layout, jobvsl, jobvsr, sort, C_NULL, n, a, lda, b, ldb,
                    sdim, ar, ai, be, vsl, ldvsl, vsr, ldvsr)
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

@testset "LAPACKE GGES/GGES3 layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 4), family in ("gges", "gges3"),
        (jobvsl, jobvsr) in (('N', 'N'), ('V', 'V')), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(6400 + n)
        a0 = randn(rng, T, n, n)
        b0 = a0 * a0' + T(n) * Matrix{T}(I, n, n)
        # Column-major reference.
        acol = vec(copy(a0)); bcol = vec(copy(b0))
        arcol = zeros(T, n); aicol = zeros(R, n); becol = zeros(T, n)
        vslcol = zeros(T, n * n); vsrcol = zeros(T, n * n)
        sd = Ref{J}(-1)
        ar, ai = T <: Complex ? (arcol, zeros(R, n)) : (arcol, aicol)
        ref = lapacke_gges_call(
            T, R, J, pointer, 102, jobvsl, jobvsr, 'N', n, acol, n, bcol, n,
            sd, ar, ai, becol, vslcol, n, vsrcol, n, fill(T(NaN), 8n + 32),
            fill(R(NaN), 8n), zeros(J, n), save, name, route,
        )
        @test ref == 0
        @test sd[] == 0
        # Row-major with padding.
        ld = n + 2
        arow = row_major(a0, ld); brow = row_major(b0, ld)
        arrow = zeros(T, n); airow = zeros(R, n); berow = zeros(T, n)
        vslrow = zeros(T, n * ld); vsrrow = zeros(T, n * ld)
        sdr = Ref{J}(-1)
        arr, air = T <: Complex ? (arrow, zeros(R, n)) : (arrow, airow)
        info = lapacke_gges_call(
            T, R, J, pointer, 101, jobvsl, jobvsr, 'N', n, arow, ld, brow, ld,
            sdr, arr, air, berow, vslrow, ld, vsrrow, ld, fill(T(NaN), 8n + 32),
            fill(R(NaN), 8n), zeros(J, n), save, name, route,
        )
        @test info == 0
        @test sdr[] == sd[]
        tol = 1000 * eps(R)
        @test isapprox(arrow, arcol; rtol=tol, atol=tol)
        @test isapprox(berow, becol; rtol=tol, atol=tol)
        if !(T <: Complex)
            @test isapprox(airow, aicol; rtol=tol, atol=tol)
        end
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        @test isapprox(from_row_major(brow, n, n, ld), reshape(bcol, n, n); rtol=tol, atol=tol)
        if jobvsl == 'V'
            @test isapprox(from_row_major(vslrow, n, n, ld), reshape(vslcol, n, n); rtol=tol, atol=tol)
        end
        if jobvsr == 'V'
            @test isapprox(from_row_major(vsrrow, n, n, ld), reshape(vsrcol, n, n); rtol=tol, atol=tol)
        end
        # Schur form and orthogonality from the column-major result.
        S = reshape(acol, n, n); P = reshape(bcol, n, n)
        @test all(isapprox(S[i, j], 0; atol=2000 * eps(R)) for i in 3:n for j in 1:i-2)
        @test all(isapprox(P[i, j], 0; atol=2000 * eps(R)) for i in 2:n for j in 1:i-1)
        if jobvsl == 'V'
            L = reshape(vslcol, n, n)
            @test isapprox(L' * L, Matrix{T}(I, n, n); rtol=2000 * eps(R), atol=2000 * eps(R))
        end
        if jobvsr == 'V'
            Zt = reshape(vsrcol, n, n)
            @test isapprox(Zt' * Zt, Matrix{T}(I, n, n); rtol=2000 * eps(R), atol=2000 * eps(R))
        end
    end
end

@testset "LAPACKE GGES/GGES3 diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64),
        family in ("gges", "gges3")

        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * family
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        a = zeros(T, n * n); b = zeros(T, n * n)
        ar = zeros(T, n); ai = zeros(R, n); be = zeros(T, n)
        vsl = zeros(T, n * n); vsr = zeros(T, n * n)
        sd = Ref{J}(0)
        arr, air = T <: Complex ? (ar, zeros(R, n)) : (ar, ai)
        @test lapacke_gges_call(
            T, R, J, pointer, 301, 'V', 'V', 'N', n, a, n, b, n, sd, arr, air,
            be, vsl, n, vsr, n, T[], R[], J[], false, name, :dll,
        ) == -1
        @test lapacke_gges_call(
            T, R, J, pointer, 101, 'V', 'V', 'N', n, a, n - 1, b, n, sd, arr,
            air, be, vsl, n, vsr, n, T[], R[], J[], false, name, :dll,
        ) == -8
        @test lapacke_gges_call(
            T, R, J, pointer, 101, 'V', 'V', 'N', n, a, n, b, n - 1, sd, arr,
            air, be, vsl, n, vsr, n, T[], R[], J[], false, name, :dll,
        ) == -10
        @test lapacke_gges_call(
            T, R, J, pointer, 101, 'V', 'V', 'N', n, a, n, b, n, sd, arr, air,
            be, vsl, n - 1, vsr, n, T[], R[], J[], false, name, :dll,
        ) == -16
        @test lapacke_gges_call(
            T, R, J, pointer, 101, 'V', 'V', 'N', n, a, n, b, n, sd, arr, air,
            be, vsl, n, vsr, n - 1, T[], R[], J[], false, name, :dll,
        ) == -18
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_gges_call(
            T, R, J, pointer, 102, 'V', 'V', 'N', n, bad, n, b, n, sd, arr,
            air, be, vsl, n, vsr, n, T[], R[], J[], false, name, :dll,
        ) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
