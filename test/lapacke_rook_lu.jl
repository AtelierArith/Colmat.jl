using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_rook_trf(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, a, lda, ipiv,
        work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                    layout, uplo, n, a, lda, ipiv, work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}),
                    layout, uplo, n, a, lda, ipiv)
            end
        end
    end
    @eval function lapacke_rook_trs(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, a, lda, ipiv,
        b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                layout, uplo, n, nrhs, a, lda, ipiv, b, ldb)
        end
    end
    @eval function lapacke_rook_sv(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, a, lda, ipiv,
        b, ldb, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J,
                     Ptr{$T}, $J),
                    layout, uplo, n, nrhs, a, lda, ipiv, b, ldb, work,
                    length(work))
            else
                ccall(pointer, $J,
                    (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                    layout, uplo, n, nrhs, a, lda, ipiv, b, ldb)
            end
        end
    end
end

function rook_row_major(matrix, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), m * ld)
    for j in 1:n, i in 1:m
        out[(i - 1) * ld + j] = matrix[i, j]
    end
    return out
end
function from_rook_row_major(vector, m, n, ld)
    out = fill(eltype(vector)(NaN), m, n)
    for j in 1:n, i in 1:m
        out[i, j] = vector[(i - 1) * ld + j]
    end
    return out
end

@testset "LAPACKE SYTRF/SYTRS/SYSV_ROOK and HETRF/HETRS_ROOK layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix, family) in
        ((Float32, "s", "sy"), (Float64, "d", "sy"),
         (ComplexF32, "c", "he"), (ComplexF64, "z", "he")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * family * "trf_rook" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * family * "trs_rook" * (save ? "_work" : "")
        vname = "LAPACKE_" * prefix * family * "sv_rook" * (save ? "_work" : "")
        for nm in (family == "sy" ? (fname, sname, vname) : (fname, sname))
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        psol = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        psv = family == "sy" ? Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix) : C_NULL
        rng = MersenneTwister(5100 + n)
        full = randn(rng, T, n, n)
        herm = T <: Complex ?
            Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n))) :
            Matrix(Symmetric(full * full' + T(n) * Matrix{T}(I, n, n)))
        herm = Matrix(T <: Complex ? Hermitian(herm, uplo == 'L' ? :L : :U) :
                                  Symmetric(herm, uplo == 'L' ? :L : :U))
        # Factorization on both layouts.
        acol = vec(copy(herm))
        ipivcol = zeros(J, n)
        @test lapacke_rook_trf(
            T, J, pfac, 102, uplo, n, acol, n, ipivcol, fill(T(NaN), n), save,
            fname, route,
        ) == 0
        ld = n + 2
        arow = rook_row_major(herm, ld)
        ipivrow = zeros(J, n)
        @test lapacke_rook_trf(
            T, J, pfac, 101, uplo, n, arow, ld, ipivrow, fill(T(NaN), n), save,
            fname, route,
        ) == 0
        @test isequal(ipivrow, ipivcol)
        tol = 4000 * eps(real(T))
        @test isapprox(from_rook_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=tol, atol=tol)
        # Solve with the shared factors.
        nrhs = 2
        b0 = randn(MersenneTwister(5200 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_rook_trs(
            T, J, psol, 102, uplo, n, nrhs, acol, n, ipivcol, bcol, n,
            sname, route,
        ) == 0
        lbs = nrhs + 2
        brow = rook_row_major(b0, lbs)
        @test lapacke_rook_trs(
            T, J, psol, 101, uplo, n, nrhs, arow, ld, ipivrow, brow, lbs,
            sname, route,
        ) == 0
        @test isapprox(
            from_rook_row_major(brow, n, nrhs, lbs), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 4000 * n * eps(real(T)) * norm(b0)
        @test isapprox(herm * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        if family == "sy"
            # Driver end to end.
            acol2 = vec(copy(herm)); arow2 = rook_row_major(herm, ld)
            ipivc = zeros(J, n); ipivr = zeros(J, n)
            bcol2 = vec(copy(b0)); brow2 = rook_row_major(b0, nrhs + 2)
            @test lapacke_rook_sv(
                T, J, psv, 102, uplo, n, nrhs, acol2, n, ipivc, bcol2, n,
                fill(T(NaN), n), save, vname, route,
            ) == 0
            @test lapacke_rook_sv(
                T, J, psv, 101, uplo, n, nrhs, arow2, ld, ipivr, brow2, nrhs + 2,
                fill(T(NaN), n), save, vname, route,
            ) == 0
            @test isequal(ipivr, ipivc)
            @test isapprox(from_rook_row_major(arow2, n, n, ld), reshape(acol2, n, n); rtol=tol, atol=tol)
            @test isapprox(
                from_rook_row_major(brow2, n, nrhs, nrhs + 2), reshape(bcol2, n, nrhs);
                rtol=tol, atol=tol,
            )
            @test isapprox(herm * reshape(bcol2, n, nrhs), b0; rtol=scale, atol=scale)
        end
    end
end

@testset "LAPACKE SYTRF/SYTRS/SYSV_ROOK diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix, family) in ((Float64, "d", "sy"), (ComplexF64, "z", "he")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * family * "trf_rook"
        sname = "LAPACKE_" * prefix * family * "trs_rook"
        vname = "LAPACKE_" * prefix * family * "sv_rook"
        pfac = Libdl.dlsym(native, fname * suffix)
        psol = Libdl.dlsym(native, sname * suffix)
        psv = family == "sy" ? Libdl.dlsym(native, vname * suffix) : C_NULL
        n, nrhs = 3, 2
        a = zeros(T, n * n)
        ipiv = zeros(J, n)
        b = zeros(T, n * nrhs)
        @test lapacke_rook_trf(
            T, J, pfac, 301, 'U', n, a, n, ipiv, T[], false, fname, :dll,
        ) == -1
        @test lapacke_rook_trf(
            T, J, pfac, 101, 'U', n, a, n - 1, ipiv, T[], false, fname, :dll,
        ) == -5
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_rook_trf(
            T, J, pfac, 102, 'U', n, bad, n, ipiv, T[], false, fname, :dll,
        ) == -4
        @test lapacke_rook_trs(
            T, J, psol, 301, 'U', n, nrhs, a, n, ipiv, b, n, sname, :dll,
        ) == -1
        @test lapacke_rook_trs(
            T, J, psol, 101, 'U', n, nrhs, a, n - 1, ipiv, b, nrhs, sname, :dll,
        ) == -6
        @test lapacke_rook_trs(
            T, J, psol, 101, 'U', n, nrhs, a, n, ipiv, b, nrhs - 1, sname, :dll,
        ) == -9
        if family == "sy"
            @test lapacke_rook_sv(
                T, J, psv, 301, 'U', n, nrhs, a, n, ipiv, b, n, T[], false,
                vname, :dll,
            ) == -1
            @test lapacke_rook_sv(
                T, J, psv, 101, 'U', n, nrhs, a, n - 1, ipiv, b, nrhs, T[], false,
                vname, :dll,
            ) == -6
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
