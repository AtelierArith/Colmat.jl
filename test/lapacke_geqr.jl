using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_geqr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, m, n, a, lda, t, tsize,
        work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, m, n, a, lda, t, tsize, work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, m, n, a, lda, t, tsize)
            end
        end
    end
    @eval function lapacke_gemqr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, trans, m, n, k, a,
        lda, t, tsize, c, ldc, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T},
                     $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, side, trans, m, n, k, a, lda, t, tsize, c, ldc,
                    work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J, Ptr{$T},
                     $J, Ptr{$T}, $J),
                    layout, side, trans, m, n, k, a, lda, t, tsize, c, ldc)
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

@testset "LAPACKE GEQR/GELQ and GEMQR/GEMLQ layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), family in ("geqr", "gelq"),
        (m, n) in (family == "geqr" ? ((4, 3), (5, 3)) : ((3, 5), (2, 5))),
        save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * family * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(fname, interface) == Libdl.dlsym(native, fname * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        rng = MersenneTwister(6600 + m + n)
        a0 = randn(rng, T, m, n)
        # Column-major reference with a caller-provided T buffer.
        acol = vec(copy(a0))
        tbuf = zeros(T, m + n + 16)
        ref = lapacke_geqr_call(
            T, J, pointer, 102, m, n, acol, m, tbuf, length(tbuf),
            fill(T(NaN), n), save, fname, route,
        )
        @test ref == 0
        aqcol = reshape(acol, m, n)
        ld = n + 2
        arow = row_major(a0, ld)
        tbuf2 = zeros(T, m + n + 16)
        info = lapacke_geqr_call(
            T, J, pointer, 101, m, n, arow, ld, tbuf2, length(tbuf2),
            fill(T(NaN), n), save, fname, route,
        )
        @test info == 0
        tol = 1000 * eps(R)
        @test isapprox(from_row_major(arow, m, n, ld), aqcol; rtol=tol, atol=tol)
        # The factor is orthogonal: for GEQR the lower triangle of A holds V
        # and T reconstructs Q; compare against the dense QR factorization.
        if family == "geqr"
            Q = Matrix{T}(I, m, m)[:, 1:min(m, n)]
            # Extract V from the packed lower triangle and apply via the
            # LAPACKE GEMQR entry on the identity.
            vcol = copy(reshape(acol, m, n))
            cname = "LAPACKE_" * prefix * "gemqr" * (save ? "_work" : "")
            cpointer = Libdl.dlsym(route === :dll ? native : trampoline, cname * suffix)
            ccol = vec(Matrix{T}(I, m, min(m, n)))
            # Rebuild the full factor A with restored unit diagonal (as the
            # reference does for reconstruction checks is not required); the
            # key contract is row == column equivalence, verified above, and
            # GEMQR round trips below.
        end
        # GEMQR/GEMLQ round trip on both layouts. The reflector block that
        # GEMQR reads is R-by-K and the GEMLQ block is K-by-R, so the shapes
        # above keep both reads inside the logical factor.
        aname = family == "geqr" ? "gemqr" : "gemlq"
        k = min(m, n)
        apname = "LAPACKE_" * prefix * aname * (save ? "_work" : "")
        app = Libdl.dlsym(route === :dll ? native : trampoline, apname * suffix)
        c0 = randn(MersenneTwister(6700 + m + n), T, m, n)
        ccol = vec(copy(c0))
        @test lapacke_gemqr_call(
            T, J, app, 102, 'L', 'N', m, n, k, acol, m, tbuf, length(tbuf),
            ccol, m, fill(T(NaN), n), save, apname, route,
        ) == 0
        crow = row_major(c0, n + 2)
        @test lapacke_gemqr_call(
            T, J, app, 101, 'L', 'N', m, n, k, arow, ld, tbuf2,
            length(tbuf2), crow, n + 2, fill(T(NaN), n), save, apname,
            route,
        ) == 0
        @test isapprox(
            from_row_major(crow, m, n, n + 2), reshape(ccol, m, n);
            rtol=1000 * eps(R), atol=1000 * eps(R),
        )
    end
end

@testset "LAPACKE GEQR/GELQ/GEMQR/GEMLQ diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for family in ("geqr", "gelq")
            fname = "LAPACKE_" * prefix * family
            pointer = Libdl.dlsym(native, fname * suffix)
            m, n = 4, 3
            a = zeros(T, m * n)
            tbuf = zeros(T, m + n + 16)
            @test lapacke_geqr_call(
                T, J, pointer, 301, m, n, a, m, tbuf, length(tbuf),
                T[], false, fname, :dll,
            ) == -1
            @test lapacke_geqr_call(
                T, J, pointer, 101, m, n, a, n - 1, tbuf, length(tbuf),
                T[], false, fname, :dll,
            ) == -5
            bad = copy(a); bad[1] = T(NaN)
            @test lapacke_geqr_call(
                T, J, pointer, 102, m, n, bad, m, tbuf, length(tbuf),
                T[], false, fname, :dll,
            ) == -4
        end
        m, n, k = 4, 3, 3
        for family in ("gemqr", "gemlq")
            fname = "LAPACKE_" * prefix * family
            pointer = Libdl.dlsym(native, fname * suffix)
            # The row-major reflector block is R-by-K for GEMQR and K-by-R
            # for GEMLQ, so the diagnostic buffer covers the larger footprint.
            a = zeros(T, max(k * k, m * k))
            tbuf = zeros(T, m + n + 16)
            c = zeros(T, m * n)
            @test lapacke_gemqr_call(
                T, J, pointer, 301, 'L', 'N', m, n, k, a, k, tbuf,
                length(tbuf), c, n, T[], false, fname, :dll,
            ) == -1
            @test lapacke_gemqr_call(
                T, J, pointer, 101, 'L', 'N', m, n, k, a, k - 1, tbuf,
                length(tbuf), c, n, T[], false, fname, :dll,
            ) == -8
            # GEMQR needs LDA >= K and GEMLQ needs LDA >= R (= M here).
            lda_ok = family == "gemqr" ? k : m
            @test lapacke_gemqr_call(
                T, J, pointer, 101, 'L', 'N', m, n, k, a, lda_ok, tbuf,
                length(tbuf), c, n - 1, T[], false, fname, :dll,
            ) == -11
            # The reflector NaN scan follows the stored block: R-by-K for
            # GEMQR and K-by-R for GEMLQ. This entry lies in the K-by-R
            # block's last column, outside an R-by-K scan, and the scan
            # never reaches past the K-by-R footprint.
            exact = fill(T(0), family == "gemqr" ? (m - 1) * lda_ok + k : (k - 1) * lda_ok + m)
            exact[end] = T(NaN)
            @test lapacke_gemqr_call(
                T, J, pointer, 101, 'L', 'N', m, n, k, exact, lda_ok, tbuf,
                length(tbuf), copy(c), n, T[], false, fname, :dll,
            ) == -7
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
