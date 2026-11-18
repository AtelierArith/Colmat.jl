using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64))
    @eval function lapacke_opgtr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, ap, tau, q, ldq,
        work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}),
                    layout, uplo, n, ap, tau, q, ldq, work)
            else
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J),
                    layout, uplo, n, ap, tau, q, ldq)
            end
        end
    end
    @eval function lapacke_opmtr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, uplo, trans, m, n,
        ap, tau, c, ldc, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, Ptr{$T},
                     Ptr{$T}, $J, Ptr{$T}),
                    layout, side, uplo, trans, m, n, ap, tau, c, ldc, work)
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, Ptr{$T},
                     Ptr{$T}, $J),
                    layout, side, uplo, trans, m, n, ap, tau, c, ldc)
            end
        end
    end
    @eval function lapacke_sptrd_setup(
        ::Type{$T}, ::Type{$J}, pointer, uplo, n, ap, d, e, tau,
    )
        info = Ref{$J}(-1)
        ccall(
            pointer, Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ref{$J}),
            uplo, $J(n), ap, d, e, tau, info,
        )
        return info[]
    end
end

function packed_col(matrix, uplo)
    n = size(matrix, 1)
    out = eltype(matrix)[]
    if uplo == 'U'
        for j in 1:n, i in 1:j
            push!(out, matrix[i, j])
        end
    else
        for j in 1:n, i in j:n
            push!(out, matrix[i, j])
        end
    end
    return out
end
function col_index(upper, n, i, j)
    return upper ? j * (j - 1) ÷ 2 + i : (j - 1) * n - (j - 1) * j ÷ 2 + i
end
function row_index(upper, n, i, j)
    return upper ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j
end
function packed_row_from_col(ap, uplo, n)
    upper = uplo == 'U'
    out = fill(eltype(ap)(NaN), length(ap))
    for j in 1:n, i in 1:n
        (upper ? i <= j : i >= j) || continue
        out[row_index(upper, n, i, j)] = ap[col_index(upper, n, i, j)]
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
function from_row_major(vector, m, n, ld)
    out = fill(eltype(vector)(NaN), m, n)
    for j in 1:n, i in 1:m
        out[i, j] = vector[(i - 1) * ld + j]
    end
    return out
end

@testset "LAPACKE OPGTR/OPMTR packed layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), n in (2, 4), uplo in ('U', 'L'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        gen = "LAPACKE_" * prefix * "opgtr" * (save ? "_work" : "")
        app = "LAPACKE_" * prefix * "opmtr" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(gen, interface) == Libdl.dlsym(native, gen * suffix)
        @test BLAS.lbt_get_forward(app, interface) == Libdl.dlsym(native, app * suffix)
        pgen = Libdl.dlsym(route === :dll ? native : trampoline, gen * suffix)
        papp = Libdl.dlsym(route === :dll ? native : trampoline, app * suffix)
        rng = MersenneTwister(3200 + n)
        full = randn(rng, T, n, n)
        sym = Matrix(Symmetric(full, uplo == 'L' ? :L : :U))
        ap0 = packed_col(sym, uplo)
        d = zeros(T, n); e = zeros(T, max(0, n - 1))
        tau = zeros(T, max(0, n - 1))
        sptrd = Libdl.dlsym(native, prefix * "sptrd_" * suffix)
        @test lapacke_sptrd_setup(T, J, sptrd, uplo, n, ap0, d, e, tau) == 0
        qcol = vec(Matrix{T}(I, n, n))
        @test lapacke_opgtr_call(
            T, J, pgen, 102, uplo, n, ap0, tau, qcol, n,
            fill(T(NaN), max(1, n - 1)), save, gen, route,
        ) == 0
        ld = n + 2
        qrow = row_major(Matrix{T}(I, n, n), ld)
        @test lapacke_opgtr_call(
            T, J, pgen, 101, uplo, n, packed_row_from_col(ap0, uplo, n), tau,
            qrow, ld, fill(T(NaN), max(1, n - 1)), save, gen, route,
        ) == 0
        tol = 1000 * eps(T)
        @test isapprox(from_row_major(qrow, n, n, ld), reshape(qcol, n, n); rtol=tol, atol=tol)
        @test isapprox(reshape(qcol, n, n)' * reshape(qcol, n, n), Matrix{T}(I, n, n); rtol=tol, atol=tol)
        for side in ('L', 'R'), trans in ('N', 'T')
            m, nn = side == 'L' ? (n, max(1, n - 1)) : (max(1, n - 1), n)
            c0 = randn(MersenneTwister(3300 + m + nn), T, m, nn)
            ccol = vec(copy(c0)); crow = row_major(c0, nn + 2)
            @test lapacke_opmtr_call(
                T, J, papp, 102, side, uplo, trans, m, nn, ap0, tau, ccol, m,
                fill(T(NaN), max(1, max(m, nn))), save, app, route,
            ) == 0
            @test lapacke_opmtr_call(
                T, J, papp, 101, side, uplo, trans, m, nn,
                packed_row_from_col(ap0, uplo, n), tau, crow, nn + 2,
                fill(T(NaN), max(1, max(m, nn))), save, app, route,
            ) == 0
            @test isapprox(
                from_row_major(crow, m, nn, nn + 2), reshape(ccol, m, nn);
                rtol=tol, atol=tol,
            )
        end
    end
end

@testset "LAPACKE OPGTR/OPMTR diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"),), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        gen = Libdl.dlsym(native, "LAPACKE_" * prefix * "opgtr" * suffix)
        app = Libdl.dlsym(native, "LAPACKE_" * prefix * "opmtr" * suffix)
        n = 3
        ap = zeros(T, n * (n + 1) ÷ 2)
        tau = zeros(T, n - 1)
        q = vec(Matrix{T}(I, n, n))
        c = vec(Matrix{T}(I, n, n))
        @test lapacke_opgtr_call(
            T, J, gen, 301, 'U', n, ap, tau, q, n, T[], false,
            "LAPACKE_" * prefix * "opgtr", :dll,
        ) == -1
        @test lapacke_opgtr_call(
            T, J, gen, 101, 'U', n, ap, tau, q, n - 1, T[], false,
            "LAPACKE_" * prefix * "opgtr", :dll,
        ) == -7
        bad = copy(ap); bad[1] = T(NaN)
        @test lapacke_opgtr_call(
            T, J, gen, 102, 'U', n, bad, tau, q, n, T[], false,
            "LAPACKE_" * prefix * "opgtr", :dll,
        ) == -4
        @test lapacke_opmtr_call(
            T, J, app, 301, 'L', 'U', 'N', n, n, ap, tau, c, n, T[], false,
            "LAPACKE_" * prefix * "opmtr", :dll,
        ) == -1
        @test lapacke_opmtr_call(
            T, J, app, 101, 'L', 'U', 'N', n, n, ap, tau, c, n - 1, T[], false,
            "LAPACKE_" * prefix * "opmtr", :dll,
        ) == -10
        @test lapacke_opmtr_call(
            T, J, app, 102, 'L', 'U', 'N', n, n, bad, tau, c, n, T[], false,
            "LAPACKE_" * prefix * "opmtr", :dll,
        ) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
