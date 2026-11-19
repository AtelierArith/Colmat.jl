using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_pftr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, n, a, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}),
                  layout, transr, uplo, n, a)
        end
    end
    @eval function lapacke_pftrs_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, n, nrhs, a, b,
        ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, Ptr{$T}, $J),
                layout, transr, uplo, n, nrhs, a, b, ldb)
        end
    end
end

function rfp_dims(transr, n)
    if uppercase(transr) == 'N'
        iseven(n) ? (n + 1, n ÷ 2) : (n, (n + 1) ÷ 2)
    else
        iseven(n) ? (n ÷ 2, n + 1) : ((n + 1) ÷ 2, n)
    end
end
# Build the column-major RFP rectangle with the native TRTTF entry.
function rfp_column(native, ::Type{T}, ::Type{J}, dense, transr, uplo, n) where {T,J}
    suffix = J === Int64 ? "64_" : ""
    p = Libdl.dlsym(native, (T <: Complex ? (T === ComplexF32 ? "c" : "z") :
                             (T === Float32 ? "s" : "d")) * "trttf_" * suffix)
    arf = zeros(T, n * (n + 1) ÷ 2)
    info = Ref{J}(-9)
    ccall(p, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}),
        transr, uplo, J(n), vec(dense), J(n), arf, info)
    @assert info[] == 0
    return arf
end
function rfp_row_from_column(acol, transr, n)
    (row, col) = rfp_dims(transr, n)
    m = reshape(acol, row, col)
    out = fill(eltype(acol)(NaN), row * col)
    for j in 1:col, i in 1:row
        out[(i - 1) * col + j] = m[i, j]
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

@testset "LAPACKE PFTRF/PFTRI/PFTRS RFP layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 4),
        transr in ('N', T <: Complex ? 'C' : 'T'), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        fname = "LAPACKE_" * prefix * "pftrf" * (save ? "_work" : "")
        iname = "LAPACKE_" * prefix * "pftri" * (save ? "_work" : "")
        sname = "LAPACKE_" * prefix * "pftrs" * (save ? "_work" : "")
        for nm in (fname, iname, sname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        pf = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        pi = Libdl.dlsym(route === :dll ? native : trampoline, iname * suffix)
        ps = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        rng = MersenneTwister(7000 + n)
        full = randn(rng, T, n, n)
        herm = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        acol0 = rfp_column(native, T, J, herm, transr, uplo, n)
        # PFTRF on both layouts.
        acol = copy(acol0)
        @test lapacke_pftr_call(
            T, J, pf, 102, transr, uplo, n, acol, fname, route,
        ) == 0
        arow = rfp_row_from_column(acol0, transr, n)
        @test lapacke_pftr_call(
            T, J, pf, 101, transr, uplo, n, arow, fname, route,
        ) == 0
        # The row-major factor is the dtf_trans transpose of the column one.
        @test isapprox(arow, rfp_row_from_column(acol, transr, n); rtol=10eps(R), atol=10eps(R))
        # PFTRI inverts the factor in place on both layouts.
        acol2 = copy(acol)
        @test lapacke_pftr_call(
            T, J, pi, 102, transr, uplo, n, acol2, iname, route,
        ) == 0
        arow2 = rfp_row_from_column(acol0, transr, n)
        @test lapacke_pftr_call(
            T, J, pf, 101, transr, uplo, n, arow2, fname, route,
        ) == 0
        @test lapacke_pftr_call(
            T, J, pi, 101, transr, uplo, n, arow2, iname, route,
        ) == 0
        @test isapprox(arow2, rfp_row_from_column(acol2, transr, n); rtol=10eps(R), atol=10eps(R))
        # PFTRS on both layouts with the shared factors.
        nrhs = 2
        b0 = randn(MersenneTwister(7100 + n), T, n, nrhs)
        bcol = vec(copy(b0))
        @test lapacke_pftrs_call(
            T, J, ps, 102, transr, uplo, n, nrhs, acol, bcol, n, sname, route,
        ) == 0
        brow = row_major(b0, nrhs + 2)
        @test lapacke_pftrs_call(
            T, J, ps, 101, transr, uplo, n, nrhs, arow, brow, nrhs + 2,
            sname, route,
        ) == 0
        tol = 4000 * eps(R)
        @test isapprox(
            from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        scale = 4000 * n * eps(R) * norm(b0)
        @test isapprox(herm * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE PFTRF/PFTRI/PFTRS diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        fname = "LAPACKE_" * prefix * "pftrf"
        sname = "LAPACKE_" * prefix * "pftrs"
        pf = Libdl.dlsym(native, fname * suffix)
        ps = Libdl.dlsym(native, sname * suffix)
        n = 3
        a = zeros(T, n * (n + 1) ÷ 2)
        @test lapacke_pftr_call(
            T, J, pf, 301, 'N', 'U', n, a, fname, :dll,
        ) == -1
        bad = copy(a); bad[1] = T(NaN)
        @test lapacke_pftr_call(
            T, J, pf, 102, 'N', 'U', n, bad, fname, :dll,
        ) == -5
        b = zeros(T, n * n)
        @test lapacke_pftrs_call(
            T, J, ps, 301, 'N', 'U', n, n, a, b, n, sname, :dll,
        ) == -1
        @test lapacke_pftrs_call(
            T, J, ps, 101, 'N', 'U', n, n, a, b, n - 1, sname, :dll,
        ) == -8
        @test lapacke_pftrs_call(
            T, J, ps, 102, 'N', 'U', n, n, bad, b, n, sname, :dll,
        ) == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
