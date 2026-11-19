using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_poequ_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, n, a, lda,
        s::Vector{$R}, cond, amax, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$R}),
                layout, n, a, lda, s, cond, amax)
        end
    end
    @eval function lapacke_ppequ_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, uplo, n, ap,
        s::Vector{$R}, cond, amax, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, Ptr{$T}, Ptr{$R}, Ptr{$R}, Ptr{$R}),
                layout, uplo, n, ap, s, cond, amax)
        end
    end
    @eval function lapacke_ppsv_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, nrhs, ap, b, ldb,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, Ptr{$T}, $J),
                layout, uplo, n, nrhs, ap, b, ldb)
        end
    end
    @eval function lapacke_pptri_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, ap, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}),
                  layout, uplo, n, ap)
        end
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
function packed_row(matrix, uplo)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), n * (n + 1) ÷ 2)
    if uplo == 'U'
        for i in 1:n, j in i:n
            out[(i - 1) * n - (i - 1) * i ÷ 2 + j] = matrix[i, j]
        end
    else
        for i in 1:n, j in 1:i
            out[(i - 1) * i ÷ 2 + j] = matrix[i, j]
        end
    end
    return out
end
function permute_row_from_col(ap, uplo, n)
    upper = uplo == 'U'
    out = fill(eltype(ap)(NaN), length(ap))
    for j in 1:n, i in 1:n
        (upper ? i <= j : i >= j) || continue
        rowidx = upper ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j
        colidx = upper ? j * (j - 1) ÷ 2 + i : (j - 1) * n - (j - 1) * j ÷ 2 + i
        out[rowidx] = ap[colidx]
    end
    return out
end
function native_pptrf(native, ::Type{T}, ::Type{J}, uplo, n, ap) where {T,J}
    suffix = J === Int64 ? "64_" : ""
    p = Libdl.dlsym(native, (T <: Complex ? (T === ComplexF32 ? "c" : "z") :
                             (T === Float32 ? "s" : "d")) * "pptrf_" * suffix)
    info = Ref{J}(-9)
    ccall(p, Cvoid, (Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}), uplo, J(n), ap, info)
    return info[]
end
function packed_matrix_row(vector, uplo, n)
    out = zeros(eltype(vector), n, n)
    if uplo == 'U'
        for i in 1:n, j in i:n
            out[i, j] = vector[(i - 1) * n - (i - 1) * i ÷ 2 + j]
        end
    else
        for i in 1:n, j in 1:i
            out[i, j] = vector[(i - 1) * i ÷ 2 + j]
        end
    end
    return out
end
function packed_matrix(vector, uplo, n)
    out = zeros(eltype(vector), n, n)
    if uplo == 'U'
        k = 1
        for j in 1:n, i in 1:j
            out[i, j] = vector[k]; k += 1
        end
    else
        k = 1
        for j in 1:n, i in j:n
            out[i, j] = vector[k]; k += 1
        end
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

@testset "LAPACKE POEQU/POEQUB/PPEQU/PPSV/PPTRI layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 4), uplo in ('U', 'L'),
        family in ("poequ", "poequb"), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(7300 + n)
        full = randn(rng, T, n, n)
        spd = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        # Dense equilibration on both layouts.
        acol = vec(copy(spd))
        scol = zeros(R, n); ccol = Ref(R(-1)); acolr = Ref(R(-1))
        @test lapacke_poequ_call(
            T, R, J, pointer, 102, n, acol, n, scol, ccol, acolr, name, route,
        ) == 0
        ld = n + 2
        arow = row_major(spd, ld)
        srow = zeros(R, n); crow = Ref(R(-1)); arowr = Ref(R(-1))
        @test lapacke_poequ_call(
            T, R, J, pointer, 101, n, arow, ld, srow, crow, arowr, name, route,
        ) == 0
        @test isapprox(srow, scol; rtol=1e-5, atol=1e-5)
        @test isapprox(crow[], ccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(arowr[], acolr[]; rtol=1e-5, atol=1e-5)
        # S is the reciprocal square root of the diagonal (binned for POEQUB).
        direct = [1 / sqrt(abs(spd[i, i])) for i in 1:n]
        if family == "poequ"
            @test isapprox(scol, R.(direct); rtol=1e-5, atol=1e-5)
        else
            @test all(0.5 .<= scol ./ R.(direct) .<= 2.0)
        end
    end
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 4), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        eqn = "LAPACKE_" * prefix * "ppequ" * (save ? "_work" : "")
        svn = "LAPACKE_" * prefix * "ppsv" * (save ? "_work" : "")
        ivn = "LAPACKE_" * prefix * "pptri" * (save ? "_work" : "")
        for nm in (eqn, svn, ivn)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        peq = Libdl.dlsym(route === :dll ? native : trampoline, eqn * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, svn * suffix)
        piv = Libdl.dlsym(route === :dll ? native : trampoline, ivn * suffix)
        rng = MersenneTwister(7400 + n)
        full = randn(rng, T, n, n)
        spd = Matrix(Hermitian(full * full' + T(n) * Matrix{T}(I, n, n)))
        # PPEQU on both layouts.
        apcol = packed_col(spd, uplo)
        scol = zeros(R, n); ccol = Ref(R(-1)); acolr = Ref(R(-1))
        @test lapacke_ppequ_call(
            T, R, J, peq, 102, uplo, n, apcol, scol, ccol, acolr, eqn, route,
        ) == 0
        aprow = packed_row(spd, uplo)
        srow = zeros(R, n); crow = Ref(R(-1)); arowr = Ref(R(-1))
        @test lapacke_ppequ_call(
            T, R, J, peq, 101, uplo, n, aprow, srow, crow, arowr, eqn, route,
        ) == 0
        @test isapprox(srow, scol; rtol=1e-5, atol=1e-5)
        @test isapprox(crow[], ccol[]; rtol=1e-5, atol=1e-5)
        @test isapprox(arowr[], acolr[]; rtol=1e-5, atol=1e-5)
        # PPSV end to end on both layouts.
        nrhs = 2
        b0 = randn(MersenneTwister(7500 + n), T, n, nrhs)
        apc2 = copy(apcol); bcol = vec(copy(b0))
        @test lapacke_ppsv_call(
            T, J, psv, 102, uplo, n, nrhs, apc2, bcol, n, svn, route,
        ) == 0
        apr2 = copy(aprow); brow = row_major(b0, nrhs + 2)
        @test lapacke_ppsv_call(
            T, J, psv, 101, uplo, n, nrhs, apr2, brow, nrhs + 2, svn, route,
        ) == 0
        tol = 4000 * eps(R)
        @test isapprox(
            from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        @test isequal(apr2, permute_row_from_col(apc2, uplo, n))
        scale = 4000 * n * eps(R) * norm(b0)
        @test isapprox(spd * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
        # PPTRI takes the packed Cholesky factor and returns inv(A); build the
        # factor with the native PPTRF entry and compare both layouts.
        fac = copy(apcol)
        @test native_pptrf(native, T, J, uplo, n, fac) == 0
        apf = copy(fac)
        @test lapacke_pptri_call(T, J, piv, 102, uplo, n, apf, ivn, route) == 0
        invc = packed_matrix(apf, uplo, n)
        aprf = permute_row_from_col(fac, uplo, n)
        @test lapacke_pptri_call(T, J, piv, 101, uplo, n, aprf, ivn, route) == 0
        @test isequal(aprf, permute_row_from_col(apf, uplo, n))
        invr = packed_matrix_row(aprf, uplo, n)
        toli = 8000 * eps(R)
        @test isapprox(invr, invc; rtol=toli, atol=toli)
        # The mirrored inverse satisfies inv * spd = I.
        minv = copy(invc)
        if uplo == 'U'
            for i in 1:n, j in 1:i-1
                minv[i, j] = conj(minv[j, i])
            end
        else
            for i in 1:n, j in i+1:n
                minv[i, j] = conj(minv[j, i])
            end
        end
        @test isapprox(minv * spd, Matrix{T}(I, n, n); rtol=toli, atol=toli)
    end
end

@testset "LAPACKE POEQU/POEQUB/PPEQU/PPSV/PPTRI diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for family in ("poequ", "poequb")
            name = "LAPACKE_" * prefix * family
            pointer = Libdl.dlsym(native, name * suffix)
            n = 3
            a = zeros(T, n * n)
            s = zeros(R, n); cond = Ref(R(0)); amax = Ref(R(0))
            @test lapacke_poequ_call(
                T, R, J, pointer, 301, n, a, n, s, cond, amax, name, :dll,
            ) == -1
            @test lapacke_poequ_call(
                T, R, J, pointer, 101, n, a, n - 1, s, cond, amax, name, :dll,
            ) == -4
            @test lapacke_poequ_call(
                T, R, J, pointer, 102, n, a, n - 1, s, cond, amax, name, :dll,
            ) == -4
            bad = copy(a); bad[1] = T(NaN)
            @test lapacke_poequ_call(
                T, R, J, pointer, 102, n, bad, n, s, cond, amax, name, :dll,
            ) == -3
        end
        n = 3
        eqn = "LAPACKE_" * prefix * "ppequ"
        svn = "LAPACKE_" * prefix * "ppsv"
        ivn = "LAPACKE_" * prefix * "pptri"
        peq = Libdl.dlsym(native, eqn * suffix)
        psv = Libdl.dlsym(native, svn * suffix)
        piv = Libdl.dlsym(native, ivn * suffix)
        ap = zeros(T, n * (n + 1) ÷ 2)
        s = zeros(R, n); cond = Ref(R(0)); amax = Ref(R(0))
        b = zeros(T, n * n)
        bad = copy(ap); bad[1] = T(NaN)
        @test lapacke_ppequ_call(
            T, R, J, peq, 301, 'U', n, ap, s, cond, amax, eqn, :dll,
        ) == -1
        @test lapacke_ppequ_call(
            T, R, J, peq, 102, 'U', n, bad, s, cond, amax, eqn, :dll,
        ) == -4
        @test lapacke_ppequ_call(
            T, R, J, peq, 102, 'X', n, ap, s, cond, amax, eqn, :dll,
        ) == -2
        @test lapacke_ppsv_call(
            T, J, psv, 301, 'U', n, n, ap, b, n, svn, :dll,
        ) == -1
        @test lapacke_ppsv_call(
            T, J, psv, 101, 'U', n, n, ap, b, n - 1, svn, :dll,
        ) == -7
        @test lapacke_ppsv_call(
            T, J, psv, 102, 'U', n, n, bad, b, n, svn, :dll,
        ) == -5
        badb = copy(b); badb[1] = T(NaN)
        @test lapacke_ppsv_call(
            T, J, psv, 102, 'U', n, n, ap, badb, n, svn, :dll,
        ) == -6
        @test lapacke_pptri_call(T, J, piv, 301, 'U', n, ap, ivn, :dll) == -1
        @test lapacke_pptri_call(T, J, piv, 102, 'U', n, bad, ivn, :dll) == -4
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
