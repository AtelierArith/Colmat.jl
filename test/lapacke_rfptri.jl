using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_tfsm_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, transr, side,
        uplo, trans, diag, m, n, alpha::$T, a, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, Cchar, Cchar, Cchar, Cchar, Cchar, $J, $J, $T,
                 Ptr{$T}, Ptr{$T}, $J),
                layout, transr, side, uplo, trans, diag, m, n, alpha, a, b, ldb)
        end
    end
    @eval function lapacke_tftri_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, transr, uplo, diag, n, a,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, Cchar, Cchar, $J, Ptr{$T}),
                  layout, transr, uplo, diag, n, a)
        end
    end
    @eval function lapacke_ptsv_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, n, nrhs,
        d::Vector{$R}, e::Vector{$T}, b, ldb, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J,
                (Cint, $J, $J, Ptr{$R}, Ptr{$T}, Ptr{$T}, $J),
                layout, n, nrhs, d, e, b, ldb)
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
function native_trttf(native, ::Type{T}, ::Type{J}, dense, transr, uplo, n) where {T,J}
    suffix = J === Int64 ? "64_" : ""
    p = Libdl.dlsym(native, (T <: Complex ? (T === ComplexF32 ? "c" : "z") :
                             (T === Float32 ? "s" : "d")) * "trttf_" * suffix)
    arf = zeros(T, n * (n + 1) ÷ 2)
    info = Ref{J}(-9)
    ccall(p, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}),
        transr, uplo, J(n), vec(copy(dense)), J(n), arf, info)
    @assert info[] == 0
    return arf
end
function native_tfttr(native, ::Type{T}, ::Type{J}, arf, transr, uplo, n) where {T,J}
    suffix = J === Int64 ? "64_" : ""
    p = Libdl.dlsym(native, (T <: Complex ? (T === ComplexF32 ? "c" : "z") :
                             (T === Float32 ? "s" : "d")) * "tfttr_" * suffix)
    a = zeros(T, n * n)
    info = Ref{J}(-9)
    ccall(p, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ptr{T}, Ref{J}, Ref{J}),
        transr, uplo, J(n), arf, a, J(n), info)
    @assert info[] == 0
    return reshape(a, n, n)
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

@testset "LAPACKE TFSM/TFTRI/PTSV layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (3, 4),
        transr in ('N', T <: Complex ? 'C' : 'T'), uplo in ('U', 'L'),
        save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        sname = "LAPACKE_" * prefix * "tfsm" * (save ? "_work" : "")
        iname = "LAPACKE_" * prefix * "tftri" * (save ? "_work" : "")
        for nm in (sname, iname)
            @test BLAS.lbt_get_forward(nm, interface) == Libdl.dlsym(native, nm * suffix)
        end
        psm = Libdl.dlsym(route === :dll ? native : trampoline, sname * suffix)
        pin = Libdl.dlsym(route === :dll ? native : trampoline, iname * suffix)
        rng = MersenneTwister(7600 + n)
        # Dense triangular matrix with a well-conditioned diagonal; the RFP
        # buffer is built with the native TRTTF over a full carrier matrix.
        tri = zeros(T, n, n)
        for j in 1:n, i in 1:n
            if uplo == 'U'
                i <= j || continue
                tri[i, j] = i == j ? T(2 + i) : T(0.3) * randn(rng, T)
            else
                i >= j || continue
                tri[i, j] = i == j ? T(2 + i) : T(0.3) * randn(rng, T)
            end
        end
        arfcol = native_trttf(native, T, J, tri, transr, uplo, n)
        arfrow = rfp_row_from_column(arfcol, transr, n)
        # TFSM on both layouts and sides/trans. SIDE='L' uses the n-by-n
        # factor; SIDE='R' uses a separate nrhs-by-nrhs factor.
        nrhs = 2
        tri2 = zeros(T, nrhs, nrhs)
        for j in 1:nrhs, i in 1:nrhs
            if uplo == 'U'
                i <= j || continue
                tri2[i, j] = i == j ? T(3 + i) : T(0.3) * randn(rng, T)
            else
                i >= j || continue
                tri2[i, j] = i == j ? T(3 + i) : T(0.3) * randn(rng, T)
            end
        end
        arfcol2 = native_trttf(native, T, J, tri2, transr, uplo, nrhs)
        arfrow2 = rfp_row_from_column(arfcol2, transr, nrhs)
        b0 = randn(MersenneTwister(7700 + n), T, n, nrhs)
        alpha = T(2)
        for side in ('L', 'R'), trans in (T <: Complex ? ('N', 'C') : ('N', 'T')),
            diagc in ('N', 'U')

            base = side == 'L' ? tri : tri2
            kdim = side == 'L' ? n : nrhs
            arfc = side == 'L' ? arfcol : arfcol2
            arfr = side == 'L' ? arfrow : arfrow2
            op = if diagc == 'U'
                unit = base + Matrix{T}(I, kdim, kdim) - Diagonal(LinearAlgebra.diag(base))
                trans == 'N' ? unit : (T <: Complex ? unit' : unit')
            else
                trans == 'N' ? base : (T <: Complex ? base' : base')
            end
            # Column-major.
            bcol = vec(copy(b0))
            @test lapacke_tfsm_call(
                T, R, J, psm, 102, transr, side, uplo, trans, diagc, n, nrhs,
                alpha, arfc, bcol, n, sname, route,
            ) == 0
            xcol = reshape(bcol, n, nrhs)
            resid = side == 'L' ? op * xcol : xcol * op
            tol = 4000 * eps(R) * (1 + norm(b0))
            @test isapprox(resid, alpha .* b0; rtol=tol, atol=tol)
            # Row-major with padding.
            brow = row_major(b0, nrhs + 2)
            @test lapacke_tfsm_call(
                T, R, J, psm, 101, transr, side, uplo, trans, diagc, n, nrhs,
                alpha, arfr, brow, nrhs + 2, sname, route,
            ) == 0
            @test isapprox(
                from_row_major(brow, n, nrhs, nrhs + 2), xcol;
                rtol=4000 * eps(R), atol=4000 * eps(R),
            )
        end
        # TFTRI on both layouts.
        acol = copy(arfcol)
        @test lapacke_tftri_call(
            T, J, pin, 102, transr, uplo, 'N', n, acol, iname, route,
        ) == 0
        invcol = native_tfttr(native, T, J, acol, transr, uplo, n)
        arow = copy(arfrow)
        @test lapacke_tftri_call(
            T, J, pin, 101, transr, uplo, 'N', n, arow, iname, route,
        ) == 0
        @test isequal(arow, rfp_row_from_column(acol, transr, n))
        # The triangular inverse multiplies back to the identity.
        toli = 8000 * eps(R)
        tmat = uplo == 'U' ? UpperTriangular(tri) : LowerTriangular(tri)
        invt = uplo == 'U' ? UpperTriangular(invcol) : LowerTriangular(invcol)
        @test isapprox(invt * tmat, Matrix{T}(I, n, n); rtol=toli, atol=toli)
    end
    # PTSV on both layouts.
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        vname = "LAPACKE_" * prefix * "ptsv" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(vname, interface) == Libdl.dlsym(native, vname * suffix)
        psv = Libdl.dlsym(route === :dll ? native : trampoline, vname * suffix)
        rng = MersenneTwister(7800 + n)
        d0 = R[2 + i for i in 1:n]
        e0 = T[0.3i - 0.1 for i in 1:max(0, n - 1)]
        nrhs = 2
        b0 = randn(MersenneTwister(7900 + n), T, n, nrhs)
        dcol = copy(d0); ecol = copy(e0); bcol = vec(copy(b0))
        @test lapacke_ptsv_call(
            T, R, J, psv, 102, n, nrhs, dcol, ecol, bcol, n, vname, route,
        ) == 0
        drow = copy(d0); erow = copy(e0); brow = row_major(b0, nrhs + 2)
        @test lapacke_ptsv_call(
            T, R, J, psv, 101, n, nrhs, drow, erow, brow, nrhs + 2, vname,
            route,
        ) == 0
        tol = 4000 * eps(R)
        @test isapprox(
            from_row_major(brow, n, nrhs, nrhs + 2), reshape(bcol, n, nrhs);
            rtol=tol, atol=tol,
        )
        @test isequal(drow, dcol) && isequal(erow, ecol)
        A = SymTridiagonal(T.(d0), T.(e0))
        scale = 4000 * n * eps(R) * norm(b0)
        @test isapprox(A * reshape(bcol, n, nrhs), b0; rtol=scale, atol=scale)
    end
end

@testset "LAPACKE TFSM/TFTRI/PTSV diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        sname = "LAPACKE_" * prefix * "tfsm"
        iname = "LAPACKE_" * prefix * "tftri"
        vname = "LAPACKE_" * prefix * "ptsv"
        psm = Libdl.dlsym(native, sname * suffix)
        pin = Libdl.dlsym(native, iname * suffix)
        psv = Libdl.dlsym(native, vname * suffix)
        n, nrhs = 3, 2
        arf = zeros(T, n * (n + 1) ÷ 2)
        b = zeros(T, n * nrhs)
        @test lapacke_tfsm_call(
            T, R, J, psm, 301, 'N', 'L', 'U', 'N', 'N', n, nrhs, T(1), arf,
            b, n, sname, :dll,
        ) == -1
        @test lapacke_tfsm_call(
            T, R, J, psm, 101, 'N', 'L', 'U', 'N', 'N', n, nrhs, T(1), arf,
            b, n - 1, sname, :dll,
        ) == -12
        bad = copy(arf); bad[1] = T(NaN)
        @test lapacke_tfsm_call(
            T, R, J, psm, 102, 'N', 'L', 'U', 'N', 'N', n, nrhs, T(1), bad,
            b, n, sname, :dll,
        ) == -10
        @test lapacke_tfsm_call(
            T, R, J, psm, 102, 'N', 'L', 'U', 'N', 'N', n, nrhs, T(NaN), arf,
            b, n, sname, :dll,
        ) == -9
        badb = copy(b); badb[1] = T(NaN)
        @test lapacke_tfsm_call(
            T, R, J, psm, 102, 'N', 'L', 'U', 'N', 'N', n, nrhs, T(1), arf,
            badb, n, sname, :dll,
        ) == -11
        @test lapacke_tftri_call(
            T, J, pin, 301, 'N', 'U', 'N', n, arf, iname, :dll,
        ) == -1
        @test lapacke_tftri_call(
            T, J, pin, 102, 'N', 'U', 'N', n, bad, iname, :dll,
        ) == -6
        d = zeros(R, n); e = zeros(T, max(0, n - 1))
        @test lapacke_ptsv_call(
            T, R, J, psv, 301, n, nrhs, d, e, b, n, vname, :dll,
        ) == -1
        @test lapacke_ptsv_call(
            T, R, J, psv, 101, n, nrhs, d, e, b, nrhs - 1, vname, :dll,
        ) == -7
        @test lapacke_ptsv_call(
            T, R, J, psv, 102, n, nrhs, d, e, badb, n, vname, :dll,
        ) == -6
        badd = copy(d); badd[1] = R(NaN)
        @test lapacke_ptsv_call(
            T, R, J, psv, 102, n, nrhs, badd, e, b, n, vname, :dll,
        ) == -4
        bade = copy(e); isempty(bade) || (bade[1] = T(NaN))
        if !isempty(bade)
            @test lapacke_ptsv_call(
                T, R, J, psv, 102, n, nrhs, d, bade, b, n, vname, :dll,
            ) == -5
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
