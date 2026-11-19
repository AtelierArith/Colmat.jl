using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_gst_call(::Type{$T}, ::Type{$J}, p, layout, itype, uplo, n, a, lda, b, ldb, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, $J, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J), layout, itype, uplo, n, a, lda, b, ldb)
        end
    end
    # kind: :gv (work: work,lwork[,rwork]) or :gvd (work: work,lwork[,rwork,lrwork],iwork,liwork)
    @eval function lapacke_gv_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, kind, work, query, layout, itype, jobz, uplo, n,
                                  a, lda, b, ldb, w::Vector{$R}, wk::Vector{$T}, rw::Vector{$R}, iw::Vector{$J}, name, route)
        lapacke288_selected_entry(name, $J, route) do
            lq = query ? $J(-1) : $J(length(wk)); lrq = query ? $J(-1) : $J(length(rw)); liq = query ? $J(-1) : $J(length(iw))
            if !work
                ccall(p, $J, (Cint, $J, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}),
                      layout, itype, jobz, uplo, n, a, lda, b, ldb, w)
            elseif kind === :gv && !($T <: Complex)
                ccall(p, $J, (Cint, $J, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J),
                      layout, itype, jobz, uplo, n, a, lda, b, ldb, w, wk, lq)
            elseif kind === :gv
                ccall(p, $J, (Cint, $J, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$R}),
                      layout, itype, jobz, uplo, n, a, lda, b, ldb, w, wk, lq, rw)
            elseif !($T <: Complex)
                ccall(p, $J, (Cint, $J, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$J}, $J),
                      layout, itype, jobz, uplo, n, a, lda, b, ldb, w, wk, lq, iw, liq)
            else
                ccall(p, $J, (Cint, $J, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$J}, $J),
                      layout, itype, jobz, uplo, n, a, lda, b, ldb, w, wk, lq, rw, lrq, iw, liq)
            end
        end
    end
end

function sg_rowmajor(m, ld)
    out = fill(eltype(m)(NaN), size(m, 1) * ld)
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[(i - 1) * ld + j] = m[i, j]
    end
    out
end
function sg_colvec(m, ld)
    out = fill(eltype(m)(NaN), ld * size(m, 2))
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[i + (j - 1) * ld] = m[i, j]
    end
    out
end
sg_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]
sg_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]

@testset "LAPACKE SYGV/HEGV/SYGVD/HEGVD" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam, famd) in ((Float32, Float32, "s", "sygv", "sygvd"), (Float64, Float64, "d", "sygv", "sygvd"),
                                      (ComplexF32, Float32, "c", "hegv", "hegvd"), (ComplexF64, Float64, "z", "hegv", "hegvd")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), itype in (1, 2, 3), jobz in ('N', 'V'),
        kind in (:gv, :gvd), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * (kind === :gv ? fam : famd) * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9900 + n + itype)
        f = randn(rng, T, n, n); g = randn(rng, T, n, n)
        A = Matrix(Hermitian(f + f')); B = Matrix(Hermitian(g * g' + T(n) * I))
        tol = 20000 * n * eps(R)
        # Reference eigenvalues of the standardized problem.
        ref = itype == 1 ? eigvals(Hermitian(B \ A) |> x -> Matrix(x)) : nothing
        Ah = Hermitian(A, uplo == 'L' ? :L : :U); Bh = Hermitian(B, uplo == 'L' ? :L : :U)
        expected = itype == 1 ? sort(real.(eigvals(Matrix(Bh) \ Matrix(Ah)))) :
                   sort(real.(eigvals(Matrix(Bh) * Matrix(Ah))))
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? sg_colvec(Matrix(Ah), ld) : sg_rowmajor(Matrix(Ah), ld)
            b = layout == 102 ? sg_colvec(Matrix(Bh), ld) : sg_rowmajor(Matrix(Bh), ld)
            w = fill(R(NaN), n)
            wk = fill(T(NaN), 8n + 64); rw = fill(R(NaN), 8n + 64); iw = zeros(J, 8n + 64)
            if work
                qwk = [T(NaN)]; qrw = [R(NaN)]; qiw = zeros(J, 1)
                @test lapacke_gv_call(T, R, J, p, kind, true, true, layout, J(itype), jobz, uplo, n, copy(a), ld, copy(b), ld,
                                      copy(w), qwk, qrw, qiw, name, route) == 0
                wk = fill(T(NaN), max(Int(real(qwk[1])), 1))
                if kind === :gvd
                    iw = zeros(J, max(Int(qiw[1]), 1))
                    T <: Complex && (rw = fill(R(NaN), max(Int(qrw[1]), 1)))
                else
                    rw = fill(R(NaN), max(3n - 2, 1))
                end
            end
            @test lapacke_gv_call(T, R, J, p, kind, work, false, layout, J(itype), jobz, uplo, n, a, ld, b, ld,
                                  w, wk, rw, iw, name, route) == 0
            @test isapprox(sort(w), expected; rtol=tol * max(1, maximum(abs, expected)), atol=tol * max(1, maximum(abs, expected)))
            if jobz == 'V'
                x = layout == 102 ? sg_fromcol(a, n, ld) : sg_fromrow(a, n, ld)
                Am, Bm = Matrix(Ah), Matrix(Bh)
                if itype == 1
                    @test isapprox(Am * x, Bm * x * Diagonal(T.(w)); rtol=tol * 50, atol=tol * 50)
                    @test isapprox(x' * Bm * x, Matrix{T}(I, n, n); rtol=tol * 50, atol=tol * 50)
                elseif itype == 2
                    @test isapprox(Am * Bm * x, x * Diagonal(T.(w)); rtol=tol * 50, atol=tol * 50)
                else
                    @test isapprox(Bm * Am * x, x * Diagonal(T.(w)); rtol=tol * 50, atol=tol * 50)
                end
            end
            if layout == 101
                for i in 1:n, j in (n + 1):ld
                    @test isnan(a[(i - 1) * ld + j]) && isnan(b[(i - 1) * ld + j])
                end
            end
        end
    end
end

@testset "LAPACKE SYGST/HEGST" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam) in ((Float32, Float32, "s", "sygst"), (Float64, Float64, "d", "sygst"),
                                (ComplexF32, Float32, "c", "hegst"), (ComplexF64, Float64, "z", "hegst")),
        J in (Int32, Int64), n in (1, 4), uplo in ('U', 'L'), itype in (1, 2, 3), work in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        rng = MersenneTwister(9950 + n + itype)
        f = randn(rng, T, n, n); g = randn(rng, T, n, n)
        A = Matrix(Hermitian(f + f')); B0 = Matrix(Hermitian(g * g' + T(n) * I))
        # B must hold the Cholesky factor of the same triangle.
        C = cholesky(Hermitian(B0, uplo == 'L' ? :L : :U))
        Bfac = uplo == 'L' ? Matrix(C.L) : Matrix(C.U)
        Ah = Matrix(Hermitian(A, uplo == 'L' ? :L : :U))
        tol = 20000 * n * eps(R) * max(1, opnorm(B0, 1))
        want = if uplo == 'U'
            U = Bfac
            itype == 1 ? (U') \ Ah / U : U * Ah * U'
        else
            L = Bfac
            itype == 1 ? L \ Ah / (L') : L' * Ah * L
        end
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? sg_colvec(Ah, ld) : sg_rowmajor(Ah, ld)
            b = layout == 102 ? sg_colvec(Bfac, ld) : sg_rowmajor(Bfac, ld)
            before_b = copy(b)
            @test lapacke_gst_call(T, J, p, layout, J(itype), uplo, n, a, ld, b, ld, name, route) == 0
            res = layout == 102 ? sg_fromcol(a, n, ld) : sg_fromrow(a, n, ld)
            for j in 1:n, i in 1:n
                (uplo == 'U' ? i <= j : i >= j) || continue
                @test isapprox(res[i, j], want[i, j]; rtol=tol, atol=tol)
            end
            @test isequal(b, before_b)   # B is read only
        end
    end
end

@testset "LAPACKE dense generalized diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, fam, famd) in ((Float64, Float64, "d", "sygv", "sygvd"), (ComplexF64, Float64, "z", "hegv", "hegvd")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for (kind, nm) in ((:gv, fam), (:gvd, famd))
            name = "LAPACKE_" * prefix * nm
            p = Libdl.dlsym(native, name * suffix)
            a = zeros(T, 16); b = zeros(T, 16); w = zeros(R, 4)
            call(layout, lda, ldb, aa, bb) = lapacke_gv_call(T, R, J, p, kind, false, false, layout, J(1), 'V', 'U', 4, aa, lda,
                                                              bb, ldb, w, T[], R[], J[], name, :dll)
            @test call(301, 4, 4, a, b) == -1
            @test call(101, 3, 4, a, b) == -7
            @test call(101, 4, 3, a, b) == -9
            bad = copy(a); bad[5] = T(NaN)     # (1,2) lies in the upper triangle
            @test call(102, 4, 4, bad, b) == -6
            @test call(102, 4, 4, a, bad) == -8
            low = copy(a); low[2] = T(NaN)     # (2,1) is outside UPLO='U'
            @test call(102, 4, 4, low, b) != -6
        end
        name = "LAPACKE_" * prefix * (T <: Complex ? "hegst" : "sygst")
        p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 16); b = zeros(T, 16)
        @test lapacke_gst_call(T, J, p, 301, J(1), 'U', 4, a, 4, b, 4, name, :dll) == -1
        @test lapacke_gst_call(T, J, p, 101, J(1), 'U', 4, a, 3, b, 4, name, :dll) == -6
        @test lapacke_gst_call(T, J, p, 101, J(1), 'U', 4, a, 4, b, 3, name, :dll) == -8
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
