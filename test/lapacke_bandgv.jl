using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

# Build and run one LAPACKE ccall from a list of (type, value) pairs. The
# signatures differ across band/packed, GV/GVD/GST, high/work and real/complex.
function bg_ccall(p, ::Type{J}, pairs) where {J}
    types = Expr(:tuple, (t for (t, _) in pairs)...)
    ex = Expr(:call, :ccall, p, J, types, (v for (_, v) in pairs)...)
    return Core.eval(@__MODULE__, ex)
end

function bg_call(::Type{T}, ::Type{R}, ::Type{J}, p, name, route; shape, kind, work, query=false, layout, itype=1,
                 jobz='N', uplo='U', n, ka=0, kb=0, a, lda=1, b, ldb=1, w=R[], z=T[], ldz=1,
                 wk=T[], rw=R[], iw=J[]) where {T,R,J}
    cplx = T <: Complex
    lq = query ? J(-1) : J(length(wk)); lrq = query ? J(-1) : J(length(rw)); liq = query ? J(-1) : J(length(iw))
    pr = Tuple{DataType,Any}[(Cint, Cint(layout))]
    if shape === :packed && kind !== :gst
        push!(pr, (J, J(itype)))
    elseif shape === :packed
        push!(pr, (J, J(itype)))
    end
    push!(pr, (Cchar, Cchar(jobz)))
    push!(pr, (Cchar, Cchar(uplo)))
    push!(pr, (J, J(n)))
    if shape === :band
        push!(pr, (J, J(ka))); push!(pr, (J, J(kb)))
        push!(pr, (Ptr{T}, a)); push!(pr, (J, J(lda)))
        push!(pr, (Ptr{T}, b)); push!(pr, (J, J(ldb)))
    else
        push!(pr, (Ptr{T}, a)); push!(pr, (Ptr{T}, b))
    end
    if kind === :gst
        if shape === :band
            push!(pr, (Ptr{T}, z)); push!(pr, (J, J(ldz)))
        end
        # SPGST/SBGST have no eigenvalue output
    else
        push!(pr, (Ptr{R}, w)); push!(pr, (Ptr{T}, z)); push!(pr, (J, J(ldz)))
    end
    if work
        if kind === :gv
            push!(pr, (Ptr{T}, wk)); cplx && push!(pr, (Ptr{R}, rw))
        elseif kind === :gst
            shape === :band && (push!(pr, (Ptr{T}, wk)); cplx && push!(pr, (Ptr{R}, rw)))
        else
            push!(pr, (Ptr{T}, wk)); push!(pr, (J, lq))
            cplx && (push!(pr, (Ptr{R}, rw)); push!(pr, (J, lrq)))
            push!(pr, (Ptr{J}, iw)); push!(pr, (J, liq))
        end
    end
    # packed GST has neither jobz nor a z argument; the helper above inserted jobz, so drop it.
    if shape === :packed && kind === :gst
        deleteat!(pr, 3)
    elseif kind === :gst && shape === :band
        # vect is the jobz slot; keep it.
    end
    # band GV/GVD/GST have no itype: remove it
    lapacke288_selected_entry(name, J, route) do
        bg_ccall(p, J, pr)
    end
end

function bg_band_col(m, uplo, kd)
    n = size(m, 1); out = fill(eltype(m)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[kd + i - j + (j - 1) * (kd + 1) + 1] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[i - j + (j - 1) * (kd + 1) + 1] = m[i, j]; end
    end
    out
end
function bg_band_row(m, uplo, kd, ld)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * ld)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; out[(kd + i - j) * ld + j] = m[i, j]
        else; 0 <= i - j <= kd || continue; out[(i - j) * ld + j] = m[i, j]; end
    end
    out
end
function bg_unband_col(v, n, kd, uplo)
    m = zeros(eltype(v), n, n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; m[i, j] = v[kd + i - j + (j - 1) * (kd + 1) + 1]
        else; 0 <= i - j <= kd || continue; m[i, j] = v[i - j + (j - 1) * (kd + 1) + 1]; end
    end
    m
end
function bg_unband_row(v, n, kd, uplo, ld)
    m = zeros(eltype(v), n, n)
    for j in 1:n, i in 1:n
        if uplo == 'U'; 0 <= j - i <= kd || continue; m[i, j] = v[(kd + i - j) * ld + j]
        else; 0 <= i - j <= kd || continue; m[i, j] = v[(i - j) * ld + j]; end
    end
    m
end
function bg_pack_col(m, uplo)
    n = size(m, 1); out = eltype(m)[]
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); push!(out, m[i, j]); end
    out
end
function bg_pack_row(m, uplo)
    n = size(m, 1); out = fill(eltype(m)(NaN), n * (n + 1) ÷ 2)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        out[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j] = m[i, j]
    end
    out
end
function bg_unpack_col(ap, n, uplo)
    m = zeros(eltype(ap), n, n); k = 0
    for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n)); k += 1; m[i, j] = ap[k]; end
    m
end
function bg_unpack_row(ap, n, uplo)
    m = zeros(eltype(ap), n, n)
    for i in 1:n, j in (uplo == 'U' ? (i:n) : (1:i))
        m[i, j] = ap[uplo == 'U' ? (i - 1) * n - (i - 1) * i ÷ 2 + j : (i - 1) * i ÷ 2 + j]
    end
    m
end
bg_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
bg_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
bg_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]
bg_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]

function bg_problem(T, n, ka, kb, uplo, seed)
    rng = MersenneTwister(seed)
    f = randn(rng, T, n, n)
    A = f + f'
    A = T[abs(i - j) <= ka ? A[i, j] : zero(T) for i in 1:n, j in 1:n]
    g = randn(rng, T, n, n)
    B = T[abs(i - j) <= kb ? (g + g')[i, j] : zero(T) for i in 1:n, j in 1:n]
    B = B + T(2 * kb + 2 + n) * Matrix{T}(I, n, n)   # diagonally dominant, hence positive definite
    herm(M) = Matrix(T <: Complex ? Hermitian(M, uplo == 'L' ? :L : :U) : Symmetric(M, uplo == 'L' ? :L : :U))
    return herm(A), herm(B)
end

@testset "LAPACKE band/packed generalized SBGV/SPGV/SBGVD/SPGVD" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, bf, pf) in ((Float32, Float32, "s", "sb", "sp"), (Float64, Float64, "d", "sb", "sp"),
                                   (ComplexF32, Float32, "c", "hb", "hp"), (ComplexF64, Float64, "z", "hb", "hp")),
        J in (Int32, Int64), n in (1, 5), uplo in ('U', 'L'), shape in (:band, :packed), kind in (:gv, :gvd),
        jobz in ('N', 'V'), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        base = (shape === :band ? bf : pf) * (kind === :gv ? "gv" : "gvd")
        name = "LAPACKE_" * prefix * base * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        ka, kb = min(2, n - 1), min(1, n - 1)
        for itype in (shape === :packed ? (1, 2, 3) : (1,))
            A, B = bg_problem(T, n, ka, kb, uplo, 9100 + n + itype)
            expected = itype == 1 ? sort(real.(eigvals(B \ A))) : sort(real.(eigvals(B * A)))
            tol = 50000 * n * eps(R) * max(1, maximum(abs, expected))
            for layout in (102, 101)
                ld = layout == 102 ? n : n + 2
                if shape === :band
                    lda = layout == 102 ? ka + 1 : n + 2; ldb = layout == 102 ? kb + 1 : n + 2
                    a = layout == 102 ? bg_band_col(A, uplo, ka) : bg_band_row(A, uplo, ka, lda)
                    b = layout == 102 ? bg_band_col(B, uplo, kb) : bg_band_row(B, uplo, kb, ldb)
                else
                    lda = 1; ldb = 1
                    a = layout == 102 ? bg_pack_col(A, uplo) : bg_pack_row(A, uplo)
                    b = layout == 102 ? bg_pack_col(B, uplo) : bg_pack_row(B, uplo)
                end
                w = fill(R(NaN), n); z = fill(T(NaN), max(n * ld, 1))
                wk = fill(T(NaN), 8n + 64); rw = fill(R(NaN), 8n + 64); iw = zeros(J, 8n + 64)
                kw = (; shape, kind, work, layout, itype, jobz, uplo, n, ka, kb, lda, ldb, ldz=ld)
                if work && kind === :gvd
                    qwk = [T(NaN)]; qrw = [R(NaN)]; qiw = zeros(J, 1)
                    @test bg_call(T, R, J, p, name, route; kw..., query=true, a=copy(a), b=copy(b), w=copy(w), z=copy(z),
                                  wk=qwk, rw=qrw, iw=qiw) == 0
                    wk = fill(T(NaN), max(Int(real(qwk[1])), 1)); iw = zeros(J, max(Int(qiw[1]), 1))
                    T <: Complex && (rw = fill(R(NaN), max(Int(qrw[1]), 1)))
                elseif work
                    rw = fill(R(NaN), max(3n, 1)); wk = fill(T(NaN), max(3n, 1))
                end
                @test bg_call(T, R, J, p, name, route; kw..., a, b, w, z, wk, rw, iw) == 0
                @test isapprox(sort(w), expected; rtol=tol, atol=tol)
                if jobz == 'V'
                    x = layout == 102 ? bg_fromcol(z, n, ld) : bg_fromrow(z, n, ld)
                    if itype == 1
                        @test isapprox(A * x, B * x * Diagonal(T.(w)); rtol=tol, atol=tol)
                        @test isapprox(x' * B * x, Matrix{T}(I, n, n); rtol=tol, atol=tol)
                    elseif itype == 2
                        @test isapprox(A * B * x, x * Diagonal(T.(w)); rtol=tol, atol=tol)
                    else
                        @test isapprox(B * A * x, x * Diagonal(T.(w)); rtol=tol, atol=tol)
                    end
                    if layout == 101
                        for i in 1:n, j in (n + 1):ld
                            @test isnan(z[(i - 1) * ld + j])
                        end
                    end
                end
            end
        end
    end
end

@testset "LAPACKE band/packed generalized diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, bf, pf) in ((Float64, Float64, "d", "sb", "sp"), (ComplexF64, Float64, "z", "hb", "hp")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        for kind in (:gv, :gvd)
            sfx = kind === :gv ? "gv" : "gvd"
            nb = "LAPACKE_" * prefix * bf * sfx; npk = "LAPACKE_" * prefix * pf * sfx
            pb = Libdl.dlsym(native, nb * suffix); pp = Libdl.dlsym(native, npk * suffix)
            a = zeros(T, 20); b = zeros(T, 20); w = zeros(R, 4); z = zeros(T, 16)
            cb(layout, lda, ldb, ldz, aa, bb) = bg_call(T, R, J, pb, nb, :dll; shape=:band, kind, work=false, layout, jobz='V',
                                                          uplo='U', n=4, ka=1, kb=1, a=aa, lda, b=bb, ldb, w, z, ldz)
            @test cb(301, 2, 2, 4, a, b) == -1
            @test cb(101, 3, 4, 4, a, b) == -8
            @test cb(101, 4, 3, 4, a, b) == -10
            @test cb(101, 4, 4, 3, a, b) == -13
            bad = copy(a); bad[2] = T(NaN)
            @test cb(102, 2, 2, 4, bad, b) == -7
            @test cb(102, 2, 2, 4, a, bad) == -9
            cp(layout, ldz, aa, bb) = bg_call(T, R, J, pp, npk, :dll; shape=:packed, kind, work=false, layout, jobz='V',
                                               uplo='U', n=4, a=aa, b=bb, w, z, ldz)
            @test cp(301, 4, a, b) == -1
            @test cp(101, 3, a, b) == -10
            @test cp(102, 4, bad, b) == -6
            @test cp(102, 4, a, bad) == -7
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_bg_pbstf(::Type{$T}, ::Type{$J}, p, uplo, n, kb, bb, ldbb, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, $J, Ptr{$T}, $J), 102, uplo, n, kb, bb, ldbb)
        end
    end
    @eval function lapacke_bg_pptrf(::Type{$T}, ::Type{$J}, p, uplo, n, ap, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}), 102, uplo, n, ap)
        end
    end
end

@testset "LAPACKE SBGST/HBGST and SPGST/HPGST" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, bf, pf) in ((Float32, Float32, "s", "sb", "sp"), (Float64, Float64, "d", "sb", "sp"),
                                   (ComplexF32, Float32, "c", "hb", "hp"), (ComplexF64, Float64, "z", "hb", "hp")),
        J in (Int32, Int64), n in (2, 5), uplo in ('U', 'L'), shape in (:band, :packed), work in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        base = (shape === :band ? bf : pf) * "gst"
        name = "LAPACKE_" * prefix * base * ((work && shape === :band) || (work && shape === :packed) ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        ka, kb = min(2, n - 1), min(1, n - 1)
        A, B = bg_problem(T, n, ka, kb, uplo, 9400 + n)
        # Split/plain Cholesky factors through the native entries.
        if shape === :band
            fname = "LAPACKE_" * prefix * "pbstf"
            fb = bg_band_col(B, uplo, kb)
            @test lapacke_bg_pbstf(T, J, Libdl.dlsym(lib, fname * suffix), uplo, n, kb, fb, kb + 1, fname, route) == 0
            Bfac = bg_unband_col(fb, n, kb, uplo)
        else
            fname = "LAPACKE_" * prefix * "pptrf"
            fb = bg_pack_col(B, uplo)
            @test lapacke_bg_pptrf(T, J, Libdl.dlsym(lib, fname * suffix), uplo, n, fb, fname, route) == 0
            Bfac = bg_unpack_col(fb, n, uplo)
        end
        for itype in (shape === :packed ? (1, 2, 3) : (1,)), vect in (shape === :band ? ('N', 'V') : ('N',)), layout in (102, 101)
            expected = itype == 1 ? sort(real.(eigvals(B \ A))) : sort(real.(eigvals(B * A)))
            tol = 50000 * n * eps(R) * max(1, maximum(abs, expected))
            ld = layout == 102 ? n : n + 2
            if shape === :band
                lda = layout == 102 ? ka + 1 : n + 2; ldb = layout == 102 ? kb + 1 : n + 2
                a = layout == 102 ? bg_band_col(A, uplo, ka) : bg_band_row(A, uplo, ka, lda)
                b = layout == 102 ? copy(fb) : bg_band_row(Bfac, uplo, kb, ldb)
            else
                lda = 1; ldb = 1
                a = layout == 102 ? bg_pack_col(A, uplo) : bg_pack_row(A, uplo)
                b = layout == 102 ? copy(fb) : bg_pack_row(Bfac, uplo)
            end
            x = fill(T(NaN), max(n * ld, 1))
            wk = fill(T(NaN), max(3n, 1) + 4); rw = fill(R(NaN), max(3n, 1) + 4)
            bcopy = copy(b)
            @test bg_call(T, R, J, p, name, route; shape, kind=:gst, work, layout, itype, jobz=vect, uplo, n, ka, kb,
                          a, lda, b, ldb, z=x, ldz=ld, wk, rw) == 0
            @test isequal(b, bcopy)   # B is read only
            C = shape === :band ? (layout == 102 ? bg_unband_col(a, n, ka, uplo) : bg_unband_row(a, n, ka, uplo, lda)) :
                                  (layout == 102 ? bg_unpack_col(a, n, uplo) : bg_unpack_row(a, n, uplo))
            Cf = Matrix(T <: Complex ? Hermitian(C, uplo == 'L' ? :L : :U) : Symmetric(C, uplo == 'L' ? :L : :U))
            @test isapprox(sort(real.(eigvals(Cf))), expected; rtol=tol, atol=tol)
            if shape === :band && vect == 'V'
                xm = layout == 102 ? bg_fromcol(x, n, ld) : bg_fromrow(x, n, ld)
                @test isapprox(xm' * B * xm, Matrix{T}(I, n, n); rtol=tol, atol=tol)
                @test isapprox(xm' * A * xm, Cf; rtol=tol, atol=tol)
            end
        end
    end
end
