using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # layout, mode (compz/jobz), n, d, e, z, ldz [, extras]
    @eval function lapacke_td_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, kind, work, layout, mode, n,
                                  d::Vector{$R}, e::Vector{$R}, z::Vector{$T}, ldz, w, name, route)
        lapacke288_selected_entry(name, $J, route) do
            if kind === :steqr || kind === :stev
                work ? ccall(p, $J, (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$R}),
                             layout, mode, n, d, e, z, ldz, w) :
                       ccall(p, $J, (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J),
                             layout, mode, n, d, e, z, ldz)
            else
                error("unsupported kind")
            end
        end
    end
end
# Query-capable drivers. STEVD/real STEDC: work, lwork, iwork, liwork; complex STEDC adds rwork.
for (T, R) in ((Float32, Float32), (Float64, Float64)), J in (Int32, Int64)
    @eval function lapacke_td_dc(::Type{$T}, ::Type{$J}, p, work, layout, mode, n, d, e, z, ldz,
                                 wk::Vector{$T}, iw::Vector{$J}, name, route; query=false)
        lapacke288_selected_entry(name, $J, route) do
            work ? ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, $J),
                         layout, mode, n, d, e, z, ldz, wk, query ? $J(-1) : $J(length(wk)), iw, query ? $J(-1) : $J(length(iw))) :
                   ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J),
                         layout, mode, n, d, e, z, ldz)
        end
    end
end
for (T, R) in ((ComplexF32, Float32), (ComplexF64, Float64)), J in (Int32, Int64)
    @eval function lapacke_td_dcc(::Type{$T}, ::Type{$R}, ::Type{$J}, p, work, layout, mode, n, d, e, z, ldz,
                                  wk::Vector{$T}, rw::Vector{$R}, iw::Vector{$J}, name, route; query=false)
        lapacke288_selected_entry(name, $J, route) do
            work ? ccall(p, $J, (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, $J, Ptr{$J}, $J),
                         layout, mode, n, d, e, z, ldz, wk, query ? $J(-1) : $J(length(wk)), rw, query ? $J(-1) : $J(length(rw)), iw, query ? $J(-1) : $J(length(iw))) :
                   ccall(p, $J, (Cint, Cchar, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J),
                         layout, mode, n, d, e, z, ldz)
        end
    end
end
for (R, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64))
    @eval function lapacke_sterf_call(::Type{$R}, ::Type{$J}, p, n, d, e, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, ($J, Ptr{$R}, Ptr{$R}), n, d, e)
        end
    end
end

td_eigs(d, e) = isempty(d) ? eltype(d)[] : eigvals(SymTridiagonal(d, e))
td_rowmajor(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
td_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
td_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]
td_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]

function td_check(T, R, n, d0, e0, dout, zout, q, tol)
    tri = Matrix(SymTridiagonal(d0, e0))
    @test isapprox(sort(dout), td_eigs(d0, e0); rtol=tol, atol=tol)
    a = q * T.(tri) * q'
    @test isapprox(a * zout, zout * Diagonal(T.(dout)); rtol=tol, atol=tol)
    @test isapprox(zout' * zout, Matrix{T}(I, n, n); rtol=tol, atol=tol)
end

@testset "LAPACKE STEQR/STEV/STEVD/STEDC/STERF" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (0, 1, 5), work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        lib = route === :dll ? native : trampoline
        rng = MersenneTwister(9500 + n)
        d0 = randn(rng, R, n) .+ R(3); e0 = randn(rng, R, max(n - 1, 0))
        tol = 2000 * max(n, 1) * eps(R)
        q = n == 0 ? zeros(T, 0, 0) : Matrix(qr(randn(rng, T, n, n)).Q)
        kinds = T <: Real ? (:steqr, :stev, :stevd, :stedc) : (:steqr, :stedc)
        for kind in kinds, mode in (kind in (:stev, :stevd) ? ('V',) : ('I', 'V')), layout in (102, 101)
            name = "LAPACKE_" * prefix * String(kind) * (work ? "_work" : "")
            @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
            p = Libdl.dlsym(lib, name * suffix)
            ld = layout == 102 ? max(n, 1) : n + 2
            start = (mode == 'V' && !(kind in (:stev, :stevd))) ? q : Matrix{T}(I, n, n)
            z = layout == 102 ? td_colvec(start, ld) : td_rowmajor(start, ld)
            if n == 0; z = fill(T(NaN), 1); end
            d = copy(d0); e = vcat(copy(e0), zeros(R, 1))
            if kind in (:steqr, :stev)
                w = fill(R(NaN), max(2n, 2) + 2)
                @test lapacke_td_call(T, R, J, p, kind, work, layout, mode, n, d, e, z, ld, w, name, route) == 0
            elseif T <: Real
                wk = fill(T(NaN), 1); iw = zeros(J, 1)
                if work
                    # query first
                    qw = [T(NaN)]; qi = zeros(J, 1)
                    @test lapacke_td_dc(T, J, p, true, layout, mode, n, d, e, z, ld, qw, qi, name, route; query=true) == 0
                    wk = fill(T(NaN), Int(qw[1])); iw = zeros(J, Int(qi[1]))
                    d = copy(d0); e = vcat(copy(e0), zeros(R, 1))
                    z = n == 0 ? fill(T(NaN), 1) : (layout == 102 ? td_colvec(start, ld) : td_rowmajor(start, ld))
                    @test lapacke_td_dc(T, J, p, true, layout, mode, n, d, e, z, ld, wk, iw, name, route) == 0
                else
                    @test lapacke_td_dc(T, J, p, false, layout, mode, n, d, e, z, ld, wk, iw, name, route) == 0
                end
            else
                if work
                    qw = [T(NaN)]; qr_ = [R(NaN)]; qi = zeros(J, 1)
                    @test lapacke_td_dcc(T, R, J, p, true, layout, mode, n, d, e, z, ld, qw, qr_, qi, name, route; query=true) == 0
                    wk = fill(T(NaN), max(Int(real(qw[1])), 1)); rw = fill(R(NaN), max(Int(qr_[1]), 1))
                    iw = zeros(J, max(Int(qi[1]), 1))
                    d = copy(d0); e = vcat(copy(e0), zeros(R, 1))
                    z = n == 0 ? fill(T(NaN), 1) : (layout == 102 ? td_colvec(start, ld) : td_rowmajor(start, ld))
                    @test lapacke_td_dcc(T, R, J, p, true, layout, mode, n, d, e, z, ld, wk, rw, iw, name, route) == 0
                else
                    @test lapacke_td_dcc(T, R, J, p, false, layout, mode, n, d, e, z, ld,
                                         T[], R[], J[], name, route) == 0
                end
            end
            n == 0 && continue
            zout = layout == 102 ? td_fromcol(z, n, ld) : td_fromrow(z, n, ld)
            qeff = (mode == 'V' && !(kind in (:stev, :stevd))) ? q : Matrix{T}(I, n, n)
            td_check(T, R, n, d0, e0, d, zout, qeff, tol)
            if layout == 101   # padding beyond the logical columns is untouched
                for i in 1:n, j in (n + 1):ld
                    @test isnan(z[(i - 1) * ld + j])
                end
            end
        end
        if T <: Real
            for w in ("", "_work")
                name = "LAPACKE_" * prefix * "sterf" * w
                @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
                p = Libdl.dlsym(lib, name * suffix)
                d = copy(d0); e = vcat(copy(e0), zeros(R, 1))
                @test lapacke_sterf_call(R, J, p, n, d, e, name, route) == 0
                @test isapprox(d, td_eigs(d0, e0); rtol=tol, atol=tol)
            end
        end
    end
end

@testset "LAPACKE tridiagonal driver diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "steqr"
        p = Libdl.dlsym(native, name * suffix)
        d = ones(R, 3); e = ones(R, 3); z = zeros(T, 9); w = zeros(R, 8)
        call(layout, ldz, dd, ee) = lapacke_td_call(T, R, J, p, :steqr, false, layout, 'I', 3, dd, ee, z, ldz, w, name, :dll)
        @test call(301, 3, d, e) == -1
        @test call(101, 2, d, e) == -7
        bd = copy(d); bd[2] = R(NaN)
        @test call(102, 3, bd, e) == -4
        be = copy(e); be[2] = R(NaN)
        @test call(102, 3, d, be) == -5
        @test call(102, 3, d, vcat(e[1:2], R(NaN))) != -5   # E has N-1 entries
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
