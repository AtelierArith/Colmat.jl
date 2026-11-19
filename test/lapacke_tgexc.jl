using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    if T <: Real
        @eval function lapacke_tgexc_call(::Type{$T}, ::Type{$J}, p, work, layout, wq, wz, n,
                                         a, lda, b, ldb, q, ldq, z, ldz, ifst::Vector{$J}, ilst::Vector{$J},
                                         w::Vector{$T}, name, route)
            lapacke288_selected_entry(name, $J, route) do
                if work
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                                  Ptr{$J}, Ptr{$J}, Ptr{$T}, $J),
                          layout, wq, wz, n, a, lda, b, ldb, q, ldq, z, ldz, ifst, ilst, w, length(w))
                else
                    ccall(p, $J, (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                                  Ptr{$J}, Ptr{$J}),
                          layout, wq, wz, n, a, lda, b, ldb, q, ldq, z, ldz, ifst, ilst)
                end
            end
        end
    else
        @eval function lapacke_tgexc_call(::Type{$T}, ::Type{$J}, p, work, layout, wq, wz, n,
                                         a, lda, b, ldb, q, ldq, z, ldz, ifst::Vector{$J}, ilst::Vector{$J},
                                         w::Vector{$T}, name, route)
            lapacke288_selected_entry(name, $J, route) do
                ccall(p, $J, (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J, $J, $J),
                      layout, wq, wz, n, a, lda, b, ldb, q, ldq, z, ldz, ifst[1], ilst[1])
            end
        end
    end
end

function tg_rowmajor(m, ld)
    out = fill(eltype(m)(NaN), size(m, 1) * ld)
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[(i - 1) * ld + j] = m[i, j]
    end
    out
end
tg_colvec(m, ld) = (out = fill(eltype(m)(NaN), ld * size(m, 2));
    for j in 1:size(m, 2), i in 1:size(m, 1); out[i + (j - 1) * ld] = m[i, j]; end; out)
tg_fromcol(v, n, ld) = [v[i + (j - 1) * ld] for i in 1:n, j in 1:n]
tg_fromrow(v, n, ld) = [v[(i - 1) * ld + j] for i in 1:n, j in 1:n]

@testset "LAPACKE TGEXC" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (1, 4), (wq, wz) in ((1, 1), (0, 0)), work in (false, true),
        route in (:dll, :lbt_trampoline)

        (!(T <: Real) && work) && continue
        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "tgexc" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        R = real(T)
        rng = MersenneTwister(9800 + n)
        a0 = triu(randn(rng, T, n, n)); b0 = triu(randn(rng, T, n, n))
        for i in 1:n
            a0[i, i] = T(i); b0[i, i] = T(1 + 0.25 * i)
        end
        tol = 5000 * eps(R) * max(n, 1)
        for layout in (102, 101)
            ld = layout == 102 ? max(n, 1) : n + 2
            pack(m) = layout == 102 ? tg_colvec(m, ld) : tg_rowmajor(m, ld)
            unpack(v) = layout == 102 ? tg_fromcol(v, n, ld) : tg_fromrow(v, n, ld)
            a = pack(a0); b = pack(b0)
            ident = Matrix{T}(I, n, n)
            q = pack(ident); z = pack(ident)
            n == 0 && (a = fill(T(NaN), 1); b = copy(a); q = copy(a); z = copy(a))
            ifst = J[n >= 2 ? 1 : 1]; ilst = J[n >= 2 ? n : 1]
            w = fill(T(NaN), max(4n + 16, 16))
            @test lapacke_tgexc_call(T, J, p, work, layout, J(wq), J(wz), n, a, ld, b, ld, q, ld, z, ld,
                                     ifst, ilst, w, name, route) == 0
            n == 0 && continue
            an, bn = unpack(a), unpack(b)
            ratios = [an[i, i] / bn[i, i] for i in 1:n]
            if n >= 2
                @test isapprox(ratios, [[a0[i, i] / b0[i, i] for i in 2:n]; a0[1, 1] / b0[1, 1]]; rtol=tol)
                @test istriu(round.(an; digits=3)) || all(abs(an[i, j]) <= tol * 100 for j in 1:n for i in j+1:n)
                if wq == 1
                    qn, zn = unpack(q), unpack(z)
                    @test isapprox(qn * an * zn', a0; rtol=tol, atol=tol)
                    @test isapprox(qn * bn * zn', b0; rtol=tol, atol=tol)
                    @test isapprox(qn' * qn, ident; rtol=tol, atol=tol)
                end
            end
        end
    end
end

@testset "LAPACKE TGEXC diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "tgexc"
        p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 16); b = zeros(T, 16); q = zeros(T, 16); z = zeros(T, 16)
        ifst = J[1]; ilst = J[2]; w = zeros(T, 32)
        call(layout, lda, ldb, ldq, ldz, aa, bb) = lapacke_tgexc_call(T, J, p, false, layout, J(1), J(1), 4, aa, lda, bb, ldb,
                                                                       q, ldq, z, ldz, ifst, ilst, w, name, :dll)
        @test call(301, 4, 4, 4, 4, a, b) == -1
        @test call(101, 3, 4, 4, 4, a, b) == -6
        @test call(101, 4, 3, 4, 4, a, b) == -8
        @test call(101, 4, 4, 3, 4, a, b) == -10
        @test call(101, 4, 4, 4, 3, a, b) == -12
        bad = copy(a); bad[1] = T(NaN)
        @test call(102, 4, 4, 4, 4, bad, b) == -5
        @test call(102, 4, 4, 4, 4, a, bad) == -7
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
