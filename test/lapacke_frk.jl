using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_frk_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, layout, transr, uplo, trans, n, k,
                                   alpha, a::Vector{$T}, lda, beta, c::Vector{$T}, name, route)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, Cchar, Cchar, $J, $J, $R, Ptr{$T}, $J, $R, Ptr{$T}),
                  layout, transr, uplo, trans, n, k, alpha, a, lda, beta, c)
        end
    end
    @eval function lapacke_frk_trttf(::Type{$T}, ::Type{$J}, p, name, route, layout, transr, uplo, n, a, lda, arf)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$T}), layout, transr, uplo, n, a, lda, arf)
        end
    end
    @eval function lapacke_frk_tfttr(::Type{$T}, ::Type{$J}, p, name, route, layout, transr, uplo, n, arf, a, lda)
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, Ptr{$T}, $J), layout, transr, uplo, n, arf, a, lda)
        end
    end
end

function frk_dims(transr, n)
    transr == 'N' ? (iseven(n) ? (n + 1, n ÷ 2) : (n, (n + 1) ÷ 2)) :
                    (iseven(n) ? (n ÷ 2, n + 1) : ((n + 1) ÷ 2, n))
end
function frk_rowbuf(colbuf, transr, n)
    r, c = frk_dims(transr, n)
    out = similar(colbuf)
    for i in 1:r, j in 1:c
        out[(i - 1) * c + j] = colbuf[i + (j - 1) * r]
    end
    out
end
function frk_colbuf(rowbuf, transr, n)
    r, c = frk_dims(transr, n)
    out = similar(rowbuf)
    for i in 1:r, j in 1:c
        out[i + (j - 1) * r] = rowbuf[(i - 1) * c + j]
    end
    out
end
function frk_rowmajor(m, ld)
    out = fill(eltype(m)(NaN), size(m, 1) * ld)
    for j in 1:size(m, 2), i in 1:size(m, 1)
        out[(i - 1) * ld + j] = m[i, j]
    end
    out
end

@testset "LAPACKE SFRK/HFRK" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix, fam) in ((Float32, Float32, "s", "sfrk"), (Float64, Float64, "d", "sfrk"),
                                (ComplexF32, Float32, "c", "hfrk"), (ComplexF64, Float64, "z", "hfrk")),
        J in (Int32, Int64), n in (0, 1, 4, 5), k in (0, 3), uplo in ('U', 'L'),
        transr in ('N', T <: Complex ? 'C' : 'T'), trans in ('N', T <: Complex ? 'C' : 'T'),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * fam * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        lib = route === :dll ? native : trampoline
        p = Libdl.dlsym(lib, name * suffix)
        tr_name = "LAPACKE_" * prefix * "trttf"; tf_name = "LAPACKE_" * prefix * "tfttr"
        ptr_trttf = Libdl.dlsym(lib, tr_name * suffix); ptr_tfttr = Libdl.dlsym(lib, tf_name * suffix)
        rng = MersenneTwister(9900 + 10n + k)
        nrowa, ka = trans == 'N' ? (n, k) : (k, n)
        a0 = randn(rng, T, nrowa, ka)
        c0 = randn(rng, T, n, n)
        c0 = T <: Complex ? Matrix(Hermitian(c0 + c0', uplo == 'L' ? :L : :U)) : Matrix(Symmetric(c0 + c0', uplo == 'L' ? :L : :U))
        alpha, beta = R(0.75), R(-0.5)
        opa = trans == 'N' ? a0 : a0'
        want = alpha * (n == 0 ? zeros(T, 0, 0) : (k == 0 ? zeros(T, n, n) : opa * opa')) + beta * c0
        # RFP array of the original C through the native TRTTF entry.
        rfp0 = fill(T(NaN), max(n * (n + 1) ÷ 2, 1))
        full = copy(c0)
        n > 0 && @test lapacke_frk_trttf(T, J, ptr_trttf, tr_name, route, 102, transr, uplo, n, vec(copy(full)), n, rfp0) == 0
        for layout in (102, 101)
            lda = layout == 102 ? max(nrowa, 1) : max(ka, 1) + 2
            a = layout == 102 ? vec(copy(a0)) : frk_rowmajor(a0, lda)
            isempty(a) && (a = fill(T(NaN), 1))
            c = layout == 102 ? copy(rfp0) : frk_rowbuf(rfp0, transr, n)
            @test lapacke_frk_call(T, R, J, p, layout, transr, uplo, trans, n, k, alpha, a, lda, beta, c,
                                   name, route) == 0
            n == 0 && continue
            ccol = layout == 102 ? c : frk_colbuf(c, transr, n)
            out = fill(T(NaN), n * n)
            @test lapacke_frk_tfttr(T, J, ptr_tfttr, tf_name, route, 102, transr, uplo, n, ccol, out, n) == 0
            res = reshape(out, n, n)
            tol = 5000 * eps(R) * max(1, norm(want, 1))
            for j in 1:n, i in 1:n
                (uplo == 'U' ? i <= j : i >= j) || continue
                @test isapprox(res[i, j], want[i, j]; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "LAPACKE SFRK/HFRK diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix, fam) in ((Float64, Float64, "d", "sfrk"), (ComplexF64, Float64, "z", "hfrk")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * fam
        p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 12); c = zeros(T, 10)
        call(layout, lda, aa, cc, al=R(1), be=R(0)) = lapacke_frk_call(T, R, J, p, layout, 'N', 'U', 'N', 4, 3, al, aa, lda, be, cc, name, :dll)
        @test call(301, 4, a, c) == -1
        @test call(101, 2, a, c) == -9
        @test call(102, 4, a, c, R(NaN)) == -7
        bad = copy(a); bad[1] = T(NaN)
        @test call(102, 4, bad, c) == -8
        @test call(102, 4, a, c, R(1), R(NaN)) == -10
        badc = copy(c); badc[3] = T(NaN)
        @test call(102, 4, a, badc) == -11
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
