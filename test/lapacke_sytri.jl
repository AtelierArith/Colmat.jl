using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, J) in ((Float32, Int32), (Float32, Int64), (Float64, Int32), (Float64, Int64),
               (ComplexF32, Int32), (ComplexF32, Int64), (ComplexF64, Int32), (ComplexF64, Int64))
    @eval function lapacke_sytri_factor(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, a, lda, ipiv, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}),
                  layout, uplo, n, a, lda, ipiv)
        end
    end
    # variant: 0 high, 1 work. kind: :tri (work n), :tri2 (work, lwork), :tri2x (work, nb)
    @eval function lapacke_sytri_call(
        ::Type{$T}, ::Type{$J}, pointer, kind, work, layout, uplo, n, a, lda, ipiv,
        extra, w::Vector{$T}, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if !work
                if kind === :tri2x
                    ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, $J),
                          layout, uplo, n, a, lda, ipiv, extra)
                else
                    ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}),
                          layout, uplo, n, a, lda, ipiv)
                end
            elseif kind === :tri
                ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}),
                      layout, uplo, n, a, lda, ipiv, w)
            else
                ccall(pointer, $J,
                      (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                      layout, uplo, n, a, lda, ipiv, w, extra)
            end
        end
    end
end

function sytri_row_major(matrix, ld)
    m, n = size(matrix)
    out = fill(eltype(matrix)(NaN), m * ld)
    for j in 1:n, i in 1:m
        out[(i - 1) * ld + j] = matrix[i, j]
    end
    return out
end
sytri_from_row(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]
in_triangle(uplo, i, j) = uplo == 'U' ? i <= j : i >= j

@testset "LAPACKE SYTRI/SYTRI2/SYTRI2X and HETRI family" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix, family) in
        ((Float32, "s", "sy"), (Float64, "d", "sy"),
         (ComplexF32, "c", "he"), (ComplexF64, "z", "he")),
        J in (Int32, Int64), n in (0, 1, 4), uplo in ('U', 'L'),
        (kind, base) in ((:tri, "tri"), (:tri2, "tri2"), (:tri2x, "tri2x")),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * base * (work ? "_work" : "")
        fname = "LAPACKE_" * prefix * family * "trf"
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        pfac = Libdl.dlsym(route === :dll ? native : trampoline, fname * suffix)
        rng = MersenneTwister(7100 + n)
        full = randn(rng, T, n, n)
        herm = full * full' + T(n) * Matrix{T}(I, n, n)
        herm = Matrix(T <: Complex ? Hermitian(herm, uplo == 'L' ? :L : :U) :
                                  Symmetric(herm, uplo == 'L' ? :L : :U))
        expected = n == 0 ? herm : inv(herm)
        tol = 20000 * n * eps(real(T)) * max(1, opnorm(expected, 1))
        nb = 2
        lw(layout) = kind === :tri ? max(n, 1) :
            kind === :tri2 ? max(n, 1) : (n + nb + 1) * (nb + 3)
        results = Dict()
        for layout in (102, 101)
            ld = layout == 102 ? max(n, 1) : n + 2
            vec_a = layout == 102 ? vcat(vec(copy(herm)), fill(T(NaN), n == 0 ? 1 : 0)) :
                    vcat(sytri_row_major(herm, ld), fill(T(NaN), n == 0 ? 1 : 0))
            ipiv = zeros(J, max(n, 1))
            @test lapacke_sytri_factor(T, J, pfac, layout, uplo, n, vec_a, ld, ipiv,
                                       fname, route) == 0
            factored = copy(vec_a)
            w = fill(T(NaN), lw(layout) + 3)
            extra = kind === :tri2x ? J(nb) : J(length(w))
            if kind === :tri2 && work
                q = fill(T(NaN), 1)
                @test lapacke_sytri_call(T, J, pointer, kind, true, layout, uplo, n,
                                         vec_a, ld, ipiv, J(-1), q, name, route) == 0
                @test real(q[1]) >= max(n, 1)
            end
            @test lapacke_sytri_call(T, J, pointer, kind, work, layout, uplo, n, vec_a,
                                     ld, ipiv, extra, w, name, route) == 0
            result = layout == 102 ? reshape(vec_a, max(n, 1), max(n, 1))[1:n, 1:n] :
                                     sytri_from_row(vec_a, n, n, ld)
            fac = layout == 102 ? reshape(factored, max(n, 1), max(n, 1))[1:n, 1:n] :
                                  sytri_from_row(factored, n, n, ld)
            for j in 1:n, i in 1:n
                if in_triangle(uplo, i, j)
                    @test isapprox(result[i, j], expected[i, j]; atol=tol, rtol=tol)
                else
                    # the opposite triangle is never written
                    @test isequal(result[i, j], fac[i, j])
                end
            end
            if layout == 101
                # padding beyond the logical columns is untouched
                for i in 1:n, j in (n + 1):ld
                    @test isnan(vec_a[(i - 1) * ld + j])
                end
            end
            results[layout] = result
        end
    end
end

@testset "LAPACKE SYTRI family diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix, family) in ((Float64, "d", "sy"), (ComplexF64, "z", "he")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        n = 3
        a = zeros(T, n * n)
        ipiv = ones(J, n)
        w = zeros(T, 64)
        for (kind, base) in ((:tri, "tri"), (:tri2, "tri2"), (:tri2x, "tri2x")), work in (false, true)
            name = "LAPACKE_" * prefix * family * base * (work ? "_work" : "")
            p = Libdl.dlsym(native, name * suffix)
            extra = kind === :tri2x ? J(2) : J(length(w))
            call(layout, lda, mat) = lapacke_sytri_call(T, J, p, kind, work, layout, 'U', n,
                                                        mat, lda, ipiv, extra, w, name, :dll)
            @test call(301, n, a) == -1
            @test call(101, n - 1, a) == -5
        end
        for (kind, base) in ((:tri, "tri"), (:tri2, "tri2"), (:tri2x, "tri2x"))
            name = "LAPACKE_" * prefix * family * base
            p = Libdl.dlsym(native, name * suffix)
            bad = copy(a); bad[4] = T(NaN)   # (1,2) is in the upper triangle
            extra = kind === :tri2x ? J(2) : J(0)
            @test lapacke_sytri_call(T, J, p, kind, false, 102, 'U', n, bad, n, ipiv,
                                     extra, w, name, :dll) == -4
            low = copy(a); low[2] = T(NaN)   # (2,1) is outside UPLO='U'
            @test lapacke_sytri_call(T, J, p, kind, false, 102, 'U', n, low, n, ipiv,
                                     extra, w, name, :dll) != -4
        end
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
