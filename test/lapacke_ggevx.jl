using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_ggevx_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, n, a, lda, b, ldb,
        ar, ai, be, vl, ldvl, vr, ldvr, ilo, ihi, ls, rs, anrm, bnrm, rc, rv,
        name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                if high
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$J}, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$R},
                         Ptr{$R}, Ptr{$R}),
                        layout, 'N', 'V', 'V', 'N', n, a, lda, b, ldb, ar, be,
                        vl, ldvl, vr, ldvr, ilo, ihi, ls, rs, anrm, bnrm, rc, rv)
                else
                    work = fill($T(NaN), max(1, 2n)); rwork = fill($R(NaN), max(1, 8n))
                    iwork = zeros($J, max(1, 6n)); bwork = zeros($J, max(1, n))
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$J}, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$R},
                         Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$R}, Ptr{$J},
                         Ptr{$J}),
                        layout, 'N', 'V', 'V', 'N', n, a, lda, b, ldb, ar, be,
                        vl, ldvl, vr, ldvr, ilo, ihi, ls, rs, anrm, bnrm, rc,
                        rv, work, length(work), rwork, iwork, bwork)
                end
            elseif high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                     Ptr{$T}, Ptr{$T}, Ptr{$T}),
                    layout, 'N', 'V', 'V', 'N', n, a, lda, b, ldb, ar, ai, be,
                    vl, ldvl, vr, ldvr, ilo, ihi, ls, rs, anrm, bnrm, rc, rv)
            else
                work = fill($T(NaN), max(1, 8n)); iwork = zeros($J, max(1, 6n))
                bwork = zeros($J, max(1, n))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, Cchar, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                     Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}),
                    layout, 'N', 'V', 'V', 'N', n, a, lda, b, ldb, ar, ai, be,
                    vl, ldvl, vr, ldvr, ilo, ihi, ls, rs, anrm, bnrm, rc, rv,
                    work, length(work), iwork, bwork)
            end
        end
    end
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

@testset "LAPACKE GGEVX layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), high in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "ggevx" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(1500 + n)
        a = randn(rng, T, n, n); b = randn(rng, T, n, n)
        acol = vec(copy(a)); bcol = vec(copy(b))
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        vlref = zeros(T, n, n); vrref = zeros(T, n, n)
        ilo = Ref{J}(-1); ihi = Ref{J}(-1)
        ls = zeros(T, n); rs = zeros(T, n); an = Ref(R(-1)); bn = Ref(R(-1))
        rc = zeros(T, n); rv = zeros(T, n)
        @test lapacke_ggevx_call(
            T, R, J, pointer, 102, n, acol, n, bcol, n, ar, ai, be,
            vec(vlref), n, vec(vrref), n, ilo, ihi, ls, rs, an, bn, rc, rv,
            name, route, high,
        ) == 0
        ld = n + 2
        arow = row_major(a, ld); brow = row_major(b, ld)
        lrow = fill(T(NaN), n * ld); rrow = fill(T(NaN), n * ld)
        ar2 = zeros(T, n); ai2 = zeros(T, n); be2 = zeros(T, n)
        info = lapacke_ggevx_call(
            T, R, J, pointer, 101, n, arow, ld, brow, ld, ar2, ai2, be2,
            lrow, ld, rrow, ld, ilo, ihi, ls, rs, an, bn, rc, rv,
            name, route, high,
        )
        @test info == 0
        tol = 1000 * eps(R)
        @test isapprox(ar2, ar; rtol=tol, atol=tol)
        @test isapprox(ai2, ai; rtol=tol, atol=tol)
        @test isapprox(be2, be; rtol=tol, atol=tol)
        @test isapprox(from_row_major(lrow, n, n, ld), vlref; rtol=tol, atol=tol)
        @test isapprox(from_row_major(rrow, n, n, ld), vrref; rtol=tol, atol=tol)
    end
end

@testset "LAPACKE GGEVX diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "ggevx"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3; a = Matrix{T}(I, n, n); b = Matrix{T}(I, n, n)
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        ilo = Ref{J}(0); ihi = Ref{J}(0); ls = zeros(T, n); rs = zeros(T, n)
        @test lapacke_ggevx_call(
            T, R, J, pointer, 301, n, vec(a), n, vec(b), n, ar, ai, be,
            zeros(T, n * n), n, zeros(T, n * n), n, ilo, ihi, ls, rs,
            Ref(R(0)), Ref(R(0)), zeros(T, n), zeros(T, n), name, :dll, true,
        ) == -1
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
