using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_orgtr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, uplo, n, a, lda, tau,
        name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}),
                    layout, uplo, n, a, lda, tau,
                )
            else
                work = fill($T(NaN), max(1, 2n))
                ccall(
                    pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J),
                    layout, uplo, n, a, lda, tau, work, length(work),
                )
            end
        end
    end
    @eval function lapacke_ormtr_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, uplo, trans, m, n,
        a, lda, tau, c, ldc, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J),
                    layout, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                )
            else
                work = fill($T(NaN), max(1, 2m))
                ccall(
                    pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                    work, length(work),
                )
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

@testset "LAPACKE ORGTR/UNGTR and ORMTR/UNMTR layouts" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (2, 5), uplo in ('L', 'U'),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        gen_name = "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr") * (high ? "" : "_work")
        app_name = "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr") * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(gen_name, interface) == Libdl.dlsym(native, gen_name * suffix)
        @test BLAS.lbt_get_forward(app_name, interface) == Libdl.dlsym(native, app_name * suffix)
        gen_pointer = Libdl.dlsym(route === :dll ? native : trampoline, gen_name * suffix)
        app_pointer = Libdl.dlsym(route === :dll ? native : trampoline, app_name * suffix)
        rng0 = MersenneTwister(50 + n)
        a0 = randn(rng0, T, n, n)
        tau = randn(rng0, T, max(0, n - 1))
        # Column-major reference.
        acol = vec(copy(a0))
        ref = lapacke_orgtr_call(
            T, J, gen_pointer, 102, uplo, n, acol, n, tau, gen_name, route, high,
        )
        @test ref == 0
        # Row-major.
        ld = n + 2
        arow = row_major(a0, ld)
        info = lapacke_orgtr_call(
            T, J, gen_pointer, 101, uplo, n, arow, ld, tau, gen_name, route, high,
        )
        @test info == 0
        @test isapprox(from_row_major(arow, n, n, ld), reshape(acol, n, n); rtol=50 * eps(real(T)), atol=50 * eps(real(T)))
        # Applicator on a rectangular C.
        rng = MersenneTwister(80 + n)
        m, nn = n, max(1, n - 1)
        c0 = randn(rng, T, m, nn)
        for side in ('L', 'R'), trans in (T <: Complex ? ('N', 'C') : ('N', 'T'))
            ccol = vec(copy(c0))
            cref = lapacke_ormtr_call(
                T, J, app_pointer, 102, side, uplo, trans, m, nn,
                acol, n, tau, ccol, m, app_name, route, high,
            )
            @test cref == 0
            lc = nn + 2
            crow = row_major(c0, lc)
            cinfo = lapacke_ormtr_call(
                T, J, app_pointer, 101, side, uplo, trans, m, nn,
                arow, ld, tau, crow, lc, app_name, route, high,
            )
            @test cinfo == 0
            @test isapprox(from_row_major(crow, m, nn, lc), reshape(ccol, m, nn); rtol=100 * eps(real(T)), atol=100 * eps(real(T)))
        end
    end
end


@testset "LAPACKE ORGTR/ORMTR diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        gen = Libdl.dlsym(native, "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr") * suffix)
        app = Libdl.dlsym(native, "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr") * suffix)
        n = 3
        a = Matrix{T}(I, n, n)
        tau = zeros(T, n - 1)
        @test lapacke_orgtr_call(
            T, J, gen, 301, 'L', n, vec(a), n, tau,
            "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr"), :dll, true,
        ) == -1
        @test lapacke_orgtr_call(
            T, J, gen, 102, '?', n, vec(a), n, tau,
            "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr"), :dll, true,
        ) == -2
        @test lapacke_orgtr_call(
            T, J, gen, 101, 'L', n, vec(a), n - 1, tau,
            "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr"), :dll, true,
        ) == -5
        bad = copy(vec(a)); bad[1] = T(NaN)
        @test lapacke_orgtr_call(
            T, J, gen, 102, 'L', n, bad, n, tau,
            "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr"), :dll, true,
        ) == -4
        badtau = copy(tau); badtau[1] = T(NaN)
        @test lapacke_orgtr_call(
            T, J, gen, 102, 'L', n, vec(a), n, badtau,
            "LAPACKE_" * prefix * (T <: Complex ? "ungtr" : "orgtr"), :dll, true,
        ) == -6
        c = Matrix{T}(I, n, n)
        @test lapacke_ormtr_call(
            T, J, app, 301, 'L', 'L', 'N', n, n, vec(a), n, tau, vec(c), n,
            "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr"), :dll, true,
        ) == -1
        @test lapacke_ormtr_call(
            T, J, app, 102, '?', 'L', 'N', n, n, vec(a), n, tau, vec(c), n,
            "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr"), :dll, true,
        ) == -2
        @test lapacke_ormtr_call(
            T, J, app, 101, 'L', 'L', 'N', n, n, vec(a), n - 1, tau, vec(c), n,
            "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr"), :dll, true,
        ) == -8
        @test lapacke_ormtr_call(
            T, J, app, 101, 'L', 'L', 'N', n, n, vec(a), n, tau, vec(c), n - 1,
            "LAPACKE_" * prefix * (T <: Complex ? "unmtr" : "ormtr"), :dll, true,
        ) == -11
    end
    ccall(
        Libdl.dlsym(native, :colmat_blas_last_error),
        Cint, (),
    )
end
