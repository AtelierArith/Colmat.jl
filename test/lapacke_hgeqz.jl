using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_hgeqz_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, job, compq, compz,
        n, ilo, ihi, h, ldh, t, ldt, ar, ai, be, q, ldq, z, ldz,
        work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                if save
                    rwork = fill($R(NaN), max(1, n))
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T},
                         $J, Ptr{$T}, $J, Ptr{$R}),
                        layout, job, compq, compz, n, ilo, ihi, h, ldh, t, ldt,
                        ar, be, q, ldq, z, ldz, work, length(work), rwork)
                else
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T},
                         $J),
                        layout, job, compq, compz, n, ilo, ihi, h, ldh, t, ldt,
                        ar, be, q, ldq, z, ldz)
                end
            elseif save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$T}, $J),
                    layout, job, compq, compz, n, ilo, ihi, h, ldh, t, ldt,
                    ar, ai, be, q, ldq, z, ldz, work, length(work))
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Cchar, $J, $J, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                     Ptr{$T}, $J),
                    layout, job, compq, compz, n, ilo, ihi, h, ldh, t, ldt,
                    ar, ai, be, q, ldq, z, ldz)
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

@testset "LAPACKE HGEQZ layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (1, 4), job in ('E',),
        compq in ('N', 'V'), compz in ('N', 'V'), save in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "hgeqz" * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(2600 + n)
        h = randn(rng, T, n, n)
        t = zeros(T, n, n)
        for j in 1:n
            h[j, j] = T(1 + j)
            t[j, j] = T(2 + j)
            j < n && (t[j, j + 1] = T(0.5j))
            j > 1 && (h[j, j - 1] = T(0.25j))
        end
        q0 = Matrix{T}(I, n, n)
        z0 = Matrix{T}(I, n, n)
        ilo, ihi = 1, n
        # Column-major reference.
        hcol = vec(copy(h)); tcol = vec(copy(t))
        qcol = vec(copy(q0)); zcol = vec(copy(z0))
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 102, job, compq, compz, n, ilo, ihi, hcol, n,
            tcol, n, ar, ai, be, qcol, n, zcol, n, fill(T(NaN), 4n), save,
            name, route,
        ) == 0
        ld = n + 2
        hrow = row_major(h, ld); trow = row_major(t, ld)
        qrow = row_major(q0, ld); zrow = row_major(z0, ld)
        ar2 = zeros(T, n); ai2 = zeros(T, n); be2 = zeros(T, n)
        info = lapacke_hgeqz_call(
            T, R, J, pointer, 101, job, compq, compz, n, ilo, ihi, hrow, ld,
            trow, ld, ar2, ai2, be2, qrow, ld, zrow, ld, fill(T(NaN), 4n),
            save, name, route,
        )
        @test info == 0
        tol = 1000 * eps(R)
        @test isapprox(ar2, ar; rtol=tol, atol=tol)
        @test isapprox(ai2, ai; rtol=tol, atol=tol)
        @test isapprox(be2, be; rtol=tol, atol=tol)
        @test isapprox(from_row_major(hrow, n, n, ld), reshape(hcol, n, n); rtol=tol, atol=tol)
        @test isapprox(from_row_major(trow, n, n, ld), reshape(tcol, n, n); rtol=tol, atol=tol)
        if compq == 'V'
            @test isapprox(from_row_major(qrow, n, n, ld), reshape(qcol, n, n); rtol=tol, atol=tol)
        end
        if compz == 'V'
            @test isapprox(from_row_major(zrow, n, n, ld), reshape(zcol, n, n); rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE HGEQZ diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "hgeqz"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        h = Matrix{T}(I, n, n); t = Matrix{T}(I, n, n)
        q = Matrix{T}(I, n, n); z = Matrix{T}(I, n, n)
        ar = zeros(T, n); ai = zeros(T, n); be = zeros(T, n)
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 301, 'E', 'V', 'V', n, 1, n, vec(h), n, vec(t),
            n, ar, ai, be, vec(q), n, vec(z), n, T[], false, name, :dll,
        ) == -1
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 101, 'E', 'V', 'V', n, 1, n, vec(h), n - 1,
            vec(t), n, ar, ai, be, vec(q), n, vec(z), n, T[], false, name, :dll,
        ) == -9
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 101, 'E', 'V', 'V', n, 1, n, vec(h), n, vec(t),
            n, ar, ai, be, vec(q), n - 1, vec(z), n, T[], false, name, :dll,
        ) == (T <: Complex ? -15 : -16)
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 101, 'E', 'V', 'V', n, 1, n, vec(h), n, vec(t),
            n, ar, ai, be, vec(q), n, vec(z), n - 1, T[], false, name, :dll,
        ) == (T <: Complex ? -17 : -18)
        bad = copy(vec(h)); bad[1] = T(NaN)
        @test lapacke_hgeqz_call(
            T, R, J, pointer, 102, 'E', 'V', 'V', n, 1, n, bad, n, vec(t), n,
            ar, ai, be, vec(q), n, vec(z), n, T[], false, name, :dll,
        ) == -8
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
