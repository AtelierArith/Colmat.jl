using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_sbtrd_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, vect, uplo, n, kd,
        ab, ldab, d::Vector{$R}, e::Vector{$R}, q, ldq, work, save, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if save
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R},
                     Ptr{$R}, Ptr{$T}, $J, Ptr{$T}),
                    layout, vect, uplo, n, kd, ab, ldab, d, e, q, ldq, work)
            else
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R},
                     Ptr{$R}, Ptr{$T}, $J),
                    layout, vect, uplo, n, kd, ab, ldab, d, e, q, ldq)
            end
        end
    end
end

function band_col(matrix, uplo, kd)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), (kd + 1) * n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[(kd + i - j + 1) + (j - 1) * (kd + 1)] = matrix[i, j]
        else
            -kd <= d <= 0 || continue
            out[(i - j + 1) + (j - 1) * (kd + 1)] = matrix[i, j]
        end
    end
    return out
end
function band_row(matrix, uplo, kd, ld)
    n = size(matrix, 1)
    out = fill(eltype(matrix)(NaN), (kd + 1) * ld)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[(kd + i - j) * ld + j] = matrix[i, j]
        else
            -kd <= d <= 0 || continue
            out[(i - j) * ld + j] = matrix[i, j]
        end
    end
    return out
end
function band_matrix_col(ab, uplo, n, kd)
    out = zeros(eltype(ab), n, n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[i, j] = ab[(kd + i - j + 1) + (j - 1) * (kd + 1)]
        else
            -kd <= d <= 0 || continue
            out[i, j] = ab[(i - j + 1) + (j - 1) * (kd + 1)]
        end
    end
    return out
end
function band_matrix_row(ab, uplo, n, kd, ld)
    out = zeros(eltype(ab), n, n)
    for j in 1:n, i in 1:n
        d = j - i
        if uplo == 'U'
            0 <= d <= kd || continue
            out[i, j] = ab[(kd + i - j) * ld + j]
        else
            -kd <= d <= 0 || continue
            out[i, j] = ab[(i - j) * ld + j]
        end
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

@testset "LAPACKE SBTRD/HBTRD band layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), (n, kd) in ((3, 1), (5, 2)), uplo in ('U', 'L'),
        vect in ('N', 'V'), save in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * (T <: Complex ? "hbtrd" : "sbtrd") * (save ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(3800 + n)
        bm = zeros(T, n, n)
        for j in 1:n, i in j:min(n, j + kd)
            bm[i, j] = i == j ? T(real(randn(rng, R)) + 2n) : randn(rng, T)
        end
        m = bm + bm'
        abcol = band_col(m, uplo, kd)
        dcol = zeros(R, n); ecol = zeros(R, max(0, n - 1))
        qcol = vec(Matrix{T}(I, n, n))
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 102, vect, uplo, n, kd, abcol, kd + 1,
            dcol, ecol, qcol, n, fill(T(NaN), n), save, name, route,
        ) == 0
        ld = n + 2
        abrow = band_row(m, uplo, kd, ld)
        drow = zeros(R, n); erow = zeros(R, max(0, n - 1))
        qrow = row_major(Matrix{T}(I, n, n), ld)
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 101, vect, uplo, n, kd, abrow, ld,
            drow, erow, qrow, ld, fill(T(NaN), n), save, name, route,
        ) == 0
        tol = 2000 * eps(R)
        @test isapprox(drow, dcol; rtol=tol, atol=tol)
        @test isapprox(erow, ecol; rtol=tol, atol=tol)
        @test isapprox(
            band_matrix_row(abrow, uplo, n, kd, ld),
            band_matrix_col(abcol, uplo, n, kd);
            rtol=tol, atol=tol,
        )
        if vect == 'V'
            @test isapprox(
                from_row_major(qrow, n, n, ld), reshape(qcol, n, n);
                rtol=tol, atol=tol,
            )
            @test isapprox(
                from_row_major(qrow, n, n, ld)' * from_row_major(qrow, n, n, ld),
                Matrix{T}(I, n, n);
                rtol=tol, atol=tol,
            )
        end
    end
end

@testset "LAPACKE SBTRD/HBTRD diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * (T <: Complex ? "hbtrd" : "sbtrd")
        pointer = Libdl.dlsym(native, name * suffix)
        n, kd = 4, 1
        ab = zeros(T, (kd + 1) * n)
        d = zeros(R, n); e = zeros(R, n - 1)
        q = vec(Matrix{T}(I, n, n))
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 301, 'V', 'U', n, kd, ab, kd + 1, d, e, q, n,
            T[], false, name, :dll,
        ) == -1
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 101, 'V', 'U', n, kd, ab, n - 1, d, e, q, n,
            T[], false, name, :dll,
        ) == -7
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 101, 'V', 'U', n, kd, ab, n, d, e, q, n - 1,
            T[], false, name, :dll,
        ) == -11
        bad = copy(ab); bad[kd + 1] = T(NaN)
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 102, 'V', 'U', n, kd, bad, kd + 1, d, e, q, n,
            T[], false, name, :dll,
        ) == -6
        badq = copy(q); badq[1] = T(NaN)
        @test lapacke_sbtrd_call(
            T, R, J, pointer, 102, 'V', 'U', n, kd, ab, kd + 1, d, e, badq, n,
            T[], false, name, :dll,
        ) == -10
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
