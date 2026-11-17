using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_trsen_call(
        ::Type{$T}, ::Type{$R}, ::Type{$J}, pointer, layout, job, compq,
        select::Vector{$J}, n, t, ldt, q, ldq, wr, wi, m, s, sep,
        name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            if $T <: Complex
                w = wr
                if high
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$J}, Ptr{$R}, Ptr{$R}),
                        layout, job, compq, select, n, t, ldt, q, ldq, w, m, s, sep)
                else
                    work = fill($T(NaN), max(1, 2n))
                    ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, Ptr{$J}, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J),
                        layout, job, compq, select, n, t, ldt, q, ldq, w, m, s,
                        sep, work, length(work))
                end
            elseif high
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, Ptr{$J}, Ptr{$T}, Ptr{$T}),
                    layout, job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s, sep)
            else
                work = fill($T(NaN), max(1, 4n))
                iwork = zeros($J, max(1, 4n))
                ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, Ptr{$T}, Ptr{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                     Ptr{$J}, $J),
                    layout, job, compq, select, n, t, ldt, q, ldq, wr, wi, m, s,
                    sep, work, length(work), iwork, length(iwork))
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

@testset "LAPACKE TRSEN layouts, JOB and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in
        ((Float32, Float32, "s"), (Float64, Float64, "d"),
         (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 4), job in ('N',), compq in ('N', 'V'),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "trsen" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(700 + n)
        t = zeros(T, n, n)
        for j in 1:n
            t[j, j] = T(j + (T <: Complex ? j / 10 : 0))
            if j < n
                t[j, j + 1] = T(0.25j)
            end
        end
        q = Matrix{T}(I, n, n)
        select = J[isodd(j) ? 1 : 0 for j in 1:n]
        select[1] = J(1)
        wr = zeros(T, n); wi = zeros(R, n)
        mref = Ref{J}(-1); sref = Ref{R}(-1); sepref = Ref{R}(-1)
        tref = vec(copy(t)); qref = vec(copy(q)); selref = copy(select)
        if T <: Complex
            wref = zeros(T, n)
            ref = lapacke_trsen_call(
                T, R, J, pointer, 102, job, compq, selref, n, tref, n, qref, n,
                wref, wi, mref, sref, sepref, name, route, high,
            )
        else
            ref = lapacke_trsen_call(
                T, R, J, pointer, 102, job, compq, selref, n, tref, n, qref, n,
                wr, wi, mref, sref, sepref, name, route, high,
            )
        end
        @test ref == 0
        ld = n + 2
        mrow = Ref{J}(-1); srow = Ref{R}(-1); seprow = Ref{R}(-1)
        trow = row_major(t, ld); qrow = row_major(q, ld); selrow = copy(select)
        if T <: Complex
            wrow = zeros(T, n)
            info = lapacke_trsen_call(
                T, R, J, pointer, 101, job, compq, selrow, n, trow, ld, qrow, ld,
                wrow, wi, mrow, srow, seprow, name, route, high,
            )
        else
            info = lapacke_trsen_call(
                T, R, J, pointer, 101, job, compq, selrow, n, trow, ld, qrow, ld,
                wr, wi, mrow, srow, seprow, name, route, high,
            )
        end
        @test info == 0
        @test mrow[] == mref[]
        @test isequal(selrow, selref)
        tol = 500 * eps(R)
        @test isapprox(from_row_major(trow, n, n, ld), reshape(tref, n, n); rtol=tol, atol=tol)
        @test isapprox(from_row_major(qrow, n, n, ld), reshape(qref, n, n); rtol=tol, atol=tol)
        if job != 'N'
            @test isapprox(srow[], sref[]; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE TRSEN diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "trsen"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3; t = Matrix{T}(I, n, n); q = Matrix{T}(I, n, n)
        select = fill(J(1), n); wr = zeros(T, n); wi = zeros(R, n)
        m = Ref{J}(-1); s = Ref{R}(-1); sep = Ref{R}(-1)
        @test lapacke_trsen_call(
            T, R, J, pointer, 301, 'N', 'V', select, n, vec(t), n, vec(q), n,
            wr, wi, m, s, sep, name, :dll, true,
        ) == -1
        @test lapacke_trsen_call(
            T, R, J, pointer, 101, 'N', 'V', select, n, vec(t), n - 1, vec(q), n,
            wr, wi, m, s, sep, name, :dll, true,
        ) == -7
        @test lapacke_trsen_call(
            T, R, J, pointer, 101, 'N', 'V', select, n, vec(t), n, vec(q), n - 1,
            wr, wi, m, s, sep, name, :dll, true,
        ) == -9
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
