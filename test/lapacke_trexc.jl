using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_trexc_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, compq, n, t, ldt, q, ldq,
        ifst, ilst, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            work = fill($T(NaN), max(1, n))
            if $T <: Complex
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, $J, $J),
                    layout, compq, n, t, ldt, q, ldq, ifst, ilst)
            elseif high
                fst = Ref{$J}(ifst); lst = Ref{$J}(ilst)
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}),
                    layout, compq, n, t, ldt, q, ldq, fst, lst)
            else
                fst = Ref{$J}(ifst); lst = Ref{$J}(ilst)
                ccall(pointer, $J,
                    (Cint, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$J}, Ptr{$T}),
                    layout, compq, n, t, ldt, q, ldq, fst, lst, work)
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

@testset "LAPACKE TREXC layouts and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (2, 4), high in (false, true),
        route in (:dll, :lbt_trampoline), compq in ('N', 'V')

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "trexc" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        t = zeros(T, n, n)
        for j in 1:n, i in 1:n
            i <= j && (t[i, j] = T(1 + i + 2j))
        end
        if !(T <: Complex) && n >= 2
            t[1, 1] = 0
            t[1, 2] = 1
            t[2, 1] = -1
            t[2, 2] = 0
        end
        q = Matrix{T}(I, n, n)
        ifst, ilst = 1, min(3, n)
        tcol = vec(copy(t)); qcol = vec(copy(q))
        @test lapacke_trexc_call(
            T, J, pointer, 102, compq, n, tcol, n, qcol, n, ifst, ilst,
            name, route, high,
        ) == 0
        ld = n + 2
        trow = row_major(t, ld); qrow = row_major(q, ld)
        @test lapacke_trexc_call(
            T, J, pointer, 101, compq, n, trow, ld, qrow, ld, ifst, ilst,
            name, route, high,
        ) == 0
        tol = 200 * eps(real(T))
        @test isapprox(from_row_major(trow, n, n, ld), reshape(tcol, n, n); rtol=tol, atol=tol)
        if compq == 'V'
            @test isapprox(from_row_major(qrow, n, n, ld), reshape(qcol, n, n); rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE TREXC diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "trexc"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        t = Matrix{T}(I, n, n); q = Matrix{T}(I, n, n)
        @test lapacke_trexc_call(
            T, J, pointer, 301, 'V', n, vec(t), n, vec(q), n, 1, 2,
            name, :dll, true,
        ) == -1
        @test lapacke_trexc_call(
            T, J, pointer, 102, '?', n, vec(t), n, vec(q), n, 1, 2,
            name, :dll, true,
        ) == -2
        @test lapacke_trexc_call(
            T, J, pointer, 101, 'V', n, vec(t), n - 1, vec(q), n, 1, 2,
            name, :dll, true,
        ) == -5
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
