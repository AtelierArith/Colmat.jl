using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_trevc_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, howmny, select::Vector{$J},
        n, t, ldt, vl::Vector{$T}, ldvl, vr::Vector{$T}, ldvr, mm, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            m = Ref{$J}(-777)
            if high
                info = ccall(pointer, $J,
                    (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                     Ptr{$T}, $J, $J, Ptr{$J}),
                    layout, side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr,
                    mm, m)
            else
                work = fill($T(NaN), max(1, $T <: Complex ? 2n : 3n))
                if $T <: Complex
                    rwork = fill($R(NaN), max(1, n))
                    info = ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, $J, Ptr{$J}, Ptr{$T}, Ptr{$R}),
                        layout, side, howmny, select, n, t, ldt, vl, ldvl, vr,
                        ldvr, mm, m, work, rwork)
                else
                    info = ccall(pointer, $J,
                        (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                         Ptr{$T}, $J, $J, Ptr{$J}, Ptr{$T}),
                        layout, side, howmny, select, n, t, ldt, vl, ldvl, vr,
                        ldvr, mm, m, work)
                end
            end
            (; info, m=m[], vl=copy(vl), vr=copy(vr))
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

@testset "LAPACKE TREVC layouts, select and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (1, 4), side in ('L', 'R', 'B'),
        howmny in ('A', 'S'), high in (false, true),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "trevc" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(400 + n + (howmny == 'S' ? 1 : 0))
        t = zeros(T, n, n)
        for j in 1:n, i in 1:n
            i <= j && (t[i, j] = randn(rng, T))
        end
        if !(T <: Complex) && n >= 2
            t[1, 1] = 0
            t[1, 2] = 1
            t[2, 1] = -1
            t[2, 2] = 0
        end
        select = fill(J(1), n)
        if howmny == 'S'
            for j in 1:n
                select[j] = isodd(j) ? J(1) : J(0)
            end
            !(T <: Complex) && n >= 2 && (select[1] = J(1); select[2] = J(1))
        end
        vl0 = zeros(T, n, n)
        vr0 = zeros(T, n, n)
        # Column-major reference.
        sel_ref = copy(select)
        ref = lapacke_trevc_call(
            T, J, pointer, 102, side, howmny, sel_ref, n, vec(t), n,
            vec(vl0), n, vec(vr0), n, n, name, route, high,
        )
        @test ref.info == 0
        # Row-major.
        ld = n + 2
        sel_row = copy(select)
        row = lapacke_trevc_call(
            T, J, pointer, 101, side, howmny, sel_row, n, row_major(t, ld), ld,
            row_major(vl0, ld), ld, row_major(vr0, ld), ld, n, name, route, high,
        )
        @test row.info == 0
        @test row.m == ref.m
        @test isequal(sel_row, sel_ref)
        tol = 200 * eps(real(T))
        if side in ('L', 'B')
            expected = reshape(ref.vl, n, n)
            actual = from_row_major(row.vl, n, n, ld)
            @test isapprox(actual[:, 1:ref.m], expected[:, 1:ref.m]; rtol=tol, atol=tol)
        end
        if side in ('R', 'B')
            expected = reshape(ref.vr, n, n)
            actual = from_row_major(row.vr, n, n, ld)
            @test isapprox(actual[:, 1:ref.m], expected[:, 1:ref.m]; rtol=tol, atol=tol)
        end
    end
end

@testset "LAPACKE TREVC diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "trevc"
        pointer = Libdl.dlsym(native, name * suffix)
        n = 3
        t = Matrix{T}(I, n, n)
        select = fill(J(1), n)
        @test lapacke_trevc_call(
            T, J, pointer, 301, 'B', 'A', select, n, vec(t), n,
            zeros(T, 9), n, zeros(T, 9), n, n, name, :dll, true,
        ).info == -1
        @test lapacke_trevc_call(
            T, J, pointer, 101, 'B', 'A', select, n, vec(t), n - 1,
            zeros(T, 9), n, zeros(T, 9), n, n, name, :dll, true,
        ).info == -7
        @test lapacke_trevc_call(
            T, J, pointer, 101, 'B', 'A', select, n, vec(t), n,
            zeros(T, 9), n, zeros(T, 9), n - 1, n, name, :dll, true,
        ).info == -11
        bad = copy(vec(t)); bad[1] = T(NaN)
        @test lapacke_trevc_call(
            T, J, pointer, 102, 'B', 'A', select, n, bad, n,
            zeros(T, 9), n, zeros(T, 9), n, n, name, :dll, true,
        ).info == -6
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
