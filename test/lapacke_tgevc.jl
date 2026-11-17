using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
isdefined(@__MODULE__, :tgevc_native_call) || include("tgevc.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke_tgevc_call(
        ::Type{$T}, ::Type{$J}, pointer, layout, side, howmny, select::Vector{$J},
        n, s, lds, t, ldp, vl, ldvl, vr, ldvr, mm, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            m = Ref{$J}(-777)
            info = ccall(
                pointer,
                $J,
                (Cint, Cchar, Cchar, Ptr{$J}, $J, Ptr{$T}, $J, Ptr{$T}, $J,
                 Ptr{$T}, $J, Ptr{$T}, $J, $J, Ptr{$J}),
                layout, side, howmny, select, n, s, lds, t, ldp, vl, ldvl,
                vr, ldvr, mm, m,
            )
            (info=info, m=m[])
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
    matrix = fill(eltype(vector)(NaN), m, n)
    for j in 1:n, i in 1:m
        matrix[i, j] = vector[(i - 1) * ld + j]
    end
    return matrix
end

@testset "LAPACKE TGEVC layouts, work entries and routes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (1, 2, 4), side in ('L', 'R', 'B'),
        howmny in ('A', 'S', 'B'), high in (false, true), layout in (101, 102),
        route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "tgevc" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        pencil = random_pencil(T, n; seed=100 + n + (howmny == 'S' ? 1 : 0))
        select = fill(J(1), n)
        if howmny == 'S' && n > 1
            for j in 1:n
                select[j] = isodd(j) ? J(1) : J(0)
            end
        end
        # Column-major reference, always the same native Fortran entry.
        ref_select = copy(select)
        ref_vl = howmny == 'B' ? copy(pencil.q) : zeros(T, max(1, n), max(1, n))
        ref_vr = howmny == 'B' ? copy(pencil.z) : zeros(T, max(1, n), max(1, n))
        reference = tgevc_native_call(
            T, J, side, howmny, n, pencil.s, pencil.t;
            select=ref_select, vl=ref_vl, vr=ref_vr, mm=max(1, n),
        )
        ld = layout == 101 ? n + 2 : max(1, n)
        vl_ld = layout == 101 ? ld : max(1, n)
        vl = howmny == 'B' ? (layout == 101 ? row_major(pencil.q, vl_ld) : vec(pencil.q)) :
             fill(T(NaN), max(1, n) * vl_ld)
        vr = howmny == 'B' ? (layout == 101 ? row_major(pencil.z, vl_ld) : vec(pencil.z)) :
             fill(T(NaN), max(1, n) * vl_ld)
        if howmny != 'B'
            fill!(vl, T(0))
            fill!(vr, T(0))
        end
        s = layout == 101 ? row_major(pencil.s, ld) : vec(pencil.s)
        t = layout == 101 ? row_major(pencil.t, ld) : vec(pencil.t)
        before_vl, before_vr = copy(vl), copy(vr)
        result = lapacke_tgevc_call(
            T, J, pointer, layout, side, howmny, select, n, s, ld, t, ld,
            vl, vl_ld, vr, vl_ld, max(1, n), name, route,
        )
        @test result.info == 0
        @test result.m == reference.m
        out_vl = layout == 101 ? from_row_major(vl, max(1, n), max(1, n), vl_ld) : reshape(vl, max(1, n), max(1, n))
        out_vr = layout == 101 ? from_row_major(vr, max(1, n), max(1, n), vl_ld) : reshape(vr, max(1, n), max(1, n))
        tol = real(T) == Float32 ? 2f-5 : 2e-13
        for j in 1:result.m
            if side in ('L', 'B')
                @test isapprox(out_vl[:, j], reference.vl[:, j]; rtol=tol, atol=tol)
            end
            if side in ('R', 'B')
                @test isapprox(out_vr[:, j], reference.vr[:, j]; rtol=tol, atol=tol)
            end
        end
        # Row-major padding is not referenced.
        if layout == 101
            for i in 1:max(1, n), j in n+1:vl_ld
                @test isequal(vl[(i - 1) * vl_ld + j], before_vl[(i - 1) * vl_ld + j])
                @test isequal(vr[(i - 1) * vl_ld + j], before_vr[(i - 1) * vl_ld + j])
            end
        end
    end
end

@testset "LAPACKE TGEVC diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        pencil = random_pencil(T, 3; seed=7)
        s = vec(pencil.s)
        t = vec(pencil.t)
        select = fill(J(1), 3)
        vl = zeros(T, 3 * 3)
        vr = zeros(T, 3 * 3)
        high = Libdl.dlsym(native, "LAPACKE_" * prefix * "tgevc" * suffix)
        work = Libdl.dlsym(native, "LAPACKE_" * prefix * "tgevc_work" * suffix)
        @test lapacke_tgevc_call(
            T, J, high, 301, 'B', 'A', select, 3, s, 3, t, 3, vl, 3, vr, 3, 3,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -1
        # Work leading-dimension order.
        for (lds, ldp, ldvl, ldvr, expected) in (
            (2, 3, 3, 3, -7),
            (3, 2, 3, 3, -9),
            (3, 3, 2, 3, -11),
            (3, 3, 3, 2, -13),
        )
            @test lapacke_tgevc_call(
                T, J, work, 101, 'B', 'A', select, 3, s, lds, t, ldp, vl,
                ldvl, vr, ldvr, 3, "LAPACKE_" * prefix * "tgevc_work", :dll,
            ).info == expected
        end
        # High-level NaN order: P -8, S -6, VL -10, VR -12.
        nan_s = copy(s); nan_s[1] = T(NaN)
        @test lapacke_tgevc_call(
            T, J, high, 102, 'B', 'A', select, 3, nan_s, 3, t, 3, vl, 3, vr, 3, 3,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -6
        nan_t = copy(t); nan_t[1] = T(NaN)
        @test lapacke_tgevc_call(
            T, J, high, 102, 'B', 'A', select, 3, s, 3, nan_t, 3, vl, 3, vr, 3, 3,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -8
        nan_vl = copy(vl); nan_vl[1] = T(NaN)
        @test lapacke_tgevc_call(
            T, J, high, 102, 'L', 'A', select, 3, s, 3, t, 3, nan_vl, 3, vr, 3, 3,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -10
        nan_vr = copy(vr); nan_vr[1] = T(NaN)
        @test lapacke_tgevc_call(
            T, J, high, 102, 'R', 'A', select, 3, s, 3, t, 3, vl, 3, nan_vr, 3, 3,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -12
        # MM too small is reported after the input checks.
        @test lapacke_tgevc_call(
            T, J, high, 102, 'B', 'A', select, 3, s, 3, t, 3, vl, 3, vr, 3, 2,
            "LAPACKE_" * prefix * "tgevc", :dll,
        ).info == -14
    end
end
