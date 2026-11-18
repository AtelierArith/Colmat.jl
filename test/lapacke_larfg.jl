using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                       (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
    J in (Int32, Int64)
    @eval function lapacke_larfg_call(
        ::Type{$T}, ::Type{$J}, pointer, n, alpha, x, incx, tau, name, route, high,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, ($J, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}),
                  n, alpha, x, incx, tau)
        end
    end
end

@testset "LAPACKE LARFG high/work and strides" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"),
                           (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (0, 1, 5), incx in (-2, -1, 1, 3),
        high in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "larfg" * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(800 + n + abs(incx))
        alpha0 = randn(rng, T)
        x0 = [randn(rng, T) for _ in 1:n]
        tau0 = T(0)
        # Direct Fortran reference.
        xref = fill(T(NaN), n == 0 ? 0 : 1 + (n - 1) * abs(incx))
        first = incx < 0 ? 1 + (n - 1) * (-incx) : 1
        for i in 1:n
            xref[first + (i - 1) * incx] = x0[i]
        end
        # Work-level reference from the same LAPACKE implementation.
        xwork = copy(xref); awork = Ref(alpha0); twork = Ref(tau0)
        workname = "LAPACKE_" * prefix * "larfg_work"
        workpointer = Libdl.dlsym(native, workname * suffix)
        @test lapacke_larfg_call(
            T, J, workpointer, J(n), awork, xwork, J(incx), twork,
            workname, :dll, false,
        ) == 0
        aref = Ref(awork[]); tref = Ref(twork[])
        # Requested high/work entry.
        xlap = fill(T(NaN), n == 0 ? 0 : 1 + (n - 1) * abs(incx))
        for i in 1:n
            xlap[first + (i - 1) * incx] = x0[i]
        end
        alap = Ref(alpha0); tlap = Ref(tau0)
        @test lapacke_larfg_call(
            T, J, pointer, J(n), alap, xlap, J(incx), tlap, name, route, high,
        ) == 0
        @test alap[] == aref[]
        @test tlap[] == tref[]
        @test isequal(xlap, xwork)
    end
end

@testset "LAPACKE LARFG diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in ((Float64, "d"), (ComplexF64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "larfg"
        pointer = Libdl.dlsym(native, name * suffix)
        alpha = Ref(T(NaN)); x = zeros(T, 3); tau = Ref(T(0))
        @test lapacke_larfg_call(
            T, J, pointer, J(3), alpha, x, J(1), tau, name, :dll, true,
        ) == -2
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
