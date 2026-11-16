using Test, Colmat, LinearAlgebra, Libdl

isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_call(
        pointer, layout, storage::Vector{$T}, pivots::Vector{$J}, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            GC.@preserve pivots begin
                ccall(
                    pointer,
                    $J,
                    (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}),
                    layout,
                    3,
                    2,
                    storage,
                    5,
                    Base.pointer(pivots, 2),
                )
            end
        end
    end
end

function lapacke288_getrf(::Type{T}, ::Type{J}) where {T,J}
    prefix = Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    suffix = J===Int64 ? "64_" : ""
    interface = J===Int64 ? :ilp64 : :lp64
    original = T[2 1; 4 3; 1 5]
    if T<:Complex
        original .*= T(0.8, -0.6)
    end
    for entry in ("getrf", "getrf_work", "getf2", "getf2_work", "getrf2", "getrf2_work"),
        route in (:dll, :lbt_trampoline),
        layout in (101, 102)

        name = "LAPACKE_"*prefix*entry
        expected = Libdl.dlsym(native, name*suffix)
        @test BLAS.lbt_get_forward(name, interface)==expected
        pointer = Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
        storage = fill(T(777), 15)
        index(i, j) = layout==101 ? (i-1)*5+j : i+(j-1)*5
        for i in 1:3, j in 1:2
            storage[index(i, j)] = original[i, j]
        end
        pivots = fill(J(-91), 4)
        info = lapacke288_call(pointer, layout, storage, pivots, name, route)
        @test info==0
        @test pivots[1]==-91 && pivots[4]==-91
        @test pivots[2]==2
        permuted = copy(original)
        for k in 1:2
            permuted[k, :], permuted[pivots[k + 1], :] = copy(permuted[pivots[k + 1], :]),
            copy(permuted[k, :])
        end
        for i in 1:3, j in 1:2
            value = zero(T)
            for k in 1:2
                l = if i==k
                    one(T)
                elseif i>k
                    storage[index(i, k)]
                else
                    zero(T)
                end
                u = k<=j ? storage[index(k, j)] : zero(T)
                value += l*u
            end
            @test isapprox(value, permuted[i, j]; atol=20eps(real(T)), rtol=20eps(real(T)))
        end
        used = Set(index(i, j) for i in 1:3 for j in 1:2)
        @test all(storage[q]==T(777) for q in eachindex(storage) if q∉used)
    end
end

@testset "LAPACKE LU factors through native and LBT calls" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    saved = ccall(Libdl.dlsym(native, :LAPACKE_get_nancheck), Cint, ())
    try
        for (interface, suffix) in ((:lp64, ""), (:ilp64, "64_")),
            handle in (native, trampoline)

            for name in ("LAPACKE_get_nancheck", "LAPACKE_set_nancheck")
                @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(
                    native, name*suffix
                )
            end
            getter = Libdl.dlsym(handle, "LAPACKE_get_nancheck"*suffix)
            setter = Libdl.dlsym(handle, "LAPACKE_set_nancheck"*suffix)
            ccall(setter, Cvoid, (Cint,), 0)
            @test ccall(getter, Cint, ())==0
            ccall(setter, Cvoid, (Cint,), -1)
            @test ccall(getter, Cint, ())==1
        end
    finally
        ccall(Libdl.dlsym(native, :LAPACKE_set_nancheck), Cvoid, (Cint,), saved)
    end

    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_getrf(T, J)
    end
end
