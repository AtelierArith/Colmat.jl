using Test, Colmat, LinearAlgebra, Libdl

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_solve(
        pointer, layout, trans, a::Vector{$T}, pivots::Vector{$J}, b::Vector{$T}
    )
        ccall(
            pointer,
            $J,
            (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
            layout,
            trans,
            3,
            2,
            a,
            5,
            pivots,
            b,
            5,
        )
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_reference_lu(pointer, a::Matrix{$T})
        pivots = zeros(Int64, 3)
        info = Ref{Int64}(-99)
        ccall(
            pointer,
            Cvoid,
            (Ref{Int64}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{Int64}, Ref{Int64}),
            3,
            3,
            a,
            3,
            pivots,
            info,
        )
        @test info[]==0
        return pivots
    end
end

function lapacke288_getrs(::Type{T}, ::Type{J}) where {T,J}
    prefix = Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    reference = only(
        filter(
            l -> l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference_handle = Libdl.dlopen(reference.libname)
    factor_pointer = Libdl.dlsym(reference_handle, prefix*"getrf_64_")
    original = T[2 1 1; 4 3 -1; 1 5 6]
    expected = T[1 2; 3 -1; 2 4]
    if T<:Complex
        for i in 1:3, j in 1:3
            original[i, j]+=T(0, (2i-j)/7)
        end
        for i in 1:3, j in 1:2
            expected[i, j]+=T(0, (i+j)/5)
        end
    end
    factors = copy(original)
    pivots = J.(lapacke288_reference_lu(factor_pointer, factors))
    suffix = J===Int64 ? "64_" : ""
    interface = J===Int64 ? :ilp64 : :lp64
    for entry in ("getrs", "getrs_work"),
        route in (:dll, :lbt_trampoline), layout in (101, 102),
        trans in ('N', 'T', 'C', 'n', 't', 'c')

        name = "LAPACKE_"*prefix*entry
        @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(native, name*suffix)
        pointer = Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
        index(i, j) = layout==101 ? (i-1)*5+j : i+(j-1)*5
        a = fill(T(777), 15)
        b = fill(T(777), 15)
        rhs = zeros(T, 3, 2)
        for i in 1:3, j in 1:2, k in 1:3
            x = if uppercase(trans)=='N'
                original[i, k]
            elseif uppercase(trans)=='T'
                original[k, i]
            else
                conj(original[k, i])
            end
            rhs[i, j]+=x*expected[k, j]
        end
        for i in 1:3, j in 1:3
            a[index(i, j)]=factors[i, j]
        end
        for i in 1:3, j in 1:2
            b[index(i, j)]=rhs[i, j]
        end
        saved_a, saved_pivots = copy(a), copy(pivots)
        @test lapacke288_solve(pointer, layout, trans, a, pivots, b)==0
        @test a==saved_a
        @test pivots==saved_pivots
        for i in 1:3, j in 1:2
            @test isapprox(
                b[index(i, j)], expected[i, j]; rtol=200eps(real(T)), atol=200eps(real(T))
            )
        end
        used = Set(index(i, j) for i in 1:3 for j in 1:2)
        @test all(b[q]==T(777) for q in eachindex(b) if q∉used)
    end
end

@testset "LAPACKE GETRS with external OpenBLAS factors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_getrs(T, J)
    end
end
