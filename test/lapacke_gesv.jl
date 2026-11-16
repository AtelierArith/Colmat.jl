using Test, Colmat, LinearAlgebra, Libdl

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_driver(
        pointer, layout, n, nrhs, a::Vector{$T}, pivots::Vector{$J}, b::Vector{$T}, ldb
    )
        ccall(
            pointer,
            $J,
            (Cint, $J, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
            layout,
            n,
            nrhs,
            a,
            5,
            pivots,
            b,
            ldb,
        )
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_external_solve(
        pointer, a::Matrix{$T}, pivots::Vector{Int64}, b::Matrix{$T}
    )
        info=Ref{Int64}(-99)
        ccall(
            pointer,
            Cvoid,
            (
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{Int64},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            UInt8('N'),
            3,
            2,
            a,
            3,
            pivots,
            b,
            3,
            info,
        )
        @test info[]==0
    end
end

function lapacke288_gesv(::Type{T}, ::Type{J}) where {T,J}
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l -> l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    standard_solve=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*"getrs_64_")
    original=T[2 1 1; 4 3 -1; 1 5 6]
    expected=T[1 2; 3 -1; 2 4]
    if T<:Complex
        for i in 1:3, j in 1:3
            original[i, j]+=T(0, (2i-j)/7)
        end
        for i in 1:3, j in 1:2
            expected[i, j]+=T(0, (i+j)/5)
        end
    end
    rhs=zeros(T, 3, 2)
    for i in 1:3, j in 1:2, k in 1:3
        rhs[i, j]+=original[i, k]*expected[k, j]
    end
    suffix=J===Int64 ? "64_" : ""
    interface=J===Int64 ? :ilp64 : :lp64
    for entry in ("gesv", "gesv_work"),
        route in (:dll, :lbt_trampoline),
        layout in (101, 102)

        name="LAPACKE_"*prefix*entry
        @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(native, name*suffix)
        pointer=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
        index(i, j) = layout==101 ? (i-1)*5+j : i+(j-1)*5
        a=fill(T(777), 15)
        b=fill(T(777), 15)
        pivots=fill(J(-91), 3)
        for i in 1:3, j in 1:3
            a[index(i, j)]=original[i, j]
        end
        for i in 1:3, j in 1:2
            b[index(i, j)]=rhs[i, j]
        end
        original_storage=copy(a)
        @test lapacke288_driver(pointer, layout, 3, 2, a, pivots, b, 5)==0
        @test pivots[1]==2
        for i in 1:3, j in 1:2
            @test isapprox(
                b[index(i, j)], expected[i, j]; rtol=200eps(real(T)), atol=200eps(real(T))
            )
        end
        for (storage, cols) in ((a, 3), (b, 2))
            used=Set(index(i, j) for i in 1:3 for j in 1:cols)
            @test all(storage[q]==T(777) for q in eachindex(storage) if q∉used)
        end
        # Standard OpenBLAS consumes the produced factors and sequential IPIV.
        external_factors=T[a[index(i, j)] for i in 1:3, j in 1:3]
        second_rhs=2rhs
        lapacke288_external_solve(
            standard_solve, external_factors, Int64.(pivots), second_rhs
        )
        @test isapprox(second_rhs, 2expected; rtol=200eps(real(T)), atol=200eps(real(T)))
        # Zero RHS must still produce the same factorization and pivots.
        zero_a=copy(original_storage)
        zero_pivots=fill(J(-91), 3)
        @test lapacke288_driver(
            pointer, layout, 3, 0, zero_a, zero_pivots, T[], layout==101 ? 0 : 3
        )==0
        @test zero_a==a
        @test zero_pivots==pivots
        # A singular driver does not overwrite B; positive INFO stays numerical.
        singular=copy(original_storage)
        for i in 1:3, j in 1:3
            singular[index(i, j)]=zero(T)
        end
        saved_b=copy(b)
        @test lapacke288_driver(pointer, layout, 3, 2, singular, pivots, b, 5)==1
        @test b==saved_b
        @test all(1 .<= pivots .<= 3)
    end
end

@testset "LAPACKE GESV and external factor reuse" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_gesv(T, J)
    end
end
