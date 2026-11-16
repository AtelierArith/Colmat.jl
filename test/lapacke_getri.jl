using Test, Colmat, LinearAlgebra, Libdl

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_inverse(pointer, layout, a::Vector{$T}, pivots::Vector{$J})
        ccall(pointer, $J, (Cint, $J, Ptr{$T}, $J, Ptr{$J}), layout, 3, a, 5, pivots)
    end
    @eval function lapacke288_inverse_work(
        pointer, layout, a::Vector{$T}, pivots::Vector{$J}, work::Vector{$T}, lwork
    )
        GC.@preserve work begin
            ccall(
                pointer,
                $J,
                (Cint, $J, Ptr{$T}, $J, Ptr{$J}, Ptr{$T}, $J),
                layout,
                3,
                a,
                5,
                pivots,
                Base.pointer(work, 2),
                lwork,
            )
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_inverse_reference_factor(pointer, a::Matrix{$T})
        pivots=zeros(Int64, 3)
        info=Ref{Int64}(-99)
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

function lapacke288_getri(::Type{T}, ::Type{J}) where {T,J}
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l -> l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    factor_pointer=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*"getrf_64_")
    original=T[2 1 1; 4 3 -1; 1 5 6]
    if T<:Complex
        for i in 1:3, j in 1:3
            original[i, j]+=T(0, (2i-j)/7)
        end
    end
    factors=copy(original)
    pivots=J.(lapacke288_inverse_reference_factor(factor_pointer, factors))
    suffix=J===Int64 ? "64_" : ""
    interface=J===Int64 ? :ilp64 : :lp64
    for entry in ("getri", "getri_work"),
        route in (:dll, :lbt_trampoline),
        layout in (101, 102)

        name="LAPACKE_"*prefix*entry
        @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(native, name*suffix)
        pointer=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
        index(i, j) = layout==101 ? (i-1)*5+j : i+(j-1)*5
        a=fill(T(777), 15)
        for i in 1:3, j in 1:3
            a[index(i, j)]=factors[i, j]
        end
        saved_a, saved_pivots=copy(a), copy(pivots)
        if entry=="getri_work"
            work=fill(T(888), 5)
            @test lapacke288_inverse_work(pointer, layout, a, pivots, work, -1)==0
            @test real(work[2])>=3 && imag(work[2])==0
            @test a==saved_a && pivots==saved_pivots
            @test work[1]==T(888) && all(==(T(888)), work[3:end])
            @test lapacke288_inverse_work(pointer, layout, a, pivots, work, 2)==-7
            @test a==saved_a && pivots==saved_pivots
            @test lapacke288_inverse_work(pointer, layout, a, pivots, work, 3)==0
            @test work[1]==T(888) && work[end]==T(888)
        else
            @test lapacke288_inverse(pointer, layout, a, pivots)==0
        end
        @test pivots==saved_pivots
        for i in 1:3, j in 1:3
            left, right=zero(T), zero(T)
            for k in 1:3
                left+=original[i, k]*a[index(k, j)]
                right+=a[index(i, k)]*original[k, j]
            end
            @test isapprox(left, T(i==j); rtol=200eps(real(T)), atol=200eps(real(T)))
            @test isapprox(right, T(i==j); rtol=200eps(real(T)), atol=200eps(real(T)))
        end
        used=Set(index(i, j) for i in 1:3 for j in 1:3)
        @test all(a[q]==T(777) for q in eachindex(a) if q∉used)
        # The singular diagonal is detected before any inversion output is stored.
        a.=saved_a
        a[index(2, 2)]=zero(T)
        singular=copy(a)
        info=if entry=="getri_work"
            lapacke288_inverse_work(pointer, layout, a, pivots, fill(T(888), 5), 3)
        else
            lapacke288_inverse(pointer, layout, a, pivots)
        end
        @test info==2
        @test a==singular && pivots==saved_pivots
    end
end

@testset "LAPACKE GETRI workspace and external factors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_getri(T, J)
    end
end
