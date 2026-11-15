using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_packed_trd!) || include("structured_tridiagonal_packed.jl")
isdefined(@__MODULE__, :direct_band_trd!) || include("structured_tridiagonal_band.jl")
@testset "Structured tridiagonal actual LBT providers and arithmetic" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        interface=J===Int64 ? :ilp64 : :lp64
        names=if T<:Real
            ("sptrd", "opgtr", "opmtr", "sbtrd")
        else
            ("hptrd", "upgtr", "upmtr", "hbtrd")
        end
        pointers=map(names) do name
            symbol=prefix*name*"_"
            provider=BLAS.lbt_find_backing_library(symbol, interface)
            @test provider!==nothing&&realpath(provider.libname)==realpath(
                Colmat.backend_library()
            )
            BLAS.lbt_get_forward(symbol, interface)
        end
        a=T[2 1; 1 3]
        packed=packed_tridiagonal_store(a, 'L')
        reduction=direct_packed_trd!(packed, J; n=2, u=('L'), pointer=pointers[1])
        q=fill(T(777), 5, 2)
        formed=direct_packed_gtr!(
            packed, reduction.tau, q, J; n=2, u=('L'), pointer=pointers[2]
        )
        @test reduction.info==reduction.diagnostic==formed.info==formed.diagnostic==0
        basis=q[1:2, :]
        @test basis*Matrix(SymTridiagonal(reduction.values, reduction.offdiag))*basis'≈a
        c=vcat(Matrix{T}(I, 2, 2), fill(T(777), 3, 2))
        applied=direct_packed_mtr!(
            packed,
            reduction.tau,
            c,
            J;
            m=2,
            n=2,
            side=('L'),
            u=('L'),
            tr=('N'),
            pointer=pointers[3],
        )
        @test applied.info==applied.diagnostic==0
        @test c[1:2, :]≈basis
        ab=tridiagonal_band_store(a, 1, 'L')
        qb=fill(T(777), 5, 2)
        band=direct_band_trd!(ab, qb, J; kd=1, u=('L'), pointer=pointers[4])
        @test band.info==band.diagnostic==0
        @test qb[1:2, :]*Matrix(SymTridiagonal(band.values, band.offdiag))*qb[1:2, :]'≈a
    end
end
