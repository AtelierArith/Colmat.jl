using Test, LinearAlgebra, Libdl, Colmat
for (name, file) in (
    (:direct_structured_gst!, "structured_generalized.jl"),
    (:direct_structured_gvd!, "structured_generalized_gvd.jl"),
    (:direct_structured_gvx!, "structured_generalized_gvx.jl"),
    (:direct_structured_bandgst!, "structured_generalized_band.jl"),
    (:direct_structured_bandgv!, "structured_generalized_band_gv.jl"),
    (:direct_structured_bandgvd!, "structured_generalized_band_gvd.jl"),
    (:direct_structured_bandgvx!, "structured_generalized_band_gvx.jl"),
)
    isdefined(@__MODULE__, name) || include(file)
end
@testset "Structured generalized LBT providers and forwarded arithmetic" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in (:pgst, :pgv, :pgvd, :pgvx, :bgst, :bgv, :bgvd, :bgvx)

        name=(T<:Real ? "s" : "h")*String(family)
        symbol=prefix*name*"_"
        interface=J===Int64 ? :ilp64 : :lp64
        provider=BLAS.lbt_find_backing_library(symbol, interface)
        @test provider!==nothing &&
            realpath(provider.libname)==realpath(Colmat.backend_library())
        pointer=BLAS.lbt_get_forward(symbol, interface)
        a=reshape(T[3], 1, 1)
        b=reshape(T[2], 1, 1)
        z=fill(T(777), 4, 1)
        q=copy(z)
        if startswith(String(family), "p")
            ap=structured_gv_pack(a, 'L')
            bp=structured_gv_pack(family===:pgst ? sqrt.(b) : b, 'L')
            result=if family===:pgst
                direct_structured_gst!(ap, bp, J; n=1, pointer)
            elseif family===:pgv
                direct_structured_gv!(ap, bp, z, J; n=1, pointer)
            elseif family===:pgvd
                direct_structured_gvd!(ap, bp, z, J; n=1, pointer)
            else
                direct_structured_gvx!(ap, bp, z, J; n=1, pointer)
            end
            @test result.info==result.diagnostic==0
            if family===:pgst
                @test ap[2]≈T(1.5)
            else
                @test result.values[1]≈real(T(1.5))
                @test abs(z[1, 1])≈inv(sqrt(real(T(2))))
            end
        else
            ab=structured_band_pack(a, 0, 'L')
            bb=structured_band_pack(family===:bgst ? sqrt.(b) : b, 0, 'L')
            result=if family===:bgst
                direct_structured_bandgst!(ab, bb, z, J; ka=0, kb=0, u=('L'), pointer)
            elseif family===:bgv
                direct_structured_bandgv!(ab, bb, z, J; ka=0, kb=0, u=('L'), pointer)
            elseif family===:bgvd
                direct_structured_bandgvd!(ab, bb, z, J; ka=0, kb=0, u=('L'), pointer)
            else
                direct_structured_bandgvx!(ab, bb, q, z, J; ka=0, kb=0, u=('L'), pointer)
            end
            @test result.info==result.diagnostic==0
            if family===:bgst
                @test ab[1, 1]≈T(1.5)
                @test abs(z[1, 1])≈inv(sqrt(real(T(2))))
            else
                @test result.values[1]≈real(T(1.5))
                @test abs(z[1, 1])≈inv(sqrt(real(T(2))))
            end
        end
    end
end
