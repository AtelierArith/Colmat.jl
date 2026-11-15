using Test, Libdl, LinearAlgebra, Colmat

@testset "Foreign-call recorder preserves actual call behavior" begin
    probe=Module(:ColmatABIEvidenceProbe)
    Core.eval(probe, :(using Libdl, LinearAlgebra, Colmat))
    expression=Meta.parse(
        raw"""
begin
    const evaluations=Ref(0)
    function dot_probe()
        ccall((evaluations[]+=1; Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :ddot_)),
              Float64,(Ref{Int32},Ptr{Float64},Ref{Int32},Ptr{Float64},Ref{Int32}),
              2,[1.0,2.0],1,[3.0,4.0],1)
    end
    for J in (Int32,Int64)
        @eval function empty_lu(::Type{$J})
            f=Libdl.dlsym(Libdl.dlopen(BLAS.libblastrampoline),
                          $J==Int32 ? :dgetrf_ : :dgetrf_64_)
            info=Ref{$J}(-777)
            ccall(f,Cvoid,(Ref{$J},Ref{$J},Ptr{Float64},Ref{$J},Ptr{$J},Ref{$J}),
                  0,0,zeros(1),1,zeros($J,1),info)
            info[]
        end
    end
    function integer_free_rotation(suffix)
        f=Libdl.dlsym(Libdl.dlopen(BLAS.libblastrampoline), Symbol("dlartg_"*suffix))
        c,s,r=Ref(0.0),Ref(0.0),Ref(0.0)
        ccall(f,Cvoid,(Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64}),
              3.0,4.0,c,s,r)
        (c[],s[],r[])
    end
    function static_call()
        ccall(:getpid,Cint,())
    end
    function failed_pointer()
        ccall(error("pointer construction failed"),Cvoid,())
    end
end
"""
    )
    file=joinpath(@__DIR__, "abi_execution_checks.jl")
    Core.eval(probe, ABIEvidence.rewrite(expression, file))
    @test Base.invokelatest(probe.dot_probe)==11
    @test probe.evaluations[]==1
    for J in (Int32, Int64)
        @test Base.invokelatest(probe.empty_lu, J)==0
        @test any(
            e->e[1]=="dgetrf_" &&
               e[2]==8sizeof(J) &&
               e[3]=="lbt_trampoline" &&
               e[4]=="test/abi_execution_checks.jl",
            ABIEvidence.witnesses,
        )
    end
    for (bits, suffix) in ((32, ""), (64, "64_"))
        c, s, r=Base.invokelatest(probe.integer_free_rotation, suffix)
        @test c≈0.6 && s≈0.8 && r≈5
        @test any(
            e->e[1]=="dlartg_" &&
               e[2]==bits &&
               e[3]=="lbt_trampoline" &&
               e[4]=="test/abi_execution_checks.jl",
            ABIEvidence.witnesses,
        )
    end
    @test Base.invokelatest(probe.static_call)==getpid()
    @test_throws ErrorException Base.invokelatest(probe.failed_pointer)
    @test isempty(ABIEvidence.errors)
end
