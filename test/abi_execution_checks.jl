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
    for J in (Int32,Int64)
        @eval function lapacke_query_probe(::Type{$J})
            f=Libdl.dlsym(Libdl.dlopen(BLAS.libblastrampoline),
                $J==Int32 ? :LAPACKE_dgetri_work : :LAPACKE_dgetri_work64_)
            work=Ref(0.0)
            info=ccall(f,$J,(Cint,$J,Ptr{Float64},$J,Ptr{$J},Ref{Float64},$J),
                101,3,C_NULL,3,C_NULL,work,-1)
            (info,work[])
        end
    end
    function static_call()
        ccall(:getpid,Cint,())
    end
    function failed_pointer()
        ccall(error("pointer construction failed"),Cvoid,())
    end
end
""",
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
    for J in (Int32, Int64)
        info, work=Base.invokelatest(probe.lapacke_query_probe, J)
        @test info==0 && work>=3
        matches=filter(
            e->e[1]=="LAPACKE_dgetri_work" &&
               e[2]==8sizeof(J) &&
               e[3]=="lbt_trampoline" &&
               e[4]=="test/abi_execution_checks.jl",
            ABIEvidence.witnesses,
        )
        @test !isempty(matches)
        @test all(ABIEvidence.return_types[e]==string(J) for e in matches)
    end
    @test Base.invokelatest(probe.static_call)==getpid()
    @test_throws ErrorException Base.invokelatest(probe.failed_pointer)
    observed=length(ABIEvidence.witnesses)
    @test ABIEvidence.with_entry(()->17, "LAPACKE_dgetri_work", 32, "dll")==17
    @test length(ABIEvidence.witnesses)==observed
    @test isnothing(get(task_local_storage(), :colmat_abi_selected_entry, nothing))
    @test_throws ErrorException ABIEvidence.with_entry(
        ()->error("selected call failed"), "LAPACKE_dgetri_work", 32, "dll"
    )
    @test isnothing(get(task_local_storage(), :colmat_abi_selected_entry, nothing))
    @test isempty(ABIEvidence.errors)
end
