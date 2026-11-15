using Test, Colmat
@testset "GEEVX availability in isolated processes" begin
    worker=joinpath(@__DIR__, "geevx_expert_worker.jl")
    project=dirname(Base.active_project())
    for mode in ("bridge", "native_lp64", "missing_after")
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$project $worker $mode`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
    for prefix in ("s", "d", "c", "z")
        cmd=`$(Base.julia_cmd()) --startup-file=no --project=$project $worker first_missing $prefix`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
end
