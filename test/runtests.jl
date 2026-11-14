using Test
using LinearAlgebra
using Random
using Libdl

# Observe the fallback before Colmat's __init__ changes the process-wide table.
const fallback_gbmv = BLAS.lbt_get_forward("dgbmv_", :ilp64)
const fallback_lapack = BLAS.lbt_get_forward("dgetrf_", :ilp64)
using Colmat

include("backend.jl")
include("blas.jl")
include("abi.jl")

# Reuse Julia's BLAS conformance suite as well as our independent references.
# It runs in the same process with Colmat forwards active.
include(joinpath(dirname(pathof(LinearAlgebra)), "..", "test", "blas.jl"))

@testset "Concurrent Julia callers" begin
    worker = joinpath(@__DIR__, "threading_worker.jl")
    project = dirname(Base.active_project())
    @test success(
        `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$project $worker`
    )
end
