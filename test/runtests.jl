using Test
using LinearAlgebra
using Random
using Libdl

# Observe the fallback before Colmat's __init__ changes the process-wide table.
const fallback_gbmv = BLAS.lbt_get_forward("dgbmv_", :ilp64)
const fallback_lapack = BLAS.lbt_get_forward("dggglm_", :ilp64)
using Colmat

include("backend.jl")
include("blas.jl")
include("abi.jl")
include("lapack_native.jl")
include("householder.jl")
include("qr_compact.jl")
include("qr_standard.jl")
include("qr_pivoted.jl")
include("lq.jl")
include("rz.jl")
include("least_squares.jl")
include("hessenberg.jl")
include("schur.jl")
include("generalized_hessenberg.jl")
include("qz.jl")
include("generalized_eigen.jl")
include("pivoted_cholesky.jl")
include("bunch_kaufman.jl")
include("balance.jl")
include("tridiagonal_evd.jl")
include("lbt_inventory.jl")
include("support_matrix.jl")

# Reuse Julia's BLAS conformance suite as well as our independent references.
# It runs in the same process with Colmat forwards active.
include(joinpath(dirname(pathof(LinearAlgebra)), "..", "test", "blas.jl"))
Colmat.load_backend() # The BLAS suite tests raw LBT mutations before returning.
include(joinpath(dirname(pathof(LinearAlgebra)), "..", "test", "lapack.jl"))

@testset "Concurrent Julia callers" begin
    worker = joinpath(@__DIR__, "threading_worker.jl")
    project = dirname(Base.active_project())
    @test success(
        `$(Base.julia_cmd()) --startup-file=no --threads=4 --project=$project $worker`
    )
end
