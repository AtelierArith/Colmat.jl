using Test
using LinearAlgebra
using Random
using Libdl

# Observe the fallback before Colmat's __init__ changes the process-wide table.
const fallback_rotm = BLAS.lbt_get_forward("drotm_", :ilp64)
const fallback_lapack = BLAS.lbt_get_forward("dggglm_", :ilp64)
using Colmat

include("backend.jl")
include("blas.jl")
include("dense_extensions.jl")
include("reduction_extensions.jl")
include("givens.jl")
include("matrix_norms.jl")
include("storage_conversions.jl")
include("matrix_copy.jl")
include("matrix_permutation.jl")
include("matrix_scaling.jl")
include("precision_conversion.jl")
include("complex_auxiliary.jl")
include("matrix_extensions.jl")
include("positive_equilibration.jl")
include("general_equilibration.jl")
include("symmetric_equilibration.jl")
include("matrix_transforms_lbt.jl")
include("structured_blas.jl")
include("structured_symmetric.jl")
include("packed_updates.jl")
include("structured_triangular.jl")
include("rfp_rank.jl")
include("structured_lu.jl")
include("band_cholesky.jl")
include("packed_cholesky.jl")
include("positive_tridiagonal.jl")
include("structured_evd.jl")
include("block_unitary.jl")
include("bidiagonal.jl")
include("selfadjoint_reduction.jl")
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
include("packed_bunch_kaufman.jl")
include("balance.jl")
include("tridiagonal_evd.jl")
include("triangular_lapack.jl")
include("generalized_reduction.jl")
include("generalized_selfadjoint.jl")
include("complete_pivot_lu.jl")
include("advanced_svd.jl")
include("selected_selfadjoint.jl")
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
