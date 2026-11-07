module Colmat

using RustCall
import LinearAlgebra
import SparseArrays
import Statistics
import Random
using LinearAlgebra: SVD

export FaerSpectralFactor,
    reconstruct,
    FaerScratch,
    scratch_capacity,
    scratch_requirements,
    FaerTeam,
    FaerParThreshold,
    team_mul,
    partial_eigen,
    partial_self_adjoint_eigen,
    partial_svd,
    FaerLinearOperator,
    FaerNpyMatrix,
    loadnpy,
    npyview,
    npyinfo,
    FaerRNG,
    randunitary,
    FaerBidiagonalization,
    FaerBidiagonalBasis,
    bidiagonalize,
    bidiagonalize!,
    bidiagonal_svd,
    FaerTridiagonalization,
    FaerTridiagonalQ,
    tridiagonalize,
    tridiagonalize!,
    tridiagonal_eigen,
    tridiagonal_eigvals,
    FaerBalance,
    balance,
    balance!,
    balance_eigenvectors,
    balance_eigenvectors!,
    FaerJacobiRotation,
    rotg,
    make_givens,
    apply_on_the_left_2x2,
    apply_on_the_right_2x2,
    FaerLsmrResult,
    FaerLsmrError,
    lsmr,
    lsmr!,
    FaerBicgstabResult,
    FaerBicgstabError,
    bicgstab,
    bicgstab!,
    FaerCgResult,
    FaerCgError,
    cg,
    cg!,
    identity_preconditioner,
    diagonal_preconditioner,
    FaerHouseholderSequence,
    reblock,
    reblock!,
    FaerHouseholder,
    householder,
    add_assign!,
    sub_assign!,
    add_into!,
    sub_into!,
    amd,
    colamd,
    FaerSparseSymbolic,
    FaerSparseFactorization,
    FaerSparsity,
    pattern_union,
    FaerSparseMatrix,
    compressed_storage,
    FaerGeneralizedHessenberg,
    FaerHessenberg,
    FaerHessenbergQ,
    FaerCompactHessenberg,
    FaerPermutation,
    permute_rows,
    permute_cols,
    permute_rows!,
    permute_cols!,
    swap_rows!,
    swap_cols!,
    FaerMatrixView,
    FaerVectorView,
    diagonalview,
    faerview,
    FaerBorrowedArray,
    FaerBorrowedMatrix,
    FaerBorrowedVector,
    FaerVector,
    FaerRow,
    FaerDiagonal,
    filldiagonal!,
    FaerMatrix,
    FaerFactorization,
    reserve!,
    capacity,
    with_capacity,
    resize_with!,
    truncate!,
    full_piv_lu,
    full_piv_lu!,
    pfaffian,
    faer_mul,
    structured_mul!,
    faer_solve,
    faer_svd

const RustCrate = @rust_crate joinpath(@__DIR__, "..", "deps", "rust")

# Materialize views, transposes and other element types into owned column-major
# buffers. All native calls are synchronous and preserve their Julia owners.
function _checked_dense(::Type{T}, a::AbstractMatrix) where {T}
    dims=size(a)
    data=Matrix{T}(a)
    data isa Matrix{T}||throw(ArgumentError("matrix conversion changed scalar storage"))
    size(data)==dims||throw(DimensionMismatch("matrix conversion changed shape"))
    return data
end
_dense(a::AbstractMatrix{<:Real}) = _checked_dense(Float64, a)

"""
    faer_mul(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})

Compute `A * B` in t4a-faer using Float64 and sequential execution.
"""
function faer_mul(a::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real})
    size(a, 2) == size(b, 1) || throw(DimensionMismatch("A columns must equal B rows"))
    aa, bb = _dense(a), _dense(b)
    c = Matrix{Float64}(undef, size(a, 1), size(b, 2))
    GC.@preserve aa bb c begin
        RustCall.result_to_exception(
            RustCrate.matmul_f64(
                pointer(aa),
                pointer(bb),
                pointer(c),
                UInt(size(aa, 1)),
                UInt(size(aa, 2)),
                UInt(size(bb, 2)),
            ),
        )
    end
    return c
end

"""
    faer_solve(A::AbstractMatrix{<:Real}, B::Union{AbstractVector,AbstractMatrix})

Solve square `A * X = B` with t4a-faer partial-pivot LU in Float64.
Singular matrices and nonfinite inputs raise an exception.
"""
function faer_solve(a::AbstractMatrix{<:Real}, b::AbstractMatrix{<:Real})
    size(a, 1) == size(a, 2) || throw(DimensionMismatch("A must be square"))
    size(a, 1) == size(b, 1) || throw(DimensionMismatch("A and B rows must match"))
    aa, bb = _dense(a), _dense(b)
    x = similar(bb)
    isempty(aa) && return x
    GC.@preserve aa bb x begin
        RustCall.result_to_exception(
            RustCrate.solve_f64(
                pointer(aa),
                pointer(bb),
                pointer(x),
                UInt(size(aa, 1)),
                UInt(size(bb, 2)),
            ),
        )
    end
    return x
end
faer_solve(a::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}) =
    vec(faer_solve(a, reshape(b, :, 1)))

"""
    faer_svd(A::AbstractMatrix{<:Real})

Compute a Float64 thin SVD with t4a-faer. Return a `LinearAlgebra.SVD`,
so `F.U * Diagonal(F.S) * F.Vt` reconstructs `A`.
"""
function faer_svd(a::AbstractMatrix{<:Real})
    aa = _dense(a)
    m, n = size(aa)
    k = min(m, n)
    u, s, v = Matrix{Float64}(undef, m, k),
    Vector{Float64}(undef, k),
    Matrix{Float64}(undef, n, k)
    if k > 0
        GC.@preserve aa u s v begin
            RustCall.result_to_exception(
                RustCrate.svd_f64(
                    pointer(aa),
                    pointer(u),
                    pointer(s),
                    pointer(v),
                    UInt(m),
                    UInt(n),
                ),
            )
        end
    end
    return SVD(u, s, copy(transpose(v)))
end

include("matrix.jl")
include("factorizations.jl")
include("reductions.jl")
include("eigen.jl")
include("structured.jl")
include("typed.jl")
include("storage.jl")
include("collections.jl")
include("views.jl")
include("borrowed.jl")
include("permutations.jl")
include("pseudoinverse.jl")
include("schur.jl")
include("qz.jl")
include("sparse.jl")
include("sparsity.jl")
include("sparse_compat.jl")
include("sparse_triangular.jl")
include("sparse_factors.jl")
include("sparse_qr_spectral.jl")
include("ordering.jl")
include("sparse_updates.jl")
include("rotations.jl")
include("householder.jl")
include("householder_sequence.jl")
include("iteration_observers.jl")
include("operators.jl")
include("operator_callbacks.jl")
include("partial_spectra.jl")
include("structured_products.jl")
include("pivoted_cholesky.jl")
include("destructive.jl")
include("teams.jl")
include("parallel_threshold.jl")
include("workspace.jl")
include("permutation_workspace.jl")
include("ordering_workspace.jl")
include("operator_workspace.jl")
include("iterative_workspace.jl")
include("partial_spectrum_workspace.jl")
include("householder_workspace.jl")
include("qz_reorder_workspace.jl")
include("qz_construction_workspace.jl")
include("schur_construction_workspace.jl")
include("schur_reorder_workspace.jl")
include("schur_eigenvectors_workspace.jl")
include("svd_workspace.jl")
include("eigen_mutation.jl")
include("evd_workspace.jl")
include("general_evd_workspace.jl")
include("selfadj_gevd_workspace.jl")
include("general_gevd_workspace.jl")
include("jacobi.jl")
include("statistics.jl")
include("balance.jl")
include("compact_reduction_q.jl")
include("tridiagonal_compact.jl")
include("tridiagonal.jl")
include("tridiagonal_workspace.jl")
include("tridiagonal_construction_workspace.jl")
include("hessenberg_compact.jl")
include("hessenberg_tuning.jl")
include("bidiagonal_compact.jl")
include("bidiagonal.jl")
include("bidiagonal_workspace.jl")
include("bidiagonal_construction_workspace.jl")
include("random.jl")
include("npy.jl")
include("reconstruction.jl")
include("spectral_factors.jl")
include("spectral_pinv_workspace.jl")
include("sparse_construction_workspace.jl")
include("spectral_solve.jl")
include("dense_solve_team.jl")
include("dense_solve_workspace.jl")
include("sparse_direct_workspace.jl")
include("sparse_qr_solve_workspace.jl")
include("spectral_solve_workspace.jl")
include("pivoted_cholesky_workspace.jl")
include("storage_capacity.jl")
include("storage_construction.jl")
include("storage_initializer.jl")
include("function_constructors.jl")
include("filled_constructors.jl")
include("diagonal_ops.jl")

end
