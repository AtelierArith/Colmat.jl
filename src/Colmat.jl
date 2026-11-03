module Colmat

using RustCall
import LinearAlgebra
import SparseArrays
import Statistics
using LinearAlgebra: SVD

export FaerLinearOperator,
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
    FaerVector,
    FaerRow,
    FaerDiagonal,
    filldiagonal!,
    FaerMatrix,
    FaerFactorization,
    reserve!,
    truncate!,
    full_piv_lu,
    pfaffian,
    faer_mul,
    faer_solve,
    faer_svd

const RustCrate = @rust_crate joinpath(@__DIR__, "..", "deps", "rust")

# Materialize views, transposes and other element types into owned column-major
# buffers. All native calls are synchronous and preserve their Julia owners.
_dense(a::AbstractMatrix{<:Real}) = Matrix{Float64}(a)

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
include("permutations.jl")
include("pseudoinverse.jl")
include("schur.jl")
include("qz.jl")
include("sparse.jl")
include("sparsity.jl")
include("sparse_compat.jl")
include("sparse_triangular.jl")
include("sparse_factors.jl")
include("ordering.jl")
include("sparse_updates.jl")
include("rotations.jl")
include("householder.jl")
include("householder_sequence.jl")
include("operators.jl")
include("jacobi.jl")
include("statistics.jl")
include("balance.jl")

end
