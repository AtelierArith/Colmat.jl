_ordering_pattern(a::FaerSparsity) = a
_ordering_pattern(a::FaerSparseMatrix) = FaerSparsity(a)
function _ordering_pattern(a::_SparseTransform)
    source=FaerSparsity(parent(a))
    try
        return copy(transpose(source))
    finally
        close(source)
    end
end
function _ordering_pattern(a::SparseArrays.AbstractSparseMatrixCSC)
    source=FaerSparseMatrix(a)
    try
        return FaerSparsity(source)
    finally
        close(source)
    end
end
function _ordering_pattern(a::Union{LinearAlgebra.Hermitian,LinearAlgebra.Symmetric})
    source=FaerSparsity(parent(a))
    try
        return _sparsity_result(
            size(a),
            source.format,
            RustCrate.sparsity_selfadjoint,
            source,
            a.uplo=='L',
        )
    finally
        close(source)
    end
end
const _OrderingInput=Union{
    FaerSparsity,
    FaerSparseMatrix,
    _SparseTransform,
    SparseArrays.AbstractSparseMatrixCSC,
    LinearAlgebra.Hermitian{<:Any,<:FaerSparseMatrix},
    LinearAlgebra.Symmetric{<:Any,<:FaerSparseMatrix},
}
"""
    amd(A; dense=10.0, aggressive=true, statistics=false, scratch=nothing)

Return a Rust-owned FaerPermutation for approximate minimum degree ordering
of the square stored pattern `A + transpose(A)`. Values, including stored
zeros, do not affect the order. Accepts native numeric/symbolic sparse owners
or Julia CSC matrices. `dense` and `aggressive` are faer's AMD controls.
With `statistics=true`, return `(permutation, n_div, n_mult_subs_ldl,
n_mult_subs_lu)` containing faer's estimates for subsequent factorizations.
The input is unchanged; the permutation is independent of its lifetime.
`scratch=FaerScratch(amd, A)` reuses the sequential ordering kernel workspace.
Pattern conversion and the returned permutation allocate separately.
"""
function amd(
    a::_OrderingInput;
    dense::Real = 10.0,
    aggressive::Bool = true,
    statistics::Bool = false,
    scratch = nothing,
)
    size(a, 1)==size(a, 2) ||
        throw(DimensionMismatch("AMD requires a square sparse pattern"))
    threshold=Float64(dense)
    isfinite(threshold) || throw(ArgumentError("dense must be finite"))
    return _ordering_options(a, true, threshold, 0.5, aggressive, statistics, scratch)
end
"""
    colamd(A; dense_row=0.5, dense_col=0.5, aggressive=true, scratch=nothing)

Return a Rust-owned FaerPermutation ordering the columns of a sparse pattern
using faer's COLAMD. Rectangular inputs and numeric-free FaerSparsity are
supported. Stored zeros participate in the pattern. Thresholds and aggressive
absorption are passed to faer unchanged, including its convention for negative
thresholds and fully dense rows/columns. Apply with
`permute_cols(A,p)` or `A[:,p]` without densifying native sparse coefficients.
`scratch=FaerScratch(colamd, A)` reuses the sequential kernel workspace;
pattern conversion and the returned permutation allocate separately.
"""
function colamd(
    a::_OrderingInput;
    dense_row::Real = 0.5,
    dense_col::Real = 0.5,
    aggressive::Bool = true,
    scratch = nothing,
)
    row, col=Float64(dense_row), Float64(dense_col)
    isfinite(row)&&isfinite(col) || throw(ArgumentError("COLAMD thresholds must be finite"))
    return _ordering_options(a, false, row, col, aggressive, false, scratch)
end
const _SparsePermutable=Union{FaerSparseMatrix,FaerSparsity}
function _permute(a::FaerSparseMatrix, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        _check_permutation(a, p, columns)
        return _sparse_result(
            eltype(a),
            size(a),
            a.format,
            RustCrate.sparse_permute,
            a,
            p,
            columns,
        )
    end
end
function _permute(a::FaerSparsity, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        _check_permutation(a, p, columns)
        return _sparsity_result(
            size(a),
            a.format,
            RustCrate.sparsity_permute,
            a,
            p,
            columns,
        )
    end
end
permute_rows(a::_SparsePermutable, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), false)
permute_cols(a::_SparsePermutable, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), true)
Base.getindex(a::_SparsePermutable, p::FaerPermutation, ::Colon) = permute_rows(a, p)
Base.getindex(a::_SparsePermutable, ::Colon, p::FaerPermutation) = permute_cols(a, p)
Base.getindex(a::_SparsePermutable, p::FaerPermutation, q::FaerPermutation) =
    permute_cols(permute_rows(a, p), q)
function _permute!(a::FaerSparseMatrix, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        return copyto!(a, _permute(a, p, columns))
    end
end
permute_rows!(a::FaerSparseMatrix, p::AbstractVector{<:Integer}) =
    _permute!(a, _as_permutation(p), false)
permute_cols!(a::FaerSparseMatrix, p::AbstractVector{<:Integer}) =
    _permute!(a, _as_permutation(p), true)
