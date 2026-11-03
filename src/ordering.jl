_ordering_pattern(a::FaerSparsity) = a
_ordering_pattern(a::FaerSparseMatrix) = FaerSparsity(a)
_ordering_pattern(a::_SparseTransform) = copy(transpose(FaerSparsity(parent(a))))
_ordering_pattern(a::SparseArrays.AbstractSparseMatrixCSC) = FaerSparsity(a)
function _ordering_pattern(a::Union{LinearAlgebra.Hermitian,LinearAlgebra.Symmetric})
    source=FaerSparsity(parent(a))
    return _sparsity_result(
        size(a),
        source.format,
        RustCrate.sparsity_selfadjoint,
        source,
        a.uplo=='L',
    )
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
    amd(A; dense=10.0, aggressive=true, statistics=false)

Return a Rust-owned FaerPermutation for approximate minimum degree ordering
of the square stored pattern `A + transpose(A)`. Values, including stored
zeros, do not affect the order. Accepts native numeric/symbolic sparse owners
or Julia CSC matrices. `dense` and `aggressive` are faer's AMD controls.
With `statistics=true`, return `(permutation, n_div, n_mult_subs_ldl,
n_mult_subs_lu)` containing faer's estimates for subsequent factorizations.
The input is unchanged; the permutation is independent of its lifetime.
"""
function amd(
    a::_OrderingInput;
    dense::Real = 10.0,
    aggressive::Bool = true,
    statistics::Bool = false,
)
    size(a, 1)==size(a, 2) ||
        throw(DimensionMismatch("AMD requires a square sparse pattern"))
    threshold=Float64(dense)
    isfinite(threshold) || throw(ArgumentError("dense must be finite"))
    lock(_matrix_lock) do
        pattern=_ordering_pattern(a)
        stats=zeros(3)
        p=GC.@preserve stats _permutation_result(
            size(a, 2),
            RustCrate.sparse_ordering,
            pattern,
            true,
            threshold,
            0.5,
            aggressive,
            pointer(stats),
        )
        return statistics ?
               (
            permutation = p,
            n_div = stats[1],
            n_mult_subs_ldl = stats[2],
            n_mult_subs_lu = stats[3],
        ) : p
    end
end
"""
    colamd(A; dense_row=0.5, dense_col=0.5, aggressive=true)

Return a Rust-owned FaerPermutation ordering the columns of a sparse pattern
using faer's COLAMD. Rectangular inputs and numeric-free FaerSparsity are
supported. Stored zeros participate in the pattern. Thresholds and aggressive
absorption are passed to faer unchanged, including its convention for negative
thresholds and fully dense rows/columns. Apply with
`permute_cols(A,p)` or `A[:,p]` without densifying native sparse coefficients.
"""
function colamd(
    a::_OrderingInput;
    dense_row::Real = 0.5,
    dense_col::Real = 0.5,
    aggressive::Bool = true,
)
    row, col=Float64(dense_row), Float64(dense_col)
    isfinite(row)&&isfinite(col) || throw(ArgumentError("COLAMD thresholds must be finite"))
    lock(_matrix_lock) do
        pattern=_ordering_pattern(a)
        stats=zeros(3)
        return GC.@preserve stats _permutation_result(
            size(a, 2),
            RustCrate.sparse_ordering,
            pattern,
            false,
            row,
            col,
            aggressive,
            pointer(stats),
        )
    end
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
