const _DensePermutationFunction = Union{
    typeof(permute_rows!),typeof(permute_cols!),typeof(permute!),typeof(invpermute!)
}
_permutation_columns(::typeof(permute_cols!)) = true
_permutation_columns(::_DensePermutationFunction) = false
function _permutation_target(
    ::Union{typeof(permute_rows!),typeof(permute_cols!)}, a::_BorrowMatrix
)
    return a
end
_permutation_target(::typeof(permute_cols!), a::FaerRow) = a
_permutation_target(::Union{typeof(permute!),typeof(invpermute!)}, v::_BorrowVector) = v
function _permutation_target(f, a)
    return throw(
        ArgumentError("$(f) has no native dense permutation workspace for $(typeof(a))")
    )
end
"""
    scratch_requirements(permute_rows!, A, p)
    scratch_requirements(permute_cols!, A, p)
    scratch_requirements(permute!, v, p)
    scratch_requirements(invpermute!, v, p)

Query faer's exact workspace bytes and alignment for an in-place dense
permutation of `A` (a `FaerMatrix`, `FaerRow` or view) or of a native vector
or vector view. `A`, its owner and `p` must be open and `p` must match the
permuted axis. The query depends only on the scalar type and shape.
`FaerScratch(fun, A, p)` and `reserve!(scratch, fun, A, p)` use it.
"""
function scratch_requirements(f::_DensePermutationFunction, a, p::AbstractVector{<:Integer})
    lock(_matrix_lock) do
        x = _permutation_target(f, a)
        _view_source(x)
        columns = _permutation_columns(f)
        _check_permutation(x, _as_permutation(p), columns)
        return _permutation_requirements(x, columns)
    end
end
function FaerScratch(f::_DensePermutationFunction, a, p::AbstractVector{<:Integer})
    req = scratch_requirements(f, a, p)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch, f::_DensePermutationFunction, a, p::AbstractVector{<:Integer}
)
    req = scratch_requirements(f, a, p)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
