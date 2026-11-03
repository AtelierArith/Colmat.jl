"""
    FaerPermutation(indices)
    FaerPermutation(F::FaerFactorization; column=false)

An immutable sequence of 1-based permutation indices owning a Rust `faer::Perm`.
`A[p, :]` gathers rows and `A[:, p]` gathers columns without elementwise FFI.
`invperm(p)` returns an independent inverse. Composition `p*q` has indices
`Vector(p)[Vector(q)]`, so `A[p,:][q,:] == A[p*q,:]`.
`FaerMatrix(p)` and `Matrix(p)` construct the matrix `I[p,:]`.
"""
mutable struct FaerPermutation{H} <: AbstractVector{Int}
    handle::Union{Nothing,H}
    len::Int
end
function _handle(p::FaerPermutation)
    h=p.handle
    isnothing(h) && throw(ArgumentError("permutation has been closed"))
    return h
end
_native_argument(p::FaerPermutation) = _handle(p)
_is_native_owner(::FaerPermutation) = true
function _permutation_result(n, f, args...)
    lock(_matrix_lock) do
        h=RustCrate.OwnedPermutation()
        _native_call(f, h, args...)
        return FaerPermutation{typeof(h)}(h, n)
    end
end
function FaerPermutation(indices::AbstractVector{<:Integer})
    Base.require_one_based_indexing(indices)
    isperm(indices) || throw(ArgumentError("indices must be a permutation of 1:length(p)"))
    data=UInt.(indices)
    GC.@preserve data return _permutation_result(
        length(data),
        RustCrate.permutation_load,
        pointer(data),
        UInt(length(data)),
    )
end
FaerPermutation(p::FaerPermutation) = copy(p)
FaerPermutation(f::FaerFactorization; column::Bool = false) =
    FaerPermutation(_permutation(f, column))
Base.size(p::FaerPermutation) = (p.len,)
Base.IndexStyle(::Type{<:FaerPermutation}) = IndexLinear()
function Base.getindex(p::FaerPermutation, i::Int)
    checkbounds(p, i)
    return Int(_native_call(RustCrate.permutation_element, p, UInt(i-1)))
end
function Base.Vector(p::FaerPermutation)
    out=Vector{UInt}(undef, length(p))
    GC.@preserve out _native_call(
        RustCrate.permutation_indices,
        p,
        pointer(out),
        UInt(length(p)),
        false,
    )
    return Int.(out)
end
Base.Array(p::FaerPermutation) = Vector(p)
Base.Vector{T}(p::FaerPermutation) where {T} = convert(Vector{T}, Vector(p))
Base.copy(p::FaerPermutation) =
    _permutation_result(length(p), RustCrate.permutation_clone, p, false)
Base.deepcopy_internal(p::FaerPermutation, d::IdDict) = get!(() -> copy(p), d, p)
Base.invperm(p::FaerPermutation) =
    _permutation_result(length(p), RustCrate.permutation_clone, p, true)
Base.inv(p::FaerPermutation) = invperm(p)
Base.isperm(p::FaerPermutation) = lock(_matrix_lock) do ;
    _handle(p)
    true
end
function Base.:*(p::FaerPermutation, q::FaerPermutation)
    length(p)==length(q) || throw(DimensionMismatch("permutation lengths must match"))
    return _permutation_result(length(p), RustCrate.permutation_compose, p, q)
end
function Base.close(p::FaerPermutation)
    lock(_matrix_lock) do ;
        p.handle=nothing
    end
    return nothing
end
Base.isopen(p::FaerPermutation) = lock(() -> !isnothing(p.handle), _matrix_lock)
_as_permutation(p::FaerPermutation) = p
_as_permutation(p::AbstractVector{<:Integer}) = FaerPermutation(p)
function _check_permutation(a, p, columns)
    length(p)==size(a, columns ? 2 : 1) ||
        throw(DimensionMismatch("permutation length must match the selected axis"))
    _handle(p)
    return nothing
end
function _permute(a::FaerMatrix, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        _check_permutation(a, p, columns)
        T=eltype(a)
        return T===Float64 ?
               _matrix_result(RustCrate.permute_from, size(a), a, p, columns) :
               _typed_result(T, RustCrate.permute_from, size(a), a, p, columns)
    end
end
"""
    permute_rows(A, p)
    permute_cols(A, p)

Return independent Rust-owned storage containing `A[p,:]` or `A[:,p]`.
`p` may be a `FaerPermutation` or a Julia vector containing a permutation.
"""
permute_rows(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), false)
permute_cols(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), true)
_permute(a::FaerMatrixView, p::FaerPermutation, columns) = _permute(copy(a), p, columns)
Base.getindex(a::FaerMatrix, p::FaerPermutation, ::Colon) = permute_rows(a, p)
Base.getindex(a::FaerMatrix, ::Colon, p::FaerPermutation) = permute_cols(a, p)
Base.getindex(a::FaerMatrix, p::FaerPermutation, q::FaerPermutation) =
    permute_cols(permute_rows(a, p), q)
function _permute!(a::FaerMatrix, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        _check_permutation(a, p, columns)
        _native_call(RustCrate.permute_storage, a, p, columns)
    end
    return a
end
function _permute!(a::FaerMatrixView, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        result=_permute(a, p, columns)
        copyto!(a, result)
    end
    return a
end
"""
    permute_rows!(A, p)
    permute_cols!(A, p)

Gather rows or columns into the same matrix. The native faer in-place routine
uses scratch storage, preserving the matrix's shape, owner and existing views.
A destination view uses an independent temporary result for alias safety.
"""
permute_rows!(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute!(a, _as_permutation(p), false)
permute_cols!(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute!(a, _as_permutation(p), true)
function _swap!(a::FaerMatrix, i::Integer, j::Integer, columns)
    lock(_matrix_lock) do
        _handle(a)
        n=size(a, columns ? 2 : 1)
        1<=i<=n && 1<=j<=n || throw(BoundsError(a, (i, j)))
        _native_call(RustCrate.swap_storage, a, UInt(i-1), UInt(j-1), columns)
    end
    return a
end
function _swap!(a::FaerMatrixView, i::Integer, j::Integer, columns)
    lock(_matrix_lock) do
        tmp=copy(a)
        _swap!(tmp, i, j, columns)
        copyto!(a, tmp)
    end
    return a
end
"""
    swap_rows!(A, i, j)
    swap_cols!(A, i, j)

Exchange two rows or columns using 1-based indices. The shape is unchanged,
and existing borrowed views remain valid.
"""
swap_rows!(a::_BorrowMatrix, i::Integer, j::Integer) = _swap!(a, i, j, false)
swap_cols!(a::_BorrowMatrix, i::Integer, j::Integer) = _swap!(a, i, j, true)
function Base.permute!(v::FaerVector, p::AbstractVector{<:Integer})
    lock(_matrix_lock) do
        pp=_as_permutation(p)
        length(v)==length(pp) ||
            throw(DimensionMismatch("vector and permutation lengths must match"))
        _native_call(RustCrate.permute_storage, v, pp, false)
    end
    return v
end
Base.invpermute!(v::FaerVector, p::AbstractVector{<:Integer}) =
    permute!(v, invperm(_as_permutation(p)))
function permute_cols!(a::FaerRow, p::AbstractVector{<:Integer})
    lock(_matrix_lock) do
        pp=_as_permutation(p)
        _check_permutation(a, pp, true)
        _native_call(RustCrate.permute_storage, a, pp, true)
    end
    return a
end
function swap_cols!(a::FaerRow, i::Integer, j::Integer)
    lock(_matrix_lock) do
        _handle(a)
        1<=i<=a.len && 1<=j<=a.len || throw(BoundsError(a, (i, j)))
        _native_call(RustCrate.swap_storage, a, UInt(i-1), UInt(j-1), true)
    end
    return a
end
FaerMatrix(p::FaerPermutation) =
    permute_rows(FaerMatrix{Float64}(LinearAlgebra.I, length(p), length(p)), p)
Base.Matrix(p::FaerPermutation) = Matrix(FaerMatrix(p))
Base.Matrix{T}(p::FaerPermutation) where {T} = convert(Matrix{T}, Matrix(p))

@doc (@doc permute_rows) permute_cols
@doc (@doc permute_rows!) permute_cols!
@doc (@doc swap_rows!) swap_cols!
