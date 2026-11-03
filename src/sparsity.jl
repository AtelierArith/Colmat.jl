"""
    FaerSparsity(A::FaerSparseMatrix; format=A.format)

An immutable sparsity pattern owning faer's value-free symbolic CSC or CSR
structure. `P[i,j]` is true at stored positions, including explicitly stored
zeros. The pattern is independent of A. Numeric owners can be constructed as
`FaerSparseMatrix{T}(P, values)` in P's physical order, or `FaerSparseMatrix{T}(P)`
with stored ones. `FaerSparseMatrix{T}(undef,P)` allocates zero-initialized
stored values while retaining P. `Matrix(P)` explicitly materializes the
Boolean pattern.
"""
mutable struct FaerSparsity{H} <: SparseArrays.AbstractSparseMatrix{Bool,Int}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    stored::Int
    format::Symbol
end
function _handle(p::FaerSparsity)
    h=p.handle
    isnothing(h) && throw(ArgumentError("sparsity pattern has been closed"))
    return h
end
_native_argument(p::FaerSparsity) = _handle(p)
_is_native_owner(::FaerSparsity) = true
function _sparsity_result(dims, format, f, args...)
    _sparse_dims(dims...)
    _sparse_format(format)
    lock(_matrix_lock) do
        h=RustCrate.SparsityOwner()
        _native_call(f, h, args...)
        n=Int(_native_call(RustCrate.sparsity_nnz, h))
        return FaerSparsity{typeof(h)}(h, dims, n, format)
    end
end
function FaerSparsity(a::FaerSparseMatrix; format = a.format)
    _sparse_format(format)
    p=_sparsity_result(size(a), a.format, RustCrate.sparsity_from_sparse, a)
    return format===a.format ? p : copy(p; format)
end
FaerSparsity(a::AbstractMatrix; kwargs...) = FaerSparsity(FaerSparseMatrix(a); kwargs...)
FaerSparsity(p::FaerSparsity; kwargs...) = copy(p; kwargs...)
Base.size(p::FaerSparsity) = p.dims
Base.IndexStyle(::Type{<:FaerSparsity}) = IndexCartesian()
Base.copy(p::FaerSparsity; format = p.format) = _sparsity_result(
    size(p),
    format,
    RustCrate.sparsity_clone_from,
    p,
    _sparse_format(format),
    false,
)
Base.deepcopy_internal(p::FaerSparsity, d::IdDict) = get!(() -> copy(p), d, p)
function Base.close(p::FaerSparsity)
    lock(_matrix_lock) do ;
        p.handle=nothing
    end
    return nothing
end
Base.isopen(p::FaerSparsity) = lock(() -> !isnothing(p.handle), _matrix_lock)
SparseArrays.nnz(p::FaerSparsity) = lock(_matrix_lock) do ;
    _handle(p)
    p.stored
end
SparseArrays.nonzeros(p::FaerSparsity) = lock(_matrix_lock) do ;
    _handle(p)
    fill(true, p.stored)
end
function Base.getindex(p::FaerSparsity, i::Int, j::Int)
    lock(_matrix_lock) do
        checkbounds(p, i, j)
        return _native_call(RustCrate.sparsity_element, p, UInt(i-1), UInt(j-1))
    end
end
"""Snapshot a symbolic pattern's physical 1-based pointers and indices."""
function compressed_storage(p::FaerSparsity)
    lock(_matrix_lock) do
        _handle(p)
        outer=size(p, p.format===:csr ? 1 : 2)
        ptr=Vector{UInt}(undef, outer+1)
        idx=Vector{UInt}(undef, p.stored)
        GC.@preserve ptr idx _native_call(
            RustCrate.sparsity_export,
            p,
            pointer(ptr),
            pointer(idx),
            UInt(length(ptr)),
            UInt(length(idx)),
        )
        return (format = p.format, pointers = Int.(ptr), indices = Int.(idx))
    end
end
function SparseArrays.SparseMatrixCSC(p::FaerSparsity)
    c=p.format===:csc ? p : copy(p; format = :csc)
    data=compressed_storage(c)
    return SparseArrays.SparseMatrixCSC(
        size(c)...,
        data.pointers,
        data.indices,
        fill(true, c.stored),
    )
end
SparseArrays.sparse(p::FaerSparsity) = SparseArrays.SparseMatrixCSC(p)
SparseArrays.findnz(p::FaerSparsity) = SparseArrays.findnz(SparseArrays.SparseMatrixCSC(p))
Base.Matrix(p::FaerSparsity) = Matrix(SparseArrays.SparseMatrixCSC(p))
Base.Array(p::FaerSparsity) = Matrix(p)
function _require_sparse_csc(p::FaerSparsity)
    _handle(p)
    p.format===:csc || throw(
        ArgumentError(
            "this CSC interface requires format=:csc; use compressed_storage for CSR",
        ),
    )
end
SparseArrays.rowvals(p::FaerSparsity) = lock(_matrix_lock) do ;
    _require_sparse_csc(p)
    compressed_storage(p).indices
end
SparseArrays.getcolptr(p::FaerSparsity) = lock(_matrix_lock) do ;
    _require_sparse_csc(p)
    compressed_storage(p).pointers
end
function SparseArrays.nzrange(p::FaerSparsity, j::Integer)
    lock(_matrix_lock) do
        _require_sparse_csc(p)
        1<=j<=size(p, 2) || throw(BoundsError(p, (:, j)))
        ptr=compressed_storage(p).pointers
        return ptr[j]:(ptr[j+1]-1)
    end
end
const _SparsityTransform=Union{
    LinearAlgebra.Transpose{Bool,<:FaerSparsity},
    LinearAlgebra.Adjoint{Bool,<:FaerSparsity},
}
function Base.copy(p::_SparsityTransform)
    a=parent(p)
    format=a.format===:csc ? :csr : :csc
    return _sparsity_result(
        reverse(size(a)),
        format,
        RustCrate.sparsity_clone_from,
        a,
        format===:csr,
        true,
    )
end
"""
    pattern_union(P::FaerSparsity, Q::FaerSparsity; format=P.format)

Return an independent native pattern containing the stored positions from P
or Q, using faer's symbolic union. Shapes must match. No numeric values are
allocated in the resulting symbolic owner.
"""
function pattern_union(p::FaerSparsity, q::FaerSparsity; format = p.format)
    size(p)==size(q) || throw(DimensionMismatch("pattern union shapes must match"))
    return _sparsity_result(
        size(p),
        format,
        RustCrate.sparsity_union_from,
        p,
        q,
        _sparse_format(format),
    )
end
function FaerSparseMatrix{T}(
    p::FaerSparsity,
    values::AbstractVector;
    format = p.format,
) where {T<:_NativeScalar}
    length(values)==p.stored ||
        throw(DimensionMismatch("pattern and value lengths must match"))
    data=Vector{T}(values)
    GC.@preserve data return _sparse_result(
        T,
        size(p),
        format,
        RustCrate.sparse_from_pattern,
        p,
        Ptr{UInt8}(pointer(data)),
        UInt(length(data)),
        _tag(T),
        _sparse_format(format),
    )
end
FaerSparseMatrix(p::FaerSparsity, values::AbstractVector{T}; kwargs...) where {T} =
    FaerSparseMatrix{_sparse_input_type(T)}(p, values; kwargs...)
FaerSparseMatrix{T}(p::FaerSparsity; kwargs...) where {T<:_NativeScalar} =
    FaerSparseMatrix{T}(p, ones(T, p.stored); kwargs...)
FaerSparseMatrix(p::FaerSparsity; kwargs...) = FaerSparseMatrix{Float64}(p; kwargs...)

"""Allocate zero-initialized numeric storage for an independent copy of a pattern."""
FaerSparseMatrix{T}(
    ::UndefInitializer,
    p::FaerSparsity;
    kwargs...,
) where {T<:_NativeScalar} = FaerSparseMatrix{T}(p, zeros(T, p.stored); kwargs...)
