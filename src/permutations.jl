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
function _permute(a::FaerMatrixView, p::FaerPermutation, columns)
    lock(_matrix_lock) do
        o, d=_view_source(a)
        _check_permutation(a, p, columns)
        T=eltype(a)
        GC.@preserve d return T===Float64 ?
                              _matrix_result(
            RustCrate.permute_from_view,
            size(a),
            o,
            pointer(d),
            p,
            columns,
        ) :
                              _typed_result(
            T,
            RustCrate.permute_from_view,
            size(a),
            o,
            pointer(d),
            p,
            columns,
        )
    end
end
"""
    permute_rows(A, p)
    permute_cols(A, p)

Return independent Rust-owned storage containing `A[p,:]` or `A[:,p]`.
`p` may be a `FaerPermutation` or a Julia vector containing a permutation.
A view source is borrowed directly; no intermediate copy is made.
"""
permute_rows(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), false)
permute_cols(a::_BorrowMatrix, p::AbstractVector{<:Integer}) =
    _permute(a, _as_permutation(p), true)
Base.getindex(a::FaerMatrix, p::FaerPermutation, ::Colon) = permute_rows(a, p)
Base.getindex(a::FaerMatrix, ::Colon, p::FaerPermutation) = permute_cols(a, p)
Base.getindex(a::FaerMatrix, p::FaerPermutation, q::FaerPermutation) =
    permute_cols(permute_rows(a, p), q)

# Dense owners and views accepted by the native in-place kernels. Vectors are
# always permuted along their single axis (descriptors of row-vector views are
# transposed to a column).
const _DensePermutable=Union{FaerMatrix,FaerMatrixView,FaerRow,FaerVector,FaerVectorView}
_permutation_dims(a) = ndims(a)==1 ? (length(a), 1) : size(a)
# A zero step over more than one index would alias a mutable faer borrow.
_injective(a) = true
_injective(v::FaerView) =
    (length(v.rows)<=1 || step(v.rows)!=0) && (length(v.cols)<=1 || step(v.cols)!=0)
function _permutation_requirements(a, columns)
    m, n=_permutation_dims(a)
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.permutation_requirements,
        UInt(m),
        UInt(n),
        _tag(eltype(a)),
        columns,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _check_permutation_scratch(a, columns, scratch)
    isnothing(scratch) && return nothing
    scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    req=_permutation_requirements(a, columns)
    cap=scratch_capacity(scratch)
    cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
        ArgumentError(
            "permutation scratch capacity/alignment is insufficient; call reserve!(scratch, permute_rows!, A, p)",
        ),
    )
    return nothing
end
# Stage a repeated-index view through an independent result and ordered writes.
function _staged_permute!(a, p, columns, inverse)
    q=inverse ? invperm(p) : p
    if ndims(a)==1
        copyto!(a, Vector(permute!(copy(a), q)))
    else
        copyto!(a, Matrix(_permute(copy(a), q, columns)))
    end
    return a
end
function _permute_dense!(a, p::FaerPermutation, columns, inverse, scratch)
    lock(_matrix_lock) do
        o, d=_view_source(a)
        _check_permutation(a, p, columns)
        _check_permutation_scratch(a, columns, scratch)
        _injective(a) || return _staged_permute!(a, p, columns, inverse)
        GC.@preserve d begin
            isnothing(scratch) ?
            _native_call(RustCrate.permute_view_storage, o, pointer(d), p, columns, inverse) :
            _native_call(
                RustCrate.permute_view_workspace,
                o,
                pointer(d),
                p,
                columns,
                inverse,
                scratch,
            )
        end
    end
    return a
end
"""
    permute_rows!(A, p; scratch=nothing)
    permute_cols!(A, p; scratch=nothing)

Gather rows or columns into the same matrix with faer's in-place kernel,
preserving the matrix's shape, owner and existing views. `A` may be a
`FaerMatrix`, a `FaerRow` (columns only) or a strided/reversed view; a view is
borrowed directly from its owner without an intermediate copy. The kernel
copies the selection into its workspace: `scratch=FaerScratch(permute_rows!, A, p)`
supplies that workspace from the caller, otherwise a call-local native buffer
is allocated. Closed or invalidated arguments, a wrong permutation length and
insufficient scratch fail before any write. A view repeating indices (zero step)
is updated through an independent temporary result and ordered writes.
faer's permutation kernels are sequential, so there is no `team=` option.
"""
permute_rows!(a::_BorrowMatrix, p::AbstractVector{<:Integer}; scratch = nothing) =
    _permute_dense!(a, _as_permutation(p), false, false, scratch)
permute_cols!(a::_BorrowMatrix, p::AbstractVector{<:Integer}; scratch = nothing) =
    _permute_dense!(a, _as_permutation(p), true, false, scratch)
permute_cols!(a::FaerRow, p::AbstractVector{<:Integer}; scratch = nothing) =
    _permute_dense!(a, _as_permutation(p), true, false, scratch)
"""
    permute!(v::FaerVector, p; scratch=nothing)
    invpermute!(v::FaerVector, p; scratch=nothing)

Permute a native vector or vector view in place (`v[:] = v[p]` or
`v[p] = v`). `invpermute!` applies faer's borrowed inverse of `p` without
building a new permutation. `scratch` follows [`permute_rows!`](@ref).
"""
Base.permute!(v::_BorrowVector, p::AbstractVector{<:Integer}; scratch = nothing) =
    _permute_dense!(v, _as_permutation(p), false, false, scratch)
Base.invpermute!(v::_BorrowVector, p::AbstractVector{<:Integer}; scratch = nothing) =
    _permute_dense!(v, _as_permutation(p), false, true, scratch)
function _swap!(a, i::Integer, j::Integer, columns)
    lock(_matrix_lock) do
        o, d=_view_source(a)
        n=size(a, columns ? 2 : 1)
        1<=i<=n && 1<=j<=n || throw(BoundsError(a, (i, j)))
        if !_injective(a)
            tmp=copy(a)
            _swap!(tmp, i, j, columns)
            copyto!(a, Matrix(tmp))
            return
        end
        GC.@preserve d _native_call(
            RustCrate.swap_view_storage,
            o,
            pointer(d),
            UInt(i-1),
            UInt(j-1),
            columns,
        )
    end
    return a
end
"""
    swap_rows!(A, i, j)
    swap_cols!(A, i, j)

Exchange two rows or columns using 1-based indices. The shape is unchanged,
and existing borrowed views remain valid. Views are swapped natively in place.
"""
swap_rows!(a::_BorrowMatrix, i::Integer, j::Integer) = _swap!(a, i, j, false)
swap_cols!(a::_BorrowMatrix, i::Integer, j::Integer) = _swap!(a, i, j, true)
swap_cols!(a::FaerRow, i::Integer, j::Integer) = _swap!(a, i, j, true)
FaerMatrix(p::FaerPermutation) =
    permute_rows(FaerMatrix{Float64}(LinearAlgebra.I, length(p), length(p)), p)
Base.Matrix(p::FaerPermutation) = Matrix(FaerMatrix(p))
Base.Matrix{T}(p::FaerPermutation) where {T} = convert(Matrix{T}, Matrix(p))

@doc (@doc permute_rows) permute_cols
@doc (@doc permute_rows!) permute_cols!
@doc (@doc swap_rows!) swap_cols!
