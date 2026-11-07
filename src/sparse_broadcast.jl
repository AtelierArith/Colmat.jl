# Whole-matrix broadcast/map for FaerSparseMatrix and its transpose/adjoint.
#
# Fast path: every array leaf of the flattened broadcast is a native sparse
# owner or a `transpose`/`adjoint` of one, with the axes of the result; all
# other leaves are scalars (`Number`, `Ref`, 0-dimensional); `f` maps zeros to
# zero and the result scalar is native. Stored values are snapshotted once per
# operand, mapped on the Julia side and attached to a native pattern:
#
# - one operand: its own pattern (a transposed operand reads the parent's
#   arrays in the other storage format, conjugated for `adjoint`), or a bulk
#   write-back when the destination is the source;
# - several operands: transposed ones are copied natively, all are brought to
#   the first operand's format, and either share one pattern (values are
#   zipped directly) or are aligned in Rust on faer's union pattern.
#
# `f` runs in Julia: a Rust-side callback would need a `@cfunction` per `f`
# and buys nothing, since the values cross once either way. Produced zeros are
# dropped, as SparseArrays does. Anything else converts the native leaves to
# `SparseMatrixCSC`, uses SparseArrays' broadcast and converts native-scalar
# sparse results back.
"""
    FaerSparseStyle()

Broadcast style of [`FaerSparseMatrix`](@ref). It wins over dense, Julia
sparse and structured matrix styles so that results follow SparseArrays'
conventions while staying native sparse owners where possible.
"""
struct FaerSparseStyle <: Broadcast.AbstractArrayStyle{2} end
FaerSparseStyle(::Val{0}) = FaerSparseStyle()
FaerSparseStyle(::Val{1}) = FaerSparseStyle()
FaerSparseStyle(::Val{2}) = FaerSparseStyle()
FaerSparseStyle(::Val{N}) where {N} = Broadcast.DefaultArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:FaerSparseMatrix}) = FaerSparseStyle()
Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,<:FaerSparseMatrix}}) =
    FaerSparseStyle()
Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,<:FaerSparseMatrix}}) =
    FaerSparseStyle()
for S in (
    SparseArrays.HigherOrderFns.SparseMatStyle,
    SparseArrays.HigherOrderFns.SparseVecStyle,
    SparseArrays.HigherOrderFns.PromoteToSparse,
    LinearAlgebra.StructuredMatrixStyle,
)
    @eval Broadcast.BroadcastStyle(::FaerSparseStyle, ::$S) = FaerSparseStyle()
end

const _SparseScalarArg=Union{Number,Base.RefValue,AbstractArray{<:Any,0}}
const _SparseLeaf=Union{FaerSparseMatrix,_SparseTransform}
_sparse_scalar_value(x::Number) = x
_sparse_scalar_value(x) = x[]
# Stored values of one operand, aligned with the result pattern.
struct _SparseLeafValues{V<:AbstractVector}
    values::V
end
_sparse_leaf_at(a::_SparseLeafValues, i) = @inbounds a.values[i]
_sparse_leaf_at(a, _) = a
_sparse_zero_arg(a::_SparseLeaf) = zero(eltype(a))
_sparse_zero_arg(a) = _sparse_scalar_value(a)

# Returns `(f, args, leaves, R)` when the native value path applies, else
# `nothing`; `leaves` are the distinct sparse operands in order.
function _sparse_map_plan(bc::Broadcast.Broadcasted)
    flat=Broadcast.flatten(bc)
    args=flat.args
    leaves=_SparseLeaf[]
    for a in args
        if a isa _SparseLeaf
            any(x -> x===a, leaves) || push!(leaves, a)
        elseif !(a isa _SparseScalarArg)
            return nothing
        end
    end
    isempty(leaves) && return nothing
    for l in leaves
        _handle(_sparse_stored(l))
        axes(l)==axes(bc) || return nothing
    end
    R=Broadcast.combine_eltypes(flat.f, args)
    R<:_NativeScalar && isconcretetype(R) || return nothing
    iszero(flat.f(map(_sparse_zero_arg, args)...)) || return nothing
    return (flat.f, args, leaves, R)
end
function _sparse_map_values!(out, f::F, prepared) where {F}
    @inbounds for i in eachindex(out)
        out[i]=f(map(a -> _sparse_leaf_at(a, i), prepared)...)
    end
    return out
end
function _sparse_check_epochs(owners, epochs)
    for (o, e) in zip(owners, epochs)
        o.epoch==e && isopen(o) || throw(
            InvalidStateException(
                "sparse broadcast source changed while mapping its values",
                :invalidated,
            ),
        )
    end
end
# Map `f` over the aligned value buffers `vals[j]` of `leaves[j]`; the
# owners must not change meanwhile.
function _sparse_mapped(f::F, args, leaves, vals, ::Type{R}, owners) where {F,R}
    epochs=map(o -> o.epoch, owners)
    prepared=map(args) do a
        a isa _SparseLeaf || return _sparse_scalar_value(a)
        return _SparseLeafValues(vals[findfirst(x -> x===a, leaves)])
    end
    v1=first(vals)
    out=length(vals)==1&&eltype(v1)===R ? v1 : Vector{R}(undef, length(v1))
    _sparse_map_values!(out, f, prepared)
    _sparse_check_epochs(owners, epochs)
    return out
end
# Stored values of `l` in its parent's physical order (conjugated for adjoint).
function _sparse_leaf_values(l::_SparseLeaf)
    v=Vector(SparseArrays.nonzeros(_sparse_stored(l)))
    l isa LinearAlgebra.Adjoint && eltype(v)<:Complex && conj!(v)
    return v
end
# A new owner with `src`'s pattern (or, `transposed`, the same arrays read as
# the transpose in the other format) and the values `vals`.
function _sparse_from_values(src::FaerSparseMatrix, vals::Vector{R}; transposed = false) where {R}
    format=transposed==(src.format===:csc) ? :csr : :csc
    f=transposed ? RustCrate.sparse_from_values_transposed : RustCrate.sparse_from_values
    out=GC.@preserve vals _sparse_result(
        R,
        transposed ? reverse(size(src)) : size(src),
        format,
        f,
        src,
        Ptr{UInt8}(pointer(vals)),
        UInt(length(vals)),
        _tag(R),
    )
    any(iszero, vals) && SparseArrays.dropzeros!(out)
    return out
end
# Several operands: plain owners in the first operand's format, with values
# aligned on one shared pattern; returns `(pattern_owner, buffers)`.
function _sparse_aligned_leaves(leaves, ::Type{R}) where {R}
    format=_oriented_format(first(leaves))
    owners=map(leaves) do l
        x=l isa FaerSparseMatrix ? l : copy(l)
        x.format===format ? x : copy(x; format)
    end
    base=first(owners)
    same(x, y)=_native_call(RustCrate.sparse_same_structure, x, y)
    all(o -> same(base, o), owners) &&
        return base, map(o -> Vector(SparseArrays.nonzeros(o)), owners)
    union=base
    for o in owners[2:end]
        same(union, o) && continue
        union=_sparse_result(
            R,
            size(base),
            format,
            RustCrate.sparse_union_from,
            union,
            o,
            _tag(R),
        )
    end
    k=union.stored
    vals=map(owners) do o
        v=Vector{eltype(o)}(undef, k)
        GC.@preserve v _native_call(
            RustCrate.sparse_aligned_values,
            o,
            union,
            Ptr{UInt8}(pointer(v)),
            UInt(k),
        )
        v
    end
    return union, vals
end
# Native result of a planned broadcast. When the single operand is the plain
# owner `dest`, the mapped values are returned for a bulk write-back instead.
function _sparse_planned_result(plan; dest = nothing)
    f, args, leaves, R=plan
    if length(leaves)==1
        l=only(leaves)
        src=_sparse_stored(l)
        vals=_sparse_mapped(f, args, leaves, [_sparse_leaf_values(l)], R, (src,))
        l===dest && return vals
        return _sparse_from_values(src, vals; transposed = !(l isa FaerSparseMatrix))
    end
    owners=map(_sparse_stored, leaves)
    pattern, aligned=_sparse_aligned_leaves(leaves, R)
    vals=_sparse_mapped(f, args, leaves, aligned, R, owners)
    return _sparse_from_values(pattern, vals)
end
function _sparse_write_values!(dest::FaerSparseMatrix{T}, vals::Vector{T}) where {T}
    GC.@preserve vals _native_call(
        RustCrate.sparse_write_values,
        dest,
        Ptr{UInt8}(pointer(vals)),
        UInt(length(vals)),
    )
    any(iszero, vals) && SparseArrays.dropzeros!(dest)
    return dest
end

_julia_sparse_broadcasted(bc::Broadcast.Broadcasted) =
    Broadcast.broadcasted(bc.f, map(_julia_sparse_broadcasted, bc.args)...)
_julia_sparse_broadcasted(a::FaerSparseMatrix) = SparseArrays.SparseMatrixCSC(a)
_julia_sparse_broadcasted(a::_SparseTransform) = SparseArrays.SparseMatrixCSC(copy(a))
_julia_sparse_broadcasted(a::FaerMatrix) = Matrix(a)
_julia_sparse_broadcasted(a) = a
function _first_faer_sparse(bc::Broadcast.Broadcasted)
    for a in bc.args
        a isa _SparseLeaf && return a
        if a isa Broadcast.Broadcasted
            found=_first_faer_sparse(a)
            isnothing(found) || return found
        end
    end
    return nothing
end
function _native_sparse_result(r, format)
    r isa SparseArrays.SparseMatrixCSC && eltype(r)<:_NativeScalar || return r
    return FaerSparseMatrix{eltype(r)}(r; format)
end

function Base.copy(bc::Broadcast.Broadcasted{FaerSparseStyle})
    lock(_matrix_lock) do
        plan=_sparse_map_plan(bc)
        isnothing(plan) || return _sparse_planned_result(plan)
        first=_first_faer_sparse(bc)
        r=Broadcast.materialize(_julia_sparse_broadcasted(bc))
        return _native_sparse_result(r, isnothing(first) ? :csc : _oriented_format(first))
    end
end
function Base.copyto!(dest::FaerSparseMatrix{T}, bc::Broadcast.Broadcasted{FaerSparseStyle}) where {T}
    lock(_matrix_lock) do
        _handle(dest)
        axes(dest)==axes(bc) ||
            throw(DimensionMismatch("sparse broadcast destination axes must match"))
        plan=_sparse_map_plan(bc)
        if isnothing(plan)
            # Broadcast into an independent Julia snapshot so that failures
            # and every destination/source alias leave `dest` consistent.
            work=SparseArrays.SparseMatrixCSC(dest)
            Broadcast.materialize!(work, _julia_sparse_broadcasted(bc))
            copyto!(dest, FaerSparseMatrix{T}(work; format = dest.format))
        else
            r=_sparse_planned_result(plan; dest)
            if r isa FaerSparseMatrix
                copyto!(dest, r)
            else
                # Convert before writing: InexactError leaves dest unchanged.
                _sparse_write_values!(dest, r isa Vector{T} ? r : Vector{T}(r))
            end
        end
    end
    return dest
end
function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{FaerSparseStyle})
    lock(() -> Broadcast.materialize!(dest, _julia_sparse_broadcasted(bc)), _matrix_lock)
    return dest
end

function _sparse_map_shapes(dest, args...)
    all(a -> size(a)==size(dest), args) ||
        throw(DimensionMismatch("sparse map arguments must have equal sizes"))
end
"""
    map(f, A::FaerSparseMatrix, Bs::FaerSparseMatrix...)
    map!(f, dest::FaerSparseMatrix, A::FaerSparseMatrix, Bs::FaerSparseMatrix...)

Element-wise map over equally sized native sparse matrices, with the same
result conventions as `broadcast` and SparseArrays. Zero-preserving `f` maps
only stored values natively: one source keeps its pattern, several sources
share one pattern or are aligned on their union pattern, and produced zeros
are dropped. `map(f, transpose(A))`/`map(f, A')` read `A`'s storage in the
other format. Other cases use SparseArrays' algorithm and return a
`FaerSparseMatrix` when the result scalar is native.
"""
Base.map(f, a::FaerSparseMatrix) = broadcast(f, a)
Base.map(f, a::_SparseTransform) = broadcast(f, a)
function Base.map(f, a::FaerSparseMatrix, b::FaerSparseMatrix, rest::FaerSparseMatrix...)
    _sparse_map_shapes(a, b, rest...)
    return broadcast(f, a, b, rest...)
end
function Base.map!(f, dest::FaerSparseMatrix, a::FaerSparseMatrix)
    _sparse_map_shapes(dest, a)
    return broadcast!(f, dest, a)
end
function Base.map!(
    f,
    dest::FaerSparseMatrix,
    a::FaerSparseMatrix,
    b::FaerSparseMatrix,
    rest::FaerSparseMatrix...,
)
    _sparse_map_shapes(dest, a, b, rest...)
    return broadcast!(f, dest, a, b, rest...)
end
