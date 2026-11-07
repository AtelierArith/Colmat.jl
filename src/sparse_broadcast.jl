# Whole-matrix broadcast/map for FaerSparseMatrix.
#
# Fast path: every array leaf of the flattened broadcast is the same native
# sparse owner, all other leaves are scalars (`Number`, `Ref`, 0-dimensional),
# `f` maps zero to zero and the result scalar is native. Stored values are
# then snapshotted once, mapped on the Julia side and attached to a copy of
# the source pattern (or bulk-written back when the destination is the
# source). Produced zeros are dropped, as SparseArrays does. Anything else
# converts the native leaves to `SparseMatrixCSC`, uses SparseArrays'
# broadcast and converts native-scalar sparse results back.
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
for S in (
    SparseArrays.HigherOrderFns.SparseMatStyle,
    SparseArrays.HigherOrderFns.SparseVecStyle,
    SparseArrays.HigherOrderFns.PromoteToSparse,
    LinearAlgebra.StructuredMatrixStyle,
)
    @eval Broadcast.BroadcastStyle(::FaerSparseStyle, ::$S) = FaerSparseStyle()
end

const _SparseScalarArg=Union{Number,Base.RefValue,AbstractArray{<:Any,0}}
_sparse_scalar_value(x::Number) = x
_sparse_scalar_value(x) = x[]
_sparse_insert(x, ::FaerSparseMatrix) = x
_sparse_insert(_, a) = _sparse_scalar_value(a)
@inline _sparse_apply(f::F, args, x) where {F} = f(map(a -> _sparse_insert(x, a), args)...)

# Returns `(f, args, src, R)` when the native value path applies, else `nothing`.
function _sparse_map_plan(bc::Broadcast.Broadcasted)
    flat=Broadcast.flatten(bc)
    args=flat.args
    src=nothing
    for a in args
        if a isa FaerSparseMatrix
            isnothing(src) && (src=a)
            a===src || return nothing
        elseif !(a isa _SparseScalarArg)
            return nothing
        end
    end
    isnothing(src) && return nothing
    _handle(src)
    axes(src)==axes(bc) || return nothing
    R=Broadcast.combine_eltypes(flat.f, args)
    R<:_NativeScalar && isconcretetype(R) || return nothing
    iszero(_sparse_apply(flat.f, args, zero(eltype(src)))) || return nothing
    return (flat.f, args, src, R)
end
function _sparse_map_values!(out, f::F, args, vals) where {F}
    @inbounds for i in eachindex(out, vals)
        out[i]=_sparse_apply(f, args, vals[i])
    end
    return out
end
# One native-to-Julia value transfer; the source must not change meanwhile.
function _sparse_mapped_values(f::F, args, src::FaerSparseMatrix{T}, ::Type{R}) where {F,T,R}
    epoch=src.epoch
    vals=Vector(SparseArrays.nonzeros(src))
    out=R===T ? vals : Vector{R}(undef, length(vals))
    _sparse_map_values!(out, f, args, vals)
    src.epoch==epoch && isopen(src) || throw(
        InvalidStateException(
            "sparse broadcast source changed while mapping its values",
            :invalidated,
        ),
    )
    return out
end
function _sparse_from_values(src::FaerSparseMatrix, vals::Vector{R}) where {R}
    out=GC.@preserve vals _sparse_result(
        R,
        size(src),
        src.format,
        RustCrate.sparse_from_values,
        src,
        Ptr{UInt8}(pointer(vals)),
        UInt(length(vals)),
        _tag(R),
    )
    any(iszero, vals) && SparseArrays.dropzeros!(out)
    return out
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
_julia_sparse_broadcasted(a::FaerMatrix) = Matrix(a)
_julia_sparse_broadcasted(a) = a
function _first_faer_sparse(bc::Broadcast.Broadcasted)
    for a in bc.args
        a isa FaerSparseMatrix && return a
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
        if !isnothing(plan)
            f, args, src, R=plan
            return _sparse_from_values(src, _sparse_mapped_values(f, args, src, R))
        end
        first=_first_faer_sparse(bc)
        r=Broadcast.materialize(_julia_sparse_broadcasted(bc))
        return _native_sparse_result(r, isnothing(first) ? :csc : first.format)
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
            f, args, src, R=plan
            vals=_sparse_mapped_values(f, args, src, R)
            if src===dest
                # Convert before writing: InexactError leaves dest unchanged.
                _sparse_write_values!(dest, vals isa Vector{T} ? vals : Vector{T}(vals))
            else
                copyto!(dest, _sparse_from_values(src, vals))
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
result conventions as `broadcast` and SparseArrays: zero-preserving `f` over a
single source maps only stored values and drops produced zeros; other cases
use SparseArrays' algorithm and return a `FaerSparseMatrix` when the result
scalar is native.
"""
Base.map(f, a::FaerSparseMatrix) = broadcast(f, a)
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
