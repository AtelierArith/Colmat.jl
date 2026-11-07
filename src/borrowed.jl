"""
    faerview(A::StridedVecOrMat{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}

Borrow Julia array memory for native faer operations without copying it. The
returned [`FaerBorrowedArray`](@ref) keeps `A` alive and hands Rust a faer
`MatRef` built from `pointer(A)`, `size(A)` and `strides(A)` during each native
call only, so negative (reversed) and non-unit strides are read in place.

No pointer, size or stride is cached: they are re-read from `A` at every call,
so a resized `Vector` parent is seen at its current length and address, and
writes to `A` are visible to later native calls (and vice versa through
`setindex!`).

`A` must be a strided `Array`/`Memory`-backed vector or matrix of one of the
four native scalars with an element-aligned pointer (for example not a
`reinterpret` of an odd byte offset), and its layout must not map two indices
to the same element (zero strides or overlapping strides throw
`ArgumentError`). Every `FaerBorrowedArray` reachable from the public API,
including its views, satisfies these checks.
"""
function faerview(a::StridedVecOrMat{T}) where {T}
    T<:_NativeScalar || throw(
        ArgumentError("faerview requires Float32, Float64, ComplexF32 or ComplexF64"),
    )
    _borrowable_memory(a) ||
        throw(ArgumentError("faerview requires memory backed by a Julia Array or Memory"))
    _aligned(a) || throw(ArgumentError("faerview requires an element-aligned pointer"))
    _injective_layout(a) ||
        throw(ArgumentError("faerview requires a layout without repeated elements"))
    return FaerBorrowedArray(a)
end
faerview(a::AbstractArray) = throw(
    ArgumentError("faerview requires a strided Julia vector or matrix; got $(typeof(a))"),
)

"""
    FaerBorrowedArray{T,N,A}
    FaerBorrowedMatrix{T,A}
    FaerBorrowedVector{T,A}

A zero-copy borrow of a Julia strided array `A`, created by [`faerview`](@ref).
Indexing, `strides` and `pointer` forward to the parent. Native consumers read
it as a faer `MatRef`: `copy`/`FaerMatrix`/`FaerVector`, `+`/`-`/`*` (including
transposes and mixed native operands), matrix-vector `mul!` into a
`FaerVector`, `mean`/`var`, and the source of `copyto!` into native owners and
views. Views of it remain borrowed.
"""
struct FaerBorrowedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    # Unchecked: faerview validates layouts; internal read-only borrows may
    # repeat elements.
    FaerBorrowedArray(a::AbstractArray{T,N}) where {T,N} = new{T,N,typeof(a)}(a)
end
const FaerBorrowedMatrix{T,A}=FaerBorrowedArray{T,2,A}
const FaerBorrowedVector{T,A}=FaerBorrowedArray{T,1,A}
faerview(b::FaerBorrowedArray) = b

function _borrowable_memory(a)
    p=parent(a)
    p===a && return a isa Union{Array,Memory}
    return _borrowable_memory(p)
end
function _injective_layout(a)
    dims=[(abs(stride(a, d)), size(a, d)) for d in 1:ndims(a) if size(a, d)>1]
    sort!(dims)
    for (k, (s, n)) in enumerate(dims)
        s==0 && return false
        k>1 && dims[k-1][1]*(dims[k-1][2]-1)>=s && return false
    end
    return true
end
_aligned(a) = UInt(pointer(a))%Base.datatype_alignment(eltype(a))==0
# Read-only internal borrows (copyto! sources) may repeat elements.
_julia_borrowable(a::StridedVecOrMat{<:_NativeScalar}) = _borrowable_memory(a) && _aligned(a)
_julia_borrowable(::Any) = false
_faerview_valid(a) = _julia_borrowable(a) && _injective_layout(a)

Base.parent(b::FaerBorrowedArray) = b.parent
Base.size(b::FaerBorrowedArray) = size(b.parent)
Base.axes(b::FaerBorrowedArray) = axes(b.parent)
Base.IndexStyle(::Type{<:FaerBorrowedArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.@propagate_inbounds Base.getindex(b::FaerBorrowedArray, i::Int...) = b.parent[i...]
Base.@propagate_inbounds Base.setindex!(b::FaerBorrowedArray, x, i::Int...) =
    (b.parent[i...]=x; b)
Base.strides(b::FaerBorrowedArray) = strides(b.parent)
Base.stride(b::FaerBorrowedArray, d::Integer) = stride(b.parent, d)
Base.elsize(::Type{<:FaerBorrowedArray{T}}) where {T} = sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, b::FaerBorrowedArray{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, b.parent)
Base.similar(b::FaerBorrowedArray, ::Type{T}, dims::Dims) where {T} = similar(b.parent, T, dims)
function Base.view(b::FaerBorrowedArray, I::Vararg{Any})
    v=view(b.parent, I...)
    # Layouts faerview would reject stay ordinary Julia views.
    return _faerview_valid(v) ? FaerBorrowedArray(v) : v
end

# Descriptor kind 4 (kind 3 is the contiguous diagonal of structured_products.jl):
# the nine view words select the whole borrowed array, then rows, cols, row
# stride, col stride (elements) and the scalar tag follow.
# `_native_argument` keeps `b` itself in `_native_call_plain`'s preserved tuple,
# which roots the parent while Rust reads `pointer(parent)`.
_owner_kind(::FaerBorrowedArray) = 4
_raw_owner_argument(x, owner::FaerBorrowedArray) = Ptr{UInt8}(pointer(owner.parent))
function _view_source(b::FaerBorrowedArray{T}) where {T}
    a=b.parent
    m, n=ndims(a)==1 ? (length(a), 1) : size(a)
    rs=stride(a, 1)
    cs=ndims(a)==1 ? m*rs : stride(a, 2)
    d=_descriptor(b, 1:m, 1:n)
    append!(d, (m, n, rs, cs, _tag(T)))
    return b, d
end

Base.copy(b::FaerBorrowedMatrix) = _copy_view_matrix(b)
Base.copy(b::FaerBorrowedVector) = _copy_view_vector(b)
FaerMatrix(b::FaerBorrowedMatrix{<:_NativeScalar}) = copy(b)
FaerVector(b::FaerBorrowedVector{<:_NativeScalar}) = copy(b)

const _ProductMatrix=Union{_BorrowMatrix,FaerBorrowedMatrix}
const _ProductVector=Union{_BorrowVector,FaerBorrowedVector}
const _BorrowedTranspose=Union{
    LinearAlgebra.Transpose{<:Any,<:FaerBorrowedMatrix},
    LinearAlgebra.Adjoint{<:Any,<:FaerBorrowedMatrix},
}
const _ProductTranspose=Union{_BorrowTranspose,_BorrowedTranspose}
const _ProductLeft=Union{_ProductMatrix,_ProductTranspose}
# Each method needs one borrowed operand; first argument types are disjoint.
for (f, op) in ((:+, 0), (:-, 1))
    @eval begin
        Base.$f(a::FaerBorrowedMatrix, b::_ProductMatrix) = _view_binary(a, b, $op)
        Base.$f(a::_BorrowMatrix, b::FaerBorrowedMatrix) = _view_binary(a, b, $op)
    end
end
Base.:*(a::FaerBorrowedMatrix, b::_ProductLeft) = _view_binary(a, b, 2)
Base.:*(a::_BorrowedTranspose, b::_ProductLeft) = _view_binary(a, b, 2)
Base.:*(a::Union{_BorrowMatrix,_BorrowTranspose}, b::Union{FaerBorrowedMatrix,_BorrowedTranspose}) =
    _view_binary(a, b, 2)
Base.:*(a::Union{FaerBorrowedMatrix,_BorrowedTranspose}, b::_ProductVector) =
    _view_binary(a, b, 2; vector = true)
Base.:*(a::Union{_BorrowMatrix,_BorrowTranspose}, b::FaerBorrowedVector) =
    _view_binary(a, b, 2; vector = true)
# Intersections with the native-matrix methods accepting Julia vectors.
Base.:*(a::FaerMatrix{Float64}, b::FaerBorrowedVector{<:Real}) =
    _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix{T}, b::FaerBorrowedVector{S}) where {T<:_ExtraScalar,S<:Number} =
    _view_binary(a, b, 2; vector = true)
for (A, B) in ((:FaerBorrowedMatrix, :_ProductVector), (:_BorrowMatrix, :FaerBorrowedVector))
    @eval begin
        LinearAlgebra.mul!(dest::FaerVector, a::$A, b::$B, alpha::Number, beta::Number) =
            _vector_mul!(dest, a, b, alpha, beta)
        LinearAlgebra.mul!(dest::FaerVector, a::$A, b::$B) = _vector_mul!(dest, a, b, 1, 0)
    end
end
