"""
    view(A::FaerMatrix, rows, cols)
    view(v::FaerVector, indices)

A borrowed view retaining its Rust storage owner. Range selections, including
strided and reversed ranges, are borrowed directly during native operations.
Reserving capacity preserves views. Changing the owner's shape invalidates
existing views, even if the original shape is later restored. Closing a view
releases its reference without closing its owner.
"""
mutable struct FaerView{T,N,P,R,C} <: AbstractArray{T,N}
    owner::Union{Nothing,P}
    rows::R
    cols::C
    epoch::UInt
    dropped::Int
end
const FaerMatrixView = FaerView{T,2} where {T}
const FaerVectorView = FaerView{T,1} where {T}
const _NativeView = FaerView
const _ViewOwner = Union{FaerMatrix,FaerVector,FaerRow,FaerDiagonal}
const _ViewRange = Union{Colon,AbstractRange{<:Integer}}
const _ViewIndex = Union{_ViewRange,Integer}

function _view_owner(v::FaerView)
    p=v.owner
    isnothing(p) && throw(ArgumentError("view has been closed"))
    _handle(p)
    p.epoch==v.epoch ||
        throw(InvalidStateException("owner shape changed; create a new view", :invalidated))
    return p
end
function Base.isopen(v::FaerView)
    lock(_matrix_lock) do
        p=v.owner
        return !isnothing(p) && isopen(p) && p.epoch==v.epoch
    end
end
function Base.close(v::FaerView)
    lock(_matrix_lock) do ;
        v.owner=nothing
    end
    return nothing
end
Base.size(v::FaerMatrixView) = (length(v.rows), length(v.cols))
Base.size(v::FaerVectorView) = (v.dropped==1 ? length(v.cols) : length(v.rows),)
Base.IndexStyle(::Type{<:FaerView}) = IndexCartesian()
function _make_view(p, r, c, dropped)
    n=dropped==0 ? 2 : 1
    return FaerView{eltype(p),n,typeof(p),typeof(r),typeof(c)}(p, r, c, p.epoch, dropped)
end
_view_range(i::Integer, n) = Int(i):Int(i)
_view_range(::Colon, n) = 1:n
_view_range(r::AbstractRange, n) = r
function _view_indices(p, r, c)
    checkbounds(p, r, c)
    return _view_range(r, size(p, 1)), _view_range(c, size(p, 2))
end
function Base.view(p::Union{FaerMatrix,FaerRow}, r::_ViewRange, c::_ViewRange)
    lock(_matrix_lock) do
        _handle(p)
        rr, cc=_view_indices(p, r, c)
        return _make_view(p, rr, cc, 0)
    end
end
function Base.view(p::Union{FaerMatrix,FaerRow}, r::Integer, c::_ViewRange)
    lock(_matrix_lock) do
        _handle(p)
        rr, cc=_view_indices(p, r, c)
        return _make_view(p, rr, cc, 1)
    end
end
function Base.view(p::Union{FaerMatrix,FaerRow}, r::_ViewRange, c::Integer)
    lock(_matrix_lock) do
        _handle(p)
        rr, cc=_view_indices(p, r, c)
        return _make_view(p, rr, cc, 2)
    end
end
function Base.view(p::FaerVector, r::_ViewRange)
    lock(_matrix_lock) do
        _handle(p)
        checkbounds(p, r)
        return _make_view(p, _view_range(r, length(p)), 1:1, 2)
    end
end
"""
    diagonalview(D::FaerDiagonal)

Borrow the stored diagonal as a mutable vector. `diag(D)` returns a copy.
"""
function diagonalview(p::FaerDiagonal)
    lock(_matrix_lock) do
        _handle(p)
        return _make_view(p, 1:p.len, 1:1, 2)
    end
end
function Base.view(v::FaerMatrixView, r::_ViewIndex, c::_ViewIndex)
    # Keep zero-dimensional views on Julia's generic SubArray path.
    lock(_matrix_lock) do
        p=_view_owner(v)
        rr, cc=_view_indices(v, r, c)
        r isa Integer && c isa Integer && return Base.SubArray(v, (Int(r), Int(c)))
        return _make_view(
            p,
            v.rows[rr],
            v.cols[cc],
            r isa Integer ? 1 : c isa Integer ? 2 : 0,
        )
    end
end
function Base.view(v::FaerVectorView, r::_ViewRange)
    lock(_matrix_lock) do
        p=_view_owner(v)
        checkbounds(v, r)
        rr=_view_range(r, length(v))
        return v.dropped==1 ? _make_view(p, v.rows, v.cols[rr], 1) :
               _make_view(p, v.rows[rr], v.cols, 2)
    end
end
_physical_get(p::FaerMatrix, i, j) = p[i, j]
_physical_get(p::FaerRow, i, j) = p[j]
_physical_get(p::Union{FaerVector,FaerDiagonal}, i, j) = _collection_element(p, i)
_physical_set!(p::FaerMatrix, x, i, j) = setindex!(p, x, i, j)
_physical_set!(p::FaerRow, x, i, j) = setindex!(p, x, j)
_physical_set!(p::Union{FaerVector,FaerDiagonal}, x, i, j) = _collection_set!(p, x, i)
function Base.getindex(v::FaerMatrixView, i::Int, j::Int)
    lock(_matrix_lock) do
        p=_view_owner(v)
        checkbounds(v, i, j)
        return _physical_get(p, v.rows[i], v.cols[j])
    end
end
function Base.getindex(v::FaerVectorView, i::Int)
    lock(_matrix_lock) do
        p=_view_owner(v)
        checkbounds(v, i)
        return _physical_get(p, v.rows[v.dropped==1 ? 1 : i], v.cols[v.dropped==1 ? i : 1])
    end
end
function Base.setindex!(v::FaerMatrixView, x, i::Int, j::Int)
    lock(_matrix_lock) do
        p=_view_owner(v)
        checkbounds(v, i, j)
        _physical_set!(p, x, v.rows[i], v.cols[j])
    end
    return v
end
function Base.setindex!(v::FaerVectorView, x, i::Int)
    lock(_matrix_lock) do
        p=_view_owner(v)
        checkbounds(v, i)
        _physical_set!(p, x, v.rows[v.dropped==1 ? 1 : i], v.cols[v.dropped==1 ? i : 1])
    end
    return v
end
_owner_kind(::FaerMatrix{Float64}) = 0
_owner_kind(::FaerMatrix) = 1
_owner_kind(::_OwnedCollection) = 2
function _descriptor(p, r, c, transposed = false, conjugated = false)
    return Int[
        _owner_kind(p),
        first(r)-1,
        step(r),
        length(r),
        first(c)-1,
        step(c),
        length(c),
        transposed,
        conjugated,
    ]
end
function _view_source(v::FaerView)
    p=_view_owner(v)
    return p, _descriptor(p, v.rows, v.cols, v.dropped==1)
end
function _view_source(p::Union{FaerMatrix,FaerRow})
    _handle(p)
    return p, _descriptor(p, 1:size(p, 1), 1:size(p, 2))
end
function _view_source(p::FaerVector)
    _handle(p)
    return p, _descriptor(p, 1:length(p), 1:1)
end
function _view_source(p::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint})
    a, d=_view_source(parent(p))
    d[8]=1-d[8]
    p isa LinearAlgebra.Adjoint && (d[9]=1-d[9])
    return a, d
end
function _copy_view_matrix(v)
    lock(_matrix_lock) do
        p, d=_view_source(v)
        T=eltype(v)
        dims=ndims(v)==1 ? (length(v), 1) : size(v)
        GC.@preserve d return T===Float64 ?
                              _matrix_result(RustCrate.view_copy, dims, p, pointer(d)) :
                              _typed_result(T, RustCrate.view_copy, dims, p, pointer(d))
    end
end
Base.copy(v::FaerMatrixView) = _copy_view_matrix(v)
Base.copy(v::FaerVectorView) = _copy_view_vector(v)
function _copy_view_vector(v)
    lock(_matrix_lock) do
        p, d=_view_source(v)
        GC.@preserve d return _collection_result(
            FaerVector,
            eltype(v),
            length(v),
            RustCrate.vector_from_view,
            p,
            pointer(d),
            Int32(0),
        )
    end
end
Base.Matrix(v::FaerMatrixView) = Matrix(copy(v))
Base.Vector(v::FaerVectorView) = Vector(copy(v))
Base.Array(v::FaerMatrixView) = Matrix(v)
Base.Array(v::FaerVectorView) = Vector(v)
FaerMatrix(v::FaerMatrixView) = copy(v)
FaerVector(v::FaerVectorView) = copy(v)
function Base.fill!(v::FaerView, x)
    lock(_matrix_lock) do
        p, d=_view_source(v)
        T=eltype(v)
        value=Ref{T}(x)
        GC.@preserve d value _native_call(
            RustCrate.view_fill,
            p,
            pointer(d),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
        )
    end
    return v
end
function _copyto_faer_view!(v::FaerView, src::AbstractArray)
    lock(_matrix_lock) do
        p, d=_view_source(v)
        size(v)==size(src) || throw(DimensionMismatch("view and source shapes must match"))
        # Snapshot before borrowing the destination mutably, including when a
        # source shares its owner. This also gives ordered repeated-index writes.
        T=eltype(v)
        dims=v.dropped==1 ? (length(v), 1) : (length(v.rows), length(v.cols))
        # Julia strided sources of the native scalar are read in place (they
        # cannot alias Rust-owned storage); others are staged in Rust.
        a=if src isa FaerBorrowedArray{T} ||
             (src isa StridedArray{T} && _julia_borrowable(src))
            src isa FaerBorrowedArray ? src : FaerBorrowedArray(src)
        elseif src isa Union{FaerMatrix{T},FaerVector{T},FaerView{T}}
            # Native sources of the same scalar type are read through their own
            # descriptor. An injective destination is written directly and Rust
            # snapshots the source only if their bytes overlap; otherwise a
            # source sharing the destination owner is first copied natively so
            # Rust never sees an aliased mutable borrow.
            if _injective_descriptor(d) && d[9]==0
                q, sd=_view_source(src)
                GC.@preserve d sd _native_call(
                    RustCrate.view_assign_into,
                    p,
                    pointer(d),
                    q,
                    pointer(sd),
                )
                return
            end
            owner=src isa FaerView ? _view_owner(src) : src
            owner===p ? copy(src) : src
        else
            FaerMatrix{T}(reshape(Array(src), dims))
        end
        q, sd=_view_source(a)
        GC.@preserve d sd _native_call(RustCrate.view_assign, p, pointer(d), q, pointer(sd))
    end
    return v
end
Base.copyto!(v::FaerView, src::AbstractArray) = _copyto_faer_view!(v, src)
function Base.similar(v::FaerView, ::Type{T}, dims::Dims) where {T}
    T<:_NativeScalar && length(dims)==1 && return FaerVector{T}(undef, dims[1])
    T<:_NativeScalar && length(dims)==2 && return FaerMatrix{T}(undef, dims...)
    return Array{T}(undef, dims)
end
const _BorrowMatrix=Union{FaerMatrix,FaerMatrixView}
const _BorrowVector=Union{FaerVector,FaerVectorView}
function _view_binary(a, b, op; vector = false)
    lock(_matrix_lock) do
        P=promote_type(eltype(a), eltype(b))
        aa=eltype(a)===P ? a :
           ndims(a)==1 ? FaerVector{P}(Vector(a)) : FaerMatrix{P}(Matrix(a))
        bb=eltype(b)===P ? b :
           ndims(b)==1 ? FaerVector{P}(Vector(b)) : FaerMatrix{P}(Matrix(b))
        p, d=_view_source(aa)
        q, e=_view_source(bb)
        dims=op==2 ? (size(a, 1), size(b, 2)) : size(a)
        if op==2
            size(a, 2)==size(b, 1) ||
                throw(DimensionMismatch("inner dimensions must match"))
        else
            size(a)==size(b) || throw(DimensionMismatch("matrix shapes must match"))
        end
        GC.@preserve d e begin
            vector && return _collection_result(
                FaerVector,
                P,
                dims[1],
                RustCrate.vector_binary_views,
                p,
                pointer(d),
                q,
                pointer(e),
                Int32(op),
            )
            return P===Float64 ?
                   _matrix_result(
                RustCrate.view_binary,
                dims,
                p,
                pointer(d),
                q,
                pointer(e),
                Int32(op),
            ) :
                   _typed_result(
                P,
                RustCrate.view_binary,
                dims,
                p,
                pointer(d),
                q,
                pointer(e),
                Int32(op),
            )
        end
    end
end
for (f, op) in ((:+, 0), (:-, 1), (:*, 2))
    @eval begin
        Base.$f(a::FaerMatrixView, b::_BorrowMatrix) = _view_binary(a, b, $op)
        Base.$f(a::FaerMatrix, b::FaerMatrixView) = _view_binary(a, b, $op)
    end
end
Base.:*(a::_BorrowMatrix, b::FaerVectorView) = _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrixView, b::FaerVector) = _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix, b::FaerVector{<:Number}) = _view_binary(a, b, 2; vector = true)
const _BorrowTranspose=Union{
    LinearAlgebra.Transpose{<:Any,<:_BorrowMatrix},
    LinearAlgebra.Adjoint{<:Any,<:_BorrowMatrix},
}
Base.:*(a::_BorrowTranspose, b::_BorrowMatrix) = _view_binary(a, b, 2)
Base.:*(a::_BorrowMatrix, b::_BorrowTranspose) = _view_binary(a, b, 2)
Base.:*(a::_BorrowTranspose, b::_BorrowTranspose) = _view_binary(a, b, 2)
Base.:*(a::_BorrowTranspose, b::_BorrowVector) = _view_binary(a, b, 2; vector = true)
# Intersections with the compatibility methods accepting Julia vectors.
Base.:*(a::FaerMatrix{Float64}, b::FaerVector{<:Real}) =
    _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix{T}, b::FaerVector{S}) where {T<:_ExtraScalar,S<:Number} =
    _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix{Float64}, b::FaerVectorView{<:Real}) =
    _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix{T}, b::FaerVectorView{S}) where {T<:_ExtraScalar,S<:Number} =
    _view_binary(a, b, 2; vector = true)

function _vector_from_source(a)
    lock(_matrix_lock) do
        size(a, 2)==1 || throw(DimensionMismatch("a column source must have one column"))
        p, d=_view_source(a)
        GC.@preserve d return _collection_result(
            FaerVector,
            eltype(a),
            size(a, 1),
            RustCrate.vector_from_view,
            p,
            pointer(d),
            Int32(0),
        )
    end
end
FaerVector(a::FaerMatrix) = _vector_from_source(a)

function Base.copyto!(dest::FaerVector, src::AbstractVector)
    length(dest)==length(src) || throw(DimensionMismatch("vector lengths must match"))
    copyto!(view(dest, :), src)
    return dest
end
# Owner destinations keep Base's linear-order copyto! semantics for non-owner
# sources (differently shaped sources and shorter prefixes are accepted), but
# stage the whole result in Julia and write it with one native view assignment.
# Staging snapshots sources aliasing dest and converts elements before writing.
# FaerMatrix sources keep the shape-strict native assignment methods.
function _copyto_native_owner!(dest::Union{FaerMatrix,FaerRow}, src::AbstractArray)
    lock(_matrix_lock) do
        _handle(dest)
        isempty(src) && return
        length(src)<=length(dest) || throw(BoundsError(dest, LinearIndices(src)))
        staged=if size(src)==size(dest)
            src
        elseif length(src)==length(dest)
            reshape(Array(src), size(dest))
        else
            copyto!(Array(dest), src)
        end
        _copyto_faer_view!(view(dest, axes(dest)...), staged)
    end
    return dest
end
Base.copyto!(dest::Union{FaerMatrix,FaerRow}, src::AbstractArray) =
    _copyto_native_owner!(dest, src)
LinearAlgebra.mul!(
    dest::FaerVector,
    a::_BorrowMatrix,
    b::_BorrowVector,
    alpha::Number,
    beta::Number;
    team = nothing,
) = _vector_mul!(dest, a, b, alpha, beta; team)
function _vector_mul!(dest, a, b, alpha, beta; team = nothing)
    lock(_matrix_lock) do
        _handle(dest)
        size(a, 2)==length(b) && length(dest)==size(a, 1) ||
            throw(DimensionMismatch("matrix/vector product dimensions must match"))
        # Written natively in place; operands aliasing dest are snapshot.
        _dense_mul!(dest, a, b, alpha, beta; team)
    end
    return dest
end
LinearAlgebra.mul!(dest::FaerVector, a::_BorrowMatrix, b::_BorrowVector; team = nothing) =
    LinearAlgebra.mul!(dest, a, b, 1, 0; team)

# Preserve Rust vector ownership through native matrix/factor solves.
_native_vector_solve(a, b) = _vector_from_source(a\_copy_view_matrix(b))
for A in (:(FaerMatrix{Float64}), :(FaerFactorization{Float64}))
    @eval begin
        Base.:\(a::$A, b::FaerVector{<:Real}) = _native_vector_solve(a, b)
        Base.:\(a::$A, b::FaerVector{<:Complex}) = _native_vector_solve(a, b)
        Base.:\(a::$A, b::FaerVectorView{<:Real}) = _native_vector_solve(a, b)
        Base.:\(a::$A, b::FaerVectorView{<:Complex}) = _native_vector_solve(a, b)
    end
end
Base.:\(a::FaerMatrix{T}, b::FaerVector{S}) where {T<:_ExtraScalar,S<:Number} =
    _native_vector_solve(a, b)
Base.:\(a::FaerMatrix{T}, b::FaerVectorView{S}) where {T<:_ExtraScalar,S<:Number} =
    _native_vector_solve(a, b)
for T in (Float32, ComplexF32, ComplexF64)
    @eval begin
        Base.:\(a::FaerFactorization{$T}, b::FaerVector{<:Number}) =
            _native_vector_solve(a, b)
        Base.:\(a::FaerFactorization{$T}, b::FaerVectorView{<:Number}) =
            _native_vector_solve(a, b)
    end
end

function Base.deepcopy_internal(v::FaerView, dict::IdDict)
    get!(dict, v) do
        lock(_matrix_lock) do
            p=Base.deepcopy_internal(_view_owner(v), dict)
            return _make_view(
                p,
                Base.deepcopy_internal(v.rows, dict),
                Base.deepcopy_internal(v.cols, dict),
                v.dropped,
            )
        end
    end
end
