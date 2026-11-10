"""
    FaerVector(v)
    FaerVector{T}(v)

A Julia vector owning a Rust `faer::Col<T>`. Construction copies values. Four
native scalar types are preserved; other real inputs convert to Float64 and other
complex inputs to ComplexF64. An explicit unsupported `T` throws `ArgumentError`.
"""
mutable struct FaerVector{T,H} <: AbstractVector{T}
    handle::Union{Nothing,H}
    len::Int
    epoch::UInt
end
"""
    FaerRow(v)

A one-row Julia matrix owning a Rust `faer::Row<T>`. Values are not implicitly
conjugated. Use `v'` for an adjoint row of a FaerVector.
"""
mutable struct FaerRow{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    len::Int
    epoch::UInt
end
"""
    FaerDiagonal(v)

A square diagonal Julia matrix owning a Rust `faer::diag::Diag<T>`. Off-diagonal
entries are implicit zeros; assigning a nonzero off the diagonal is an error.
Products with a `FaerMatrix`, its range views or native triangular wrappers
(either side, `*` and `mul!`), `D * v::FaerVector` and `r::FaerRow * D` scale in
Rust from the stored diagonal without forming a dense matrix.
"""
mutable struct FaerDiagonal{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    len::Int
    epoch::UInt
end
const _OwnedCollection = Union{FaerVector,FaerRow,FaerDiagonal}
_collection_kind(::Type{<:FaerVector}) = Int32(0)
_collection_kind(::Type{<:FaerRow}) = Int32(1)
_collection_kind(::Type{<:FaerDiagonal}) = Int32(2)
_collection_kind(x::_OwnedCollection) = _collection_kind(typeof(x))
_collection_type(::FaerVector) = FaerVector
_collection_type(::FaerRow) = FaerRow
_collection_type(::FaerDiagonal) = FaerDiagonal
function _handle(x::_OwnedCollection)
    h = x.handle
    isnothing(h) && throw(ArgumentError("native collection has been closed"))
    return h
end
_native_argument(x::_OwnedCollection) = _handle(x)
_is_native_owner(::_OwnedCollection) = true
function _collection_result(C, ::Type{T}, n, f, args...) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        h = RustCrate.VectorOwner()
        _native_call(f, h, args...)
        return C{T,typeof(h)}(h, n, UInt(0))
    end
end
function _collection_load(C, ::Type{T}, v::AbstractVector) where {T<:_NativeScalar}
    data = Vector{T}(v)
    GC.@preserve data return _collection_result(
        C,
        T,
        length(data),
        RustCrate.collection_load,
        Ptr{UInt8}(pointer(data)),
        UInt(length(data)),
        _tag(T),
        _collection_kind(C),
    )
end
for C in (:FaerVector, :FaerRow, :FaerDiagonal)
    @eval begin
        $C{T}(v::AbstractVector) where {T<:_NativeScalar} = _collection_load($C, T, v)
        $C{T}(::AbstractVector) where {T} = _unsupported_scalar(T)
        $C(v::AbstractVector{T}) where {T<:Number} = $C{_native_input_type(T)}(v)
        function $C{T}(::UndefInitializer, n::Integer) where {T<:_NativeScalar}
            n >= 0 || throw(ArgumentError("collection length must be nonnegative"))
            return _collection_result(
                $C,
                T,
                Int(n),
                RustCrate.collection_zeroes,
                UInt(n),
                _tag(T),
                _collection_kind($C),
            )
        end
        $C(::UndefInitializer, n::Integer) = $C{Float64}(undef, n)
    end
end
function FaerRow(a::AbstractMatrix)
    return if size(a, 1) == 1
        FaerRow(vec(a))
    else
        throw(DimensionMismatch("row must have exactly one row"))
    end
end
FaerDiagonal(d::LinearAlgebra.Diagonal) = FaerDiagonal(d.diag)
Base.size(x::FaerVector) = (x.len,)
Base.size(x::FaerRow) = (1, x.len)
Base.size(x::FaerDiagonal) = (x.len, x.len)
Base.IndexStyle(::Type{<:FaerVector}) = IndexLinear()
Base.IndexStyle(::Type{<:FaerRow}) = IndexLinear()
Base.IndexStyle(::Type{<:FaerDiagonal}) = IndexCartesian()
function _collection_element(x::_OwnedCollection, i::Int)
    T = eltype(x)
    value = Ref{T}()
    GC.@preserve value _native_call(
        RustCrate.collection_element,
        x,
        UInt(i - 1),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
    )
    return value[]
end
function _collection_set!(x::_OwnedCollection, value, i::Int)
    T = eltype(x)
    v = Ref{T}(value)
    GC.@preserve v _native_call(
        RustCrate.collection_set, x, UInt(i - 1), Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v))
    )
    return x
end
function Base.getindex(x::Union{FaerVector,FaerRow}, i::Int)
    @boundscheck checkbounds(x, i)
    return _collection_element(x, i)
end
function Base.setindex!(x::Union{FaerVector,FaerRow}, value, i::Int)
    @boundscheck checkbounds(x, i)
    return _collection_set!(x, value, i)
end
function Base.getindex(x::FaerDiagonal{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(x, i, j)
    i == j && return _collection_element(x, i)
    return lock(_matrix_lock) do ;
        _handle(x)
        return zero(T)
    end
end
function Base.setindex!(x::FaerDiagonal{T}, value, i::Int, j::Int) where {T}
    @boundscheck checkbounds(x, i, j)
    if i != j
        lock(_matrix_lock) do ;
            return _handle(x)
        end
        iszero(value) || throw(ArgumentError("cannot store nonzero off-diagonal values"))
        return x
    end
    return _collection_set!(x, value, i)
end
function Base.Vector(x::_OwnedCollection)
    T = eltype(x)
    out = Vector{T}(undef, x.len)
    GC.@preserve out _native_call(
        RustCrate.collection_buffer, x, Ptr{UInt8}(pointer(out)), UInt(length(out))
    )
    return out
end
Base.Vector{T}(x::_OwnedCollection) where {T} = convert(Vector{T}, Vector(x))
function FaerMatrix(x::_OwnedCollection)
    dims = x isa FaerVector ? (x.len, 1) : size(x)
    T = eltype(x)
    return if T === Float64
        _matrix_result(RustCrate.matrix_from_collection, dims, x)
    else
        _typed_result(T, RustCrate.matrix_from_collection, dims, x)
    end
end
Base.Matrix(x::_OwnedCollection) = Matrix(FaerMatrix(x))
Base.Matrix{T}(x::_OwnedCollection) where {T} = convert(Matrix{T}, Matrix(x))
Base.Array(x::FaerVector) = Vector(x)
Base.Array(x::Union{FaerRow,FaerDiagonal}) = Matrix(x)
function Base.copy(x::_OwnedCollection)
    return _collection_result(
        _collection_type(x),
        eltype(x),
        x.len,
        RustCrate.collection_copy,
        x,
        Int32(-1),
        false,
    )
end
function Base.deepcopy_internal(x::_OwnedCollection, dict::IdDict)
    get!(dict, x) do ;
        return copy(x)
    end
end
FaerVector(x::FaerVector) = copy(x)
FaerRow(x::FaerRow) = copy(x)
FaerDiagonal(x::FaerDiagonal) = copy(x)
function Base.close(x::_OwnedCollection)
    lock(_matrix_lock) do ;
        x.handle = nothing
        return x.epoch += 1
    end
    return nothing
end
Base.isopen(x::_OwnedCollection) = lock(() -> !isnothing(x.handle), _matrix_lock)
function Base.fill!(x::Union{FaerVector,FaerRow}, value)
    T = eltype(x)
    v = Ref{T}(value)
    GC.@preserve v _native_call(
        RustCrate.collection_fill, x, Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v))
    )
    return x
end
function Base.fill!(x::FaerDiagonal, value)
    iszero(value) || throw(
        ArgumentError(
            "fill! of a diagonal matrix requires zero; use filldiagonal! for stored values",
        ),
    )
    return filldiagonal!(x, value)
end
"""Fill only the stored diagonal entries of a FaerDiagonal."""
function filldiagonal!(x::FaerDiagonal, value)
    T = eltype(x)
    v = Ref{T}(value)
    GC.@preserve v _native_call(
        RustCrate.collection_fill, x, Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v))
    )
    return x
end
function Base.similar(x::_OwnedCollection, ::Type{T}, dims::Dims) where {T}
    if T <: _NativeScalar
        length(dims) == 1 && return FaerVector{T}(undef, dims[1])
        length(dims) == 2 && return FaerMatrix{T}(undef, dims...)
    end
    return Array{T}(undef, dims)
end
Base.zero(x::FaerVector{T}) where {T} = FaerVector{T}(undef, x.len)
Base.zero(x::FaerRow{T}) where {T} = FaerRow{T}(undef, x.len)
Base.zero(x::FaerDiagonal{T}) where {T} = FaerDiagonal{T}(undef, x.len)
function _collection_coerce(x::_OwnedCollection, ::Type{T}) where {T}
    T === eltype(x) && return x
    return _collection_type(x){T}(Vector(x))
end
function _collection_binary(x::_OwnedCollection, y::_OwnedCollection, op)
    _collection_kind(x) == _collection_kind(y) && x.len == y.len ||
        throw(DimensionMismatch("collection shapes must match"))
    P = promote_type(eltype(x), eltype(y))
    return _collection_result(
        _collection_type(x),
        P,
        x.len,
        RustCrate.collection_arithmetic,
        _collection_coerce(x, P),
        _collection_coerce(y, P),
        Int32(op),
    )
end
for C in (FaerVector, FaerRow, FaerDiagonal)
    @eval begin
        Base.:+(x::$C, y::$C) = _collection_binary(x, y, 0)
        Base.:-(x::$C, y::$C) = _collection_binary(x, y, 1)
        Base.:+(x::$C) = copy(x)
        Base.:-(x::$C) = -one(_realtype(eltype(x))) * x
    end
end
Base.:*(x::FaerDiagonal, y::FaerDiagonal) = _collection_binary(x, y, 2)
function Base.:*(value::Number, x::_OwnedCollection)
    P = _native_promotion(typeof(value), eltype(x))
    # Real factors scale complex entries componentwise, as in Julia.
    R = value isa Real ? _realtype(P) : P
    v = Ref{R}(value)
    GC.@preserve v return _collection_result(
        _collection_type(x),
        P,
        x.len,
        R === P ? RustCrate.collection_scale : RustCrate.collection_scale_real,
        _collection_coerce(x, P),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, v)),
    )
end
Base.:*(x::_OwnedCollection, value::Number) = value * x
Base.:/(x::_OwnedCollection, value::Number) = (one(eltype(x)) / value) * x
Base.:/(x::_OwnedCollection, value::Real) = (one(_realtype(eltype(x))) / value) * x
function _collection_transform(x::_OwnedCollection, C, conjugate=false)
    return _collection_result(
        C, eltype(x), x.len, RustCrate.collection_copy, x, _collection_kind(C), conjugate
    )
end
Base.conj(x::_OwnedCollection) = _collection_transform(x, _collection_type(x), true)
function Base.sum(x::_OwnedCollection; dims=:, kw...)
    dims isa Colon && isempty(kw) || return sum(Array(x); dims, kw...)
    T = eltype(x)
    v = Ref{T}()
    GC.@preserve v _native_call(
        RustCrate.collection_sum, x, Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v))
    )
    return v[]
end
function LinearAlgebra.norm(x::_OwnedCollection, p::Real=2)
    p in (1, 2, Inf) ||
        return x isa FaerDiagonal ? LinearAlgebra.norm(Vector(x), p) : _native_norm(x, p)
    T = _realtype(eltype(x))
    v = Ref{T}()
    code = if p == 2
        0
    elseif p == 1
        1
    else
        2
    end
    GC.@preserve v _native_call(
        RustCrate.collection_norm,
        x,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v)),
        Int32(code),
    )
    return v[]
end
function _collection_dot(x::_OwnedCollection, y::_OwnedCollection, conjugated)
    x.len == y.len || throw(DimensionMismatch("collection lengths must match"))
    P = promote_type(eltype(x), eltype(y))
    v = Ref{P}()
    GC.@preserve v _native_call(
        RustCrate.collection_dot,
        _collection_coerce(x, P),
        _collection_coerce(y, P),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, v)),
        conjugated,
    )
    return v[]
end
LinearAlgebra.dot(x::FaerVector, y::FaerVector) = _collection_dot(x, y, true)
LinearAlgebra.dot(x::FaerRow, y::FaerRow) = _collection_dot(x, y, true)
LinearAlgebra.dot(x::FaerDiagonal, y::FaerDiagonal) = _collection_dot(x, y, true)
function Base.:*(x::LinearAlgebra.Adjoint{<:Number,<:FaerVector}, y::FaerVector{<:Number})
    return _collection_dot(parent(x), y, true)
end
function Base.:*(x::LinearAlgebra.Transpose{<:Number,<:FaerVector}, y::FaerVector{<:Number})
    return _collection_dot(parent(x), y, false)
end
function Base.copy(x::LinearAlgebra.Adjoint{<:Any,<:FaerVector})
    return _collection_transform(parent(x), FaerRow, true)
end
function Base.copy(x::LinearAlgebra.Transpose{<:Any,<:FaerVector})
    return _collection_transform(parent(x), FaerRow, false)
end
function Base.copy(x::LinearAlgebra.Adjoint{<:Any,<:FaerRow})
    return FaerMatrix(_collection_transform(parent(x), FaerVector, true))
end
function Base.copy(x::LinearAlgebra.Transpose{<:Any,<:FaerRow})
    return FaerMatrix(_collection_transform(parent(x), FaerVector, false))
end
LinearAlgebra.diag(x::FaerDiagonal) = _collection_transform(x, FaerVector, false)
function reserve!(x::Union{FaerVector,FaerRow}, n::Integer)
    n >= 0 || throw(ArgumentError("capacity must be nonnegative"))
    _native_call(RustCrate.collection_reserve, x, UInt(n))
    return x
end
Base.sizehint!(x::Union{FaerVector,FaerRow}, n::Integer) = reserve!(x, n)
function Base.resize!(x::Union{FaerVector,FaerRow}, n::Integer; fill=zero(eltype(x)))
    n >= 0 || throw(ArgumentError("length must be nonnegative"))
    T = eltype(x)
    v = Ref{T}(fill)
    lock(_matrix_lock) do
        GC.@preserve v _native_call(
            RustCrate.collection_resize,
            x,
            UInt(n),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v)),
        )
        n != x.len && (x.epoch += 1)
        return x.len = Int(n)
    end
    return x
end
function truncate!(x::Union{FaerVector,FaerRow}, n::Integer)
    0 <= n <= x.len || throw(DimensionMismatch("truncate! must not grow the collection"))
    return resize!(x, n)
end
function Base.:*(
    x::LinearAlgebra.Transpose{T,<:FaerVector}, y::FaerVector{T}
) where {T<:Real}
    return _collection_dot(parent(x), y, false)
end
