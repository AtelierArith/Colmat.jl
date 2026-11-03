"""
    FaerMatrix{T}(undef, m, n)
    FaerMatrix(undef, m, n)

Allocate Rust-owned matrix storage. The default scalar type is Float64. Storage
is initialized to zero; callers must not depend on the contents of `undef`.
"""
FaerMatrix{T}(::UndefInitializer, m::Integer, n::Integer) where {T<:_NativeScalar} =
    _zeros(T, (Int(m), Int(n)))
FaerMatrix(::UndefInitializer, m::Integer, n::Integer) = FaerMatrix{Float64}(undef, m, n)
FaerMatrix{T}(::UndefInitializer, dims::Tuple{Integer,Integer}) where {T<:_NativeScalar} =
    FaerMatrix{T}(undef, dims...)
function FaerMatrix{T}(
    u::LinearAlgebra.UniformScaling,
    m::Integer,
    n::Integer,
) where {T<:_NativeScalar}
    dims=(Int(m), Int(n))
    all(>=(0), dims) || throw(ArgumentError("matrix dimensions must be nonnegative"))
    a=if T===Float64
        _matrix_result(RustCrate.identity_storage, dims, UInt(m), UInt(n), _tag(T))
    else
        _typed_result(T, RustCrate.identity_storage, dims, UInt(m), UInt(n), _tag(T))
    end
    return isone(u.λ) ? a : T(u.λ)*a
end
FaerMatrix(u::LinearAlgebra.UniformScaling, m::Integer, n::Integer) =
    FaerMatrix{Float64}(u, m, n)

"""
    reserve!(A::FaerMatrix, rows, cols)

Reserve native capacity without changing dimensions or values. The matrix
continues to own its buffer; a failed allocation leaves its contents intact.
"""
function reserve!(a::FaerMatrix, m::Integer, n::Integer)
    m>=0&&n>=0 || throw(ArgumentError("matrix capacities must be nonnegative"))
    _native_call(RustCrate.reserve_storage, a, UInt(m), UInt(n))
    return a
end
Base.sizehint!(a::FaerMatrix, dims::Tuple{Integer,Integer}) = reserve!(a, dims...)

"""
    resize!(A::FaerMatrix, rows, cols; fill=zero(eltype(A)))

Resize the Rust-owned matrix, preserving the overlapping upper-left block.
New entries are initialized with `fill`. Reserved capacity is reused by faer.
"""
function Base.resize!(a::FaerMatrix{T}, m::Integer, n::Integer; fill = zero(T)) where {T}
    dims=(Int(m), Int(n))
    all(>=(0), dims) || throw(ArgumentError("matrix dimensions must be nonnegative"))
    value=Ref{T}(fill)
    lock(_matrix_lock) do
        GC.@preserve value _native_call(
            RustCrate.resize_storage,
            a,
            UInt(m),
            UInt(n),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
        )
        dims!=a.dims && (a.epoch+=1)
        a.dims=dims
    end
    return a
end

"""
    truncate!(A::FaerMatrix, rows, cols)

Shrink native storage to its upper-left block without growing the matrix.
"""
function truncate!(a::FaerMatrix, m::Integer, n::Integer)
    dims=(Int(m), Int(n))
    all(>=(0), dims) || throw(ArgumentError("matrix dimensions must be nonnegative"))
    lock(_matrix_lock) do
        m<=size(a, 1)&&n<=size(a, 2) ||
            throw(DimensionMismatch("truncate! cannot grow a matrix"))
        _native_call(RustCrate.truncate_storage, a, UInt(m), UInt(n))
        dims!=a.dims && (a.epoch+=1)
        a.dims=dims
    end
    return a
end
