_initializer_components(::Type{<:Real}) = 1
_initializer_components(::Type{<:Complex}) = 2
function _initializer_put!(values, index, value::Real)
    values[index]=value
    return index+1
end
function _initializer_put!(values, index, value::Complex)
    values[index]=real(value)
    values[index+1]=imag(value)
    return index+2
end
_matrix_initializer(::Type{Float32}) = RustCrate.resize_matrix_values_f32
_matrix_initializer(::Type{Float64}) = RustCrate.resize_matrix_values_f64
_collection_initializer(::Type{Float32}) = RustCrate.resize_collection_values_f32
_collection_initializer(::Type{Float64}) = RustCrate.resize_collection_values_f64

"""
    resize_with!(f, A::FaerMatrix, rows, cols)
    resize_with!(f, x::Union{FaerVector,FaerRow}, length)

Resize native storage, preserving overlapping elements. Call `f(i,j)` or `f(i)`
only for new elements, with one-based indices in column-major order. Supports
do-block syntax. Shrinking and unchanged dimensions do not invoke `f`.

Evaluate and convert all new values before resizing; an initializer exception
leaves the owner unchanged. The initializer must not mutate or close that owner:
its own side effects cannot be rolled back. Shape/lifetime changes during the
initializer are rejected. New values use a temporary Julia primitive buffer;
the native update reserves capacity before truncating either dimension.
Successful shape changes invalidate existing views as with `resize!`.
"""
function resize_with!(f, a::FaerMatrix{T}, rows::Integer, cols::Integer) where {T}
    m, n=Int(rows), Int(cols)
    m>=0&&n>=0 || throw(ArgumentError("matrix dimensions must be nonnegative"))
    total=Base.checked_mul(m, n)
    lock(_matrix_lock) do
        handle=_handle(a)
        old=a.dims
        epoch=a.epoch
        overlap=Base.checked_mul(min(m, old[1]), min(n, old[2]))
        values=Vector{real(T)}(
            undef,
            Base.checked_mul(total-overlap, _initializer_components(T)),
        )
        index=1
        if m>old[1]
            for j = 1:min(n, old[2]), i = (old[1]+1):m
                index=_initializer_put!(values, index, T(f(i, j)))
            end
        end
        if n>old[2]
            for j = (old[2]+1):n, i = 1:m
                index=_initializer_put!(values, index, T(f(i, j)))
            end
        end
        a.handle===handle&&a.epoch==epoch&&a.dims==old ||
            throw(ArgumentError("initializer changed the matrix owner or shape"))
        _native_call(
            _matrix_initializer(real(T)),
            a,
            UInt(m),
            UInt(n),
            UInt(old[1]),
            UInt(old[2]),
            _tag(T),
            values,
        )
        (m, n)!=old && (a.epoch+=1)
        a.dims=(m, n)
        return a
    end
end
function resize_with!(f, a::Union{FaerVector{T},FaerRow{T}}, length::Integer) where {T}
    n=Int(length)
    n>=0 || throw(ArgumentError("length must be nonnegative"))
    lock(_matrix_lock) do
        handle=_handle(a)
        old=a.len
        epoch=a.epoch
        values=Vector{real(T)}(
            undef,
            Base.checked_mul(max(n-old, 0), _initializer_components(T)),
        )
        index=1
        if n>old
            for i = (old+1):n
                index=_initializer_put!(values, index, T(f(i)))
            end
        end
        a.handle===handle&&a.epoch==epoch&&a.len==old ||
            throw(ArgumentError("initializer changed the collection owner or length"))
        _native_call(
            _collection_initializer(real(T)),
            a,
            UInt(n),
            UInt(old),
            _collection_kind(a),
            _tag(T),
            values,
        )
        n!=old && (a.epoch+=1)
        a.len=n
        return a
    end
end
