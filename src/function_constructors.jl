function _from_initializer(f, C, dims...)
    a = with_capacity(C, dims...)
    try
        return resize_with!(f, a, dims...)
    catch
        close(a)
        rethrow()
    end
end

"""
    FaerMatrix{T}(f, rows, cols)
    FaerMatrix(f, rows, cols)

Construct owned native storage from `f(i,j)`, evaluated once per element with
one-based indices in column-major order. The default scalar type is Float64;
specify a native `T` for other real/complex results. Empty dimensions do not
invoke `f`. Evaluation and conversion precede the native bulk initialization.
"""
function FaerMatrix{T}(f, m::Integer, n::Integer) where {T<:_NativeScalar}
    return _from_initializer(f, FaerMatrix{T}, m, n)
end
FaerMatrix(f, m::Integer, n::Integer) = FaerMatrix{Float64}(f, m, n)

for C in (:FaerVector, :FaerRow)
    @eval begin
        """
            FaerVector{T}(f, length) / FaerRow{T}(f, length)
            FaerVector(f, length) / FaerRow(f, length)

        Construct owned native storage from `f(i)` with one-based indices,
        evaluated exactly once in increasing order. The default scalar type
        is Float64. An empty length does not invoke `f`.
        """
        $C{T}(f, n::Integer) where {T<:_NativeScalar} = _from_initializer(f, $C{T}, n)
        $C(f, n::Integer) = $C{Float64}(f, n)
    end
end
