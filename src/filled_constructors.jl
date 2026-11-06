_constant_components(x::Real) = [x]
_constant_components(x::Complex) = [real(x), imag(x)]
_filled_matrix_function(::Type{Float32}) = RustCrate.filled_matrix_f32
_filled_matrix_function(::Type{Float64}) = RustCrate.filled_matrix_f64
_filled_collection_function(::Type{Float32}) = RustCrate.filled_collection_f32
_filled_collection_function(::Type{Float64}) = RustCrate.filled_collection_f64

"""
    FaerMatrix{T}(rows, cols; fill=zero(T))
    FaerMatrix(rows, cols; fill=0.0)
    zeros(FaerMatrix{T}, rows, cols)
    ones(FaerMatrix{T}, rows, cols)

Construct native matrix storage directly with a constant value. No dense Julia
array is materialized. The default scalar type is Float64. Tuple dimensions are
also accepted. Values are converted to T before constructing the native owner.
"""
function FaerMatrix{T}(
    rows::Integer,
    cols::Integer;
    fill = zero(T),
) where {T<:_NativeScalar}
    m, n=Int(rows), Int(cols)
    m>=0&&n>=0 || throw(ArgumentError("matrix dimensions must be nonnegative"))
    Base.checked_mul(m, n)
    values=_constant_components(T(fill))
    f=_filled_matrix_function(real(T))
    if T===Float64
        return _matrix_result(f, (m, n), UInt(m), UInt(n), _tag(T), values)
    end
    return _typed_result(T, f, (m, n), UInt(m), UInt(n), _tag(T), values)
end
FaerMatrix(m::Integer, n::Integer; kwargs...) = FaerMatrix{Float64}(m, n; kwargs...)
FaerMatrix{T}(dims::Tuple{Integer,Integer}; kwargs...) where {T<:_NativeScalar} =
    FaerMatrix{T}(dims...; kwargs...)
FaerMatrix(dims::Tuple{Integer,Integer}; kwargs...) =
    FaerMatrix{Float64}(dims...; kwargs...)

for C in (:FaerVector, :FaerRow, :FaerDiagonal)
    @eval begin
        """
            FaerVector{T}(n; fill=zero(T))
            FaerRow{T}(n; fill=zero(T))
            FaerDiagonal{T}(n; fill=zero(T))

        Construct n stored entries with a constant value directly in native
        storage. The unparameterized form defaults to Float64. Diagonal
        off-diagonal entries remain implicit zeros.
        """
        function $C{T}(length::Integer; fill = zero(T)) where {T<:_NativeScalar}
            n=Int(length)
            n>=0 || throw(ArgumentError("length must be nonnegative"))
            values=_constant_components(T(fill))
            return _collection_result(
                $C,
                T,
                n,
                _filled_collection_function(real(T)),
                UInt(n),
                _collection_kind($C),
                _tag(T),
                values,
            )
        end
        $C(n::Integer; kwargs...) = $C{Float64}(n; kwargs...)
        Base.zeros(::Type{$C{T}}, n::Integer) where {T<:_NativeScalar} = $C{T}(n)
        Base.ones(::Type{$C{T}}, n::Integer) where {T<:_NativeScalar} =
            $C{T}(n; fill = one(T))
        Base.zeros(::Type{$C}, n::Integer) = $C{Float64}(n)
        Base.ones(::Type{$C}, n::Integer) = $C{Float64}(n; fill = 1.0)
        Base.zeros(::Type{$C{T}}, dims::Tuple{Integer}) where {T<:_NativeScalar} =
            zeros($C{T}, dims...)
        Base.ones(::Type{$C{T}}, dims::Tuple{Integer}) where {T<:_NativeScalar} =
            ones($C{T}, dims...)
        Base.zeros(::Type{$C}, dims::Tuple{Integer}) = zeros($C, dims...)
        Base.ones(::Type{$C}, dims::Tuple{Integer}) = ones($C, dims...)

    end
end
Base.zeros(::Type{FaerMatrix{T}}, m::Integer, n::Integer) where {T<:_NativeScalar} =
    FaerMatrix{T}(m, n)
Base.ones(::Type{FaerMatrix{T}}, m::Integer, n::Integer) where {T<:_NativeScalar} =
    FaerMatrix{T}(m, n; fill = one(T))
Base.zeros(::Type{FaerMatrix}, m::Integer, n::Integer) = FaerMatrix{Float64}(m, n)
Base.ones(::Type{FaerMatrix}, m::Integer, n::Integer) =
    FaerMatrix{Float64}(m, n; fill = 1.0)
Base.zeros(::Type{FaerMatrix{T}}, dims::Tuple{Integer,Integer}) where {T<:_NativeScalar} =
    zeros(FaerMatrix{T}, dims...)
Base.ones(::Type{FaerMatrix{T}}, dims::Tuple{Integer,Integer}) where {T<:_NativeScalar} =
    ones(FaerMatrix{T}, dims...)
Base.zeros(::Type{FaerMatrix}, dims::Tuple{Integer,Integer}) = zeros(FaerMatrix, dims...)
Base.ones(::Type{FaerMatrix}, dims::Tuple{Integer,Integer}) = ones(FaerMatrix, dims...)
