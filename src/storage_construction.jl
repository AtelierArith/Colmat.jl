"""
    with_capacity(FaerMatrix{T}, rows, cols)
    with_capacity(FaerVector{T}, length)
    with_capacity(FaerRow{T}, length)

Construct an empty Rust-owned array with at least the requested capacity.
Omitting `{T}` selects Float64. Logical size remains `(0,0)`, `0`, or `(1,0)`;
use `resize!` to initialize elements. Capacity includes native alignment padding.
The reservation uses faer's fallible allocator and throws on failure.
"""
function with_capacity(
    ::Type{FaerMatrix{T}}, rows::Integer, cols::Integer
) where {T<:_NativeScalar}
    m, n = Int(rows), Int(cols)
    m >= 0 && n >= 0 || throw(ArgumentError("matrix capacities must be nonnegative"))
    Base.checked_mul(m, n)
    a = FaerMatrix{T}(undef, 0, 0)
    try
        return reserve!(a, m, n)
    catch
        close(a)
        rethrow()
    end
end
function with_capacity(::Type{FaerMatrix}, rows::Integer, cols::Integer)
    return with_capacity(FaerMatrix{Float64}, rows, cols)
end

for C in (:FaerVector, :FaerRow)
    @eval begin
        function with_capacity(::Type{$C{T}}, length::Integer) where {T<:_NativeScalar}
            n = Int(length)
            n >= 0 || throw(ArgumentError("capacity must be nonnegative"))
            a = $C{T}(undef, 0)
            try
                return reserve!(a, n)
            catch
                close(a)
                rethrow()
            end
        end
        with_capacity(::Type{$C}, length::Integer) = with_capacity($C{Float64}, length)
    end
end
