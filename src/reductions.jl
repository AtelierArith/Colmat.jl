"""Compute a Frobenius/entrywise norm using t4a-faer reductions."""
function LinearAlgebra.norm(a::FaerMatrix{Float64}, p::Real = 2)
    p == 2 && return _native_call(RustCrate.reduction, a, Int32(1))
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(2))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(3))
    return LinearAlgebra.norm(Matrix(a), p)
end
function Base.sum(a::FaerMatrix{Float64}; dims = :)
    dims isa Colon && return _native_call(RustCrate.reduction, a, Int32(0))
    dims == 1 &&
        return _matrix_result(RustCrate.sum_axis_from, (1, size(a, 2)), a, Int32(1))
    dims == 2 &&
        return _matrix_result(RustCrate.sum_axis_from, (size(a, 1), 1), a, Int32(2))
    return sum(Matrix(a); dims)
end
function LinearAlgebra.dot(a::FaerMatrix{Float64}, b::FaerMatrix{Float64})
    size(a)==size(b) || throw(DimensionMismatch("inner product dimensions must match"))
    return _native_call(RustCrate.inner_product, a, b)
end
function LinearAlgebra.det(a::FaerMatrix{Float64})
    LinearAlgebra.checksquare(a)
    return _native_call(RustCrate.reduction, a, Int32(5))
end
function LinearAlgebra.tr(a::FaerMatrix{Float64})
    LinearAlgebra.checksquare(a)
    return _native_call(RustCrate.reduction, a, Int32(8))
end
function _real_extremum(
    a::FaerMatrix{T},
    minimum::Bool,
) where {T <: Union{Float32, Float64}}
    lock(_matrix_lock) do
        _dense_rhs_validate(a)
        isempty(a) && throw(ArgumentError("empty matrix has no extrema"))
        T===Float32 && return _native_call(RustCrate.real_extremum_f32, a, minimum)
        return _native_call(RustCrate.reduction, a, Int32(minimum ? 6 : 7))
    end
end
Base.minimum(a::FaerMatrix{T}) where {T <: Union{Float32, Float64}} =
    _real_extremum(a, true)
Base.maximum(a::FaerMatrix{T}) where {T <: Union{Float32, Float64}} =
    _real_extremum(a, false)
Base.extrema(a::FaerMatrix{T}) where {T <: Union{Float32, Float64}} =
    (minimum(a), maximum(a))
Base.all(::typeof(isfinite), a::FaerMatrix{Float64}) =
    _native_call(RustCrate.predicate, a, Int32(0))
Base.any(::typeof(isnan), a::FaerMatrix{Float64}) =
    _native_call(RustCrate.predicate, a, Int32(1))
Base.isone(a::FaerMatrix{Float64}) = _native_call(RustCrate.predicate, a, Int32(2))
LinearAlgebra.issymmetric(a::FaerMatrix{Float64}) =
    _native_call(RustCrate.predicate, a, Int32(3))
LinearAlgebra.ishermitian(a::FaerMatrix{Float64}) = LinearAlgebra.issymmetric(a)
function LinearAlgebra.opnorm(a::FaerMatrix{Float64}, p::Real = 2)
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(9))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(10))
    if p==2
        values=LinearAlgebra.svdvals(a)
        return isempty(values) ? 0.0 : maximum(values)
    end
    throw(ArgumentError("operator norm supports p=1, 2 or Inf"))
end
function LinearAlgebra.kron(a::FaerMatrix{Float64}, b::FaerMatrix{Float64})
    dims=(
        Base.checked_mul(size(a, 1), size(b, 1)),
        Base.checked_mul(size(a, 2), size(b, 2)),
    )
    return _matrix_result(RustCrate.kron_from, dims, a, b)
end
