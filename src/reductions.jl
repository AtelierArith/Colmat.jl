"""Compute a Frobenius/entrywise norm using t4a-faer reductions."""
function LinearAlgebra.norm(a::FaerMatrix, p::Real = 2)
    p == 2 && return _native_call(RustCrate.reduction, a, Int32(1))
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(2))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(3))
    return LinearAlgebra.norm(Matrix(a), p)
end
function Base.sum(a::FaerMatrix; dims = :)
    dims isa Colon && return _native_call(RustCrate.reduction, a, Int32(0))
    dims == 1 &&
        return _matrix_result(RustCrate.sum_axis_from, (1, size(a, 2)), a, Int32(1))
    dims == 2 &&
        return _matrix_result(RustCrate.sum_axis_from, (size(a, 1), 1), a, Int32(2))
    return sum(Matrix(a); dims)
end
function LinearAlgebra.dot(a::FaerMatrix, b::FaerMatrix)
    size(a)==size(b) || throw(DimensionMismatch("inner product dimensions must match"))
    return _native_call(RustCrate.inner_product, a, b)
end
function LinearAlgebra.det(a::FaerMatrix)
    LinearAlgebra.checksquare(a)
    return _native_call(RustCrate.reduction, a, Int32(5))
end
function LinearAlgebra.tr(a::FaerMatrix)
    LinearAlgebra.checksquare(a)
    return _native_call(RustCrate.reduction, a, Int32(8))
end
function Base.minimum(a::FaerMatrix)
    isempty(a) && throw(ArgumentError("empty matrix has no minimum"))
    return _native_call(RustCrate.reduction, a, Int32(6))
end
function Base.maximum(a::FaerMatrix)
    isempty(a) && throw(ArgumentError("empty matrix has no maximum"))
    return _native_call(RustCrate.reduction, a, Int32(7))
end
Base.extrema(a::FaerMatrix) = (minimum(a), maximum(a))
Base.all(::typeof(isfinite), a::FaerMatrix) = _native_call(RustCrate.predicate, a, Int32(0))
Base.any(::typeof(isnan), a::FaerMatrix) = _native_call(RustCrate.predicate, a, Int32(1))
Base.isone(a::FaerMatrix) = _native_call(RustCrate.predicate, a, Int32(2))
LinearAlgebra.issymmetric(a::FaerMatrix) = _native_call(RustCrate.predicate, a, Int32(3))
LinearAlgebra.ishermitian(a::FaerMatrix) = LinearAlgebra.issymmetric(a)
function LinearAlgebra.opnorm(a::FaerMatrix, p::Real = 2)
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(9))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(10))
    p == 2 && return isempty(a) ? 0.0 : maximum(LinearAlgebra.svdvals(a))
    throw(ArgumentError("operator norm supports p=1, 2 or Inf"))
end
function LinearAlgebra.kron(a::FaerMatrix, b::FaerMatrix)
    dims=(
        Base.checked_mul(size(a, 1), size(b, 1)),
        Base.checked_mul(size(a, 2), size(b, 2)),
    )
    return _matrix_result(RustCrate.kron_from, dims, a, b)
end
