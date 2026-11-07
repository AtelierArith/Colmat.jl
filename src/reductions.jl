"""Compute a Frobenius/entrywise norm using t4a-faer reductions."""
function LinearAlgebra.norm(a::FaerMatrix{Float64}, p::Real = 2)
    p == 2 && return _native_call(RustCrate.reduction, a, Int32(1))
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(2))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(3))
    return _native_norm(a, p)
end
function Base.sum(a::FaerMatrix{Float64}; dims = :, kw...)
    isempty(kw) || return sum(Matrix(a); dims, kw...)
    dims isa Colon && return _native_call(RustCrate.reduction, a, Int32(0))
    dims == 1 &&
        return _matrix_result(RustCrate.sum_axis_from, (1, size(a, 2)), a, Int32(1))
    dims == 2 &&
        return _matrix_result(RustCrate.sum_axis_from, (size(a, 1), 1), a, Int32(2))
    return _native_sum(a, dims)
end
function LinearAlgebra.dot(a::FaerMatrix{Float64}, b::FaerMatrix{Float64})
    size(a)==size(b) || return _native_dot(a, b)
    return _native_call(RustCrate.inner_product, a, b)
end
function LinearAlgebra.tr(a::FaerMatrix{Float64})
    LinearAlgebra.checksquare(a)
    return _native_call(RustCrate.reduction, a, Int32(8))
end
Base.all(::typeof(isfinite), a::FaerMatrix{Float64}) =
    _native_call(RustCrate.predicate, a, Int32(0))
Base.any(::typeof(isnan), a::FaerMatrix{Float64}) =
    _native_call(RustCrate.predicate, a, Int32(1))
# Julia's `==` semantics (NaN breaks symmetry), see `_native_structure`.
Base.isone(a::FaerMatrix{Float64}) = _native_structure(a, 0)
LinearAlgebra.issymmetric(a::FaerMatrix{Float64}) = _native_structure(a, 1)
LinearAlgebra.ishermitian(a::FaerMatrix{Float64}) = _native_structure(a, 1)
function LinearAlgebra.opnorm(a::FaerMatrix{Float64}, p::Real = 2)
    p == 1 && return _native_call(RustCrate.reduction, a, Int32(9))
    p == Inf && return _native_call(RustCrate.reduction, a, Int32(10))
    if p==2
        values=LinearAlgebra.svdvals(a)
        return isempty(values) ? 0.0 : maximum(values)
    end
    throw(ArgumentError("operator norm supports p=1, 2 or Inf"))
end
