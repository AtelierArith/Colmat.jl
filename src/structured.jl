const _FaerTriangular = Union{
    LinearAlgebra.LowerTriangular{Float64,<:FaerMatrix},
    LinearAlgebra.UpperTriangular{Float64,<:FaerMatrix},
    LinearAlgebra.UnitLowerTriangular{Float64,<:FaerMatrix},
    LinearAlgebra.UnitUpperTriangular{Float64,<:FaerMatrix},
}
_islower(::Union{LinearAlgebra.LowerTriangular,LinearAlgebra.UnitLowerTriangular}) = true
_islower(::Union{LinearAlgebra.UpperTriangular,LinearAlgebra.UnitUpperTriangular}) = false
_isunit(::Union{LinearAlgebra.UnitLowerTriangular,LinearAlgebra.UnitUpperTriangular}) = true
_isunit(::Union{LinearAlgebra.LowerTriangular,LinearAlgebra.UpperTriangular}) = false
function Base.:\(a::_FaerTriangular, b::FaerMatrix)
    size(a, 1)==size(b, 1) ||
        throw(DimensionMismatch("triangular solve dimensions must match"))
    return _matrix_result(
        RustCrate.triangular_solve_from,
        size(b),
        parent(a),
        b,
        _islower(a),
        _isunit(a),
    )
end
Base.:\(a::_FaerTriangular, b::StridedMatrix{<:Real}) = a\FaerMatrix(b)
Base.:\(a::_FaerTriangular, b::AbstractVector{<:Real}) =
    vec(Matrix(a\FaerMatrix(reshape(b, :, 1))))
Base.inv(a::_FaerTriangular) = _matrix_result(
    RustCrate.triangular_inverse_from,
    size(a),
    parent(a),
    _islower(a),
    _isunit(a),
)
function LinearAlgebra.ldiv!(a::_FaerTriangular, b::FaerMatrix)
    return copyto!(b, a\b)
end
function LinearAlgebra.ldiv!(dest::FaerMatrix, a::_FaerTriangular, b::FaerMatrix)
    size(dest)==size(b) || throw(DimensionMismatch("triangular output size must match RHS"))
    return copyto!(dest, a\b)
end

const _FaerTransposedMatrix=Union{
    LinearAlgebra.Transpose{Float64,<:FaerMatrix},
    LinearAlgebra.Adjoint{Float64,<:FaerMatrix},
}
Base.:\(a::_FaerTransposedMatrix, b::FaerMatrix) = copy(a)\b
Base.:\(a::_FaerTransposedMatrix, b::AbstractMatrix{<:Real}) = copy(a)\b
Base.:\(a::_FaerTransposedMatrix, b::AbstractVector{<:Real}) = copy(a)\b
Base.:/(b::FaerMatrix, a::FaerMatrix) = copy(adjoint(copy(adjoint(a))\copy(adjoint(b))))
Base.:/(b::FaerMatrix, f::FaerFactorization) = copy(adjoint(f'\copy(adjoint(b))))
function LinearAlgebra.rdiv!(b::FaerMatrix, f::FaerFactorization)
    return copyto!(b, b/f)
end
function LinearAlgebra.ldiv!(f::FaerFactorization, b::AbstractVector)
    result=f\b
    length(result)==length(b) ||
        throw(DimensionMismatch("in-place RHS cannot change shape"))
    return copyto!(b, result)
end
function LinearAlgebra.ldiv!(f::FaerFactorization, b::AbstractMatrix)
    result=f\b
    size(result)==size(b) || throw(DimensionMismatch("in-place RHS cannot change shape"))
    return copyto!(b, Matrix(result))
end
# Resolve the intersection with the native two-argument ldiv! method.
LinearAlgebra.ldiv!(f::FaerFactorization, b::FaerMatrix) = LinearAlgebra.ldiv!(b, f, b)

Base.:/(b::_FaerTransposedMatrix, a::FaerMatrix) = copy(b)/a
Base.:/(b::FaerMatrix, a::_FaerTransposedMatrix) = b/copy(a)
Base.:/(b::_FaerTransposedMatrix, a::_FaerTransposedMatrix) = copy(b)/copy(a)
Base.:*(a::_FaerTransposedMatrix, b::FaerMatrix) = copy(a)*b
Base.:*(a::FaerMatrix, b::_FaerTransposedMatrix) = a*copy(b)
Base.:*(a::_FaerTransposedMatrix, b::_FaerTransposedMatrix) = copy(a)*copy(b)
