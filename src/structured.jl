const _FaerTriangular = Union{
    LinearAlgebra.LowerTriangular{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.UpperTriangular{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.UnitLowerTriangular{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.UnitUpperTriangular{Float64,<:FaerMatrix{Float64}},
}
_islower(::Union{LinearAlgebra.LowerTriangular,LinearAlgebra.UnitLowerTriangular}) = true
_islower(::Union{LinearAlgebra.UpperTriangular,LinearAlgebra.UnitUpperTriangular}) = false
_isunit(::Union{LinearAlgebra.UnitLowerTriangular,LinearAlgebra.UnitUpperTriangular}) = true
_isunit(::Union{LinearAlgebra.LowerTriangular,LinearAlgebra.UpperTriangular}) = false
function Base.:\(a::_FaerTriangular, b::FaerMatrix{Float64})
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
function LinearAlgebra.ldiv!(a::_FaerTriangular, b::FaerMatrix{Float64})
    return copyto!(b, a\b)
end
function LinearAlgebra.ldiv!(
    dest::FaerMatrix{Float64},
    a::_FaerTriangular,
    b::FaerMatrix{Float64},
)
    size(dest)==size(b) || throw(DimensionMismatch("triangular output size must match RHS"))
    return copyto!(dest, a\b)
end

const _FaerTransposedMatrix=Union{
    LinearAlgebra.Transpose{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.Adjoint{Float64,<:FaerMatrix{Float64}},
}
Base.:\(a::_FaerTransposedMatrix, b::FaerMatrix{Float64}) = copy(a)\b
Base.:\(a::_FaerTransposedMatrix, b::AbstractMatrix{<:Real}) = copy(a)\b
Base.:\(a::_FaerTransposedMatrix, b::AbstractVector{<:Real}) = copy(a)\b
Base.:/(b::FaerMatrix{Float64}, a::FaerMatrix{Float64}) =
    copy(adjoint(copy(adjoint(a))\copy(adjoint(b))))
Base.:/(b::FaerMatrix{Float64}, f::FaerFactorization{Float64}) =
    copy(adjoint(f'\copy(adjoint(b))))

Base.:/(b::_FaerTransposedMatrix, a::FaerMatrix{Float64}) = copy(b)/a
Base.:/(b::FaerMatrix{Float64}, a::_FaerTransposedMatrix) = b/copy(a)
Base.:/(b::_FaerTransposedMatrix, a::_FaerTransposedMatrix) = copy(b)/copy(a)
Base.:*(a::_FaerTransposedMatrix, b::FaerMatrix{Float64}) = copy(a)*b
Base.:*(a::FaerMatrix{Float64}, b::_FaerTransposedMatrix) = a*copy(b)
Base.:*(a::_FaerTransposedMatrix, b::_FaerTransposedMatrix) = copy(a)*copy(b)
