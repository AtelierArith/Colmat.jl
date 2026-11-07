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
Base.inv(a::_FaerTriangular) = _matrix_result(
    RustCrate.triangular_inverse_from,
    size(a),
    parent(a),
    _islower(a),
    _isunit(a),
)

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
