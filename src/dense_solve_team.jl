# Retained dense factor solves share the existing scalar-promotion policy.
const _DenseTransformedFactor = Union{
    LinearAlgebra.TransposeFactorization{T,<:FaerFactorization{T}},
    LinearAlgebra.AdjointFactorization{T,<:FaerFactorization{T}},
} where {T}
const _DenseSolveFactor = Union{FaerFactorization,_DenseTransformedFactor}
_dense_parent(f::FaerFactorization) = f
_dense_parent(f::_DenseTransformedFactor) = parent(f)
_dense_op(::FaerFactorization) = 0
_dense_op(::LinearAlgebra.TransposeFactorization{T,<:FaerFactorization{T}}) where {T} = 1
_dense_op(::LinearAlgebra.AdjointFactorization{T,<:FaerFactorization{T}}) where {T} = 2
_dense_kind(f::FaerFactorization{T,K}) where {T,K} = Int32(getproperty(_FactorCodes, K))
function _dense_solve_validate(f::FaerFactorization)
    _native_call(
        RustCrate.factor_solve_validate,
        f,
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        _tag(eltype(f)),
        _dense_kind(f),
    )
end
function _dense_team_native_result(f::FaerFactorization{T}, b, op, team) where {T}
    dims = (size(f, op in (1, 2) ? 1 : 2), size(b, 2))
    args = (
        f,
        b,
        Int32(op),
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        UInt(size(b, 2)),
        _tag(T),
        _dense_kind(f),
        team,
    )
    return T===Float64 ? _matrix_result(RustCrate.factor_solve_team, dims, args...) :
           _typed_result(T, RustCrate.factor_solve_team, dims, args...)
end
function _dense_solve_rhs(
    f::_DenseSolveFactor,
    b;
    scratch = nothing,
    team = nothing,
    right = false,
)
    original=_dense_parent(f)
    _handle(original)
    _spectral_rhs_open(b)
    _dense_rhs_validate(b)
    Base.require_one_based_indexing(b)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    LinearAlgebra.issuccess(original) ||
        throw(LinearAlgebra.SingularException(original.info))
    _dense_solve_validate(original)
    (right ? size(b, 2)==size(f, 2) : size(b, 1)==size(f, 1)) ||
        throw(DimensionMismatch("factor and RHS dimensions do not match"))
    op = _dense_op(f)
    right && (op = (2, 3, 0)[op+1])
    input =
        right ? reshape(adjoint(b), size(b, 2), size(b, 1)) :
        ndims(b)==1 ? reshape(b, :, 1) : b
    rhs = FaerMatrix(input)
    try
        result = _typed_solve(original, rhs, op; scratch, team)
        try
            return right ? copy(adjoint(Matrix(result))) :
                   ndims(b)==1 ? vec(Matrix(result)) : Matrix(result)
        finally
            close(result)
        end
    finally
        close(rhs)
    end
end
function _dense_copy_result!(dest, result)
    buffer = ndims(dest)==1 ? Vector{eltype(dest)}(result) : Matrix{eltype(dest)}(result)
    copyto!(dest, buffer)
    return dest
end
function _dense_ldiv!(dest, f, b; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("factor solve destination shape mismatch"))
        return _dense_copy_result!(dest, _dense_solve_rhs(f, b; scratch, team))
    end
end
"""
    ldiv!(X, F::FaerFactorization, B; scratch=nothing, team=nothing)
    ldiv!(F, B; scratch=nothing, team=nothing)

Solve with retained dense factors and an optional caller-owned native team.
Transpose/adjoint, rectangular LU/full LU/QR and rank-deficient QR solves are supported.
Three-argument calls stage conversion before writeback and support overlapping
inputs. Two-argument calls require square factors. Native workspace and result
staging allocate separately. Caller scratch covers native solve work, including
R SVD/projection for rectangular or deficient QR. Mixed scalar
promotion retains the existing factor reconstruction/refactorization policy;
component extraction remains sequential, while products and new factorization
borrow the team. QR rank cutoffs keep the source precision after promotion.
The factor never retains the team token.
"""
LinearAlgebra.ldiv!(
    dest::AbstractMatrix,
    f::_DenseSolveFactor,
    b::AbstractMatrix;
    scratch = nothing,
    team = nothing,
) = _dense_ldiv!(dest, f, b; scratch, team)
LinearAlgebra.ldiv!(
    dest::AbstractVector,
    f::_DenseSolveFactor,
    b::AbstractVector;
    scratch = nothing,
    team = nothing,
) = _dense_ldiv!(dest, f, b; scratch, team)
function LinearAlgebra.ldiv!(
    f::_DenseSolveFactor,
    b::AbstractVecOrMat;
    scratch = nothing,
    team = nothing,
)
    size(f, 1)==size(f, 2) || throw(
        DimensionMismatch("two-argument ldiv! requires square factors; use ldiv!(X,F,B)"),
    )
    return _dense_ldiv!(b, f, b; scratch, team)
end
function LinearAlgebra.rdiv!(
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_DenseSolveFactor;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(size(b, 1), size(f, 1)) ||
            throw(DimensionMismatch("right factor solve destination shape mismatch"))
        return _dense_copy_result!(
            dest,
            _dense_solve_rhs(f, b; scratch, team, right = true),
        )
    end
end
function LinearAlgebra.rdiv!(
    b::AbstractMatrix,
    f::_DenseSolveFactor;
    scratch = nothing,
    team = nothing,
)
    size(f, 1)==size(f, 2) || throw(
        DimensionMismatch("two-argument rdiv! requires square factors; use rdiv!(X,B,F)"),
    )
    return LinearAlgebra.rdiv!(b, b, f; scratch, team)
end
# Resolve LinearAlgebra's transform-specific two-argument right-solve methods.
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval function LinearAlgebra.rdiv!(
        b::AbstractMatrix,
        f::$Wrapper{T,<:FaerFactorization{T}};
        scratch = nothing,
        team = nothing,
    ) where {T}
        size(f, 1)==size(f, 2) || throw(
            DimensionMismatch(
                "two-argument rdiv! requires square factors; use rdiv!(X,B,F)",
            ),
        )
        return LinearAlgebra.rdiv!(b, b, f; scratch, team)
    end
end
