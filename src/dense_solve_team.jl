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
const _DenseNativeRhs=Union{FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}
# One native copy of the right-hand side, transposed/conjugated for right
# solves through the view descriptor. Native operands and strided Julia arrays
# of a native scalar type are never staged in Julia.
function _dense_native_rhs(b, right)
    src=if b isa _DenseNativeRhs
        b
    elseif b isa StridedVecOrMat{<:_NativeScalar} && _julia_borrowable(b)
        FaerBorrowedArray(b)
    else
        nothing
    end
    if isnothing(src)
        input=right ? reshape(adjoint(b), size(b, 2), size(b, 1)) :
              ndims(b)==1 ? reshape(b, :, 1) : b
        return FaerMatrix(Matrix(input))
    end
    return _copy_view_matrix(right ? adjoint(src) : src)
end
# Validate and solve; returns a fresh native result in the solve orientation.
function _dense_solve_native(
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
    rhs = _dense_native_rhs(b, right)
    try
        return _typed_solve(original, rhs, op; scratch, team)
    finally
        close(rhs)
    end
end
function _dense_solve_rhs(
    f::_DenseSolveFactor,
    b;
    scratch = nothing,
    team = nothing,
    right = false,
)
    result=_dense_solve_native(f, b; scratch, team, right)
    try
        return right ? copy(adjoint(Matrix(result))) :
               ndims(b)==1 ? vec(Matrix(result)) : Matrix(result)
    finally
        close(result)
    end
end
function _dense_copy_result!(dest, result)
    buffer = ndims(dest)==1 ? Vector{eltype(dest)}(result) : Matrix{eltype(dest)}(result)
    copyto!(dest, buffer)
    return dest
end
# Write a fresh native result (optionally adjointed) into a native destination
# through the view descriptors; the source never aliases the destination.
function _dense_assign_native!(dest, result, right)
    p, d=_view_source(dest)
    q, e=_view_source(right ? adjoint(result) : result)
    GC.@preserve d e _native_call(RustCrate.view_assign, p, pointer(d), q, pointer(e))
    return dest
end
_dense_native_dest(dest, result) =
    dest isa Union{FaerMatrix,FaerVector,FaerView} && eltype(dest)===eltype(result)
# Julia destinations of the result type take one copy of the native result.
function _dense_julia_dest!(dest, x)
    eltype(dest)===eltype(x) || return _dense_copy_result!(dest, x)
    return copyto!(dest, x)
end
function _dense_ldiv!(dest, f, b; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("factor solve destination shape mismatch"))
        result=_dense_solve_native(f, b; scratch, team)
        try
            _dense_native_dest(dest, result) &&
                return _dense_assign_native!(dest, result, false)
            return _dense_julia_dest!(
                dest,
                ndims(dest)==1 ? vec(Matrix(result)) : Matrix(result),
            )
        finally
            close(result)
        end
    end
end
"""
    ldiv!(X, F::FaerFactorization, B; scratch=nothing, team=nothing)
    ldiv!(F, B; scratch=nothing, team=nothing)

Solve with retained dense factors and an optional caller-owned native team.
Transpose/adjoint, rectangular LU/full LU/QR and rank-deficient QR solves are supported.
Three-argument calls compute a fresh native result before writeback and support
overlapping inputs. Two-argument calls require square factors. Native and
strided Julia right-hand sides are copied once into native storage (right solves
transpose through the view descriptor), and native destinations of the result
type are written natively, so no Julia staging array is made; Julia
destinations receive one copy of the result. Native workspace allocates
separately. Caller scratch covers native solve work, including
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
        result=_dense_solve_native(f, b; scratch, team, right = true)
        try
            _dense_native_dest(dest, result) &&
                return _dense_assign_native!(dest, result, true)
            return _dense_julia_dest!(dest, copy(adjoint(Matrix(result))))
        finally
            close(result)
        end
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
