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
function _dense_solve_check(f::_DenseSolveFactor, b; team = nothing, right = false)
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
    return original
end
# Validate and solve; returns a fresh native result in the solve orientation.
function _dense_solve_native(
    f::_DenseSolveFactor,
    b;
    scratch = nothing,
    team = nothing,
    right = false,
)
    original=_dense_solve_check(f, b; team, right)
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
# Write a fresh native result into a borrowed or Julia strided destination of
# its scalar type natively (no Julia staging array); `false` otherwise.
function _dense_write_julia!(dest, result, right)
    target=eltype(dest)===eltype(result) ? _gemm_destination(dest) : nothing
    isnothing(target) && return false
    p, d=_view_source(target)
    q, e=_view_source(right ? adjoint(result) : result)
    GC.@preserve d e _native_call(RustCrate.view_assign_into, p, pointer(d), q, pointer(e))
    return true
end
# faer's in-place solve with the retained factor directly in `target` (a
# native owner/view, borrowed or Julia strided array named by its view
# descriptor). Returns `false`, writing nothing, when the factor needs the
# rectangular, rank-revealing or SVD path.
#
# With caller scratch (already checked by `_dense_in_place_scratch`) the
# per-kind low-level `solve_in_place` kernels run on it, so the solve makes no
# native allocation of its own. Right solves pass `transpose(B)` and the
# remapped op: `X A = B` as `Aᵀ Xᵀ = Bᵀ`, `X Aᴴ = B` as `conj(A) Xᵀ = Bᵀ`.
function _dense_in_place_solve!(
    original::FaerFactorization,
    target,
    op,
    team;
    scratch = nothing,
)
    p, d=_view_source(target)
    isnothing(scratch) && return GC.@preserve d _native_call(
        RustCrate.solve_in_place_view,
        original,
        p,
        pointer(d),
        Int32(op),
        team,
    )
    T=eltype(original)
    return GC.@preserve p d _native_call(
        RustCrate.factor_solve_into,
        original,
        p,
        pointer(d),
        scratch,
        Int32(op),
        UInt(size(original, 1)),
        UInt(size(original, 2)),
        UInt(size(target, 2)),
        _tag(T),
        _dense_kind(original),
        Float64(eps(_realtype(T))),
        team,
    )::Bool
end
# Validate caller scratch for an in-place solve before any write.
function _dense_in_place_scratch(original, columns, op, scratch, team)
    isnothing(scratch) && return nothing
    scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    precision=Float64(eps(_realtype(eltype(original))))
    req=_dense_native_requirements(original, columns, op; team, precision)
    cap=scratch_capacity(scratch)
    cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
        ArgumentError("dense solve scratch capacity/alignment is insufficient; call reserve!"),
    )
    return scratch
end
# Same-scalar square solves run in place in the destination: `B` is assigned
# into `X` natively first (inputs overlapping `X` are snapshot), so no RHS
# copy, fresh result or writeback is made. Returns `nothing` when solved,
# otherwise the right-hand side for the fresh-result path (`X` once `B` has
# been assigned into it).
function _dense_ldiv_in_place!(dest, f, b, scratch, team)
    original=_dense_parent(f)
    T=eltype(original)
    eltype(dest)===T && eltype(b)===T && size(f, 1)==size(f, 2) || return b
    _dense_solve_check(f, b; team)
    target=_gemm_destination(dest)
    input=b===dest ? target : _gemm_input(b)
    (isnothing(target) || isnothing(input)) && return b
    work=_dense_in_place_scratch(original, size(target, 2), _dense_op(f), scratch, team)
    if b!==dest
        # A non-finite `B` fails in the fresh-result path before `X` is written.
        all(isfinite, b) || return b
        p, d=_view_source(target)
        q, e=_view_source(input)
        GC.@preserve d e _native_call(RustCrate.view_assign_into, p, pointer(d), q, pointer(e))
    end
    _dense_in_place_solve!(original, target, _dense_op(f), team; scratch = work) &&
        return nothing
    # Declined (rectangular-path factor): `X` now holds `B` and is the input of
    # the fresh-result path; a later failure there leaves `X == B`.
    return dest
end
# Two-argument right solves of a same-scalar square factor run in place on
# `transpose(B)`; `false` (nothing written) when the factor declines.
function _dense_rdiv_in_place!(b, f, scratch, team)
    original=_dense_parent(f)
    T=eltype(original)
    eltype(b)===T && size(f, 1)==size(f, 2) || return false
    _dense_solve_check(f, b; team, right = true)
    target=_gemm_destination(b)
    isnothing(target) && return false
    op=(1, 0, 3)[_dense_op(f)+1]
    work=_dense_in_place_scratch(original, size(b, 1), op, scratch, team)
    return _dense_in_place_solve!(original, transpose(target), op, team; scratch = work)
end
function _dense_ldiv!(dest, f, b; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("factor solve destination shape mismatch"))
        b=_dense_ldiv_in_place!(dest, f, b, scratch, team)
        isnothing(b) && return dest
        result=_dense_solve_native(f, b; scratch, team)
        try
            _dense_native_dest(dest, result) &&
                return _dense_assign_native!(dest, result, false)
            _dense_write_julia!(dest, result, false) && return dest
            return _dense_julia_dest!(
                dest,
                ndims(dest)==1 ? vec(Matrix(result)) : Matrix(result),
            )
        finally
            close(result)
        end
    end
end
# `x` is a fresh native copy of a right-hand side that nothing else holds:
# same-scalar square solves overwrite it in place and return it; otherwise
# the ordinary solve runs and the copy is released.
function _dense_solve_temporary(f::_DenseSolveFactor, x)
    original=_dense_parent(f)
    if eltype(x)===eltype(original) &&
       size(f, 1)==size(f, 2)==size(x, 1) &&
       LinearAlgebra.issuccess(original)
        solved=lock(_matrix_lock) do
            _dense_in_place_solve!(original, x, _dense_op(f), nothing)
        end
        solved && return x
    end
    result=f\x
    result===x || close(x)
    return result
end
# One native copy of a same-scalar borrowed or Julia strided right-hand side.
function _dense_rhs_copy(b::AbstractMatrix, ::Type{T}) where {T}
    eltype(b)===T || return FaerMatrix{T}(b)
    b isa FaerBorrowedMatrix && return _copy_view_matrix(b)
    b isa StridedMatrix && _julia_borrowable(b) && return _copy_view_matrix(FaerBorrowedArray(b))
    return FaerMatrix{T}(b)
end
# Without a target scalar the copy keeps the right-hand side's native scalar.
_dense_rhs_copy(b::AbstractMatrix) =
    eltype(b)<:_NativeScalar ? _dense_rhs_copy(b, eltype(b)) : FaerMatrix(b)
_dense_vector_solve(f, b::AbstractVector) =
    _dense_vector_solve(f, b, eltype(b)<:_NativeScalar ? eltype(b) : nothing)
function _dense_vector_solve(f, b::AbstractVector, ::Type{T}) where {T}
    x=_dense_solve_temporary(f, _dense_rhs_copy(reshape(b, :, 1), T))
    try
        return vec(Matrix(x))
    finally
        close(x)
    end
end
function _dense_vector_solve(f, b::AbstractVector, ::Nothing)
    x=_dense_solve_temporary(f, FaerMatrix(reshape(b, :, 1)))
    try
        return vec(Matrix(x))
    finally
        close(x)
    end
end
"""
    ldiv!(X, F::FaerFactorization, B; scratch=nothing, team=nothing)
    ldiv!(F, B; scratch=nothing, team=nothing)

Solve with retained dense factors and an optional caller-owned native team.
Transpose/adjoint, rectangular LU/full LU/QR and rank-deficient QR solves are supported.
Three-argument calls compute a fresh native result before writeback and support
overlapping inputs. Two-argument calls require square factors. Same-scalar square,
numerically nonsingular factors solve in place in the destination with faer's
`solve_in_place` when it is a native owner/view, a `faerview` or an injective
Julia strided array: two-argument calls make no right-hand-side copy, result or
writeback, and three-argument calls first assign `B` into `X` natively (inputs
overlapping `X` are snapshot). Two-argument `rdiv!(B, F)` solves in place
on `transpose(B)` the same way. With caller scratch the low-level per-kind
kernels run on it, so these in-place solves make no native allocation of their
own. Otherwise native and strided Julia right-hand
sides are copied once into native storage (right solves transpose through the
view descriptor) and the fresh result is written natively into native,
borrowed and Julia strided destinations of the result type. Native workspace allocates
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
        dest===b && _dense_rdiv_in_place!(b, f, scratch, team) && return b
        result=_dense_solve_native(f, b; scratch, team, right = true)
        try
            _dense_native_dest(dest, result) &&
                return _dense_assign_native!(dest, result, true)
            _dense_write_julia!(dest, result, true) && return dest
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
