const _PivotedNativeCholesky=LinearAlgebra.CholeskyPivoted{T,<:FaerMatrix} where {T}
"""
    cholesky(A, RowMaximum(); tol=0.0, check=true, block_size=128, scratch=nothing, team=nothing)

Pivoted Cholesky with independently owned native factors. Hermitian/Symmetric
wrappers select the stored triangle. Negative tol uses faer's fixed automatic
threshold eps(T)*n*maximum(diag(A)); nonnegative tol uses an absolute cutoff in
the Rust adapter's complete diagonal-pivot algorithm. Rank-deficient factors
require check=false; solves require full rank. F.L, F.U, F.p, F.rank and Matrix(F)
follow LinearAlgebra's CholeskyPivoted conventions.
`block_size` must be positive and controls both the automatic native kernel and
the explicit-tolerance panel algorithm. Query matching workspace with
`scratch_requirements(cholesky, A, RowMaximum(); tol, block_size, team)`.
Native views and selected-triangle wrappers are supported. Declared scratch
includes permutation/kernel temporaries and the explicit-tolerance working
matrix. Returned factors, view snapshots, promotion and output arrays allocate
separately. Saved `ldiv!`, `rdiv!`, native `\`/`/`, `inv` and `reconstruct` accept
`scratch`/`team`; use factor-first ordering for their workspace queries.
Bang methods stage writeback; in-place solves reject factor-storage aliases and
publish the destination only after successful computation and conversion.
"""
_pivoted_cholesky(a; kwargs...) = _pivoted_cholesky_workspace(a; kwargs...)
LinearAlgebra.cholesky(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    _pivoted_cholesky(a; kwargs...)
for W in (:Hermitian, :Symmetric)
    @eval LinearAlgebra.cholesky(
        a::LinearAlgebra.$W{T,<:FaerMatrix},
        ::LinearAlgebra.RowMaximum;
        kwargs...,
    ) where {T} = _pivoted_cholesky(a; kwargs...)
end
function _pivoted_apply(F::_PivotedNativeCholesky, b, op::Int32; kwargs...)
    result=_pivoted_native_apply(F, op in (0, 3) ? b : nothing, op; kwargs...)
    try
        array=Matrix(result)
        return b isa AbstractVector ? vec(array) : array
    finally
        close(result)
    end
end
_pivoted_solve(F::_PivotedNativeCholesky, b; kwargs...) =
    _pivoted_apply(F, b, Int32(0); kwargs...)
function _pivoted_divide_into!(dest, F, b, op; kwargs...)
    lock(_matrix_lock) do
        _pivoted_destination_check(F, dest)
        Base.require_one_based_indexing(dest, b)
        size(dest)==size(b)||throw(
            DimensionMismatch("pivoted Cholesky destination and RHS shapes must match"),
        )
        result=_pivoted_apply(F, b, Int32(op); kwargs...)
        staged=eltype(dest)===eltype(result) ? result : Array{eltype(dest)}(result)
        all(isfinite, staged)||throw(
            ArgumentError(
                "pivoted Cholesky result cannot be represented finitely in the destination",
            ),
        )
        _pivoted_destination_check(F, dest)
        size(dest)==size(staged)||throw(
            DimensionMismatch(
                "pivoted Cholesky destination changed shape during conversion",
            ),
        )
        copyto!(dest, staged)
    end
end
LinearAlgebra.ldiv!(F::_PivotedNativeCholesky, b::AbstractVector; kwargs...) =
    _pivoted_divide_into!(b, F, b, 0; kwargs...)
LinearAlgebra.ldiv!(F::_PivotedNativeCholesky, b::AbstractMatrix; kwargs...) =
    _pivoted_divide_into!(b, F, b, 0; kwargs...)
LinearAlgebra.ldiv!(
    dest::AbstractVector,
    F::_PivotedNativeCholesky,
    b::AbstractVector;
    kwargs...,
) = _pivoted_divide_into!(dest, F, b, 0; kwargs...)
LinearAlgebra.ldiv!(
    dest::AbstractMatrix,
    F::_PivotedNativeCholesky,
    b::AbstractMatrix;
    kwargs...,
) = _pivoted_divide_into!(dest, F, b, 0; kwargs...)
LinearAlgebra.rdiv!(b::AbstractMatrix, F::_PivotedNativeCholesky; kwargs...) =
    _pivoted_divide_into!(b, F, b, 3; kwargs...)
Base.:/(b::FaerMatrix, F::_PivotedNativeCholesky; kwargs...) =
    _pivoted_native_apply(F, b, Int32(3); kwargs...)
Base.:\(F::_PivotedNativeCholesky, b::FaerMatrix; kwargs...) =
    _pivoted_native_apply(F, b, Int32(0); kwargs...)
Base.inv(F::_PivotedNativeCholesky; kwargs...) =
    _pivoted_native_apply(F, nothing, Int32(1); kwargs...)
reconstruct(F::_PivotedNativeCholesky; kwargs...) =
    _pivoted_native_apply(F, nothing, Int32(2); kwargs...)
function Base.Matrix(F::_PivotedNativeCholesky; kwargs...)
    result=reconstruct(F; kwargs...)
    try
        return Matrix(result)
    finally
        close(result)
    end
end
FaerMatrix(F::_PivotedNativeCholesky; kwargs...) = reconstruct(F; kwargs...)
Base.close(F::_PivotedNativeCholesky) = close(F.factors)
Base.isopen(F::_PivotedNativeCholesky) = isopen(F.factors)
