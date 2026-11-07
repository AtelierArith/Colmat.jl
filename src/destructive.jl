"""
    lu!(A::FaerMatrix; check=true, allowsingular=false, team=nothing, scratch=nothing)
    qr!(A::FaerMatrix[, pivot]; team=nothing[, scratch=nothing])
    cholesky!(A::FaerMatrix; check=true, team=nothing, scratch=nothing)

Without `scratch`, factor using native faer, then replace A's entries with
packed factors after a successful computation. LU stores U on/above the
diagonal and the strict L below it. QR stores R on/above the diagonal and
faer's Householder basis below; its block coefficients remain owned by the
returned factorization. This QR layout is not LAPACK's tau format. Cholesky
stores L (or U for an upper wrapper). F.packed returns independent
lower-oriented packed storage.

These methods allocate an independent factorization and stage the writeback;
F stays usable if A changes or closes, and existing same-shape views observe
the new entries. Errors before writeback preserve A. Failed check=false
Cholesky leaves A unchanged and returns the unsuccessful factorization.

With `scratch=work::FaerScratch` (sized by `scratch_requirements(lu!, A)`,
`scratch_requirements(qr!, A)` or `scratch_requirements(cholesky!, A)`), LU,
unpivoted QR and Cholesky instead run faer's `lu_in_place`, `qr_in_place` or
`cholesky_in_place` directly on A's native storage and return a
[`FaerInPlaceFactorization`](@ref) that aliases A (`F.factors === A`), as
Julia's `lu!` does. Colmat itself allocates no native memory there (only faer's
internal matrix products of blocked shapes may); the LU permutation and QR
coefficient buffers of F are allocated in Julia. Shape, scalar, scratch, team and
finiteness checks precede any write, but after a numerical failure (singular
`check=true` LU, non-positive-definite Cholesky) A holds partially factored
data, as with LAPACK. Column-pivoted `qr!` has no direct path.
"""
function _factor_writeback!(a::FaerMatrix, f::FaerFactorization; upper = false)
    if getfield(f, :info)!=0 &&
       (f isa FaerFactorization{<:Any,:cholesky} || f isa FaerFactorization{<:Any,:ldlt})
        return f
    end
    packed=f.packed
    copyto!(a, upper ? copy(adjoint(packed)) : packed)
    return f
end
function LinearAlgebra.lu!(
    a::FaerMatrix;
    check::Bool = true,
    allowsingular::Bool = false,
    team = nothing,
    scratch = nothing,
)
    isnothing(scratch) || return _direct_lu!(a, check, allowsingular, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.lu(a; check, allowsingular, team))
    end
end
LinearAlgebra.lu!(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu!(a; kwargs...)
LinearAlgebra.qr!(a::FaerMatrix; team = nothing, scratch = nothing) =
    LinearAlgebra.qr!(a, LinearAlgebra.NoPivot(); team, scratch)
function LinearAlgebra.qr!(
    a::FaerMatrix,
    pivot::LinearAlgebra.NoPivot;
    team = nothing,
    scratch = nothing,
)
    isnothing(scratch) || return _direct_qr!(a, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.qr(a, pivot; team))
    end
end
function LinearAlgebra.qr!(a::FaerMatrix, pivot::LinearAlgebra.ColumnNorm; team = nothing)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.qr(a, pivot; team))
    end
end
function LinearAlgebra.cholesky!(
    a::FaerMatrix;
    check::Bool = true,
    team = nothing,
    scratch = nothing,
)
    isnothing(scratch) || return _direct_cholesky!(
        a,
        true,
        LinearAlgebra.ishermitian(a),
        check,
        scratch,
        team,
    )
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.cholesky(a; check, team))
    end
end
for W in (:Hermitian, :Symmetric)
    bound=W===:Symmetric ? Real : Any
    @eval function LinearAlgebra.cholesky!(
        a::LinearAlgebra.$W{T,<:FaerMatrix};
        check::Bool = true,
        team = nothing,
        scratch = nothing,
    ) where {T<:$bound}
        isnothing(scratch) ||
            return _direct_cholesky!(parent(a), a.uplo=='L', true, check, scratch, team)
        lock(_matrix_lock) do
            return _factor_writeback!(
                parent(a),
                LinearAlgebra.cholesky(a; check, team);
                upper = a.uplo=='U',
            )
        end
    end
end
function _pivoted_writeback!(a; kwargs...)
    lock(_matrix_lock) do
        source=a isa Union{FaerMatrix,FaerMatrixView} ? a : parent(a)
        f=LinearAlgebra.cholesky(a, LinearAlgebra.RowMaximum(); kwargs...)
        upper=!(a isa Union{FaerMatrix,FaerMatrixView}) && a.uplo=='U'
        copyto!(source, upper ? copy(adjoint(f.factors)) : f.factors)
        return f
    end
end
LinearAlgebra.cholesky!(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    _pivoted_writeback!(a; kwargs...)
for W in (:Hermitian, :Symmetric)
    bound=W===:Symmetric ? Real : Any
    @eval LinearAlgebra.cholesky!(
        a::LinearAlgebra.$W{T,<:FaerMatrix},
        ::LinearAlgebra.RowMaximum;
        kwargs...,
    ) where {T<:$bound} = _pivoted_writeback!(a; kwargs...)
end

"""
    full_piv_lu!(A::FaerMatrix; check=true, team=nothing)

Compute native full-pivot LU, then replace A by packed L/U storage. F.p/F.q
retain the row/column permutations. Returned factors are independent of A.
"""
function full_piv_lu!(a::FaerMatrix; check::Bool = true, team = nothing)
    lock(_matrix_lock) do
        return _factor_writeback!(a, full_piv_lu(a; check, team))
    end
end
for fn in (:ldlt, :bunchkaufman)
    bang=Symbol(fn, :!)
    # `bunchkaufman!(A, rook=false)` keeps Julia's positional rook argument.
    pivot=fn===:bunchkaufman ? (Expr(:kw, :(rook::Bool), false),) : ()
    forward=fn===:bunchkaufman ? (:rook,) : ()
    @eval function LinearAlgebra.$bang(
        a::FaerMatrix,
        $(pivot...);
        check::Bool = true,
        team = nothing,
    )
        lock(_matrix_lock) do
            return _factor_writeback!(a, LinearAlgebra.$fn(a, $(forward...); check, team))
        end
    end
    for W in (:Hermitian, :Symmetric)
        bound=W===:Symmetric ? Real : Any
        @eval function LinearAlgebra.$bang(
            a::LinearAlgebra.$W{T,<:FaerMatrix},
            $(pivot...);
            check::Bool = true,
            team = nothing,
        ) where {T<:$bound}
            lock(_matrix_lock) do
                return _factor_writeback!(
                    parent(a),
                    LinearAlgebra.$fn(a, $(forward...); check, team);
                    upper = a.uplo=='U',
                )
            end
        end
    end
end
