"""
    lu!(A::FaerMatrix; check=true, allowsingular=false, team=nothing)
    qr!(A::FaerMatrix[, pivot]; team=nothing)
    cholesky!(A::FaerMatrix; check=true, team=nothing)

Factor using native faer, then replace A's entries with packed factors after a
successful computation. LU stores U on/above the diagonal and the strict L
below it. QR stores R on/above the diagonal and faer's Householder basis below;
its block coefficients remain owned by the returned factorization. This QR
layout is not LAPACK's tau format. Cholesky stores L (or U for an upper wrapper).
F.packed returns independent lower-oriented packed storage.

These methods allocate an independent factorization and stage the writeback;
F stays usable if A changes or closes, and existing same-shape views observe
the new entries. They do not provide allocation-free factorization. Errors
before writeback preserve A. Failed check=false Cholesky leaves A unchanged
and returns the unsuccessful factorization.
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
)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.lu(a; check, allowsingular, team))
    end
end
LinearAlgebra.lu!(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu!(a; kwargs...)
LinearAlgebra.qr!(a::FaerMatrix; team = nothing) =
    LinearAlgebra.qr!(a, LinearAlgebra.NoPivot(); team)
for P in (:NoPivot, :ColumnNorm)
    @eval function LinearAlgebra.qr!(a::FaerMatrix, pivot::LinearAlgebra.$P; team = nothing)
        lock(_matrix_lock) do
            return _factor_writeback!(a, LinearAlgebra.qr(a, pivot; team))
        end
    end
end
function LinearAlgebra.cholesky!(a::FaerMatrix; check::Bool = true, team = nothing)
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
    ) where {T<:$bound}
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
    @eval function LinearAlgebra.$bang(a::FaerMatrix; check::Bool = true, team = nothing)
        lock(_matrix_lock) do
            return _factor_writeback!(a, LinearAlgebra.$fn(a; check, team))
        end
    end
    for W in (:Hermitian, :Symmetric)
        bound=W===:Symmetric ? Real : Any
        @eval function LinearAlgebra.$bang(
            a::LinearAlgebra.$W{T,<:FaerMatrix};
            check::Bool = true,
            team = nothing,
        ) where {T<:$bound}
            lock(_matrix_lock) do
                return _factor_writeback!(
                    parent(a),
                    LinearAlgebra.$fn(a; check, team);
                    upper = a.uplo=='U',
                )
            end
        end
    end
end
