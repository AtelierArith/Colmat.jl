# Transposed and adjoint native matrices. LinearAlgebra's generic fallbacks copy
# these wrappers element by element through `getindex` (one native call per
# entry), so the conversion hooks below make one native copy through the view
# descriptor instead, and entrywise/operator norms and rank use the parent.
_transposed_copy(a, ::Type{T}) where {T} = _coerce(_copy_view_matrix(a), T)
function LinearAlgebra.copy_similar(a::_BorrowTranspose, ::Type{T}) where {T<:_NativeScalar}
    return _transposed_copy(a, T)
end
function LinearAlgebra.eigencopy_oftype(
    a::_BorrowTranspose, ::Type{T}
) where {T<:_NativeScalar}
    return _transposed_copy(a, T)
end
Base.copymutable(a::_BorrowTranspose) = _copy_view_matrix(a)
Base.Matrix(a::_BorrowTranspose) = Matrix(_copy_view_matrix(a))
Base.Matrix{T}(a::_BorrowTranspose) where {T} = Matrix{T}(_copy_view_matrix(a))

for f in (:eigen, :eigvals, :svd, :svdvals, :schur, :hessenberg, :pinv)
    @eval LinearAlgebra.$f(a::_BorrowTranspose; kwargs...) =
        LinearAlgebra.$f(_copy_view_matrix(a); kwargs...)
end
function LinearAlgebra.rank(a::_BorrowTranspose; kwargs...)
    return LinearAlgebra.rank(parent(a); kwargs...)
end
# Entrywise norms are invariant under transposition and conjugation.
LinearAlgebra.norm(a::_BorrowTranspose, p::Real=2) = LinearAlgebra.norm(parent(a), p)
function LinearAlgebra.opnorm(a::_BorrowTranspose, p::Real=2)
    q = if p == 1
        Inf
    elseif p == Inf
        1
    else
        p
    end
    return LinearAlgebra.opnorm(parent(a), q)
end

# mul! with transposed/adjoint operands into native matrix or Julia destinations
# is handled by dense_destinations.jl; a native vector destination with a
# transposed matrix is the remaining combination and uses the same gemm path.
function LinearAlgebra.mul!(
    y::FaerVector,
    a::_GemmTransposed,
    x::_GemmNativeVector,
    alpha::Number,
    beta::Number;
    team=nothing,
)
    return _dense_mul!(y, a, x, alpha, beta; team)
end
function LinearAlgebra.mul!(
    y::FaerVector, a::_GemmTransposed, x::_GemmNativeVector; team=nothing
)
    return _dense_mul!(y, a, x, true, false; team)
end
