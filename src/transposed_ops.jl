# Transposed and adjoint native matrices. LinearAlgebra's generic fallbacks copy
# these wrappers element by element through `getindex` (one native call per
# entry), so the conversion hooks below make one native copy through the view
# descriptor instead, and entrywise/operator norms and rank use the parent.
_transposed_copy(a, ::Type{T}) where {T} = _coerce(_copy_view_matrix(a), T)
LinearAlgebra.copy_similar(a::_BorrowTranspose, ::Type{T}) where {T<:_NativeScalar} =
    _transposed_copy(a, T)
LinearAlgebra.eigencopy_oftype(a::_BorrowTranspose, ::Type{T}) where {T<:_NativeScalar} =
    _transposed_copy(a, T)
Base.copymutable(a::_BorrowTranspose) = _copy_view_matrix(a)
Base.Matrix(a::_BorrowTranspose) = Matrix(_copy_view_matrix(a))
Base.Matrix{T}(a::_BorrowTranspose) where {T} = Matrix{T}(_copy_view_matrix(a))

for f in (:eigen, :eigvals, :svd, :svdvals, :schur, :hessenberg, :pinv)
    @eval LinearAlgebra.$f(a::_BorrowTranspose; kwargs...) =
        LinearAlgebra.$f(_copy_view_matrix(a); kwargs...)
end
LinearAlgebra.rank(a::_BorrowTranspose; kwargs...) =
    LinearAlgebra.rank(parent(a); kwargs...)
# Entrywise norms are invariant under transposition and conjugation.
LinearAlgebra.norm(a::_BorrowTranspose, p::Real = 2) = LinearAlgebra.norm(parent(a), p)
function LinearAlgebra.opnorm(a::_BorrowTranspose, p::Real = 2)
    q=p==1 ? Inf : p==Inf ? 1 : p
    return LinearAlgebra.opnorm(parent(a), q)
end

# Accumulate products with transposed/adjoint native operands directly into a
# native destination through `matmul::triangular` (dense structure).
const _TransposedOperand=Union{_BorrowMatrix,_BorrowTranspose}
for (A, B) in
    ((:_BorrowTranspose, :_TransposedOperand), (:_BorrowMatrix, :_BorrowTranspose))
    @eval begin
        LinearAlgebra.mul!(c::_ProductDense, a::$A, b::$B, alpha::Number, beta::Number) =
            _structured_mul!(c, a, b, alpha, beta)
        LinearAlgebra.mul!(c::_ProductDense, a::$A, b::$B) =
            _structured_mul!(c, a, b, true, false)
    end
end
LinearAlgebra.mul!(
    y::FaerVector,
    a::_BorrowTranspose,
    x::_BorrowVector,
    alpha::Number,
    beta::Number,
) = _structured_mul!(y, a, x, alpha, beta)
LinearAlgebra.mul!(y::FaerVector, a::_BorrowTranspose, x::_BorrowVector) =
    _structured_mul!(y, a, x, true, false)
