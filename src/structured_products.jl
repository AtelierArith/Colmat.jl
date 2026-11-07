const _ProductTriangular=Union{
    LinearAlgebra.LowerTriangular,
    LinearAlgebra.UpperTriangular,
    LinearAlgebra.UnitLowerTriangular,
    LinearAlgebra.UnitUpperTriangular,
}
const _NativeProductTriangular=Union{
    LinearAlgebra.LowerTriangular{<:Any,<:FaerMatrix},
    LinearAlgebra.UpperTriangular{<:Any,<:FaerMatrix},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:FaerMatrix},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:FaerMatrix},
}
const _ProductDense=Union{FaerMatrix,FaerMatrixView}
const _ProductDiagonal=Union{FaerDiagonal,LinearAlgebra.Diagonal}
_product_structure(a::_ProductDiagonal) = Int32(5)
_product_structure(a) = Int32(0)
_product_structure(a::_ProductTriangular) =
    Int32(_islower(a) ? (_isunit(a) ? 3 : 1) : (_isunit(a) ? 4 : 2))
_product_parent(a) = a
_product_parent(a::_ProductTriangular) = parent(a)
function _structured_product(a, b)
    # Diagonal-dense products borrow the stored diagonal and dense operand.
    a isa _ProductDiagonal && b isa _ProductDense && return _diagonal_matrix(a, b, 0)
    a isa _ProductDense && b isa _ProductDiagonal && return _diagonal_matrix(b, a, 1)
    lock(_matrix_lock) do
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("structured product inner dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        P<:_NativeScalar ||
            throw(ArgumentError("structured product requires native scalar types"))
        # Snapshot physical parent data: faer's block structure ignores unused
        # entries and the stored diagonal for unit-triangular operands.
        x=Matrix{P}(_product_parent(a))
        y=Matrix{P}(_product_parent(b))
        sa, sb=_product_structure(a), _product_structure(b)
        m, k=size(x)
        n=size(y, 2)
        GC.@preserve x y begin
            args=(
                Ptr{UInt8}(pointer(x)),
                Ptr{UInt8}(pointer(y)),
                UInt(m),
                UInt(k),
                UInt(n),
                sa,
                sb,
            )
            return P===Float64 ?
                   _matrix_result(RustCrate.structured_product, (m, n), args...) :
                   _typed_result(P, RustCrate.structured_product, (m, n), args..., _tag(P))
        end
    end
end
Base.:*(a::_ProductTriangular, b::_ProductDense) = _structured_product(a, b)
Base.:*(a::_ProductDense, b::_ProductTriangular) = _structured_product(a, b)
Base.:*(a::_NativeProductTriangular, b::_ProductTriangular) = _structured_product(a, b)
Base.:*(a::_ProductTriangular, b::_NativeProductTriangular) = _structured_product(a, b)
Base.:*(a::_NativeProductTriangular, b::_NativeProductTriangular) =
    _structured_product(a, b)
function _structured_mul!(dest, a, b, alpha, beta)
    lock(_matrix_lock) do
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("structured product inner dimensions must match"))
        size(dest)==(size(a, 1), size(b, 2)) ||
            throw(DimensionMismatch("structured product destination size must match"))
        for x in (dest, _product_parent(a), _product_parent(b))
            x isa _RotationArray && _rotation_open(x)
            x isa _OwnedCollection && _handle(x)
        end
        # Snapshot before writes; zero alpha/beta never read the ignored values.
        P=promote_type(eltype(a), eltype(b), eltype(dest), typeof(alpha), typeof(beta))
        values=iszero(alpha) ? zeros(P, size(dest)) :
               Matrix{P}(Matrix(_structured_product(a, b)))
        !isone(alpha) && (values .*= alpha)
        !iszero(beta) && (values .+= beta .* Matrix(dest))
        return copyto!(dest, values)
    end
end

for (A, B) in (
    (:_ProductTriangular, :_ProductDense),
    (:_ProductDense, :_ProductTriangular),
    (:_NativeProductTriangular, :_ProductTriangular),
    (:_ProductTriangular, :_NativeProductTriangular),
    (:_NativeProductTriangular, :_NativeProductTriangular),
)
    @eval begin
        LinearAlgebra.mul!(dest::_ProductDense, a::$A, b::$B, alpha::Number, beta::Number) =
            _structured_mul!(dest, a, b, alpha, beta)
        LinearAlgebra.mul!(dest::_ProductDense, a::$A, b::$B) =
            _structured_mul!(dest, a, b, 1, 0)
    end
end

for (A, B) in (
    (:_ProductDiagonal, :_ProductDense),
    (:_ProductDense, :_ProductDiagonal),
    (:FaerDiagonal, :_ProductTriangular),
    (:_ProductTriangular, :FaerDiagonal),
    (:_NativeProductTriangular, :_ProductDiagonal),
    (:_ProductDiagonal, :_NativeProductTriangular),
    (:_NativeProductTriangular, :FaerDiagonal),
    (:FaerDiagonal, :_NativeProductTriangular),
)
    @eval begin
        Base.:*(a::$A, b::$B) = _structured_product(a, b)
        LinearAlgebra.mul!(dest::_ProductDense, a::$A, b::$B, alpha::Number, beta::Number) =
            _structured_mul!(dest, a, b, alpha, beta)
        LinearAlgebra.mul!(dest::_ProductDense, a::$A, b::$B) =
            _structured_mul!(dest, a, b, 1, 0)
    end
end
