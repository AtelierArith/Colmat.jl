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

# Diagonal scaling borrows the stored diagonal and the dense or native
# triangular operand in Rust; neither is expanded into a dense snapshot.
const _DiagonalScaled=Union{_ProductDense,_NativeProductTriangular}
const _julia_diagonal_kind=3
_diagonal_scaling(a::_ProductDiagonal, b::_DiagonalScaled, ::Type{P}) where {P} =
    eltype(_product_parent(b))===P && promote_type(eltype(a), P)===P
_diagonal_scaling(a::_DiagonalScaled, b::_ProductDiagonal, ::Type{P}) where {P} =
    _diagonal_scaling(b, a, P)
_diagonal_scaling(a, b, ::Type) = false
# A contiguous Julia buffer crosses for one call; it is not validated, so
# LinearAlgebra.Diagonal operands may hold non-finite values as before.
_diagonal_buffer(v::Vector) =
    (Ptr{UInt8}(pointer(v)), Int[_julia_diagonal_kind, 0, 1, length(v), 0, 1, 1, 0, 0], v)
function _diagonal_source(d::FaerDiagonal, ::Type{P}) where {P}
    _handle(d)
    eltype(d)===P || return _diagonal_buffer(Vector{P}(Vector(d)))
    return d, _descriptor(d, 1:d.len, 1:1), d
end
_diagonal_source(d::LinearAlgebra.Diagonal, ::Type{P}) where {P} =
    _diagonal_buffer(d.diag isa Vector{P} ? d.diag : Vector{P}(d.diag))
_diagonal_owner(x::FaerMatrixView) = _view_owner(x)
_diagonal_owner(x) = x
# Return (diagonal, dense operand, left?) in the product's operand order.
_diagonal_operands(a::_ProductDiagonal, b::Union{_DiagonalScaled,FaerVector}) = (a, b, true)
_diagonal_operands(a::Union{_DiagonalScaled,FaerRow}, b::_ProductDiagonal) = (b, a, false)
function _diagonal_call(f, ::Type{P}, a, b, dest...) where {P}
    g, x, left=_diagonal_operands(a, b)
    dp, dd, dkeep=_diagonal_source(g, P)
    y=_product_parent(x)
    structure=x isa _ProductTriangular ? _product_structure(x) : Int32(0)
    q, qd=_view_source(y)
    if !isempty(dest) && _diagonal_owner(dest[1])===q
        # mul!(A, D, A) and overlapping views: snapshot the operand natively.
        y=copy(y)
        q, qd=_view_source(y)
    end
    GC.@preserve dd qd dkeep y return f(dp, dd, q, qd, left, structure)
end
function _diagonal_product(a, b, ::Type{P}) where {P}
    dims=(size(a, 1), size(b, 2))
    return _diagonal_call(P, a, b) do dp, dd, q, qd, left, structure
        args=(dp, pointer(dd), q, pointer(qd), left, structure)
        P===Float64 ? _matrix_result(RustCrate.diagonal_scaled, dims, args...) :
        _typed_result(P, RustCrate.diagonal_scaled, dims, args...)
    end
end
function _diagonal_mul!(dest, a, b, alpha, beta)
    Q=eltype(dest)
    # Converting the coefficients matches Julia's InexactError for complex
    # coefficients into a real destination.
    al, be=Ref{Q}(Q(alpha)), Ref{Q}(Q(beta))
    p, d=_view_source(dest)
    _diagonal_call(Q, a, b, dest) do dp, dd, q, qd, left, structure
        GC.@preserve d al be _native_call(
            RustCrate.diagonal_scale_into,
            p,
            pointer(d),
            dp,
            pointer(dd),
            q,
            pointer(qd),
            left,
            structure,
            Ptr{UInt8}(Base.unsafe_convert(Ptr{Q}, al)),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{Q}, be)),
            iszero(alpha),
            iszero(beta),
        )
    end
    return dest
end
function _structured_product(a, b)
    lock(_matrix_lock) do
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("structured product inner dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        P<:_NativeScalar ||
            throw(ArgumentError("structured product requires native scalar types"))
        _diagonal_scaling(a, b, P) && return _diagonal_product(a, b, P)
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
        _diagonal_scaling(a, b, eltype(dest)) && return _diagonal_mul!(dest, a, b, alpha, beta)
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

# D * column and row * D scale the stored vector in Rust and stay native.
function _diagonal_vector(a, b, C)
    lock(_matrix_lock) do
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("diagonal product dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        g, x, left=_diagonal_operands(a, b)
        y=_collection_coerce(x, P)
        dp, dd, dkeep=_diagonal_source(g, P)
        q, qd=_view_source(y)
        GC.@preserve dd qd dkeep y return _collection_result(
            C,
            P,
            y.len,
            RustCrate.diagonal_scaled_vector,
            dp,
            pointer(dd),
            q,
            pointer(qd),
            left,
            _collection_kind(C),
        )
    end
end
Base.:*(a::FaerDiagonal, b::FaerVector) = _diagonal_vector(a, b, FaerVector)
Base.:*(a::FaerRow, b::FaerDiagonal) = _diagonal_vector(a, b, FaerRow)
