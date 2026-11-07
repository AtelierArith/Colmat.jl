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
# Operands are borrowed: native owners and ranges through view descriptors,
# strided Julia buffers of the product type through a kind-3 descriptor that
# preserves the buffer for the call. Only conversions and non-strided Julia
# inputs are copied. Returns (pointer or owner, descriptor, retained object).
function _raw_operand(a::StridedMatrix{P}) where {P<:_NativeScalar}
    m, n=size(a)
    rs, cs=m*n==0 ? (1, 1) : (stride(a, 1), stride(a, 2))
    d=Int[3, _tag(P), rs, m, 0, cs, n, 0, 0]
    return Ptr{UInt8}(pointer(a)), d, a
end
const _ProductNativeSource=Union{
    FaerMatrix,
    FaerView,
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerView}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerView}},
}
function _structured_operand(x, ::Type{P}) where {P}
    if x isa _ProductDiagonal
        if x isa FaerDiagonal && eltype(x)===P
            _handle(x)
            return x, _descriptor(x, 1:x.len, 1:1), x
        end
        v=Vector{P}(x isa FaerDiagonal ? Vector(x) : x.diag)
        return _raw_operand(reshape(v, :, 1))
    end
    y=_product_parent(x)
    if y isa _ProductNativeSource
        eltype(y)===P || (y=_diagonal_coerce(y, P))
        p, d=_view_source(y)
        return p, d, y
    end
    if y isa StridedMatrix{P} && all(>=(1), strides(y))
        return _raw_operand(y)
    end
    return _raw_operand(Matrix{P}(y))
end
function _structured_scalar(a, b)
    size(a, 2)==size(b, 1) ||
        throw(DimensionMismatch("structured product inner dimensions must match"))
    P=promote_type(eltype(a), eltype(b))
    P<:_NativeScalar ||
        throw(ArgumentError("structured product requires native scalar types"))
    return P
end
function _structured_product(a, b)
    # Diagonal-dense products borrow the stored diagonal and dense operand.
    a isa _ProductDiagonal && b isa _ProductDense && return _diagonal_matrix(a, b, 0)
    a isa _ProductDense && b isa _ProductDiagonal && return _diagonal_matrix(b, a, 1)
    lock(_matrix_lock) do
        P=_structured_scalar(a, b)
        p, pd, ka=_structured_operand(a, P)
        q, qd, kb=_structured_operand(b, P)
        dims=(size(a, 1), size(b, 2))
        h=P===Float64 ? RustCrate.OwnedMatrix() : RustCrate.TypedMatrix()
        GC.@preserve pd qd ka kb _native_call(
            RustCrate.structured_product,
            h,
            p,
            pointer(pd),
            _product_structure(a),
            q,
            pointer(qd),
            _product_structure(b),
        )
        return P===Float64 ? _owned(h, dims) : FaerMatrix{P,typeof(h)}(h, dims)
    end
end
function _structured_vector(a, b)
    lock(_matrix_lock) do
        P=_structured_scalar(a, b)
        p, pd, ka=_structured_operand(a, P)
        bb=eltype(b)===P ? b : FaerVector{P}(Vector(b))
        q, qd=_view_source(bb)
        h=RustCrate.VectorOwner()
        GC.@preserve pd qd ka _native_call(
            RustCrate.structured_vector,
            h,
            p,
            pointer(pd),
            _product_structure(a),
            q,
            pointer(qd),
        )
        return FaerVector{P,typeof(h)}(h, size(a, 1), UInt(0))
    end
end
Base.:*(a::_NativeProductTriangular, b::_BorrowVector) = _structured_vector(a, b)
# Native dense operands include transposed and adjoint native wrappers.
const _ProductOperand=Union{
    _ProductDense,
    LinearAlgebra.Transpose{<:Any,<:_ProductDense},
    LinearAlgebra.Adjoint{<:Any,<:_ProductDense},
}
Base.:*(a::_ProductTriangular, b::_ProductOperand) = _structured_product(a, b)
Base.:*(a::_ProductOperand, b::_ProductTriangular) = _structured_product(a, b)
Base.:*(a::_NativeProductTriangular, b::_ProductTriangular) = _structured_product(a, b)
Base.:*(a::_ProductTriangular, b::_NativeProductTriangular) = _structured_product(a, b)
Base.:*(a::_NativeProductTriangular, b::_NativeProductTriangular) =
    _structured_product(a, b)

_alias_owner(x::Union{FaerMatrix,_OwnedCollection}) = x
_alias_owner(x::FaerView) = x.owner
_alias_owner(x::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint}) =
    _alias_owner(parent(x))
_alias_owner(x) = nothing
_unalias(x::Union{FaerMatrix,FaerView,_OwnedCollection}) = copy(x)
_unalias(x::LinearAlgebra.Transpose) = transpose(_unalias(parent(x)))
_unalias(x::LinearAlgebra.Adjoint) = adjoint(_unalias(parent(x)))
# Snapshot an operand only when it shares the destination's native owner.
function _unalias_operand(x, owner)
    if x isa _ProductTriangular
        p=parent(x)
        return _alias_owner(p)===owner ? typeof(x).name.wrapper(_unalias(p)) : x
    end
    return _alias_owner(x)===owner ? _unalias(x) : x
end
const _structured_destinations=(
    full = 0,
    lower = 1,
    upper = 2,
    strict_lower = 6,
    strict_upper = 7,
)
function _structured_mul!(dest, a, b, alpha, beta, ds = Int32(0))
    lock(_matrix_lock) do
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("structured product inner dimensions must match"))
        size(dest, 1)==size(a, 1) && size(dest, 2)==size(b, 2) && ndims(dest)==ndims(b) ||
            throw(DimensionMismatch("structured product destination size must match"))
        ds==0 ||
            size(dest, 1)==size(dest, 2) ||
            throw(DimensionMismatch("a structured destination must be square"))
        for x in (dest, _product_parent(a), _product_parent(b))
            x isa _RotationArray && _rotation_open(x)
            x isa _OwnedCollection && _handle(x)
            x isa FaerView && _view_owner(x)
        end
        P=promote_type(eltype(a), eltype(b), eltype(dest), typeof(alpha), typeof(beta))
        native=P===eltype(dest) && P<:_NativeScalar && !(dest isa FaerVectorView)
        if !native
            ds==0 || throw(
                ArgumentError(
                    "structured destinations require the promoted destination type",
                ),
            )
            # Promotion beyond the destination type is computed, then converted.
            values=iszero(alpha) ? zeros(P, size(dest)) :
                   Array{P}(Array(_structured_product(a, b)))
            !isone(alpha) && (values .*= alpha)
            !iszero(beta) && (values .+= beta .* Array(dest))
            return copyto!(dest, values)
        end
        owner, dd=_view_source(dest)
        aa=_unalias_operand(a, owner)
        bb=_unalias_operand(b, owner)
        p, pd, ka=_structured_operand(aa, P)
        q, qd, kb=_structured_operand(bb, P)
        α=Ref{P}(alpha)
        β=Ref{P}(beta)
        GC.@preserve dd pd qd ka kb α β _native_call(
            RustCrate.structured_product_into,
            owner,
            pointer(dd),
            Int32(ds),
            p,
            pointer(pd),
            _product_structure(a),
            q,
            pointer(qd),
            _product_structure(b),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, α)),
            iszero(beta) ? Ptr{UInt8}(C_NULL) : Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, β)),
        )
        return dest
    end
end

"""
    structured_mul!(C, A, B, α=true, β=false; dest=:full)

Compute `C = α*A*B + β*C` with faer's structured `matmul::triangular`, where
`A` and `B` may be dense native matrices or views, triangular wrappers or
diagonals, and `C` is a native matrix or range view. `dest` selects the block of
`C` that is computed and written: `:full`, `:lower` / `:upper` (including the
diagonal) or `:strict_lower` / `:strict_upper`; the rest of `C` is not touched.
Operands sharing `C`'s storage are snapshotted first. A zero `β` does not read
`C` and a zero `α` does not read the operands.
"""
function structured_mul!(
    c,
    a,
    b,
    alpha::Number = true,
    beta::Number = false;
    dest::Symbol = :full,
)
    haskey(_structured_destinations, dest) ||
        throw(ArgumentError("unknown structured destination $(repr(dest))"))
    c isa Union{FaerMatrix,FaerMatrixView} ||
        throw(ArgumentError("structured_mul! requires a native matrix destination"))
    return _structured_mul!(c, a, b, alpha, beta, Int32(_structured_destinations[dest]))
end
LinearAlgebra.mul!(
    dest::FaerVector,
    a::_NativeProductTriangular,
    b::_BorrowVector,
    alpha::Number,
    beta::Number,
) = _structured_mul!(dest, a, b, alpha, beta)
LinearAlgebra.mul!(dest::FaerVector, a::_NativeProductTriangular, b::_BorrowVector) =
    _structured_mul!(dest, a, b, true, false)

for (A, B) in (
    (:_ProductTriangular, :_ProductOperand),
    (:_ProductOperand, :_ProductTriangular),
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
