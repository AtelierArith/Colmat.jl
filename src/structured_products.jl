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

# Diagonal scaling borrows the stored diagonal and the dense (possibly
# transposed or adjoint) or native triangular operand in Rust; neither is
# expanded into a dense snapshot.
const _DiagonalDenseWrap=Union{
    LinearAlgebra.Transpose{<:Any,<:_ProductDense},
    LinearAlgebra.Adjoint{<:Any,<:_ProductDense},
}
const _DiagonalScaled=Union{_ProductDense,_DiagonalDenseWrap,_NativeProductTriangular}
_diagonal_scaling(a::_ProductDiagonal, b::_DiagonalScaled, ::Type{P}) where {P} =
    eltype(_product_parent(b))===P && promote_type(eltype(a), P)===P
_diagonal_scaling(a::_DiagonalScaled, b::_ProductDiagonal, ::Type{P}) where {P} =
    _diagonal_scaling(b, a, P)
_diagonal_scaling(a, b, ::Type) = false
# A Julia buffer crosses as a borrowed array (owner kind 4) for one call; it
# is not validated, so LinearAlgebra.Diagonal operands may hold non-finite
# values as before.
_diagonal_buffer(v::Vector) = (_view_source(FaerBorrowedArray(v))..., v)
function _diagonal_source(d::FaerDiagonal, ::Type{P}) where {P}
    _handle(d)
    eltype(d)===P || return _diagonal_buffer(Vector{P}(Vector(d)))
    return d, _descriptor(d, 1:d.len, 1:1), d
end
_diagonal_source(d::LinearAlgebra.Diagonal, ::Type{P}) where {P} =
    _diagonal_buffer(d.diag isa Vector{P} ? d.diag : Vector{P}(d.diag))
_diagonal_owner(x::FaerView) = _view_owner(x)
_diagonal_owner(x) = x
# Return (diagonal, dense operand, left?) in the product's operand order.
_diagonal_operands(a::_ProductDiagonal, b::Union{_DiagonalScaled,_BorrowVector}) =
    (a, b, true)
_diagonal_operands(a::Union{_DiagonalScaled,FaerRow}, b::_ProductDiagonal) = (b, a, false)
function _diagonal_call(f, ::Type{P}, a, b, dest...) where {P}
    g, x, left=_diagonal_operands(a, b)
    dp, dd, dkeep=_diagonal_source(g, P)
    y=_product_parent(x)
    structure=x isa _ProductTriangular ? _product_structure(x) : Int32(0)
    q, qd=_view_source(y)
    if !isempty(dest)
        owner=_diagonal_owner(dest[1])
        # mul!(A, D, A), overlapping views and transposed operands sharing the
        # destination: snapshot the operand natively.
        if owner===q
            y=_unalias(y)
            q, qd=_view_source(y)
        end
        # A destination viewing the diagonal's own storage (`diagonalview`).
        owner===g && ((dp, dd, dkeep)=_diagonal_buffer(Vector{P}(Vector(g))))
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
# Operands are borrowed: native owners and ranges through view descriptors,
# strided Julia buffers of the product type as borrowed arrays (kind 4). Only conversions and non-strided Julia
# inputs are copied. Returns (pointer or owner, descriptor, retained object).
# Strided Julia buffers cross as #152's kind-4 borrowed arrays (read-only, so
# repeated elements are allowed); other inputs are copied once.
function _raw_operand(a::AbstractMatrix{P}) where {P<:_NativeScalar}
    a isa StridedMatrix && _julia_borrowable(a) || (a=Matrix{P}(a))
    b=FaerBorrowedArray(a)
    p, d=_view_source(b)
    return p, d, b
end
const _ProductNativeSource=Union{
    FaerMatrix,
    FaerView,
    FaerVector,
    FaerRow,
    FaerBorrowedArray,
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
        eltype(y)===P || (y=_dop_coerce(y, P))
        p, d=_view_source(y)
        return p, d, y
    end
    y isa AbstractVector && (y=reshape(y, :, 1))
    return _raw_operand(eltype(y)===P ? y : Matrix{P}(y))
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
    lock(_matrix_lock) do
        P=_structured_scalar(a, b)
        _diagonal_scaling(a, b, P) && return _diagonal_product(a, b, P)
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
        ds==0 &&
            _diagonal_scaling(a, b, eltype(dest)) &&
            return _diagonal_mul!(dest, a, b, alpha, beta)
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
    (:_ProductDiagonal, :_DiagonalDenseWrap),
    (:_DiagonalDenseWrap, :_ProductDiagonal),
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
