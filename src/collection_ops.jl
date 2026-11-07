# Remaining native operations of owned Row/Col/Diag collections. They reuse
# the diagonal kernels of diagonal_ops.jl and structured_products.jl: a native
# operand gives a native result, and Julia `Diagonal` operands cross as their n
# stored entries only.

# Diagonal with diagonal operands returns a FaerDiagonal, as Julia's
# Diagonal-with-Diagonal operations return a Diagonal.
_faer_diagonal(x::FaerDiagonal, ::Type{P}) where {P} = _collection_coerce(x, P)
_faer_diagonal(x::LinearAlgebra.Diagonal, ::Type{P}) where {P} = FaerDiagonal{P}(x.diag)
function _diagonal_pair(a, b, op)
    P=_dop_scalar(a, b)
    size(a)==size(b) || throw(DimensionMismatch("diagonal sizes must match"))
    return _collection_binary(_faer_diagonal(a, P), _faer_diagonal(b, P), op)
end
# Modes as in `_dop_update!`: 2 `d \ b`, 3 `b / d`. The quotient is computed in
# place in a fresh promoted copy of `b`; a zero divisor throws
# `SingularException` before anything is written.
function _diagonal_quotient(d, b, mode)
    lock(_matrix_lock) do
        P=_dop_scalar(d, b)
        size(d)==size(b) || throw(DimensionMismatch("diagonal sizes must match"))
        x=b isa FaerDiagonal && eltype(b)===P ? copy(b) : _faer_diagonal(b, P)
        return _dop_update!(x, d, mode)
    end
end
for (f, op) in ((:+, 0), (:-, 1), (:*, 2))
    @eval begin
        Base.$f(a::FaerDiagonal, b::LinearAlgebra.Diagonal) = _diagonal_pair(a, b, $op)
        Base.$f(a::LinearAlgebra.Diagonal, b::FaerDiagonal) = _diagonal_pair(a, b, $op)
    end
end
Base.:\(d::FaerDiagonal, b::Union{FaerDiagonal,LinearAlgebra.Diagonal}) =
    _diagonal_quotient(d, b, 2)
Base.:\(d::LinearAlgebra.Diagonal, b::FaerDiagonal) = _diagonal_quotient(d, b, 2)
Base.:/(b::Union{FaerDiagonal,LinearAlgebra.Diagonal}, d::FaerDiagonal) =
    _diagonal_quotient(d, b, 3)
Base.:/(b::FaerDiagonal, d::LinearAlgebra.Diagonal) = _diagonal_quotient(d, b, 3)

# `transpose(D)` is `D` and `D'` is `conj(D)` (one O(n) native copy for complex
# entries), so wrapped diagonals use the native diagonal methods.
const _WrappedFaerDiagonal=Union{
    LinearAlgebra.Transpose{<:Any,<:FaerDiagonal},
    LinearAlgebra.Adjoint{<:Any,<:FaerDiagonal},
}
_undiag(x::LinearAlgebra.Transpose{<:Any,<:FaerDiagonal}) = parent(x)
_undiag(x::LinearAlgebra.Adjoint{<:Any,<:FaerDiagonal}) =
    eltype(x)<:Real ? parent(x) : conj(parent(x))
_undiag(x) = x
const _DiagonalPeer=Union{FaerDiagonal,LinearAlgebra.Diagonal}
const _WrappedDiagonalRight=Union{
    _DiagonalScaled,
    _BorrowVector,
    _DiagonalPeer,
    _WrappedFaerDiagonal,
}
const _WrappedDiagonalLeft=Union{_DiagonalScaled,FaerRow,_DiagonalPeer}
Base.:*(a::_WrappedFaerDiagonal, b::_WrappedDiagonalRight) = _undiag(a)*_undiag(b)
Base.:*(a::_WrappedDiagonalLeft, b::_WrappedFaerDiagonal) = a*_undiag(b)
Base.:\(a::_WrappedFaerDiagonal, b::Union{_DiagonalDense,_BorrowVector,_DiagonalPeer}) =
    _undiag(a)\b
Base.:/(a::Union{_DiagonalDense,FaerRow,_DiagonalPeer}, b::_WrappedFaerDiagonal) =
    a/_undiag(b)
for f in (:+, :-)
    @eval begin
        Base.$f(
            a::_WrappedFaerDiagonal,
            b::Union{_DiagonalDense,_DiagonalPeer,_WrappedFaerDiagonal},
        ) = $f(_undiag(a), _undiag(b))
        Base.$f(a::Union{_DiagonalDense,_DiagonalPeer}, b::_WrappedFaerDiagonal) =
            $f(a, _undiag(b))
    end
end
LinearAlgebra.mul!(
    c::_ProductDense,
    a::_WrappedFaerDiagonal,
    b::_DiagonalScaled,
    alpha::Number,
    beta::Number,
) = LinearAlgebra.mul!(c, _undiag(a), b, alpha, beta)
LinearAlgebra.mul!(
    c::_ProductDense,
    a::_DiagonalScaled,
    b::_WrappedFaerDiagonal,
    alpha::Number,
    beta::Number,
) = LinearAlgebra.mul!(c, a, _undiag(b), alpha, beta)
LinearAlgebra.mul!(c::_ProductDense, a::_WrappedFaerDiagonal, b::_DiagonalScaled) =
    LinearAlgebra.mul!(c, _undiag(a), b, true, false)
LinearAlgebra.mul!(c::_ProductDense, a::_DiagonalScaled, b::_WrappedFaerDiagonal) =
    LinearAlgebra.mul!(c, a, _undiag(b), true, false)

# Julia diagonals scale native columns and rows like FaerDiagonal.
Base.:*(a::LinearAlgebra.Diagonal, b::FaerVector) = _diagonal_vector(a, b, FaerVector)
Base.:*(a::FaerRow, b::LinearAlgebra.Diagonal) = _diagonal_vector(a, b, FaerRow)

# `D*x` into native column destinations and `r*D` into native rows write
# natively from the borrowed diagonal and operand (`diagonal_scale_into`).
# Operands sharing the destination's storage, including `diagonalview(D)`, are
# read from a snapshot. Promotion beyond the destination type is computed
# out of place and converted on assignment, as in LinearAlgebra.
function _diagonal_vector_mul!(dest, a, b, alpha, beta)
    lock(_matrix_lock) do
        g, x, _=_diagonal_operands(a, b)
        n=size(g, 1)
        length(x)==n && length(dest)==n ||
            throw(DimensionMismatch("diagonal product dimensions must match"))
        for y in (dest, x, g)
            y isa _OwnedCollection && _handle(y)
            y isa FaerView && _view_owner(y)
        end
        Q=eltype(dest)
        eltype(x)===Q && promote_type(eltype(g), Q)===Q &&
            return _diagonal_mul!(dest, a, b, alpha, beta)
        P=promote_type(eltype(a), eltype(b), Q, typeof(alpha), typeof(beta))
        values=iszero(alpha) ? zeros(P, size(dest)) : alpha .* Array{P}(Array(a*b))
        iszero(beta) || (values=values .+ beta .* Array(dest))
        return copyto!(dest, values)
    end
end
const _DiagonalColumnDest=Union{FaerVector,FaerVectorView}
for D in (:FaerDiagonal, :(LinearAlgebra.Diagonal))
    @eval begin
        LinearAlgebra.mul!(
            y::_DiagonalColumnDest,
            d::$D,
            x::_BorrowVector,
            alpha::Number,
            beta::Number,
        ) = _diagonal_vector_mul!(y, d, x, alpha, beta)
        LinearAlgebra.mul!(y::_DiagonalColumnDest, d::$D, x::_BorrowVector) =
            _diagonal_vector_mul!(y, d, x, true, false)
        LinearAlgebra.mul!(z::FaerRow, r::FaerRow, d::$D, alpha::Number, beta::Number) =
            _diagonal_vector_mul!(z, r, d, alpha, beta)
        LinearAlgebra.mul!(z::FaerRow, r::FaerRow, d::$D) =
            _diagonal_vector_mul!(z, r, d, true, false)
    end
end

# Row vectors times or over a diagonal keep LinearAlgebra's wrapper type:
# `v'*D == (D'*v)'` and `transpose(v)*D == transpose(D*v)`, computed natively.
_adjoint_diagonal(d::FaerDiagonal) = _undiag(d')
_adjoint_diagonal(d::LinearAlgebra.Diagonal) = LinearAlgebra.Diagonal(conj(d.diag))
const _AdjointColumn=LinearAlgebra.Adjoint{<:Any,<:_BorrowVector}
const _TransposeColumn=LinearAlgebra.Transpose{<:Any,<:_BorrowVector}
_column(x::_BorrowVector) = x isa FaerVector ? x : copy(x)
for D in (:FaerDiagonal, :(LinearAlgebra.Diagonal))
    @eval begin
        Base.:*(x::_AdjointColumn, d::$D) =
            adjoint(_adjoint_diagonal(d)*_column(parent(x)))
        Base.:*(x::_TransposeColumn, d::$D) = transpose(d*_column(parent(x)))
        Base.:/(x::_AdjointColumn, d::$D) =
            adjoint(_adjoint_diagonal(d)\_column(parent(x)))
        Base.:/(x::_TransposeColumn, d::$D) = transpose(d\_column(parent(x)))
    end
end

# faer's `is_all_finite` / `has_nan` over the stored entries; implicit zeros of
# a diagonal are finite and not NaN.
function _collection_predicate(x::_OwnedCollection, code)
    lock(_matrix_lock) do
        _handle(x)
        d=x isa FaerRow ? _descriptor(x, 1:1, 1:x.len) : _descriptor(x, 1:x.len, 1:1)
        GC.@preserve d return _native_call(RustCrate.view_predicate, x, pointer(d), code)
    end
end
Base.all(::typeof(isfinite), x::_OwnedCollection; dims = :) =
    dims isa Colon ? _collection_predicate(x, Int32(0)) : all(isfinite, Array(x); dims)
Base.any(::typeof(isnan), x::_OwnedCollection; dims = :) =
    dims isa Colon ? _collection_predicate(x, Int32(1)) : any(isnan, Array(x); dims)

"""
    map(f, D::FaerDiagonal)

Apply `f` to the stored entries with one bulk read. As for `LinearAlgebra.Diagonal`,
the result is diagonal when `f(zero(eltype(D)))` is zero, and dense otherwise; it
is a native `FaerDiagonal`/`FaerMatrix` for the four native scalar types and a
Julia `Diagonal`/`Matrix` for other result types.
"""
function Base.map(f, d::FaerDiagonal)
    values=map(f, Vector(d))
    z=f(zero(eltype(d)))
    R=promote_type(eltype(values), typeof(z))
    if iszero(z)
        R<:_NativeScalar && return FaerDiagonal{R}(values)
        return LinearAlgebra.Diagonal(Vector{R}(values))
    end
    out=fill(R(z), size(d))
    for i in eachindex(values)
        out[i, i]=values[i]
    end
    return R<:_NativeScalar ? FaerMatrix{R}(out) : out
end

# A row plus or minus an adjoint or transposed column is a native FaerRow.
const _ColumnRow=Union{_AdjointColumn,_TransposeColumn}
_row(x::_AdjointColumn) = copy(adjoint(_column(parent(x))))
_row(x::_TransposeColumn) = copy(transpose(_column(parent(x))))
for (f, op) in ((:+, 0), (:-, 1))
    @eval begin
        Base.$f(a::FaerRow, b::_ColumnRow) = _collection_binary(a, _row(b), $op)
        Base.$f(a::_ColumnRow, b::FaerRow) = _collection_binary(_row(a), b, $op)
    end
end
# A column times a row is a native outer product.
Base.:*(a::_BorrowVector, b::FaerRow) = _view_binary(a, b, 2)
