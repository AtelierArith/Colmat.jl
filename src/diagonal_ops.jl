# Native diagonal operations borrowing faer DiagRef/DiagMut. A FaerDiagonal is
# borrowed from its native owner; a Julia Diagonal (or a FaerDiagonal needing a
# scalar conversion) passes only its n stored entries, never an n×n snapshot.
const _DiagonalOperand=Union{FaerDiagonal,LinearAlgebra.Diagonal}
const _DiagonalDense=Union{FaerMatrix,FaerMatrixView}
const _DiagonalVector=Union{FaerVector,FaerVectorView}

function _dop_scalar(d, a)
    P=promote_type(eltype(d), eltype(a))
    P<:_NativeScalar ||
        throw(ArgumentError("diagonal operation requires native scalar types"))
    return P
end
_dop_values(d::FaerDiagonal, ::Type{P}) where {P} = Vector{P}(Vector(d))
_dop_values(d::LinearAlgebra.Diagonal, ::Type{P}) where {P} = Vector{P}(d.diag)
# Calls `owner(h, D, args...)` when D is a native FaerDiagonal of scalar type P
# that does not share h's handle; otherwise `values(h, ptr, n, args...)`.
function _dop_call(owner, values, h, d, ::Type{P}, args...) where {P}
    if d isa FaerDiagonal && eltype(d)===P && _handle(d)!==_native_argument(h)
        return _native_call(owner, h, d, args...)
    end
    v=_dop_values(d, P)
    GC.@preserve v return _native_call(
        values,
        h,
        Ptr{UInt8}(pointer(v)),
        UInt(length(v)),
        args...,
    )
end
_dop_check(info) =
    iszero(info) ? nothing : throw(LinearAlgebra.SingularException(Int(info)))
_dop_coerce(a, ::Type{P}) where {P} =
    eltype(a)===P ? a :
    a isa _PromotableDense ? _native_promote(a, P) :
    ndims(a)==1 ? FaerVector{P}(Vector(a)) : FaerMatrix{P}(Matrix(a))
# Native dense operands of a narrower scalar type are converted once in Rust,
# applying any transpose or conjugation, instead of through a Julia snapshot.
const _PromotableDense=Union{
    FaerMatrix,
    FaerMatrixView,
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
}
function _native_promote(a, ::Type{P}) where {P<:_NativeScalar}
    lock(_matrix_lock) do
        p, d=_view_source(a)
        dims=size(a)
        GC.@preserve d return P===Float64 ?
                              _matrix_result(
            RustCrate.view_promoted,
            dims,
            p,
            pointer(d),
            _tag(P),
        ) : _typed_result(P, RustCrate.view_promoted, dims, p, pointer(d), _tag(P))
    end
end

# Modes: 0 D*A, 1 A*D, 2 D\A, 3 A/D, 4 A+D, 5 A-D, 6 D-A.
function _dop_matrix(d, a, mode)
    lock(_matrix_lock) do
        P=_dop_scalar(d, a)
        n=size(d, 1)
        m, k=size(a)
        fits=mode in (0, 2) ? m==n : mode in (1, 3) ? k==n : m==n==k
        fits || throw(DimensionMismatch("diagonal and matrix dimensions must match"))
        aa=_dop_coerce(a, P)
        p, desc=_view_source(aa)
        h=P===Float64 ? RustCrate.OwnedMatrix() : RustCrate.TypedMatrix()
        info=GC.@preserve desc _dop_call(
            RustCrate.diagonal_matrix,
            RustCrate.diagonal_matrix_values,
            h,
            d,
            P,
            p,
            pointer(desc),
            Int32(mode),
        )
        _dop_check(info)
        return P===Float64 ? _owned(h, (m, k)) : FaerMatrix{P,typeof(h)}(h, (m, k))
    end
end
# Modes: 0 D*v (column), 1 r*D (row), 2 D\v (column), 3 r/D (row).
function _dop_vector(d, a, mode)
    lock(_matrix_lock) do
        P=_dop_scalar(d, a)
        n=size(d, 1)
        length(a)==n && size(a, mode in (0, 2) ? 1 : 2)==n ||
            throw(DimensionMismatch("diagonal and vector dimensions must match"))
        aa=_dop_coerce(a, P)
        p, desc=_view_source(aa)
        h=RustCrate.VectorOwner()
        info=GC.@preserve desc _dop_call(
            RustCrate.diagonal_vector,
            RustCrate.diagonal_vector_values,
            h,
            d,
            P,
            p,
            pointer(desc),
            Int32(mode),
        )
        _dop_check(info)
        C=mode in (0, 2) ? FaerVector : FaerRow
        return C{P,typeof(h)}(h, n, UInt(0))
    end
end

for (D, A) in ((:FaerDiagonal, :_DiagonalDense), (:(LinearAlgebra.Diagonal), :FaerMatrix))
    @eval begin
        Base.:\(d::$D, a::$A) = _dop_matrix(d, a, 2)
        Base.:/(a::$A, d::$D) = _dop_matrix(d, a, 3)
    end
end
Base.:+(d::FaerDiagonal, a::_DiagonalDense) = _dop_matrix(d, a, 4)
Base.:+(a::_DiagonalDense, d::FaerDiagonal) = _dop_matrix(d, a, 4)
Base.:-(a::_DiagonalDense, d::FaerDiagonal) = _dop_matrix(d, a, 5)
Base.:-(d::FaerDiagonal, a::_DiagonalDense) = _dop_matrix(d, a, 6)

# FaerDiagonal*FaerVector and FaerRow*FaerDiagonal are defined with the
# diagonal scaling products; views of vectors use the same native path here.
Base.:*(d::FaerDiagonal, v::FaerVectorView) = _dop_vector(d, v, 0)
Base.:\(d::FaerDiagonal, v::_DiagonalVector) = _dop_vector(d, v, 2)
Base.:/(r::FaerRow, d::FaerDiagonal) = _dop_vector(d, r, 3)

"""
    inv(D::FaerDiagonal)

Native reciprocal diagonal. A zero entry throws `SingularException` with its
index, as for `LinearAlgebra.Diagonal`.
"""
function Base.inv(d::FaerDiagonal)
    lock(_matrix_lock) do
        _handle(d)
        h=RustCrate.VectorOwner()
        _dop_check(_native_call(RustCrate.diagonal_inverse, h, d))
        return FaerDiagonal{eltype(d),typeof(h)}(h, d.len, UInt(0))
    end
end
Base.copy(x::LinearAlgebra.Adjoint{<:Any,<:FaerDiagonal}) = conj(parent(x))
Base.copy(x::LinearAlgebra.Transpose{<:Any,<:FaerDiagonal}) = copy(parent(x))

function _dop_of(a, C)
    lock(_matrix_lock) do
        p, desc=_view_source(a)
        GC.@preserve desc return _collection_result(
            C,
            eltype(a),
            minimum(size(a)),
            RustCrate.diagonal_of_view,
            p,
            pointer(desc),
            _collection_kind(C),
        )
    end
end
"""
    diag(A::FaerMatrix)
    FaerDiagonal(A::FaerMatrix)

Copy the main diagonal of a native matrix or range view natively into a
`FaerVector`, or into a `FaerDiagonal` owner. `diag(A, k)` with `k != 0` uses
the generic LinearAlgebra method.
"""
LinearAlgebra.diag(a::_DiagonalDense) = _dop_of(a, FaerVector)
FaerDiagonal(a::_DiagonalDense) = _dop_of(a, FaerDiagonal)

# In-place modes: 0 lmul!(D,A), 1 rmul!(A,D), 2 ldiv!(D,A), 3 rdiv!(A,D).
# Singular divisors throw before any write, so the destination is unchanged.
function _dop_update!(a, d, mode)
    lock(_matrix_lock) do
        _handle(a)
        n=size(d, 1)
        size(a, mode in (0, 2) ? 1 : 2)==n ||
            throw(DimensionMismatch("diagonal and destination dimensions must match"))
        # Entries convert to the destination type; nonreal values for a real
        # destination throw InexactError before the native call.
        P=eltype(a)
        f, g=a isa FaerMatrix ?
             (RustCrate.diagonal_update, RustCrate.diagonal_update_values) :
             (
            RustCrate.diagonal_collection_update,
            RustCrate.diagonal_collection_update_values,
        )
        _dop_check(_dop_call(f, g, a, d, P, Int32(mode)))
        return a
    end
end
# Views are computed out of place, then assigned; this also covers views
# sharing the diagonal's storage, such as `diagonalview(D)`.
function _dop_update_view!(a, d, mode)
    lock(_matrix_lock) do
        _view_owner(a)
        result=ndims(a)==1 ? _dop_vector(d, a, mode) : _dop_matrix(d, a, mode)
        copyto!(a, result)
        return a
    end
end
for D in (:FaerDiagonal, :(LinearAlgebra.Diagonal))
    @eval begin
        LinearAlgebra.lmul!(d::$D, a::Union{FaerMatrix,FaerVector}) =
            _dop_update!(a, d, 0)
        LinearAlgebra.rmul!(a::Union{FaerMatrix,FaerRow}, d::$D) =
            _dop_update!(a, d, 1)
        LinearAlgebra.ldiv!(d::$D, a::Union{FaerMatrix,FaerVector}) =
            _dop_update!(a, d, 2)
        LinearAlgebra.rdiv!(a::Union{FaerMatrix,FaerRow}, d::$D) =
            _dop_update!(a, d, 3)
    end
end
LinearAlgebra.lmul!(d::FaerDiagonal, a::FaerDiagonal) = _dop_update!(a, d, 0)
LinearAlgebra.rmul!(a::FaerDiagonal, d::FaerDiagonal) = _dop_update!(a, d, 1)
LinearAlgebra.ldiv!(d::FaerDiagonal, a::FaerDiagonal) = _dop_update!(a, d, 2)
LinearAlgebra.rdiv!(a::FaerDiagonal, d::FaerDiagonal) = _dop_update!(a, d, 3)
LinearAlgebra.lmul!(d::FaerDiagonal, a::FaerView) = _dop_update_view!(a, d, 0)
LinearAlgebra.rmul!(a::FaerMatrixView, d::FaerDiagonal) = _dop_update_view!(a, d, 1)
LinearAlgebra.ldiv!(d::FaerDiagonal, a::FaerView) = _dop_update_view!(a, d, 2)
LinearAlgebra.rdiv!(a::FaerMatrixView, d::FaerDiagonal) = _dop_update_view!(a, d, 3)
function LinearAlgebra.ldiv!(
    y::Union{FaerVector,FaerMatrix},
    d::FaerDiagonal,
    x::Union{_DiagonalVector,_DiagonalDense},
)
    lock(_matrix_lock) do
        size(y)==size(x) || throw(DimensionMismatch("destination and source must match"))
        result=ndims(x)==1 ? _dop_vector(d, x, 2) : _dop_matrix(d, x, 2)
        return copyto!(y, result)
    end
end

# Outer products return native matrices through borrowed descriptors.
const _VectorAdjoint=Union{
    LinearAlgebra.Adjoint{<:Any,<:_DiagonalVector},
    LinearAlgebra.Transpose{<:Any,<:_DiagonalVector},
}
Base.:*(a::_DiagonalVector, b::_VectorAdjoint) = _view_binary(a, b, 2)
