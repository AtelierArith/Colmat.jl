# Products of native sparse owners and their `transpose`/`adjoint` wrappers.
#
# Every owner is read in place through one column-major faer view (a CSR owner
# as its transpose); the wrapper selects the orientation and a conjugate flag.
# Sparse×dense products and in-place `mul!` never copy the sparse operand.
# Sparse×sparse products need column-major faer operands: same-orientation
# pairs are read in place (two row-major operands give `Cᵀ = op(B)ᵀ op(A)ᵀ`,
# returned as CSR); mixed pairs convert the right operand once in Rust.

# A native sparse owner or its transpose/adjoint.
const _OwnerSparseOperand=Union{FaerSparseMatrix,_SparseTransform}
_sparse_orientation(::FaerSparseMatrix) = Int32(0)
_sparse_stored(a::FaerSparseMatrix) = a
_sparse_stored(a::_SparseTransform) = parent(a)
_owner_sparse_promote(a::FaerSparseMatrix, ::Type{P}) where {P} = _sparse_coerce(a, P)
_owner_sparse_promote(a::LinearAlgebra.Transpose{<:Any,<:FaerSparseMatrix}, ::Type{P}) where {P} =
    transpose(_sparse_coerce(parent(a), P))
_owner_sparse_promote(a::LinearAlgebra.Adjoint{<:Any,<:FaerSparseMatrix}, ::Type{P}) where {P} =
    adjoint(_sparse_coerce(parent(a), P))
# Storage format of `op(A)` when read in place: CSC exactly when the stored
# column-major view is `op(A)` itself.
_oriented_format(a::_OwnerSparseOperand) =
    (_sparse_stored(a).format===:csc)==(a isa FaerSparseMatrix) ? :csc : :csr

function _sparse_oriented_product(a::_OwnerSparseOperand, b::_OwnerSparseOperand)
    lock(_matrix_lock) do
        _handle(_sparse_stored(a))
        _handle(_sparse_stored(b))
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("sparse operation dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        aa, bb=_owner_sparse_promote(a, P), _owner_sparse_promote(b, P)
        return _sparse_result(
            P,
            (size(a, 1), size(b, 2)),
            _oriented_format(aa),
            RustCrate.sparse_oriented_product,
            _sparse_stored(aa),
            _sparse_orientation(aa),
            _sparse_stored(bb),
            _sparse_orientation(bb),
        )
    end
end
for (A, B) in (
    (:_SparseTransform, :FaerSparseMatrix),
    (:FaerSparseMatrix, :_SparseTransform),
    (:_SparseTransform, :_SparseTransform),
)
    @eval Base.:*(a::$A, b::$B) = _sparse_oriented_product(a, b)
end

# Dense operands read in place by the owner kernels. Plain `FaerMatrix`
# operands of a plain owner keep the dedicated `sparse_product_from` method.
const _OwnerSparseDense=Union{FaerMatrixView,_BorrowTranspose,FaerBorrowedMatrix,_BorrowedTranspose}

function _owner_sparse_product(a::_OwnerSparseOperand, b, left::Bool)
    lock(_matrix_lock) do
        _handle(_sparse_stored(a))
        (left ? size(a, 2)==size(b, 1) : size(b, 2)==size(a, 1)) ||
            throw(DimensionMismatch("sparse/dense dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        aa=_owner_sparse_promote(a, P)
        bb=_borrowed_dense_promote(b, P)
        q, d=_view_source(bb)
        dims=left ? (size(a, 1), size(b, 2)) : (size(b, 1), size(a, 2))
        f=RustCrate.sparse_owner_product
        s, op=_sparse_stored(aa), _sparse_orientation(aa)
        GC.@preserve d return P===Float64 ?
                              _matrix_result(f, dims, s, op, q, pointer(d), left) :
                              _typed_result(P, f, dims, s, op, q, pointer(d), left)
    end
end
for D in (:FaerMatrix, :_OwnerSparseDense)
    @eval begin
        Base.:*(a::_SparseTransform, b::$D) = _owner_sparse_product(a, b, true)
        Base.:*(a::$D, b::_SparseTransform) = _owner_sparse_product(b, a, false)
    end
end
Base.:*(a::FaerSparseMatrix, b::_OwnerSparseDense) = _owner_sparse_product(a, b, true)
Base.:*(a::_OwnerSparseDense, b::FaerSparseMatrix) = _owner_sparse_product(b, a, false)
Base.:*(a::_SparseTransform, b::_BorrowVector) =
    _vector_from_source(_owner_sparse_product(a, b, true))
Base.:*(a::_SparseTransform, b::StridedMatrix{<:Number}) =
    a*FaerMatrix{promote_type(eltype(a), eltype(b))}(b)
Base.:*(a::StridedMatrix{<:Number}, b::_SparseTransform) =
    FaerMatrix{promote_type(eltype(a), eltype(b))}(a)*b
Base.:*(a::_SparseTransform, b::StridedVector{<:Number}) =
    Vector(a*FaerVector{promote_type(eltype(a), eltype(b))}(b))

function _sparse_dense_mul_call!(c, s::_OwnerSparseOperand, q, desc, y, z, acc, left)
    _handle(_sparse_stored(s))
    _native_call(
        RustCrate.sparse_owner_mul,
        c,
        _sparse_stored(s),
        _sparse_orientation(s),
        q,
        desc,
        y,
        z,
        acc,
        left,
    )
end
const _OwnerMulDense=Union{_BorrowMatrix,_BorrowTranspose,FaerBorrowedMatrix,_BorrowedTranspose}
LinearAlgebra.mul!(
    c::FaerMatrix,
    a::_OwnerSparseOperand,
    b::_OwnerMulDense,
    alpha::Number,
    beta::Number,
) = _borrowed_sparse_mul!(c, a, b, alpha, beta, true)
LinearAlgebra.mul!(
    c::FaerMatrix,
    a::_OwnerMulDense,
    b::_OwnerSparseOperand,
    alpha::Number,
    beta::Number,
) = _borrowed_sparse_mul!(c, a, b, alpha, beta, false)
LinearAlgebra.mul!(c::FaerMatrix, a::_OwnerSparseOperand, b::_OwnerMulDense) =
    _borrowed_sparse_mul!(c, a, b, true, false, true)
LinearAlgebra.mul!(c::FaerMatrix, a::_OwnerMulDense, b::_OwnerSparseOperand) =
    _borrowed_sparse_mul!(c, a, b, true, false, false)
