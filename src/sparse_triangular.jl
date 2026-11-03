const _SparseTriangularParent{T}=Union{
    FaerSparseMatrix{T},
    LinearAlgebra.Transpose{T,<:FaerSparseMatrix{T}},
    LinearAlgebra.Adjoint{T,<:FaerSparseMatrix{T}},
}
const _SparseTriangular=Union{
    LinearAlgebra.LowerTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UpperTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:_SparseTriangularParent},
}
function _sparse_triangular_solve(a::_SparseTriangular, b::AbstractMatrix)
    lock(_matrix_lock) do
        size(a, 1)==size(b, 1) ||
            throw(DimensionMismatch("sparse triangular solve dimensions must match"))
        source=parent(a) isa FaerSparseMatrix ? parent(a) : copy(parent(a))
        _handle(source)
        if !_isunit(a)
            i=Int(_native_call(RustCrate.sparse_zero_diagonal, source))
            iszero(i) || throw(LinearAlgebra.SingularException(i))
        end
        T=promote_type(eltype(a), _sparse_input_type(eltype(b)))
        sparse=_sparse_coerce(source, T)
        rhs=b isa FaerMatrix ? _coerce(b, T) : FaerMatrix{T}(b)
        args=(sparse, rhs, _islower(a), _isunit(a))
        return T===Float64 ?
               _matrix_result(RustCrate.sparse_triangular_from, size(rhs), args...) :
               _typed_result(T, RustCrate.sparse_triangular_from, size(rhs), args...)
    end
end
function _sparse_triangular_vector(a, b)
    result=_sparse_triangular_solve(a, reshape(b, :, 1))
    return b isa _BorrowVector ? _vector_from_source(result) : vec(Matrix(result))
end
for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    @eval begin
        Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::AbstractMatrix{<:Number},
        ) where {T<:_NativeScalar} = _sparse_triangular_solve(a, b)
        Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::AbstractVector{<:Number},
        ) where {T<:_NativeScalar} = _sparse_triangular_vector(a, b)
        Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::SparseArrays.AbstractCompressedVector{<:Number},
        ) where {T<:_NativeScalar} = _sparse_triangular_vector(a, b)
    end
end
function LinearAlgebra.ldiv!(a::_SparseTriangular, b::Union{AbstractVector,AbstractMatrix})
    return copyto!(b, a\b)
end
function LinearAlgebra.ldiv!(
    dest::Union{AbstractVector,AbstractMatrix},
    a::_SparseTriangular,
    b::Union{AbstractVector,AbstractMatrix},
)
    size(dest)==size(b) || throw(DimensionMismatch("triangular output size must match RHS"))
    return copyto!(dest, a\b)
end

# Resolve structured RHS methods provided by LinearAlgebra and SparseArrays.
for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    for R in (
        LinearAlgebra.LowerTriangular,
        LinearAlgebra.UpperTriangular,
        LinearAlgebra.UnitLowerTriangular,
        LinearAlgebra.UnitUpperTriangular,
        LinearAlgebra.Bidiagonal,
        LinearAlgebra.UpperHessenberg,
        SparseArrays.AbstractSparseMatrixCSC,
    )
        @eval Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::$R{<:Number},
        ) where {T<:_NativeScalar} = _sparse_triangular_solve(a, b)
    end
end
