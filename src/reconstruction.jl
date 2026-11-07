const _ReconstructableDense=Union{
    FaerFactorization,
    FaerHessenberg,
    FaerCompactHessenberg,
    _NativeSchur,
    FaerTridiagonalization,
    FaerBidiagonalization,
}
const _SparseLU=FaerSparseFactorization{T,:lu} where {T}
const _ReconstructableSparse=Union{_SparseLU,_SparseCholesky}

# Typed reconstruction goes through one independent native reconstruction,
# which is closed unless it is already the requested result.
function _typed_dense_reconstruction(::Type{T}, f) where {T<:_NativeScalar}
    native=FaerMatrix(f)
    T===eltype(native) && return native
    try
        return FaerMatrix{T}(native)
    finally
        close(native)
    end
end
function _convert_dense_reconstruction(convert_result, f)
    native=FaerMatrix(f)
    try
        return convert_result(native)
    finally
        close(native)
    end
end
FaerMatrix{T}(f::_ReconstructableDense) where {T<:_NativeScalar} =
    _typed_dense_reconstruction(T, f)
Base.Matrix{T}(f::_ReconstructableDense) where {T} =
    _convert_dense_reconstruction(Matrix{T}, f)
Base.Array(f::_NativeSchur) = Matrix(f)

"""
    FaerSparseMatrix(F::FaerSparseFactorization{T,:lu})

Reconstruct independent native CSC storage from saved LU factors for every algorithm
and inverse row/column permutations. Uses current numeric factors, including
updates; original input snapshots are not used. Sparse QR supports independent dense reconstruction through `reconstruct(F)`.
"""
function FaerSparseMatrix(f::_SparseLU)
    lock(_matrix_lock) do
        _handle(f)
        LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
        L=U=p=q=pi=qi=product=rows=nothing
        try
            L=f.L
            U=f.U
            p=f.p
            q=f.q
            pi=invperm(p)
            qi=invperm(q)
            product=L*U
            rows=permute_rows(product, pi)
            return permute_cols(rows, qi)
        finally
            for value in (L, U, p, q, pi, qi, product, rows)
                isnothing(value) || close(value)
            end
        end
    end
end
function _convert_sparse_reconstruction(convert_result, f)
    native=FaerSparseMatrix(f)
    try
        return convert_result(native)
    finally
        close(native)
    end
end
FaerSparseMatrix{T}(f::_SparseLU) where {T<:_NativeScalar} =
    _convert_sparse_reconstruction(FaerSparseMatrix{T}, f)
FaerMatrix(f::_SparseLU) = _convert_sparse_reconstruction(FaerMatrix, f)
FaerMatrix{T}(f::_SparseLU) where {T<:_NativeScalar} =
    _convert_sparse_reconstruction(FaerMatrix{T}, f)
Base.Matrix(f::_SparseLU) = _convert_sparse_reconstruction(Matrix, f)
Base.Matrix{T}(f::_SparseLU) where {T} = _convert_sparse_reconstruction(Matrix{T}, f)
Base.Array(f::_SparseLU) = Matrix(f)
SparseArrays.SparseMatrixCSC(f::_SparseLU) =
    _convert_sparse_reconstruction(SparseArrays.SparseMatrixCSC, f)
SparseArrays.SparseMatrixCSC{T,I}(f::_ReconstructableSparse) where {T,I<:Integer} =
    _convert_sparse_reconstruction(SparseArrays.SparseMatrixCSC{T,I}, f)
SparseArrays.SparseMatrixCSC{T}(f::_ReconstructableSparse) where {T} =
    SparseArrays.SparseMatrixCSC{T,Int}(f)
SparseArrays.sparse(f::_SparseLU) = SparseArrays.SparseMatrixCSC(f)

function _reconstruct_product(left, middle, right)
    temp=left*middle
    try
        return temp*right'
    finally
        close(temp)
    end
end
"""
    reconstruct(F)

Reconstruct independent native matrices from owned factors. Dense LU/QR/
Cholesky/LDLT (including the in-place `lu!/qr!/cholesky!(A; scratch)` results),
Schur/Hessenberg and tridiagonal/bidiagonal reductions return a
`FaerMatrix`; retained spectral factors also support reconstruction. Sparse Cholesky and explicit simplicial LU return a
`FaerSparseMatrix` in original row/column order. Explicit simplicial sparse QR
returns a dense `FaerMatrix` in original column order. Generalized Schur/Hessenberg
return `(A=...,B=...)` with `A=Q*S*Z'` (or `Q*H*Z'`) and `B=Q*T*Z'`.
Inputs and factors are preserved; returned owners survive closing F.
"""
reconstruct(f::_ReconstructableDense) = FaerMatrix(f)
reconstruct(f::_ReconstructableSparse) = FaerSparseMatrix(f)
function reconstruct(f::_NativePairFactor)
    lock(_matrix_lock) do
        _check_pair_factor(f)
        A=_reconstruct_product(f.Q, _pair_first(f), f.Z)
        B=try
            _reconstruct_product(f.Q, f.T, f.Z)
        catch
            close(A)
            rethrow()
        end
        return (A = A, B = B)
    end
end
