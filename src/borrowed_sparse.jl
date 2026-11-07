"""
    faerview(S::SparseMatrixCSC{T,Ti}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}

Borrow a Julia CSC matrix for native faer operations. The returned
[`FaerBorrowedSparseMatrix`](@ref) keeps `S` alive and hands Rust a faer
`SparseColMatRef<usize,T>` during each native call only.

Stored values are read in place from `nonzeros(S)` (zero-copy). faer's sparse
structure is always 0-based, so Julia's 1-based `getcolptr(S)` / `rowvals(S)`
cannot be borrowed as they are: each native call converts them into two
`UInt` buffers owned by the view and reused across calls (`n+1` and `nnz`
words; no allocation once their sizes are reached) and validates them
(pointers monotone, rows in bounds, sorted and unique) before Rust reads them.
Because nothing is cached across calls, value and structural changes to `S`
(for example `S[i,j]=x` inserting an entry) are seen by the next call. Any
`Ti<:Integer` index type is accepted; faer receives `usize` indices.
"""
function faerview(a::SparseArrays.SparseMatrixCSC{T,Ti}) where {T,Ti}
    T<:_NativeScalar || throw(
        ArgumentError("faerview requires Float32, Float64, ComplexF32 or ComplexF64"),
    )
    b=FaerBorrowedSparseMatrix{T,Ti}(a, UInt[], UInt[])
    lock(() -> _borrowed_indices!(b), _matrix_lock)
    return b
end

"""
    FaerBorrowedSparseMatrix{T,Ti}

A borrowed Julia `SparseMatrixCSC{T,Ti}` created by `faerview(S)`. Indexing,
`nnz`, `nonzeros`, `rowvals`, `getcolptr` and `nzrange` forward to the parent
(`nonzeros(B) === nonzeros(S)`). Native consumers read it as a faer
`SparseColMatRef`: products with dense native, view, transposed and borrowed
Julia operands (`*` and `mul!` into a `FaerMatrix`, both sides), `copy` /
`FaerSparseMatrix` / `FaerSparsity` (one owned copy), and the sparse `lu`,
`qr`, `cholesky`, `factorize(plan, B)`, `FaerSparseSymbolic(B)`, `B \\ x` and
the `lu!`/`qr!`/`cholesky!` refactorizations. Factorizations still keep their
own owned snapshot, so later writes to `S` do not change them.
"""
struct FaerBorrowedSparseMatrix{T,Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{T,Ti}
    parent::SparseArrays.SparseMatrixCSC{T,Ti}
    # 0-based copies of colptr/rowval, rewritten under `_matrix_lock` per call.
    colptr::Vector{UInt}
    rowval::Vector{UInt}
end
faerview(b::FaerBorrowedSparseMatrix) = b
Base.parent(b::FaerBorrowedSparseMatrix) = b.parent
Base.size(b::FaerBorrowedSparseMatrix) = size(b.parent)
Base.@propagate_inbounds Base.getindex(b::FaerBorrowedSparseMatrix, i::Int, j::Int) =
    b.parent[i, j]
Base.@propagate_inbounds Base.setindex!(b::FaerBorrowedSparseMatrix, x, i::Int, j::Int) =
    (b.parent[i, j]=x; b)
SparseArrays.nnz(b::FaerBorrowedSparseMatrix) = SparseArrays.nnz(b.parent)
SparseArrays.nonzeros(b::FaerBorrowedSparseMatrix) = SparseArrays.nonzeros(b.parent)
SparseArrays.rowvals(b::FaerBorrowedSparseMatrix) = SparseArrays.rowvals(b.parent)
SparseArrays.getcolptr(b::FaerBorrowedSparseMatrix) = SparseArrays.getcolptr(b.parent)
SparseArrays.nzrange(b::FaerBorrowedSparseMatrix, j::Integer) =
    SparseArrays.nzrange(b.parent, j)
SparseArrays.findnz(b::FaerBorrowedSparseMatrix) = SparseArrays.findnz(b.parent)
SparseArrays.SparseMatrixCSC(b::FaerBorrowedSparseMatrix) = copy(b.parent)
Base.Matrix(b::FaerBorrowedSparseMatrix) = Matrix(b.parent)
Base.dataids(b::FaerBorrowedSparseMatrix) = (
    Base.dataids(SparseArrays.getcolptr(b.parent))...,
    Base.dataids(SparseArrays.rowvals(b.parent))...,
    Base.dataids(SparseArrays.nonzeros(b.parent))...,
)
LinearAlgebra.ishermitian(b::FaerBorrowedSparseMatrix) = LinearAlgebra.ishermitian(b.parent)
LinearAlgebra.issymmetric(b::FaerBorrowedSparseMatrix) = LinearAlgebra.issymmetric(b.parent)
function Base.show(io::IO, m::MIME"text/plain", b::FaerBorrowedSparseMatrix)
    println(io, "faerview of ", summary(b.parent), ":")
    show(io, m, b.parent)
end

# Rewrite the 0-based buffers from the parent's current structure and validate
# it as FaerSparseMatrix(S) does. Must run under `_matrix_lock`.
function _borrowed_indices!(b::FaerBorrowedSparseMatrix)
    a=b.parent
    m, n=size(a)
    cp=SparseArrays.getcolptr(a)
    rv=SparseArrays.rowvals(a)
    length(cp)==n+1 && first(cp)==1 || throw(ArgumentError("invalid CSC column pointers"))
    k=Int(last(cp))-1
    0<=k<=length(rv) && k<=length(SparseArrays.nonzeros(a)) ||
        throw(ArgumentError("invalid CSC stored count"))
    p=resize!(b.colptr, n+1)
    i=resize!(b.rowval, k)
    p[1]=0
    @inbounds for j in 1:n
        lo, hi=Int(cp[j]), Int(cp[j+1])
        lo<=hi<=k+1 || throw(ArgumentError("invalid CSC column pointers"))
        p[j+1]=hi-1
        previous=0
        for q in lo:(hi-1)
            r=Int(rv[q])
            previous<r<=m ||
                throw(ArgumentError("CSC row indices must be in bounds, sorted and unique"))
            i[q]=r-1
            previous=r
        end
    end
    return m, n, k
end
# Run `f(raw...)` with the validated borrowed parts `(m, n, colptr, rowval,
# nzval, nnz, tag)`; the parent's vectors and the buffers stay rooted.
function _with_borrowed_sparse(f, b::FaerBorrowedSparseMatrix{T}) where {T}
    lock(_matrix_lock) do
        m, n, k=_borrowed_indices!(b)
        values=SparseArrays.nonzeros(b.parent)
        p, i=b.colptr, b.rowval
        GC.@preserve values p i return f(
            UInt(m),
            UInt(n),
            pointer(p),
            pointer(i),
            Ptr{UInt8}(pointer(values)),
            UInt(k),
            _tag(T),
        )
    end
end

function FaerSparseMatrix(b::FaerBorrowedSparseMatrix{T}; format = :csc) where {T}
    _sparse_format(format)
    owned=_with_borrowed_sparse(b) do raw...
        _sparse_result(T, size(b), :csc, RustCrate.sparse_from_borrowed, raw...)
    end
    format===:csc && return owned
    converted=copy(owned; format)
    close(owned)
    return converted
end
FaerSparseMatrix{T}(b::FaerBorrowedSparseMatrix{T}; kwargs...) where {T<:_NativeScalar} =
    FaerSparseMatrix(b; kwargs...)
FaerSparseMatrix{T}(b::FaerBorrowedSparseMatrix; kwargs...) where {T<:_NativeScalar} =
    FaerSparseMatrix{T}(b.parent; kwargs...)
Base.copy(b::FaerBorrowedSparseMatrix) = FaerSparseMatrix(b)
FaerMatrix(b::FaerBorrowedSparseMatrix) = FaerMatrix(Matrix(b.parent))
function FaerSparsity(b::FaerBorrowedSparseMatrix; format = :csc)
    _sparse_format(format)
    pattern=_with_borrowed_sparse(b) do m, n, p, i, values, k, tag
        _sparsity_result(size(b), :csc, RustCrate.sparsity_from_borrowed, m, n, p, i, k)
    end
    format===:csc && return pattern
    converted=copy(pattern; format)
    close(pattern)
    return converted
end

# Dense operands read in place by the borrowed sparse kernels.
const _BorrowedSparseDense=Union{
    _BorrowMatrix,
    _BorrowTranspose,
    FaerBorrowedMatrix,
    _BorrowedTranspose,
}
_borrowed_sparse_promote(b::FaerBorrowedSparseMatrix{T}, ::Type{T}) where {T} = b
_borrowed_sparse_promote(b::FaerBorrowedSparseMatrix{T,Ti}, ::Type{P}) where {T,Ti,P} =
    faerview(SparseArrays.SparseMatrixCSC{P,Ti}(b.parent))
_borrowed_dense_promote(x, ::Type{P}) where {P} =
    eltype(x)===P ? x : FaerMatrix{P}(Matrix(x))
# Julia arrays are borrowed when faerview accepts them, otherwise copied.
_borrowed_dense_operand(x::AbstractVecOrMat, ::Type{P}) where {P} =
    eltype(x)===P && _faerview_valid(x) ? faerview(x) :
    ndims(x)==1 ? FaerVector{P}(Vector(x)) : FaerMatrix{P}(Matrix(x))

function _borrowed_sparse_product(a::FaerBorrowedSparseMatrix, b, left::Bool)
    lock(_matrix_lock) do
        (left ? size(a, 2)==size(b, 1) : size(b, 2)==size(a, 1)) ||
            throw(DimensionMismatch("sparse/dense dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        P<:_NativeScalar || _unsupported_scalar(P)
        aa=_borrowed_sparse_promote(a, P)
        bb=_borrowed_dense_promote(b, P)
        q, d=_view_source(bb)
        dims=left ? (size(a, 1), size(b, 2)) : (size(b, 1), size(a, 2))
        f=RustCrate.borrowed_sparse_product
        GC.@preserve d return _with_borrowed_sparse(aa) do raw...
            P===Float64 ? _matrix_result(f, dims, raw..., q, pointer(d), left) :
            _typed_result(P, f, dims, raw..., q, pointer(d), left)
        end
    end
end
Base.:*(a::FaerBorrowedSparseMatrix, b::_BorrowedSparseDense) =
    _borrowed_sparse_product(a, b, true)
Base.:*(a::_BorrowedSparseDense, b::FaerBorrowedSparseMatrix) =
    _borrowed_sparse_product(b, a, false)
Base.:*(a::FaerBorrowedSparseMatrix, b::StridedMatrix{<:Number}) =
    a*_borrowed_dense_operand(b, promote_type(eltype(a), eltype(b)))
Base.:*(a::StridedMatrix{<:Number}, b::FaerBorrowedSparseMatrix) =
    _borrowed_dense_operand(a, promote_type(eltype(a), eltype(b)))*b
Base.:*(a::FaerBorrowedSparseMatrix, b::Union{_BorrowVector,FaerBorrowedVector}) =
    _vector_from_source(_borrowed_sparse_product(a, b, true))
function Base.:*(a::FaerBorrowedSparseMatrix, b::StridedVector{<:Number})
    x=_borrowed_dense_operand(b, promote_type(eltype(a), eltype(b)))
    return vec(Matrix(_borrowed_sparse_product(a, x, true)))
end
Base.:*(a::FaerBorrowedSparseMatrix, b::FaerSparseMatrix) = FaerSparseMatrix(a)*b
Base.:*(a::FaerSparseMatrix, b::FaerBorrowedSparseMatrix) = a*FaerSparseMatrix(b)

function _borrowed_sparse_mul!(c::FaerMatrix, a, b, alpha, beta, left::Bool)
    lock(_matrix_lock) do
        _handle(c)
        s=left ? a : b
        x=left ? b : a
        rows, inner, cols=left ? (size(a, 1), size(a, 2), size(b, 2)) :
                          (size(a, 1), size(b, 1), size(b, 2))
        (left ? size(x, 1)==inner : size(x, 2)==inner) && size(c)==(rows, cols) ||
            throw(DimensionMismatch("product and destination dimensions must match"))
        T=eltype(c)
        if eltype(s)!==T || eltype(x)!==T
            # Mixed scalar types: promoted native product, then one assignment.
            r=Matrix(left ? s*x : x*s)
            copyto!(c, FaerMatrix{T}(iszero(beta) ? alpha*r : alpha*r+beta*Matrix(c)))
            return c
        end
        q, d=_view_source(x)
        # A source viewing the destination owner is staged before Rust writes.
        q===c && ((q, d)=_view_source(_copy_view_matrix(x)))
        y=Ref{T}(alpha)
        z=Ref{T}(beta)
        GC.@preserve d y z _with_borrowed_sparse(s) do raw...
            _native_call(
                RustCrate.borrowed_sparse_mul,
                c,
                raw...,
                q,
                pointer(d),
                Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, y)),
                Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, z)),
                !iszero(beta),
                left,
            )
        end
        return c
    end
end
LinearAlgebra.mul!(
    c::FaerMatrix,
    a::FaerBorrowedSparseMatrix,
    b::_BorrowedSparseDense,
    alpha::Number,
    beta::Number,
) = _borrowed_sparse_mul!(c, a, b, alpha, beta, true)
LinearAlgebra.mul!(
    c::FaerMatrix,
    a::_BorrowedSparseDense,
    b::FaerBorrowedSparseMatrix,
    alpha::Number,
    beta::Number,
) = _borrowed_sparse_mul!(c, a, b, alpha, beta, false)
LinearAlgebra.mul!(c::FaerMatrix, a::FaerBorrowedSparseMatrix, b::_BorrowedSparseDense) =
    _borrowed_sparse_mul!(c, a, b, true, false, true)
LinearAlgebra.mul!(c::FaerMatrix, a::_BorrowedSparseDense, b::FaerBorrowedSparseMatrix) =
    _borrowed_sparse_mul!(c, a, b, true, false, false)
