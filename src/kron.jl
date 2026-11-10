# Kronecker products. Dense operands (native owners, strided/reversed views,
# FaerVector/FaerRow, borrowed Julia arrays and their transposes/adjoints) are
# read in place through view descriptors and faer `linalg::kron::kron` writes
# the new native owner. Sparse products build their compressed structure in Rust.
const _KronMatrix = Union{FaerMatrix,FaerMatrixView,FaerRow,FaerBorrowedMatrix}
const _KronVector = Union{FaerVector,FaerVectorView,FaerBorrowedVector}
const _KronAdjointVector = LinearAlgebra.Adjoint{<:Any,<:_KronVector}
const _KronWrappedVector = Union{
    _KronAdjointVector,LinearAlgebra.Transpose{<:Any,<:_KronVector}
}
const _KronDense = Union{
    _KronMatrix,
    _KronVector,
    LinearAlgebra.Transpose{<:Any,<:Union{_KronMatrix,_KronVector}},
    LinearAlgebra.Adjoint{<:Any,<:Union{_KronMatrix,_KronVector}},
}
const _KronJuliaVector = StridedVector{<:Number}
const _KronJuliaAdjointVector = LinearAlgebra.Adjoint{<:Any,<:_KronJuliaVector}
const _KronJuliaWrappedVector = Union{
    _KronJuliaAdjointVector,LinearAlgebra.Transpose{<:Any,<:_KronJuliaVector}
}
const _KronJulia = Union{
    StridedVecOrMat{<:Number},
    LinearAlgebra.Transpose{<:Any,<:StridedVecOrMat{<:Number}},
    LinearAlgebra.Adjoint{<:Any,<:StridedVecOrMat{<:Number}},
}

_kron_dims(x) = ndims(x) == 1 ? (length(x), 1) : size(x)
function _kron_dims(a, b)
    (ma, na), (mb, nb) = _kron_dims(a), _kron_dims(b)
    return Base.checked_mul(ma, mb), Base.checked_mul(na, nb)
end
_kron_scalar(a, b) = _kron_scalar(promote_type(eltype(a), eltype(b)))
_kron_scalar(::Type{P}) where {P} = P <: _NativeScalar ? P : _unsupported_scalar(P)

# An operand of scalar P that `_view_source` reads in place. Julia arrays are
# borrowed when they already hold P (read-only borrows may repeat elements);
# conversions for promotion are the only copies.
function _kron_operand(x::LinearAlgebra.Transpose, ::Type{P}) where {P}
    return transpose(_kron_operand(parent(x), P))
end
function _kron_operand(x::LinearAlgebra.Adjoint, ::Type{P}) where {P}
    return adjoint(_kron_operand(parent(x), P))
end
_kron_operand(x::FaerMatrix, ::Type{P}) where {P} = _coerce(x, P)
_kron_operand(x::Union{FaerVector,FaerRow}, ::Type{P}) where {P} = _collection_coerce(x, P)
function _kron_operand(x::Union{FaerView,FaerBorrowedArray}, ::Type{P}) where {P}
    if eltype(x) === P
        x
    elseif ndims(x) == 1
        FaerVector{P}(Vector(x))
    else
        FaerMatrix{P}(Matrix(x))
    end
end
function _kron_operand(x::StridedVecOrMat, ::Type{P}) where {P}
    if eltype(x) === P && _julia_borrowable(x)
        FaerBorrowedArray(x)
    elseif ndims(x) == 1
        FaerVector{P}(Vector(x))
    else
        FaerMatrix{P}(Matrix(x))
    end
end

function _kron_matrix(a, b)
    lock(_matrix_lock) do
        P = _kron_scalar(a, b)
        dims = _kron_dims(a, b)
        p, d = _view_source(_kron_operand(a, P))
        q, e = _view_source(_kron_operand(b, P))
        f = RustCrate.view_kron
        GC.@preserve d e return if P === Float64
            _matrix_result(f, dims, p, pointer(d), q, pointer(e))
        else
            _typed_result(P, f, dims, p, pointer(d), q, pointer(e))
        end
    end
end
# A collection `C` holding the column-major values of a product with one unit
# dimension.
function _kron_collection(C, a, b)
    lock(_matrix_lock) do
        P = _kron_scalar(a, b)
        m, n = _kron_dims(a, b)
        p, d = _view_source(_kron_operand(a, P))
        q, e = _view_source(_kron_operand(b, P))
        GC.@preserve d e return _collection_result(
            C,
            P,
            m * n,
            RustCrate.vector_view_kron,
            p,
            pointer(d),
            q,
            pointer(e),
            _collection_kind(C),
        )
    end
end
_kron_vector(a, b) = _kron_collection(FaerVector, a, b)
# kron(a', b') == kron(a, b)' (and likewise for transposes), as in LinearAlgebra.
function _kron_wrapped_vector(a, b)
    a isa LinearAlgebra.Adjoint &&
        b isa LinearAlgebra.Adjoint &&
        return adjoint(_kron_vector(parent(a), parent(b)))
    a isa LinearAlgebra.Transpose &&
        b isa LinearAlgebra.Transpose &&
        return transpose(_kron_vector(parent(a), parent(b)))
    return transpose(_kron_vector(a, b))
end

"""
    kron(A, B)

Native Kronecker products. Dense operands may be `FaerMatrix`, strided or
reversed `FaerMatrixView`s, `FaerVector`/`FaerRow` and their views, borrowed
Julia arrays (`faerview`) and transposes/adjoints of these, mixed with Julia
strided arrays; they are read in place (no operand snapshot) and the result is
a new native owner. Vectors follow `LinearAlgebra`: `kron(v, w)` is a
`FaerVector`, `kron(v', w')` is `kron(v, w)'`, `kron(FaerRow, FaerRow)` is a
`FaerRow`, and other shapes give a `FaerMatrix`. Mixed scalar types promote;
only an operand needing conversion is copied.

With a `FaerSparseMatrix`, a borrowed `faerview(::SparseMatrixCSC)` or their
transposes/adjoints as an operand, the result is a `FaerSparseMatrix` built
directly in compressed form (`nnz(A)*nnz(B)` stored entries, sorted columns),
in the first sparse operand's format (CSR for a transposed CSC operand). As in
SparseArrays, a dense operand is sparsified (its zeros are not stored) and the
stored zeros of a sparse operand are kept. The other operand may also be a
Julia `SparseMatrixCSC` (borrowed in place) or `SparseVector` (one CSC copy).
"""
LinearAlgebra.kron(a::_KronDense, b::_KronDense) = _kron_matrix(a, b)
LinearAlgebra.kron(a::_KronDense, b::_KronJulia) = _kron_matrix(a, b)
LinearAlgebra.kron(a::_KronJulia, b::_KronDense) = _kron_matrix(a, b)
LinearAlgebra.kron(a::FaerRow, b::FaerRow) = _kron_collection(FaerRow, a, b)
for (A, B) in (
    (:_KronVector, :_KronVector),
    (:_KronVector, :_KronJuliaVector),
    (:_KronJuliaVector, :_KronVector),
)
    @eval LinearAlgebra.kron(a::$A, b::$B) = _kron_vector(a, b)
end
for (A, B) in (
    (:_KronWrappedVector, :_KronWrappedVector),
    (:_KronWrappedVector, :_KronJuliaWrappedVector),
    (:_KronJuliaWrappedVector, :_KronWrappedVector),
    (:_KronAdjointVector, :_KronAdjointVector),
    (:_KronAdjointVector, :_KronJuliaAdjointVector),
    (:_KronJuliaAdjointVector, :_KronAdjointVector),
)
    @eval LinearAlgebra.kron(a::$A, b::$B) = _kron_wrapped_vector(a, b)
end

const _KronDestination = Union{FaerMatrix,FaerMatrixView,FaerVector,FaerVectorView}
"""
    kron!(C, A, B)

Write `kron(A, B)` into the native destination `C` (`FaerMatrix`, `FaerVector`
or a view of either) without allocating the product, for the dense operands
accepted by `kron`. An operand sharing `C`'s owner is snapshot first, so
aliasing calls give the same result as `kron`; a failed call leaves `C`
unchanged. Operands of another scalar type than `C` are converted (or, when
their promotion differs from `eltype(C)`, the product is computed and then
assigned).
"""
function LinearAlgebra.kron!(
    c::_KronDestination, a::Union{_KronDense,_KronJulia}, b::Union{_KronDense,_KronJulia}
)
    lock(_matrix_lock) do
        dims = _kron_dims(a, b)
        _kron_dims(c) == dims || throw(
            DimensionMismatch("Kronecker destination has size $(size(c)), expected $dims"),
        )
        T = eltype(c)
        if promote_type(eltype(a), eltype(b)) !== T
            copyto!(c, kron(a, b))
            return c
        end
        o, cd = _view_source(c)
        p, d = _view_source(_kron_operand(a, T))
        q, e = _view_source(_kron_operand(b, T))
        # Rust borrows the destination mutably: snapshot operands of its owner.
        p === o && ((p, d) = _view_source(_copy_view_matrix(_kron_operand(a, T))))
        q === o && ((q, e) = _view_source(_copy_view_matrix(_kron_operand(b, T))))
        GC.@preserve cd d e _native_call(
            RustCrate.view_kron_into, o, pointer(cd), p, pointer(d), q, pointer(e)
        )
        return c
    end
end

# ------------------------------------------------------------------ sparse ---
const _KronSparseOwner = Union{FaerSparseMatrix,FaerBorrowedSparseMatrix}
const _KronSparse = Union{
    _KronSparseOwner,
    LinearAlgebra.Transpose{<:Any,<:_KronSparseOwner},
    LinearAlgebra.Adjoint{<:Any,<:_KronSparseOwner},
}
const _KronJuliaSparseParent = Union{SparseArrays.SparseMatrixCSC,SparseArrays.SparseVector}
const _KronJuliaSparse = Union{
    _KronJuliaSparseParent,
    LinearAlgebra.Transpose{<:Any,<:_KronJuliaSparseParent},
    LinearAlgebra.Adjoint{<:Any,<:_KronJuliaSparseParent},
}
_kron_sparse_parent(x::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint}) = parent(x)
_kron_sparse_parent(x) = x
_kron_sparse_flags(x::LinearAlgebra.Adjoint) = (1, 1)
_kron_sparse_flags(x::LinearAlgebra.Transpose) = (1, 0)
_kron_sparse_flags(x) = (0, 0)
function _kron_is_sparse(x)
    return _kron_sparse_parent(x) isa Union{_KronSparseOwner,_KronJuliaSparseParent}
end
# Result format: the first sparse operand's effective layout.
function _kron_sparse_format(x)
    p = _kron_sparse_parent(x)
    csr = p isa FaerSparseMatrix && p.format === :csr
    return (csr != (_kron_sparse_flags(x)[1] == 1)) ? :csr : :csc
end
_kron_sparse_promote(x::FaerSparseMatrix, ::Type{P}) where {P} = _sparse_coerce(x, P)
function _kron_sparse_promote(x::FaerBorrowedSparseMatrix, ::Type{P}) where {P}
    return _borrowed_sparse_promote(x, P)
end
function _kron_sparse_promote(x::SparseArrays.SparseMatrixCSC, ::Type{P}) where {P}
    return faerview(eltype(x) === P ? x : SparseArrays.SparseMatrixCSC{P}(x))
end
# A Julia sparse vector is one CSC column (one structural copy).
function _kron_sparse_promote(x::SparseArrays.SparseVector, ::Type{P}) where {P}
    return _kron_sparse_promote(SparseArrays.SparseMatrixCSC(x), P)
end
# `(owner, descriptor, rooted)` in the layout of kron.rs `sparse_operand`.
# Must run under `_matrix_lock`.
function _kron_sparse_operand(x, ::Type{P}) where {P}
    _kron_is_sparse(x) || begin
        o, d = _view_source(_kron_operand(x, P))
        return o, pushfirst!(d, 2), nothing
    end
    t, c = _kron_sparse_flags(x)
    s = _kron_sparse_promote(_kron_sparse_parent(x), P)
    s isa FaerSparseMatrix && return s, Int[0, t, c], nothing
    m, n, k = _borrowed_indices!(s)
    values = SparseArrays.nonzeros(s.parent)
    d = Int[
        1,
        t,
        c,
        m,
        n,
        Int(pointer(s.colptr)),
        Int(pointer(s.rowval)),
        Int(pointer(values)),
        k,
        _tag(P),
    ]
    return Ptr{UInt8}(C_NULL), d, s
end
function _kron_sparse(a, b)
    lock(_matrix_lock) do
        P = _kron_scalar(a, b)
        dims = _kron_dims(a, b)
        format = _kron_sparse_format(_kron_is_sparse(a) ? a : b)
        p, d, ra = _kron_sparse_operand(a, P)
        q, e, rb = _kron_sparse_operand(b, P)
        # Borrowed index buffers are shared when `a` and `b` are one view; both
        # rewrites above produced the same structure.
        GC.@preserve d e ra rb return _sparse_result(
            P,
            dims,
            format,
            RustCrate.sparse_kron,
            p,
            pointer(d),
            q,
            pointer(e),
            format === :csr,
        )
    end
end
LinearAlgebra.kron(a::_KronSparse, b::_KronSparse) = _kron_sparse(a, b)
function LinearAlgebra.kron(
    a::_KronSparse, b::Union{_KronDense,_KronJulia,_KronJuliaSparse}
)
    return _kron_sparse(a, b)
end
function LinearAlgebra.kron(
    a::Union{_KronDense,_KronJulia,_KronJuliaSparse}, b::_KronSparse
)
    return _kron_sparse(a, b)
end
