"""
    FaerSparseMatrix(A; format=:csc)
    FaerSparseMatrix{T}(A; format=:csc)
    FaerSparseMatrix(I, J, V, m, n; format=:csc)

A sparse Julia matrix owning faer CSC or CSR storage. Four native scalar types
are preserved; other real inputs convert to Float64 and other complex inputs to
ComplexF64. An explicit unsupported `T` throws `ArgumentError`. Triplet duplicates are summed by faer. Construction retains stored zeros; `dropzeros!` removes them. Native
addition, subtraction, products and scalar arithmetic preserve sparse storage; `Matrix(A)` and
`FaerMatrix(A)` explicitly materialize a dense matrix. Broadcast, `map` and
`map!` follow SparseArrays' sparse result and zero-dropping conventions;
zero-preserving maps over one native operand and scalars map only the stored
values, and native-scalar results are `FaerSparseMatrix`.

`nonzeros(A)` is a live view in the owner's physical storage order. Structural
changes invalidate earlier value views. `compressed_storage(A)` returns an
independent snapshot of the physical pointers, indices and values, using
1-based Julia indices. CSC-only `rowvals`/`getcolptr` return index snapshots.
"""
mutable struct FaerSparseMatrix{T,H} <: SparseArrays.AbstractSparseMatrix{T,Int}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    stored::Int
    format::Symbol
    epoch::UInt
end
function _sparse_dims(m, n)
    dims=(Int(m), Int(n))
    all(>=(0), dims) || throw(ArgumentError("sparse dimensions must be nonnegative"))
    Base.checked_mul(dims...)
    return dims
end
function _sparse_format(format)
    format in (:csc, :csr) || throw(ArgumentError("format must be :csc or :csr"))
    return format===:csr
end
function _handle(a::FaerSparseMatrix)
    h=a.handle
    isnothing(h) && throw(ArgumentError("sparse matrix has been closed"))
    return h
end
_native_argument(a::FaerSparseMatrix) = _handle(a)
_is_native_owner(::FaerSparseMatrix) = true
function _sparse_result(::Type{T}, dims, format, f, args...) where {T<:_NativeScalar}
    _sparse_dims(dims...)
    _sparse_format(format)
    lock(_matrix_lock) do
        h=RustCrate.SparseOwner()
        _native_call(f, h, args...)
        k=Int(_native_call(RustCrate.sparse_nnz, h))
        return FaerSparseMatrix{T,typeof(h)}(h, dims, k, format, UInt(0))
    end
end
_sparse_input_type(::Type{T}) where {T} = _native_input_type(T)
function FaerSparseMatrix{T}(
    a::SparseArrays.SparseMatrixCSC;
    format = :csc,
) where {T<:_NativeScalar}
    csr=_sparse_format(format)
    dims=_sparse_dims(size(a)...)
    cp=SparseArrays.getcolptr(a)
    length(cp)==size(a, 2)+1 && first(cp)==1 && issorted(cp) ||
        throw(ArgumentError("invalid CSC column pointers"))
    k=Int(last(cp)-1)
    0<=k<=length(SparseArrays.rowvals(a)) && k<=length(SparseArrays.nonzeros(a)) ||
        throw(ArgumentError("invalid CSC stored count"))
    rows=SparseArrays.rowvals(a)[1:k]
    all(i->1<=i<=size(a, 1), rows) ||
        throw(ArgumentError("CSC row indices must be in bounds"))
    p=UInt.(cp .- 1)
    i=UInt.(rows .- 1)
    v=Vector{T}(SparseArrays.nonzeros(a)[1:k])
    GC.@preserve p i v return _sparse_result(
        T,
        dims,
        format,
        RustCrate.sparse_load_csc,
        UInt(dims[1]),
        UInt(dims[2]),
        pointer(p),
        pointer(i),
        Ptr{UInt8}(pointer(v)),
        UInt(k),
        _tag(T),
        csr,
    )
end
FaerSparseMatrix(a::SparseArrays.SparseMatrixCSC{T}; kwargs...) where {T} =
    FaerSparseMatrix{_sparse_input_type(T)}(a; kwargs...)
FaerSparseMatrix{T}(a::AbstractMatrix; kwargs...) where {T<:_NativeScalar} =
    FaerSparseMatrix{T}(SparseArrays.sparse(a); kwargs...)
FaerSparseMatrix{T}(::AbstractMatrix; kwargs...) where {T} = _unsupported_scalar(T)
FaerSparseMatrix(a::AbstractMatrix{T}; kwargs...) where {T} =
    FaerSparseMatrix{_sparse_input_type(T)}(a; kwargs...)
function FaerSparseMatrix{T}(
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
    values::AbstractVector,
    m::Integer,
    n::Integer;
    format = :csc,
) where {T<:_NativeScalar}
    dims=_sparse_dims(m, n)
    csr=_sparse_format(format)
    length(rows)==length(cols)==length(values) ||
        throw(DimensionMismatch("triplet vectors must have matching lengths"))
    all(i->1<=i<=m, rows)&&all(j->1<=j<=n, cols) ||
        throw(ArgumentError("triplet indices must be in bounds"))
    i=UInt.(rows .- 1)
    j=UInt.(cols .- 1)
    v=Vector{T}(values)
    GC.@preserve i j v return _sparse_result(
        T,
        dims,
        format,
        RustCrate.sparse_load_triplets,
        UInt(m),
        UInt(n),
        pointer(i),
        pointer(j),
        Ptr{UInt8}(pointer(v)),
        UInt(length(v)),
        _tag(T),
        csr,
    )
end
FaerSparseMatrix(
    i::AbstractVector{<:Integer},
    j::AbstractVector{<:Integer},
    v::AbstractVector{T},
    m::Integer,
    n::Integer;
    kwargs...,
) where {T} = FaerSparseMatrix{_sparse_input_type(T)}(i, j, v, m, n; kwargs...)
function Base.copy(a::FaerSparseMatrix; format = a.format)
    return _sparse_result(
        eltype(a),
        size(a),
        format,
        RustCrate.sparse_clone_from,
        a,
        _sparse_format(format),
        false,
        false,
    )
end
FaerSparseMatrix(a::FaerSparseMatrix; kwargs...) = copy(a; kwargs...)
function FaerSparseMatrix{T}(
    a::FaerSparseMatrix;
    format = a.format,
) where {T<:_NativeScalar}
    T===eltype(a) && return copy(a; format)
    data=compressed_storage(a)
    v=T.(data.values)
    p=UInt.(data.pointers .- 1)
    i=UInt.(data.indices .- 1)
    dims=a.format===:csr ? reverse(size(a)) : size(a)
    temp=GC.@preserve p i v _sparse_result(
        T,
        dims,
        :csc,
        RustCrate.sparse_load_csc,
        UInt(dims[1]),
        UInt(dims[2]),
        pointer(p),
        pointer(i),
        Ptr{UInt8}(pointer(v)),
        UInt(length(v)),
        _tag(T),
        false,
    )
    return a.format===:csr ?
           _sparse_result(
        T,
        size(a),
        format,
        RustCrate.sparse_clone_from,
        temp,
        _sparse_format(format),
        true,
        false,
    ) : copy(temp; format)
end
Base.deepcopy_internal(a::FaerSparseMatrix, d::IdDict) = get!(() -> copy(a), d, a)
Base.size(a::FaerSparseMatrix) = a.dims
Base.IndexStyle(::Type{<:FaerSparseMatrix}) = IndexCartesian()
function Base.close(a::FaerSparseMatrix)
    lock(_matrix_lock) do ;
        a.handle=nothing
        a.epoch+=1
    end
    return nothing
end
Base.isopen(a::FaerSparseMatrix) = lock(() -> !isnothing(a.handle), _matrix_lock)
SparseArrays.nnz(a::FaerSparseMatrix) = lock(_matrix_lock) do ;
    _handle(a)
    a.stored
end
"""
    compressed_storage(A::FaerSparseMatrix)

Snapshot `(format, pointers, indices, values)` in physical CSC/CSR order, using
1-based indices. All returned arrays are independent; editing them does not
change A. CSR snapshots use row pointers and column indices.
"""
function compressed_storage(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        _handle(a)
        outer=size(a, a.format===:csr ? 1 : 2)
        p=Vector{UInt}(undef, outer+1)
        i=Vector{UInt}(undef, a.stored)
        v=Vector{eltype(a)}(undef, a.stored)
        GC.@preserve p i v _native_call(
            RustCrate.sparse_export,
            a,
            pointer(p),
            pointer(i),
            Ptr{UInt8}(pointer(v)),
            UInt(length(p)),
            UInt(length(i)),
        )
        return (format = a.format, pointers = Int.(p), indices = Int.(i), values = v)
    end
end
function SparseArrays.SparseMatrixCSC(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        c=a.format===:csc ? a : copy(a; format = :csc)
        data=compressed_storage(c)
        return SparseArrays.SparseMatrixCSC(
            size(c)...,
            data.pointers,
            data.indices,
            data.values,
        )
    end
end
SparseArrays.SparseMatrixCSC{T,I}(a::FaerSparseMatrix) where {T,I<:Integer} =
    SparseArrays.SparseMatrixCSC{T,I}(SparseArrays.SparseMatrixCSC(a))
SparseArrays.sparse(a::FaerSparseMatrix) = SparseArrays.SparseMatrixCSC(a)
SparseArrays.findnz(a::FaerSparseMatrix) =
    SparseArrays.findnz(SparseArrays.SparseMatrixCSC(a))
function _require_sparse_csc(a::FaerSparseMatrix)
    _handle(a)
    a.format===:csc || throw(
        ArgumentError(
            "this CSC interface requires format=:csc; use compressed_storage for CSR",
        ),
    )
end
SparseArrays.rowvals(a::FaerSparseMatrix) = lock(_matrix_lock) do ;
    _require_sparse_csc(a)
    compressed_storage(a).indices
end
SparseArrays.getcolptr(a::FaerSparseMatrix) = lock(_matrix_lock) do ;
    _require_sparse_csc(a)
    compressed_storage(a).pointers
end
function SparseArrays.nzrange(a::FaerSparseMatrix, j::Integer)
    lock(_matrix_lock) do
        _require_sparse_csc(a)
        1<=j<=size(a, 2) || throw(BoundsError(a, (:, j)))
        p=compressed_storage(a).pointers
        return p[j]:(p[j+1]-1)
    end
end
function Base.getindex(a::FaerSparseMatrix{T}, i::Int, j::Int) where {T}
    lock(_matrix_lock) do
        checkbounds(a, i, j)
        v=Ref{T}()
        GC.@preserve v _native_call(
            RustCrate.sparse_element,
            a,
            UInt(i-1),
            UInt(j-1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v)),
        )
        return v[]
    end
end
function Base.setindex!(a::FaerSparseMatrix{T}, x, i::Int, j::Int) where {T}
    lock(_matrix_lock) do
        checkbounds(a, i, j)
        v=Ref{T}(x)
        changed=GC.@preserve v _native_call(
            RustCrate.sparse_set,
            a,
            UInt(i-1),
            UInt(j-1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, v)),
        )
        if changed
            a.stored=Int(_native_call(RustCrate.sparse_nnz, a))
            a.epoch+=1
        end
    end
    return a
end
function SparseArrays.dropzeros!(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        changed=_native_call(RustCrate.sparse_dropzeros, a)
        if changed
            a.stored=Int(_native_call(RustCrate.sparse_nnz, a))
            a.epoch+=1
        end
    end
    return a
end
mutable struct FaerSparseValues{T,P} <: AbstractVector{T}
    owner::Union{Nothing,P}
    len::Int
    epoch::UInt
end
function _sparse_values_owner(v::FaerSparseValues)
    a=v.owner
    isnothing(a) && throw(ArgumentError("sparse value view has been closed"))
    _handle(a)
    a.epoch==v.epoch || throw(
        InvalidStateException(
            "sparse structure changed; request nonzeros(A) again",
            :invalidated,
        ),
    )
    return a
end
function SparseArrays.nonzeros(a::FaerSparseMatrix{T}) where {T}
    lock(_matrix_lock) do ;
        _handle(a)
        FaerSparseValues{T,typeof(a)}(a, a.stored, a.epoch)
    end
end
Base.size(v::FaerSparseValues) = (v.len,)
Base.IndexStyle(::Type{<:FaerSparseValues}) = IndexLinear()
function Base.getindex(v::FaerSparseValues{T}, i::Int) where {T}
    lock(_matrix_lock) do
        a=_sparse_values_owner(v)
        checkbounds(v, i)
        out=Ref{T}()
        GC.@preserve out _native_call(
            RustCrate.sparse_value,
            a,
            UInt(i-1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
        )
        return out[]
    end
end
function Base.setindex!(v::FaerSparseValues{T}, x, i::Int) where {T}
    lock(_matrix_lock) do
        a=_sparse_values_owner(v)
        checkbounds(v, i)
        value=Ref{T}(x)
        GC.@preserve value _native_call(
            RustCrate.sparse_set_value,
            a,
            UInt(i-1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
        )
    end
    return v
end
function Base.Vector(v::FaerSparseValues{T}) where {T}
    lock(_matrix_lock) do
        a=_sparse_values_owner(v)
        out=Vector{T}(undef, length(v))
        GC.@preserve out _native_call(
            RustCrate.sparse_values_buffer,
            a,
            Ptr{UInt8}(pointer(out)),
            UInt(length(out)),
        )
        return out
    end
end
Base.copy(v::FaerSparseValues) = Vector(v)
function Base.deepcopy_internal(v::FaerSparseValues, d::IdDict)
    get!(d, v) do
        lock(_matrix_lock) do
            a=Base.deepcopy_internal(_sparse_values_owner(v), d)
            return SparseArrays.nonzeros(a)
        end
    end
end
function Base.close(v::FaerSparseValues)
    lock(_matrix_lock) do ;
        v.owner=nothing
    end
    return nothing
end
Base.isopen(v::FaerSparseValues) = lock(_matrix_lock) do ;
    a=v.owner
    !isnothing(a)&&isopen(a)&&a.epoch==v.epoch
end
function Base.fill!(v::FaerSparseValues{T}, x) where {T}
    lock(_matrix_lock) do
        a=_sparse_values_owner(v)
        value=Ref{T}(x)
        GC.@preserve value _native_call(
            RustCrate.sparse_fill_values,
            a,
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
        )
    end
    return v
end
function Base.fill!(a::FaerSparseMatrix, x)
    lock(_matrix_lock) do
        _handle(a)
        iszero(x)||a.stored==length(a) ||
            throw(
                ArgumentError(
                    "nonzero fill would create implicit entries; use fill!(nonzeros(A),x) for stored values",
                ),
            )
        fill!(SparseArrays.nonzeros(a), x)
    end
    return a
end
Base.similar(a::FaerSparseMatrix, ::Type{T}, dims::Dims{2}) where {T<:_NativeScalar} =
    FaerSparseMatrix{T}(Int[], Int[], T[], dims...; format = a.format)
Base.zero(a::FaerSparseMatrix) = similar(a, eltype(a), size(a))
function FaerMatrix(a::FaerSparseMatrix{T}) where {T}
    T===Float64 ? _matrix_result(RustCrate.sparse_dense_from, size(a), a) :
    _typed_result(T, RustCrate.sparse_dense_from, size(a), a)
end
Base.Matrix(a::FaerSparseMatrix) = Matrix(FaerMatrix(a))
Base.Matrix{T}(a::FaerSparseMatrix) where {T} = Matrix{T}(Matrix(a))
Base.Array(a::FaerSparseMatrix) = Matrix(a)
_sparse_coerce(a::FaerSparseMatrix{T}, ::Type{T}) where {T} = a
_sparse_coerce(a::FaerSparseMatrix, ::Type{T}) where {T} = FaerSparseMatrix{T}(a)
function _sparse_binary(a::FaerSparseMatrix, b::FaerSparseMatrix, op)
    lock(_matrix_lock) do
        dims=op==2 ? (size(a, 1), size(b, 2)) : size(a)
        (op==2 ? size(a, 2)==size(b, 1) : size(a)==size(b)) ||
            throw(DimensionMismatch("sparse operation dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        return _sparse_result(
            P,
            dims,
            a.format,
            RustCrate.sparse_arithmetic,
            _sparse_coerce(a, P),
            _sparse_coerce(b, P),
            Int32(op),
            a.format===:csr,
        )
    end
end
Base.:+(a::FaerSparseMatrix, b::FaerSparseMatrix) = _sparse_binary(a, b, 0)
Base.:-(a::FaerSparseMatrix, b::FaerSparseMatrix) = _sparse_binary(a, b, 1)
Base.:*(a::FaerSparseMatrix, b::FaerSparseMatrix) = _sparse_binary(a, b, 2)
Base.:+(a::FaerSparseMatrix) = copy(a)
Base.:-(a::FaerSparseMatrix) = -one(eltype(a))*a
function Base.:*(x::Number, a::FaerSparseMatrix)
    P=_native_promotion(typeof(x), eltype(a))
    v=Ref{P}(x)
    GC.@preserve v return _sparse_result(
        P,
        size(a),
        a.format,
        RustCrate.sparse_scale,
        _sparse_coerce(a, P),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, v)),
    )
end
Base.:*(a::FaerSparseMatrix, x::Number) = x*a
Base.:/(a::FaerSparseMatrix, x::Number) = (one(eltype(a))/x)*a
Base.conj(a::FaerSparseMatrix) = _sparse_result(
    eltype(a),
    size(a),
    a.format,
    RustCrate.sparse_clone_from,
    a,
    a.format===:csr,
    false,
    true,
)
const _SparseTransform=Union{
    LinearAlgebra.Transpose{<:Any,<:FaerSparseMatrix},
    LinearAlgebra.Adjoint{<:Any,<:FaerSparseMatrix},
}
function Base.copy(a::_SparseTransform)
    p=parent(a)
    format=p.format===:csc ? :csr : :csc
    return _sparse_result(
        eltype(p),
        reverse(size(p)),
        format,
        RustCrate.sparse_clone_from,
        p,
        format===:csr,
        true,
        a isa LinearAlgebra.Adjoint,
    )
end
function _sparse_dense_product(a::FaerSparseMatrix, b::FaerMatrix, left = true)
    lock(_matrix_lock) do
        (left ? size(a, 2)==size(b, 1) : size(b, 2)==size(a, 1)) ||
            throw(DimensionMismatch("sparse/dense dimensions must match"))
        dims=left ? (size(a, 1), size(b, 2)) : (size(b, 1), size(a, 2))
        P=promote_type(eltype(a), eltype(b))
        aa, bb=_sparse_coerce(a, P), _coerce(b, P)
        return P===Float64 ?
               _matrix_result(RustCrate.sparse_product_from, dims, aa, bb, left) :
               _typed_result(P, RustCrate.sparse_product_from, dims, aa, bb, left)
    end
end
Base.:*(a::FaerSparseMatrix, b::FaerMatrix) = _sparse_dense_product(a, b)
Base.:*(a::FaerMatrix, b::FaerSparseMatrix) = _sparse_dense_product(b, a, false)
Base.:*(a::FaerSparseMatrix, b::FaerMatrixView) = a*copy(b)
Base.:*(a::FaerMatrixView, b::FaerSparseMatrix) = copy(a)*b
Base.:*(a::FaerSparseMatrix, b::FaerVector{<:Number}) = _vector_from_source(a*FaerMatrix(b))
Base.:*(a::FaerSparseMatrix, b::FaerVectorView{<:Number}) =
    _vector_from_source(a*_copy_view_matrix(b))
for (A, B) in (
    (:_SparseTransform, :FaerSparseMatrix),
    (:FaerSparseMatrix, :_SparseTransform),
    (:_SparseTransform, :_SparseTransform),
    (:_SparseTransform, :FaerMatrix),
    (:FaerMatrix, :_SparseTransform),
    (:_BorrowTranspose, :FaerSparseMatrix),
    (:FaerSparseMatrix, :_BorrowTranspose),
)
    @eval Base.:*(a::$A, b::$B) =
        $(A===:FaerSparseMatrix||A===:FaerMatrix ? :a : :(copy(a)))*$(
            B===:FaerSparseMatrix||B===:FaerMatrix ? :b : :(copy(b))
        )
end
function Base.sum(a::FaerSparseMatrix{T}; dims = :) where {T}
    dims isa Colon || return sum(SparseArrays.SparseMatrixCSC(a); dims)
    out=Ref{T}()
    GC.@preserve out _native_call(
        RustCrate.sparse_sum,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
    )
    return out[]
end
function LinearAlgebra.norm(a::FaerSparseMatrix{T}, p::Real = 2) where {T}
    p in (1, 2, Inf) || return LinearAlgebra.norm(SparseArrays.SparseMatrixCSC(a), p)
    R=_realtype(T)
    out=Ref{R}()
    code=p==2 ? 0 : p==1 ? 1 : 2
    GC.@preserve out _native_call(
        RustCrate.sparse_norm,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, out)),
        Int32(code),
    )
    return out[]
end

function Base.copyto!(dest::FaerSparseMatrix, src::FaerSparseMatrix)
    lock(_matrix_lock) do
        _handle(dest)
        _handle(src)
        size(dest)==size(src) || throw(DimensionMismatch("sparse copy shapes must match"))
        dest===src && return dest
        source=_sparse_coerce(src, eltype(dest))
        changed=_native_call(RustCrate.sparse_assign, dest, source, dest.format===:csr)
        if changed
            dest.stored=Int(_native_call(RustCrate.sparse_nnz, dest))
            dest.epoch+=1
        end
    end
    return dest
end
function _sparse_mul!(dest, a, b, alpha, beta)
    lock(_matrix_lock) do
        _handle(dest)
        _handle(a)
        _handle(b)
        size(a, 2)==size(b, 1)&&size(dest)==(size(a, 1), size(b, 2)) ||
            throw(DimensionMismatch("product and destination dimensions must match"))
        result=iszero(alpha) ? zero(dest) : alpha*(a*b)
        if dest isa FaerMatrix && result isa FaerSparseMatrix
            result=FaerMatrix(result)
        end
        !iszero(beta) && (result=result+beta*dest)
        copyto!(dest, result)
    end
    return dest
end
LinearAlgebra.mul!(
    c::FaerSparseMatrix,
    a::FaerSparseMatrix,
    b::FaerSparseMatrix,
    alpha::Number,
    beta::Number,
) = _sparse_mul!(c, a, b, alpha, beta)
LinearAlgebra.mul!(c::FaerSparseMatrix, a::FaerSparseMatrix, b::FaerSparseMatrix) =
    _sparse_mul!(c, a, b, 1, 0)
for (A, B) in (
    (:FaerSparseMatrix, :FaerMatrix),
    (:FaerMatrix, :FaerSparseMatrix),
    (:FaerSparseMatrix, :FaerSparseMatrix),
)
    @eval begin
        LinearAlgebra.mul!(c::FaerMatrix, a::$A, b::$B, alpha::Number, beta::Number) =
            _sparse_mul!(c, a, b, alpha, beta)
        LinearAlgebra.mul!(c::FaerMatrix, a::$A, b::$B) = _sparse_mul!(c, a, b, 1, 0)
    end
end
for (f, op) in ((:+, 0), (:-, 1), (:*, 2))
    @eval begin
        Base.$f(a::FaerSparseMatrix, b::SparseArrays.SparseMatrixCSC) =
            _sparse_binary(a, FaerSparseMatrix{promote_type(eltype(a), eltype(b))}(b), $op)
        Base.$f(a::SparseArrays.SparseMatrixCSC, b::FaerSparseMatrix) =
            _sparse_binary(FaerSparseMatrix{promote_type(eltype(a), eltype(b))}(a), b, $op)
    end
end
Base.:*(a::FaerSparseMatrix, b::StridedMatrix{<:Number}) =
    a*FaerMatrix{promote_type(eltype(a), eltype(b))}(b)
Base.:*(a::StridedMatrix{<:Number}, b::FaerSparseMatrix) =
    FaerMatrix{promote_type(eltype(a), eltype(b))}(a)*b
Base.:*(a::FaerSparseMatrix, b::StridedVector{<:Number}) =
    Vector(a*FaerVector{promote_type(eltype(a), eltype(b))}(b))
function LinearAlgebra.mul!(
    dest::FaerVector,
    a::FaerSparseMatrix,
    b::_BorrowVector,
    alpha::Number,
    beta::Number,
)
    lock(_matrix_lock) do
        _handle(dest)
        _handle(a)
        _view_source(b)
        length(dest)==size(a, 1)&&size(a, 2)==length(b) ||
            throw(DimensionMismatch("sparse matrix/vector dimensions must match"))
        result=iszero(alpha) ? zero(dest) : alpha*(a*b)
        !iszero(beta) && (result=result+beta*dest)
        copyto!(dest, result)
    end
    return dest
end
LinearAlgebra.mul!(dest::FaerVector, a::FaerSparseMatrix, b::_BorrowVector) =
    LinearAlgebra.mul!(dest, a, b, 1, 0)

Base.dataids(a::FaerSparseMatrix) = (UInt(objectid(a)),)
Base.dataids(v::FaerSparseValues) =
    lock(() -> Base.dataids(_sparse_values_owner(v)), _matrix_lock)
Base.unaliascopy(v::FaerSparseValues) =
    lock(() -> SparseArrays.nonzeros(copy(_sparse_values_owner(v))), _matrix_lock)
