"""
    bidiagonalize(A; full=false, blocksize=0)

Native faer reduction `A ≈ U * B * V'`, where B is a square `Bidiagonal` of
order min(size(A)...). Tall/square inputs give upper B; wide inputs give lower B.
The result owns native U/V with thin orthonormal columns by default; full=true
retains complete bases. Inputs are preserved; complex phases stay in B.
Compact reflectors remain internal. Zero blocksize selects faer's recommendation.
"""
struct FaerBidiagonalization{T,M<:FaerMatrix{T}} <: LinearAlgebra.Factorization{T}
    U::M
    B::LinearAlgebra.Bidiagonal{T,Vector{T}}
    V::M
end
Base.size(f::FaerBidiagonalization) = (size(f.U, 1), size(f.V, 1))
Base.size(f::FaerBidiagonalization, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(f)[d] : 1
Base.isopen(f::FaerBidiagonalization) = isopen(f.U)&&isopen(f.V)
function Base.close(f::FaerBidiagonalization)
    lock(_matrix_lock) do ;
        close(f.U)
        close(f.V)
    end
    return nothing
end
function Base.copy(f::FaerBidiagonalization)
    lock(_matrix_lock) do
        _handle(f.U)
        _handle(f.V)
        return FaerBidiagonalization(copy(f.U), copy(f.B), copy(f.V))
    end
end
Base.deepcopy_internal(f::FaerBidiagonalization, d::IdDict) = get!(() -> copy(f), d, f)
function _bidiagonal_bases(f)
    _handle(f.U)
    _handle(f.V)
    m, n=size(f)
    k=size(f.B, 1)
    return copy(view(f.U, 1:m, 1:k)), copy(view(f.V, 1:n, 1:k))
end
function FaerMatrix(f::FaerBidiagonalization)
    lock(_matrix_lock) do
        U, V=_bidiagonal_bases(f)
        return U*FaerMatrix(Matrix(f.B))*V'
    end
end
Base.Matrix(f::FaerBidiagonalization) = Matrix(FaerMatrix(f))
Base.Array(f::FaerBidiagonalization) = Matrix(f)
function bidiagonalize(
    a::AbstractMatrix{<:Number};
    full::Bool = false,
    blocksize::Integer = 0,
)
    lock(_matrix_lock) do
        _tridiagonal_open(a)
        T=_reflector_type(eltype(a))
        data=Matrix{T}(a)
        all(isfinite, data) ||
            throw(ArgumentError("bidiagonalization input must be finite"))
        m, n=size(a)
        k=min(m, n)
        bs=Int(blocksize)
        0<=bs<=max(1, k) || throw(
            ArgumentError(
                "blocksize must be zero (automatic) or between 1 and max(1,min(size(A)...))",
            ),
        )
        U=zeros(T, m, full ? m : k)
        V=zeros(T, n, full ? n : k)
        d=zeros(T, k)
        e=zeros(T, max(0, k-1))
        GC.@preserve data U V d e _native_call(
            RustCrate.bidiagonalize_data,
            Ptr{UInt8}(pointer(data)),
            UInt(m),
            UInt(n),
            Ptr{UInt8}(pointer(U)),
            Ptr{UInt8}(pointer(V)),
            Ptr{UInt8}(pointer(d)),
            Ptr{UInt8}(pointer(e)),
            _tag(T),
            full,
            UInt(bs),
        )
        all(isfinite, U)&&all(isfinite, V)&&all(isfinite, d)&&all(isfinite, e) ||
            throw(ArgumentError("bidiagonalization overflowed the working type"))
        return FaerBidiagonalization(
            FaerMatrix(U),
            LinearAlgebra.Bidiagonal(d, e, m<n ? :L : :U),
            FaerMatrix(V),
        )
    end
end
"""Stage native bidiagonalization, then replace A by the rectangular embedding of B."""
function bidiagonalize!(a::FaerMatrix; kwargs...)
    lock(_matrix_lock) do
        f=bidiagonalize(a; kwargs...)
        m, n=size(a)
        k=size(f.B, 1)
        buffer=zeros(eltype(a), m, n)
        buffer[1:k, 1:k]=Matrix(f.B)
        copyto!(a, buffer)
        return f
    end
end
"""
    bidiagonal_svd(d, e; uplo=:U, recursion_threshold=128)

Call faer's square bidiagonal SVD directly. d and e are the diagonal and selected
offdiagonal (`:U` superdiagonal, `:L` subdiagonal). All four native scalar types
and promotion are supported. Inputs remain unchanged. Returns a Julia SVD with
native U/Vt and nonnegative descending real S. Native workspace includes dense
square factors, even when the caller only needs singular values.
"""
function bidiagonal_svd(
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    uplo = :U,
    recursion_threshold::Integer = 128,
)
    lock(_matrix_lock) do
        _tridiagonal_open(d)
        _tridiagonal_open(e)
        n=length(d)
        length(e)==max(0, n-1) ||
            throw(DimensionMismatch("offdiagonal length must be max(0,length(diag)-1)"))
        uplo in (:U, :L, 'U', 'L') || throw(ArgumentError("uplo must be :U or :L"))
        threshold=Int(recursion_threshold)
        threshold>=2 || throw(ArgumentError("recursion_threshold must be at least two"))
        T=_reflector_type(promote_type(eltype(d), eltype(e)))
        R=_realtype(T)
        diag=Vector{T}(d)
        off=Vector{T}(e)
        all(isfinite, diag)&&all(isfinite, off) ||
            throw(ArgumentError("bidiagonal coefficients must be finite"))
        U=zeros(T, n, n)
        V=zeros(T, n, n)
        s=zeros(R, n)
        GC.@preserve diag off U V s _native_call(
            RustCrate.bidiagonal_svd_data,
            Ptr{UInt8}(pointer(diag)),
            Ptr{UInt8}(pointer(off)),
            UInt(n),
            Ptr{UInt8}(pointer(s)),
            Ptr{UInt8}(pointer(U)),
            Ptr{UInt8}(pointer(V)),
            _tag(T),
            uplo in (:U, 'U'),
            UInt(threshold),
        )
        all(isfinite, U)&&all(isfinite, V)&&all(isfinite, s) ||
            throw(ArgumentError("bidiagonal SVD overflowed the working type"))
        return LinearAlgebra.SVD(FaerMatrix(U), s, FaerMatrix(copy(adjoint(V))))
    end
end
function LinearAlgebra.svd(f::FaerBidiagonalization; full::Bool = false, kwargs...)
    lock(_matrix_lock) do
        _handle(f.U)
        _handle(f.V)
        m, n=size(f)
        k=size(f.B, 1)
        T=eltype(f)
        core=bidiagonal_svd(f.B.dv, f.B.ev; uplo = f.B.uplo, kwargs...)
        if full
            size(f.U, 2)==m && size(f.V, 2)==n ||
                throw(ArgumentError("full SVD requires a reduction made with full=true"))
            left=Matrix{T}(LinearAlgebra.I, m, m)
            left[1:k, 1:k]=Matrix(core.U)
            right=Matrix{T}(LinearAlgebra.I, n, n)
            right[1:k, 1:k]=Matrix(core.Vt)
            return LinearAlgebra.SVD(f.U*FaerMatrix(left), core.S, FaerMatrix(right)*f.V')
        end
        U, V=_bidiagonal_bases(f)
        return LinearAlgebra.SVD(U*core.U, core.S, core.Vt*V')
    end
end
function LinearAlgebra.svdvals(f::FaerBidiagonalization; kwargs...)
    lock(_matrix_lock) do
        _handle(f.U)
        _handle(f.V)
        return bidiagonal_svd(f.B.dv, f.B.ev; uplo = f.B.uplo, kwargs...).S
    end
end
const _NativeBidiagonal=LinearAlgebra.Bidiagonal{
    T,
    V,
} where {T<:_NativeScalar,V<:FaerVector{T}}
LinearAlgebra.svd(a::_NativeBidiagonal; full::Bool = false, kwargs...) =
    bidiagonal_svd(a.dv, a.ev; uplo = a.uplo, kwargs...)
LinearAlgebra.svdvals(a::_NativeBidiagonal; kwargs...) =
    bidiagonal_svd(a.dv, a.ev; uplo = a.uplo, kwargs...).S
