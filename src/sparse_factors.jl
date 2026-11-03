"""
    FaerSparseSymbolic(P::FaerSparsity; kind=:lu, uplo=:L)
    FaerSparseSymbolic(A::FaerSparseMatrix; kind=:lu, uplo=:L)

Own faer's symbolic sparse LU, QR or Cholesky analysis (`kind=:lu/:qr/:cholesky`).
`LinearAlgebra.factorize(plan,A)` computes independent numeric factors without repeating the
analysis. A must have exactly the original stored pattern, including stored
zeros; CSC/CSR orientation and numeric scalar type may differ. Sparse QR
currently requires at least as many rows as columns. Cholesky accesses `uplo`.

Cholesky accepts `ordering=:default/:amd/:identity` or a permutation, AMD tuning
`dense=10.0, aggressive=true`, and `algorithm=:auto/:simplicial/:supernodal`.
`plan.p` returns an independent FaerPermutation; `plan.algorithm` reports the
actual chosen algorithm. `plan.factor_storage` counts numeric scalar slots,
including supernodal padding. These controls currently apply only to Cholesky.
"""
mutable struct FaerSparseSymbolic{K,H}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    uplo::Symbol
end
"""
    FaerSparseFactorization

Reusable Rust-owned sparse factors returned by `lu`, `qr`, `cholesky` and
`LinearAlgebra.factorize(plan,A)` for FaerSparseMatrix. `F \\ B` reuses numeric factors; QR
solves full-column-rank least-squares problems. LU/Cholesky also support
transpose/adjoint solves. Inputs and symbolic plans may be closed independently.
Mixed precision RHS are handled by reusing symbolic analysis to factor a sparse
snapshot in the promoted type. `ldiv!` stages the result before overwriting.
Sparse Cholesky additionally exposes independent `F.p`, the actual `F.algorithm`
and `F.factor_storage` (numeric scalar slots, including supernodal padding).
`cholesky(A; perm=p)` selects a custom ordering using Julia's sparse keyword;
`ordering=p` is an alternative. Numeric updates preserve the symbolic controls.
"""
mutable struct FaerSparseFactorization{T,K,H,P,A} <: LinearAlgebra.Factorization{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    info::Int
    plan::P
    source::A
end
const _SparseNativeFactor=Union{FaerSparseSymbolic,FaerSparseFactorization}
function _handle(f::_SparseNativeFactor)
    h=getfield(f, :handle)
    isnothing(h) && throw(ArgumentError("sparse analysis/factorization has been closed"))
    return h
end
_native_argument(f::_SparseNativeFactor) = _handle(f)
_is_native_owner(::_SparseNativeFactor) = true
Base.size(f::_SparseNativeFactor) = getfield(f, :dims)
Base.size(f::_SparseNativeFactor, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(f)[d] : 1
Base.isopen(f::_SparseNativeFactor) =
    lock(()->!isnothing(getfield(f, :handle)), _matrix_lock)
function Base.close(f::_SparseNativeFactor)
    lock(_matrix_lock) do
        f.handle=nothing
        if f isa FaerSparseFactorization
            close(f.plan)
            close(f.source)
        end
    end
    return nothing
end
LinearAlgebra.issuccess(f::FaerSparseFactorization) = iszero(f.info)
function FaerSparseSymbolic(
    p::FaerSparsity;
    kind = :lu,
    uplo = :L,
    ordering = :default,
    dense::Real = 10.0,
    aggressive::Bool = true,
    algorithm = :auto,
)
    code=kind===:lu ? 0 :
         kind===:qr ? 1 :
         kind===:cholesky ? 2 : throw(ArgumentError("kind must be :lu, :qr or :cholesky"))
    uplo in (:L, :U) || throw(ArgumentError("uplo must be :L or :U"))
    kind!==:qr &&
        size(p, 1)!=size(p, 2) &&
        throw(DimensionMismatch("sparse LU/Cholesky requires a square matrix"))
    kind===:qr &&
        size(p, 1)<size(p, 2) &&
        throw(ArgumentError("sparse QR requires at least as many rows as columns"))
    threshold=Float64(dense)
    isfinite(threshold) || throw(ArgumentError("dense must be finite"))
    algo=algorithm===:auto ? 0 :
         algorithm===:simplicial ? 1 :
         algorithm===:supernodal ? 2 :
         throw(ArgumentError("algorithm must be :auto, :simplicial or :supernodal"))
    if kind!==:cholesky &&
       (ordering!==:default || threshold!=10.0 || !aggressive || algo!=0)
        throw(
            ArgumentError(
                "ordering, AMD controls and algorithm selection currently require kind=:cholesky",
            ),
        )
    end
    custom=ordering isa AbstractVector{<:Integer}
    ordercode=custom ? 2 :
              ordering in (:default, :amd) ? 0 :
              ordering===:identity ? 1 :
              throw(
        ArgumentError("ordering must be :default, :amd, :identity or a permutation"),
    )
    lock(_matrix_lock) do
        perm=custom ? _as_permutation(ordering) : Ptr{UInt8}(C_NULL)
        custom && _check_permutation(p, perm, true)
        h=RustCrate.SparseSymbolic()
        _native_call(
            RustCrate.sparse_analyze,
            h,
            p,
            Int32(code),
            uplo===:L,
            Int32(ordercode),
            perm,
            threshold,
            aggressive,
            Int32(algo),
        )
        return FaerSparseSymbolic{kind,typeof(h)}(h, size(p), uplo)
    end
end
FaerSparseSymbolic(a::FaerSparseMatrix; kwargs...) =
    FaerSparseSymbolic(FaerSparsity(a); kwargs...)
function Base.copy(p::FaerSparseSymbolic{K}) where {K}
    lock(_matrix_lock) do
        h=RustCrate.SparseSymbolic()
        _native_call(RustCrate.sparse_symbolic_clone, h, p)
        return FaerSparseSymbolic{K,typeof(h)}(h, size(p), p.uplo)
    end
end
Base.deepcopy_internal(p::FaerSparseSymbolic, d::IdDict) = get!(() -> copy(p), d, p)
function LinearAlgebra.factorize(
    p::FaerSparseSymbolic{K},
    a::FaerSparseMatrix{T};
    check::Bool = true,
) where {K,T}
    size(p)==size(a) ||
        throw(DimensionMismatch("symbolic analysis and numeric matrix shapes must match"))
    lock(_matrix_lock) do
        snapshot=copy(a)
        finite = if K===:cholesky
            rows, cols, values=SparseArrays.findnz(snapshot)
            all(
                i ->
                    (p.uplo===:L ? rows[i]<cols[i] : rows[i]>cols[i]) ||
                    isfinite(values[i]),
                eachindex(values),
            )
        else
            all(isfinite, Vector(SparseArrays.nonzeros(snapshot)))
        end
        finite || throw(
            ArgumentError(
                "sparse factorization requires finite values in the accessed entries",
            ),
        )
        plan=copy(p)
        h=RustCrate.SparseFactor()
        info=try
            Int(_native_call(RustCrate.sparse_factorize, h, plan, snapshot))
        catch e
            if e isa RustCall.RustError &&
               occursin("numeric matrix sparsity pattern must match", sprint(showerror, e))
                throw(
                    ArgumentError(
                        "numeric matrix sparsity pattern must match the symbolic analysis exactly",
                    ),
                )
            end
            rethrow()
        end
        if check && !iszero(info)
            K===:cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerSparseFactorization{T,K,typeof(h),typeof(plan),typeof(snapshot)}(
            h,
            size(a),
            info,
            plan,
            snapshot,
        )
    end
end
LinearAlgebra.lu(a::FaerSparseMatrix; check::Bool = true) =
    LinearAlgebra.factorize(FaerSparseSymbolic(a; kind = :lu), a; check)
LinearAlgebra.lu(a::FaerSparseMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu(a; kwargs...)
LinearAlgebra.qr(a::FaerSparseMatrix) =
    LinearAlgebra.factorize(FaerSparseSymbolic(a; kind = :qr), a)
const _SparseSelfAdjoint=Union{
    LinearAlgebra.Hermitian{<:Any,<:FaerSparseMatrix},
    LinearAlgebra.Symmetric{<:Real,<:FaerSparseMatrix},
}
function LinearAlgebra.cholesky(
    a::_SparseSelfAdjoint;
    check::Bool = true,
    perm = nothing,
    ordering = :default,
    kwargs...,
)
    !isnothing(perm) &&
        ordering!==:default &&
        throw(ArgumentError("specify either perm or ordering, not both"))
    selected=isnothing(perm) ? ordering : perm
    source=parent(a)
    return LinearAlgebra.factorize(
        FaerSparseSymbolic(
            source;
            kind = :cholesky,
            uplo = a.uplo=='L' ? :L : :U,
            ordering = selected,
            kwargs...,
        ),
        source;
        check,
    )
end
function LinearAlgebra.cholesky(a::FaerSparseMatrix; check::Bool = true, kwargs...)
    LinearAlgebra.ishermitian(a) || throw(
        ArgumentError(
            "Cholesky requires Hermitian input; use Hermitian(A,:L) or Hermitian(A,:U) to select a stored triangle",
        ),
    )
    return LinearAlgebra.cholesky(LinearAlgebra.Hermitian(a, :L); check, kwargs...)
end
LinearAlgebra.factorize(a::FaerSparseMatrix) =
    size(a, 1)==size(a, 2) ? LinearAlgebra.lu(a) : LinearAlgebra.qr(a)
Base.:\(a::FaerSparseMatrix, b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}}) =
    LinearAlgebra.factorize(a)\b
function Base.copy(f::FaerSparseFactorization{T,K}) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        h=RustCrate.SparseFactor()
        LinearAlgebra.issuccess(f) && _native_call(RustCrate.sparse_factor_clone, h, f)
        plan=copy(f.plan)
        source=copy(f.source)
        return FaerSparseFactorization{T,K,typeof(h),typeof(plan),typeof(source)}(
            h,
            size(f),
            f.info,
            plan,
            source,
        )
    end
end
Base.deepcopy_internal(f::FaerSparseFactorization, d::IdDict) = get!(() -> copy(f), d, f)
const _SparseTransposedFactor=Union{
    LinearAlgebra.TransposeFactorization{<:Any,<:FaerSparseFactorization},
    LinearAlgebra.AdjointFactorization{<:Any,<:FaerSparseFactorization},
}
const _SparseSolveFactor=Union{FaerSparseFactorization,_SparseTransposedFactor}
function _sparse_factor_solve(
    f::FaerSparseFactorization{T,K},
    b::AbstractMatrix,
    trans = false,
    adj = false,
) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        size(b, 1)==size(f, trans ? 2 : 1) ||
            throw(DimensionMismatch("sparse factor and RHS rows must match"))
        LinearAlgebra.issuccess(f) || (
            K===:cholesky ? throw(LinearAlgebra.PosDefException(f.info)) :
            throw(LinearAlgebra.SingularException(f.info))
        )
        K===:qr &&
            trans &&
            throw(ArgumentError("faer does not provide transpose/adjoint sparse QR solves"))
        P=promote_type(T, _sparse_input_type(eltype(b)))
        if P!==T
            promoted=LinearAlgebra.factorize(f.plan, FaerSparseMatrix{P}(f.source))
            return _sparse_factor_solve(promoted, b, trans, adj)
        end
        rhs=b isa FaerMatrix ? _coerce(b, T) : FaerMatrix{T}(b)
        dims=(size(f, trans ? 1 : 2), size(b, 2))
        args=(f, rhs, trans, adj)
        return T===Float64 ? _matrix_result(RustCrate.sparse_factor_solve, dims, args...) :
               _typed_result(T, RustCrate.sparse_factor_solve, dims, args...)
    end
end
_sparse_solve(f::FaerSparseFactorization, b) = _sparse_factor_solve(f, b)
_sparse_solve(f::_SparseTransposedFactor, b) =
    _sparse_factor_solve(parent(f), b, true, f isa LinearAlgebra.AdjointFactorization)
Base.:\(f::_SparseSolveFactor, b::AbstractMatrix{<:Number}) = _sparse_solve(f, b)
function Base.:\(f::_SparseSolveFactor, b::AbstractVector{<:Number})
    result=_sparse_solve(f, reshape(b, :, 1))
    return b isa _BorrowVector ? _vector_from_source(result) : vec(Matrix(result))
end
function LinearAlgebra.ldiv!(f::_SparseSolveFactor, b::Union{AbstractVector,AbstractMatrix})
    size(f, 1)==size(f, 2) ||
        throw(DimensionMismatch("two-argument ldiv! requires a square sparse factor"))
    return copyto!(b, f\b)
end
function LinearAlgebra.ldiv!(dest::AbstractMatrix, f::_SparseSolveFactor, b::AbstractMatrix)
    size(dest)==(size(f, 2), size(b, 2)) ||
        throw(DimensionMismatch("sparse solve destination has the wrong shape"))
    return copyto!(dest, f\b)
end
function LinearAlgebra.ldiv!(dest::AbstractVector, f::_SparseSolveFactor, b::AbstractVector)
    length(dest)==size(f, 2) ||
        throw(DimensionMismatch("sparse solve destination has the wrong length"))
    return copyto!(dest, f\b)
end
LinearAlgebra.ishermitian(a::FaerSparseMatrix) = lock(_matrix_lock) do
    _native_call(RustCrate.sparse_is_selfadjoint, a, true)
end
LinearAlgebra.issymmetric(a::FaerSparseMatrix) = lock(_matrix_lock) do
    _native_call(RustCrate.sparse_is_selfadjoint, a, false)
end
# LinearAlgebra specializes real factorization / matching complex Array RHS.
const _SparseSolveFactorOf{T}=Union{
    FaerSparseFactorization{T},
    LinearAlgebra.TransposeFactorization{T,<:FaerSparseFactorization},
    LinearAlgebra.AdjointFactorization{T,<:FaerSparseFactorization},
}
Base.:\(
    f::_SparseSolveFactorOf{T},
    b::Matrix{Complex{T}},
) where {T<:Union{Float32,Float64}} = _sparse_solve(f, b)
Base.:\(
    f::_SparseSolveFactorOf{T},
    b::Vector{Complex{T}},
) where {T<:Union{Float32,Float64}} = vec(Matrix(_sparse_solve(f, reshape(b, :, 1))))
function _refactor_sparse!(
    f::FaerSparseFactorization{T},
    a::FaerSparseMatrix{T};
    check = true,
) where {T}
    lock(_matrix_lock) do
        _handle(f)
        next=LinearAlgebra.factorize(f.plan, a; check)
        oldplan=f.plan
        oldsource=f.source
        f.handle=next.handle
        f.dims=next.dims
        f.info=next.info
        f.plan=next.plan
        f.source=next.source
        close(oldplan)
        close(oldsource)
        return f
    end
end
LinearAlgebra.lu!(
    f::FaerSparseFactorization{T,:lu},
    a::FaerSparseMatrix{T};
    check::Bool = true,
) where {T} = _refactor_sparse!(f, a; check)
LinearAlgebra.qr!(f::FaerSparseFactorization{T,:qr}, a::FaerSparseMatrix{T}) where {T} =
    _refactor_sparse!(f, a)
LinearAlgebra.cholesky!(
    f::FaerSparseFactorization{T,:cholesky},
    a::FaerSparseMatrix{T};
    check::Bool = true,
) where {T} = _refactor_sparse!(f, a; check)

"""Return an independent copy of the permutation used by sparse Cholesky analysis."""
FaerPermutation(p::FaerSparseSymbolic{:cholesky}) =
    _permutation_result(size(p, 1), RustCrate.sparse_cholesky_permutation, p)
function FaerPermutation(f::FaerSparseFactorization{T,:cholesky}) where {T}
    lock(_matrix_lock) do
        _handle(f)
        return FaerPermutation(f.plan)
    end
end
function Base.getproperty(p::FaerSparseSymbolic{:cholesky}, name::Symbol)
    name===:p && return FaerPermutation(p)
    name===:factor_storage && return Int(_native_call(RustCrate.sparse_cholesky_storage, p))
    if name===:algorithm
        return _native_call(RustCrate.sparse_cholesky_algorithm, p)==1 ? :simplicial :
               :supernodal
    end
    return getfield(p, name)
end
Base.propertynames(p::FaerSparseSymbolic{:cholesky}, private::Bool = false) =
    private ? (fieldnames(typeof(p))..., :p, :algorithm, :factor_storage) :
    (:dims, :uplo, :p, :algorithm, :factor_storage)
function Base.getproperty(f::FaerSparseFactorization{T,:cholesky}, name::Symbol) where {T}
    name===:p && return FaerPermutation(f)
    if name===:factor_storage
        return lock(_matrix_lock) do
            _handle(f)
            return f.plan.factor_storage
        end
    end
    if name===:algorithm
        return lock(_matrix_lock) do
            _handle(f)
            return f.plan.algorithm
        end
    end
    return getfield(f, name)
end
Base.propertynames(
    f::FaerSparseFactorization{T,:cholesky},
    private::Bool = false,
) where {T} =
    private ? (fieldnames(typeof(f))..., :p, :algorithm, :factor_storage) :
    (:dims, :info, :p, :algorithm, :factor_storage)
