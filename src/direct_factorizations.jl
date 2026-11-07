# Direct in-place dense LU / QR / Cholesky: `lu!(A; scratch)`, `qr!(A; scratch)`
# and `cholesky!(A; scratch)` factor A's own native storage with faer's
# low-level `*_in_place` kernels, as Julia's LAPACK-backed `lu!` does.

"""
    FaerInPlaceFactorization

Factorization returned by `lu!(A; scratch)`, `qr!(A[, NoPivot()]; scratch)` and
`cholesky!(A; scratch)` for a `FaerMatrix` `A`. Following Julia's `lu!`, the
packed factor lives in A itself: `F.factors === A`. LU stores U on/above the
diagonal and the strict unit-lower L below it; QR stores R on/above the
diagonal and faer's Householder basis below it, with the block Householder
coefficients (`F.Qcoeff`, not LAPACK's τ) kept in a Julia-owned buffer of F;
Cholesky stores L (or U for an upper `Hermitian`/`Symmetric` wrapper) and
zeroes the other triangle. The LU row permutation is kept in Julia vectors.

F aliases A: later writes to A change F, and closing, resizing or truncating A
invalidates F (its operations then throw `ArgumentError`). `close(F)` only
invalidates F. Solves (`\\`, `ldiv!` with optional `scratch=`/`team=`, transpose
and adjoint for square factors, tall QR least squares), `L/U/p/P/R/Q/Qthin/
Qbasis/Qcoeff/packed`, `reconstruct(F)`/`Matrix(F)`/`FaerMatrix(F)` and their
typed forms, `inv`, and `det/logabsdet/logdet` (LU and
Cholesky) read A's storage. `copy(F)` copies A and F's buffers into a new direct
factor. Other retained-factor operations (`rdiv!`, linear-operator and
preconditioner use, pseudoinverse, mixed-scalar promotion, rectangular-LU or
wide-QR solves) require the default `lu!(A)`/`qr!(A)`/`cholesky!(A)` result.
"""
mutable struct FaerInPlaceFactorization{T,K,M<:FaerMatrix{T}} <:
               LinearAlgebra.Factorization{T}
    factors::M
    perm::Vector{UInt}
    perm_inv::Vector{UInt}
    coeff::Matrix{T}
    dims::Tuple{Int,Int}
    epoch::UInt
    info::Int
    uplo::Symbol
    open::Bool
end

_direct_kind(::Val{:lu}) = Int32(0)
_direct_kind(::Val{:qr}) = Int32(2)
_direct_kind(::Val{:cholesky}) = Int32(4)
_direct_kind(::FaerInPlaceFactorization{T,K}) where {T,K} = _direct_kind(Val(K))

# Every native call goes through this check: A must be open, unresized and of
# the recorded shape, so a stale factor never reads reallocated storage.
function _direct_source(f::FaerInPlaceFactorization)
    getfield(f, :open) || throw(ArgumentError("FaerInPlaceFactorization has been closed"))
    a=getfield(f, :factors)
    isopen(a) &&
        a.epoch==getfield(f, :epoch) &&
        size(a)==getfield(f, :dims) ||
        throw(
            ArgumentError(
                "the matrix holding this in-place factorization was closed, resized or truncated",
            ),
        )
    _handle(a)
    return a
end

function _direct_requirements(kind, op, m, n, k, ::Type{T}, team) where {T}
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    info=zeros(UInt, 3)
    _native_call(
        RustCrate.direct_factor_requirements,
        Int32(kind),
        Int32(op),
        UInt(m),
        UInt(n),
        UInt(k),
        _tag(T),
        team,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]), block = Int(info[3]))
end
function _direct_scratch_check(scratch, req, name)
    scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    cap=scratch_capacity(scratch)
    cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
        ArgumentError("$name scratch capacity/alignment is insufficient; call reserve!"),
    )
    return scratch
end
_coeff_pointer(c::Matrix) = Ptr{UInt8}(isempty(c) ? C_NULL : pointer(c))
# A zero-column coefficient buffer still needs a valid, non-null pointer.
_qr_coeff_pointer(c::Matrix) = Ptr{UInt8}(pointer(c))

function _direct_factor!(a::FaerMatrix{T}, ::Val{K}, lower::Bool, scratch, team) where {T,K}
    lock(_matrix_lock) do
        _handle(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        m, n=size(a)
        K===:cholesky &&
            m!=n &&
            throw(DimensionMismatch("factorization requires a square matrix"))
        kind=_direct_kind(Val(K))
        req=_direct_requirements(kind, 0, m, n, 0, T, team)
        _direct_scratch_check(scratch, req, "factorization")
        perm=Vector{UInt}(undef, K===:lu ? m : 0)
        perm_inv=Vector{UInt}(undef, K===:lu ? m : 0)
        coeff=Matrix{T}(undef, K===:qr ? req.block : 0, K===:qr ? min(m, n) : 0)
        info=GC.@preserve coeff Int(
            _native_call(
                RustCrate.direct_factor,
                a,
                kind,
                lower,
                UInt(m),
                UInt(n),
                _tag(T),
                perm,
                perm_inv,
                K===:qr ? _qr_coeff_pointer(coeff) : _coeff_pointer(coeff),
                UInt(size(coeff, 1)),
                scratch,
                team,
            ),
        )
        return FaerInPlaceFactorization{T,K,typeof(a)}(
            a,
            perm,
            perm_inv,
            coeff,
            (m, n),
            a.epoch,
            info,
            lower ? :L : :U,
            true,
        )
    end
end
function _direct_non_hermitian(a::FaerMatrix{T}, check::Bool) where {T}
    LinearAlgebra.checksquare(a)
    check && throw(LinearAlgebra.PosDefException(-1))
    return FaerInPlaceFactorization{T,:cholesky,typeof(a)}(
        a,
        UInt[],
        UInt[],
        Matrix{T}(undef, 0, 0),
        size(a),
        a.epoch,
        -1,
        :L,
        true,
    )
end

function _direct_lu!(a::FaerMatrix, check, allowsingular, scratch, team)
    f=_direct_factor!(a, Val(:lu), true, scratch, team)
    check && !allowsingular && f.info!=0 && throw(LinearAlgebra.SingularException(f.info))
    return f
end
_direct_qr!(a::FaerMatrix, scratch, team) = _direct_factor!(a, Val(:qr), true, scratch, team)
function _direct_cholesky!(a::FaerMatrix, lower::Bool, hermitian::Bool, check, scratch, team)
    lock(_matrix_lock) do
        _handle(a)
        hermitian || return _direct_non_hermitian(a, check)
        f=_direct_factor!(a, Val(:cholesky), lower, scratch, team)
        check && f.info!=0 && throw(LinearAlgebra.PosDefException(f.info))
        return f
    end
end

"""
    scratch_requirements(lu!, A::FaerMatrix; team=nothing)
    scratch_requirements(qr!, A::FaerMatrix[, NoPivot()]; team=nothing)
    scratch_requirements(cholesky!, A; team=nothing)

Query faer's exact workspace for the direct in-place factorization of A (a
`FaerMatrix`, or for Cholesky also a `Hermitian`/real `Symmetric` wrapper of one).
Pass the reserved `FaerScratch` as `scratch=` to `lu!`, `qr!` or `cholesky!` to
select the direct path; see [`FaerInPlaceFactorization`](@ref). The query does
not read A's coefficients. Use the same team for query and computation.
"""
function scratch_requirements(::typeof(LinearAlgebra.lu!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(0, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function scratch_requirements(::typeof(LinearAlgebra.qr!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(2, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
scratch_requirements(
    ::typeof(LinearAlgebra.qr!),
    a::FaerMatrix,
    ::LinearAlgebra.NoPivot;
    team = nothing,
) = scratch_requirements(LinearAlgebra.qr!, a; team)
const _DirectCholeskyInput = Union{
    FaerMatrix,
    LinearAlgebra.Hermitian{<:Any,<:FaerMatrix},
    LinearAlgebra.Symmetric{<:Any,<:FaerMatrix},
}
function scratch_requirements(
    ::typeof(LinearAlgebra.cholesky!),
    a::_DirectCholeskyInput;
    team = nothing,
)
    lock(_matrix_lock) do
        source=a isa FaerMatrix ? a : parent(a)
        _handle(source)
        n=LinearAlgebra.checksquare(source)
        req=_direct_requirements(4, 0, n, n, 0, eltype(source), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
for (fn, signature, args) in (
    (:(LinearAlgebra.lu!), :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix, p::LinearAlgebra.NoPivot), :(a, p)),
    (:(LinearAlgebra.cholesky!), :(a::_DirectCholeskyInput,), :(a,)),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); team = nothing)
        req=scratch_requirements($fn, $(args.args...); team)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        team = nothing,
    )
        req=scratch_requirements($fn, $(args.args...); team)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end

Base.size(f::FaerInPlaceFactorization) = getfield(f, :dims)
Base.size(f::FaerInPlaceFactorization, d::Integer) =
    d < 1 ? throw(ArgumentError("dimension must be positive")) : d <= 2 ? size(f)[d] : 1
LinearAlgebra.issuccess(f::FaerInPlaceFactorization) = getfield(f, :info)==0
Base.isopen(f::FaerInPlaceFactorization) = lock(_matrix_lock) do
    getfield(f, :open) || return false
    a=getfield(f, :factors)
    return isopen(a) && a.epoch==getfield(f, :epoch) && size(a)==getfield(f, :dims)
end
function Base.close(f::FaerInPlaceFactorization)
    lock(_matrix_lock) do
        f.open=false
    end
    return nothing
end

function _direct_part(f::FaerInPlaceFactorization{T}, code, dims) where {T}
    lock(_matrix_lock) do
        a=_direct_source(f)
        coeff=getfield(f, :coeff)
        args=(
            a,
            _direct_kind(f),
            Int32(code),
            getfield(f, :uplo)===:L,
            UInt.(size(f))...,
            _tag(T),
            getfield(f, :perm),
            getfield(f, :perm_inv),
            _direct_kind(f)==2 ? _qr_coeff_pointer(coeff) : _coeff_pointer(coeff),
            UInt(size(coeff, 1)),
        )
        GC.@preserve coeff begin
            T===Float64 ? _matrix_result(RustCrate.direct_part, dims, args...) :
            _typed_result(T, RustCrate.direct_part, dims, args...)
        end
    end
end
function _direct_permutation(f::FaerInPlaceFactorization)
    lock(_matrix_lock) do
        _direct_source(f)
        return Int.(getfield(f, :perm)) .+ 1
    end
end
function Base.getproperty(f::FaerInPlaceFactorization{T,K}, name::Symbol) where {T,K}
    m, n=size(f)
    k=min(m, n)
    if name===:factors
        lock(() -> _direct_source(f), _matrix_lock)
        return getfield(f, :factors)
    elseif name===:packed
        K===:cholesky && return _direct_part(f, 1, (n, n))
        return lock(() -> copy(_direct_source(f)), _matrix_lock)
    elseif K===:lu && name===:L
        return _direct_part(f, 1, (m, k))
    elseif K===:lu && name===:U
        return _direct_part(f, 2, (k, n))
    elseif K===:lu && name===:p
        return _direct_permutation(f)
    elseif K===:lu && name===:P
        return FaerMatrix(Matrix{T}(LinearAlgebra.I, m, m)[f.p, :])
    elseif K===:qr && name===:Q
        return _direct_part(f, 3, (m, m))
    elseif K===:qr && name===:Qthin
        return _direct_part(f, 6, (m, k))
    elseif K===:qr && name===:R
        return _direct_part(f, 4, (k, n))
    elseif K===:qr && name===:Qbasis
        return _direct_part(f, 8, (m, k))
    elseif K===:qr && name===:Qcoeff
        return _direct_part(f, 9, size(getfield(f, :coeff)))
    elseif K===:cholesky && name===:L
        return _direct_part(f, 1, (n, n))
    elseif K===:cholesky && name===:U
        return _direct_part(f, 2, (n, n))
    elseif K===:cholesky && name===:UL
        return getfield(f, :uplo)===:U ? f.U : f.L
    end
    return getfield(f, name)
end
function Base.propertynames(f::FaerInPlaceFactorization{T,K}, private::Bool = false) where {T,K}
    props =
        K===:lu ? (:L, :U, :p, :P) :
        K===:qr ? (:Q, :Qthin, :R, :Qbasis, :Qcoeff) : (:L, :U, :UL)
    props=(:factors, props..., :packed, :info, :uplo)
    return private ? (props..., :perm, :perm_inv, :coeff, :dims, :epoch, :open) : props
end

function _cholesky_failure(f::FaerInPlaceFactorization{T,K}) where {T,K}
    info=getfield(f, :info)
    K===:cholesky ? LinearAlgebra.PosDefException(info) :
    LinearAlgebra.SingularException(info)
end
FaerMatrix(f::FaerInPlaceFactorization) = _direct_part(f, 0, size(f))
FaerMatrix{T}(f::FaerInPlaceFactorization) where {T<:_NativeScalar} =
    _typed_dense_reconstruction(T, f)
reconstruct(f::FaerInPlaceFactorization) = FaerMatrix(f)
Base.Matrix(f::FaerInPlaceFactorization) = _convert_dense_reconstruction(Matrix, f)
Base.Matrix{T}(f::FaerInPlaceFactorization) where {T} =
    _convert_dense_reconstruction(Matrix{T}, f)
Base.Array(f::FaerInPlaceFactorization) = Matrix(f)
function Base.inv(f::FaerInPlaceFactorization{T,K}) where {T,K}
    size(f, 1)==size(f, 2) || throw(DimensionMismatch("inverse requires a square factor"))
    LinearAlgebra.issuccess(f) || throw(_cholesky_failure(f))
    if K===:qr
        d=Matrix(_direct_part(f, 11, (size(f, 1), 1)))
        i=findfirst(iszero, d)
        isnothing(i) || throw(LinearAlgebra.SingularException(i[1]))
    end
    return _direct_part(f, 10, size(f))
end
function LinearAlgebra.logabsdet(f::FaerInPlaceFactorization{T,K}) where {T,K}
    K===:qr && throw(
        ArgumentError(
            "logabsdet/det are not provided for in-place QR; use the default qr! result",
        ),
    )
    n=LinearAlgebra.checksquare(f)
    LinearAlgebra.issuccess(f) ||
        K===:lu ||
        throw(LinearAlgebra.PosDefException(getfield(f, :info)))
    d=vec(Matrix(_direct_part(f, 11, (n, 1))))
    if K===:cholesky
        return (2*sum(log∘real, d; init = zero(real(T))), one(T))
    end
    s=_permutation_parity(_direct_permutation(f)) ? -one(T) : one(T)
    return (sum(log∘abs, d; init = zero(real(T))), s*prod(sign, d; init = one(T)))
end
# True when the permutation has odd parity (an odd number of even cycles).
function _permutation_parity(p::AbstractVector{<:Integer})
    seen=falses(length(p))
    odd=false
    for i in eachindex(p)
        seen[i] && continue
        len=0
        j=i
        while !seen[j]
            seen[j]=true
            j=p[j]
            len+=1
        end
        iseven(len) && (odd=!odd)
    end
    return odd
end
function LinearAlgebra.det(f::FaerInPlaceFactorization)
    value, phase=LinearAlgebra.logabsdet(f)
    return phase*exp(value)
end
function LinearAlgebra.logdet(f::FaerInPlaceFactorization)
    value, phase=LinearAlgebra.logabsdet(f)
    eltype(f)<:Real &&
        phase<0 &&
        throw(DomainError(phase, "real logdet requires a nonnegative determinant"))
    return value + log(phase)
end
LinearAlgebra.isposdef(f::FaerInPlaceFactorization{T,:cholesky}) where {T} =
    LinearAlgebra.issuccess(f)

function Base.copy(f::FaerInPlaceFactorization{T,K}) where {T,K}
    lock(_matrix_lock) do
        a=copy(_direct_source(f))
        return FaerInPlaceFactorization{T,K,typeof(a)}(
            a,
            copy(getfield(f, :perm)),
            copy(getfield(f, :perm_inv)),
            copy(getfield(f, :coeff)),
            size(f),
            a.epoch,
            getfield(f, :info),
            getfield(f, :uplo),
            true,
        )
    end
end
Base.deepcopy_internal(f::FaerInPlaceFactorization, dict::IdDict) =
    get!(() -> copy(f), dict, f)

const _DirectTransposed = Union{
    LinearAlgebra.TransposeFactorization{<:Any,<:FaerInPlaceFactorization},
    LinearAlgebra.AdjointFactorization{<:Any,<:FaerInPlaceFactorization},
}
_direct_op(::FaerInPlaceFactorization) = 1
_direct_op(::LinearAlgebra.TransposeFactorization) = 2
_direct_op(::LinearAlgebra.AdjointFactorization) = 3
_direct_parent(f::FaerInPlaceFactorization) = f
_direct_parent(f::_DirectTransposed) = parent(f)
const _DirectSolveFactor = Union{FaerInPlaceFactorization,_DirectTransposed}

function _direct_solve_rows(f::FaerInPlaceFactorization{T,K}, op) where {T,K}
    m, n=size(f)
    if K===:qr && op==1
        m>=n || throw(
            DimensionMismatch(
                "in-place QR solves require a tall or square factor; wide QR uses the default qr! result",
            ),
        )
        return m
    end
    m==n || throw(
        DimensionMismatch(
            "in-place solves require a square factor (or a tall QR least-squares solve)",
        ),
    )
    return n
end

"""
    scratch_requirements(ldiv!, F::FaerInPlaceFactorization, B::FaerMatrix; team=nothing)

Query the workspace of `ldiv!(F, B; scratch, team)` (also for `transpose(F)` and
`adjoint(F)`). The solve overwrites B in place without native allocation of its
own; tall QR writes the least-squares solution into `B[1:n, :]`.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        _direct_source(f)
        _handle(b)
        op=_direct_op(g)
        _direct_solve_rows(f, op)
        req=_direct_requirements(_direct_kind(f), op, size(f)..., size(b, 2), eltype(f), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function FaerScratch(
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.ldiv!, g, b; team)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    work::FaerScratch,
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.ldiv!, g, b; team)
    return reserve!(work, req.bytes; alignment = req.alignment)
end

function _direct_ldiv!(g::_DirectSolveFactor, b::FaerMatrix, scratch, team)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        T=eltype(f)
        a=_direct_source(f)
        eltype(b)===T || throw(
            ArgumentError("in-place factor solves require a right-hand side of eltype $T"),
        )
        _handle(b)===_handle(a) && throw(
            ArgumentError(
                "the right-hand side must not share storage with the in-place factorization",
            ),
        )
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        LinearAlgebra.issuccess(f) || throw(_cholesky_failure(f))
        op=_direct_op(g)
        rows=_direct_solve_rows(f, op)
        size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
        kind=_direct_kind(f)
        req=_direct_requirements(kind, op, size(f)..., size(b, 2), T, team)
        temporary=isnothing(scratch)
        work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        try
            _direct_scratch_check(work, req, "solve")
            coeff=getfield(f, :coeff)
            GC.@preserve coeff _native_call(
                RustCrate.direct_solve,
                a,
                b,
                kind,
                Int32(op),
                getfield(f, :uplo)===:L,
                UInt.(size(f))...,
                _tag(T),
                getfield(f, :perm),
                getfield(f, :perm_inv),
                kind==2 ? _qr_coeff_pointer(coeff) : _coeff_pointer(coeff),
                UInt(size(coeff, 1)),
                work,
                team,
            )
        finally
            temporary && close(work)
        end
        return b
    end
end
LinearAlgebra.ldiv!(
    f::FaerInPlaceFactorization,
    b::FaerMatrix;
    scratch = nothing,
    team = nothing,
) = _direct_ldiv!(f, b, scratch, team)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval LinearAlgebra.ldiv!(
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
        b::FaerMatrix;
        scratch = nothing,
        team = nothing,
    ) = _direct_ldiv!(f, b, scratch, team)
end

function _direct_backslash(g::_DirectSolveFactor, b::AbstractVecOrMat)
    f=_direct_parent(g)
    T=eltype(f)
    op=_direct_op(g)
    rows=lock(() -> (_direct_source(f); _direct_solve_rows(f, op)), _matrix_lock)
    size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
    x=FaerMatrix{T}(b isa AbstractVector ? reshape(collect(b), :, 1) : b)
    _direct_ldiv!(g, x, nothing, nothing)
    n=size(f, 2)
    rows==n || truncate!(x, n, size(x, 2))
    return b isa AbstractVector ? vec(Matrix(x)) : x
end
# A real factor solves a complex RHS as two real solves, like LinearAlgebra.
_direct_backslash_complex(f, b) =
    _direct_backslash(f, real.(b)) .+ im .* _direct_backslash(f, imag.(b))
Base.:\(f::FaerInPlaceFactorization, b::AbstractVecOrMat) = _direct_backslash(f, b)
Base.:\(
    f::FaerInPlaceFactorization{T},
    b::VecOrMat{Complex{T}},
) where {T<:Union{Float32,Float64}} = _direct_backslash_complex(f, b)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval Base.:\(
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
        b::AbstractVecOrMat,
    ) = _direct_backslash(f, b)
    @eval Base.:\(
        f::LinearAlgebra.$W{T,<:FaerInPlaceFactorization},
        b::VecOrMat{Complex{T}},
    ) where {T<:Union{Float32,Float64}} = _direct_backslash_complex(f, b)
end
