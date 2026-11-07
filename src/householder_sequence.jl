"""
    FaerHouseholderSequence(basis, tau; blocksize=min(32,max(1,size(basis,2))), team=nothing)
    FaerHouseholderSequence(basis, factor::AbstractMatrix; blocksize=size(factor,1), prev_block_size=nothing, team=nothing)

Own the compact unitary product `Q = H₁*H₂*...*Hₖ` in faer's block format.
Basis is m-by-k with k<=m; diagonal ones and upper zeros are implicit, and only
strict-lower essentials are used. Each real denominator tau[j] defines
`Hⱼ = I - vⱼ*vⱼ'/tau[j]`; finite denominators must agree with `vⱼ'vⱼ/2`.
Positive infinity denotes identity and its unused essentials are canonicalized
to zero. Source arrays are copied. The factor-matrix form accepts faer's
blocksize-by-k layout and validates relevant upper blocks against the basis.

`Q.basis` and `Q.factor` return independent native matrices, `Q.tau` returns a
Julia real vector. These expose the base representation even for transformed
objects; Q.transposed/Q.conjugated record its orientation. `reblock(Q,b)` and
`reblock!(Q,b)` rebuild factors using faer's upgrade API. Array products and
lmul!/rmul!/mul! do not materialize Q; Matrix(Q) and operator composition do.
All four native scalars, promotion, views, left solve, transpose/adjoint/conj,
copy/deepcopy and close are supported. In-place application accepts `scratch=`
and `team=`; see `scratch_requirements`. Construction and reblock accept `team=`.
Factor-input construction and reblock accept a previous block size dividing both
old and target sizes (default gcd), and reuse saved upper blocks. The native
upgrade has no external scratch API and may allocate internal reduction buffers;
construction/reblock do not claim caller-controlled workspace.
"""
mutable struct FaerHouseholderSequence{T,H} <: AbstractFaerHouseholder{T}
    handle::Union{Nothing,H}
    n::Int
    k::Int
    blocksize::Int
    transposed::Bool
    conjugated::Bool
end
function _handle(q::FaerHouseholderSequence)
    h=getfield(q, :handle)
    isnothing(h) && throw(ArgumentError("Householder sequence has been closed"))
    return h
end
_native_argument(q::FaerHouseholderSequence) = _handle(q)
_is_native_owner(::FaerHouseholderSequence) = true
Base.size(q::FaerHouseholderSequence) = (q.n, q.n)
Base.size(q::FaerHouseholderSequence, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? q.n : 1
Base.isopen(q::FaerHouseholderSequence) =
    lock(()->!isnothing(getfield(q, :handle)), _matrix_lock)
function Base.close(q::FaerHouseholderSequence)
    lock(_matrix_lock) do ;
        q.handle=nothing
    end
    return nothing
end
function _sequence_build(
    ::Type{T},
    source,
    tau,
    blocksize;
    check = true,
    provided = nothing,
    transposed = false,
    conjugated = false,
    team = nothing,
) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        _dense_rhs_validate(source)
        _dense_rhs_validate(tau)
        isnothing(provided)||_dense_rhs_validate(provided)
        m, k=size(source)
        k<=m || throw(DimensionMismatch("Householder basis requires rows >= columns"))
        length(tau)==k || throw(
            DimensionMismatch("Householder denominator count must match basis columns"),
        )
        b=Int(blocksize)
        b>0 || throw(ArgumentError("Householder blocksize must be positive"))
        Base.checked_mul(m, k)
        Base.checked_mul(b, k)
        R=_realtype(T)
        original=Matrix(source)
        basis=Matrix{T}(undef, m, k)
        denominators=Vector{T}(undef, k)
        for j = 1:k
            isreal(tau[j]) || throw(ArgumentError("Householder denominators must be real"))
            t=convert(R, real(tau[j]))
            t>0 || throw(ArgumentError("Householder denominators must be positive or Inf"))
            denominators[j]=t
            for i = 1:m
                basis[i, j]=i<j ? zero(T) :
                            i==j ? one(T) : isinf(t) ? zero(T) : convert(T, original[i, j])
                isfinite(basis[i, j]) ||
                    throw(ArgumentError("Householder essentials must be finite"))
            end
            if check && isfinite(t)
                isapprox(
                    LinearAlgebra.norm(view(basis, j:m, j))/sqrt(t),
                    sqrt(R(2));
                    rtol = sqrt(eps(R)),
                ) || throw(
                    ArgumentError(
                        "Householder denominator must equal the squared basis norm / 2",
                    ),
                )
            end
        end
        factor=isnothing(provided) ? nothing : Matrix{T}(provided)
        isnothing(factor) ||
            size(factor)==(b, k) ||
            throw(DimensionMismatch("provided Householder factor shape mismatch"))
        h=RustCrate.OwnedHouseholderSequence()
        ptr=isnothing(factor) ? Ptr{UInt8}(C_NULL) : Ptr{UInt8}(pointer(factor))
        args=(
            h,
            Ptr{UInt8}(pointer(basis)),
            UInt(m),
            UInt(k),
            Ptr{UInt8}(pointer(denominators)),
            UInt(b),
            _tag(T),
            ptr,
        )
        GC.@preserve basis denominators factor begin
            _native_call(RustCrate.sequence_load, args..., team)
        end
        return FaerHouseholderSequence{T,typeof(h)}(h, m, k, b, transposed, conjugated)
    end
end
function FaerHouseholderSequence{T}(
    basis::AbstractMatrix,
    tau::AbstractVector;
    blocksize = min(32, max(1, size(basis, 2))),
    team = nothing,
) where {T<:_NativeScalar}
    return _sequence_build(T, basis, tau, blocksize; team)
end
function FaerHouseholderSequence(basis::AbstractMatrix, tau::AbstractVector; kwargs...)
    T=promote_type(
        _reflector_type(eltype(basis)),
        _reflector_type(typeof(real(zero(eltype(tau))))),
    )
    return FaerHouseholderSequence{T}(basis, tau; kwargs...)
end
function FaerHouseholderSequence{T}(
    basis::AbstractMatrix,
    factor::AbstractMatrix;
    blocksize = size(factor, 1),
    prev_block_size = nothing,
    team = nothing,
) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        _dense_rhs_validate(basis)
        _dense_rhs_validate(factor)
        old, k=size(factor)
        old>0 || throw(ArgumentError("Householder factor must have at least one row"))
        size(basis, 2)==k ||
            throw(DimensionMismatch("Householder factor and basis columns must match"))
        b, prev=_sequence_reblock_sizes(old, blocksize, prev_block_size)
        tau=[factor[mod1(j, old), j] for j = 1:k]
        generated=_sequence_build(T, basis, tau, old; team)
        canonical=zeros(T, old, k)
        try
            expected=Matrix(generated.factor)
            for j = 1:k, i = 1:mod1(j, old)
                value=convert(T, factor[i, j])
                (
                    value==expected[i, j] || isapprox(
                        value,
                        expected[i, j];
                        rtol = sqrt(eps(_realtype(T))),
                        atol = 10eps(_realtype(T)),
                    )
                ) || throw(
                    ArgumentError(
                        "Householder factor is inconsistent with the basis/denominators",
                    ),
                )
                canonical[i, j]=value
            end
        finally
            close(generated)
        end
        prepared=_sequence_build(
            T,
            basis,
            tau,
            old;
            check = false,
            provided = canonical,
            team,
        )
        try
            return reblock(prepared, b; prev_block_size = prev, team)
        finally
            close(prepared)
        end
    end
end
function FaerHouseholderSequence(basis::AbstractMatrix, factor::AbstractMatrix; kwargs...)
    T=promote_type(_reflector_type(eltype(basis)), _reflector_type(eltype(factor)))
    return FaerHouseholderSequence{T}(basis, factor; kwargs...)
end
function _sequence_copy(
    q::FaerHouseholderSequence{T};
    transposed = q.transposed,
    conjugated = q.conjugated,
) where {T}
    lock(_matrix_lock) do
        h=RustCrate.OwnedHouseholderSequence()
        _native_call(RustCrate.sequence_clone, h, q)
        return FaerHouseholderSequence{T,typeof(h)}(
            h,
            q.n,
            q.k,
            q.blocksize,
            transposed,
            conjugated,
        )
    end
end
Base.copy(q::FaerHouseholderSequence) = _sequence_copy(q)
Base.deepcopy_internal(q::FaerHouseholderSequence, d::IdDict) = get!(() -> copy(q), d, q)
Base.transpose(q::FaerHouseholderSequence) = _sequence_copy(q; transposed = !q.transposed)
Base.conj(q::FaerHouseholderSequence) = _sequence_copy(q; conjugated = !q.conjugated)
Base.adjoint(q::FaerHouseholderSequence) =
    _sequence_copy(q; transposed = !q.transposed, conjugated = !q.conjugated)
Base.inv(q::FaerHouseholderSequence) = adjoint(q)
function Base.getproperty(q::FaerHouseholderSequence{T}, name::Symbol) where {T}
    if name in (:basis, :factor)
        factor=name===:factor
        dims=factor ? (q.blocksize, q.k) : (q.n, q.k)
        return T===Float64 ? _matrix_result(RustCrate.sequence_component, dims, q, factor) :
               _typed_result(T, RustCrate.sequence_component, dims, q, factor)
    elseif name===:tau
        out=Vector{T}(undef, q.k)
        GC.@preserve out _native_call(
            RustCrate.sequence_taus,
            q,
            Ptr{UInt8}(pointer(out)),
            UInt(length(out)),
        )
        return real.(out)
    elseif name===:nreflections
        return getfield(q, :k)
    end
    return getfield(q, name)
end
Base.propertynames(q::FaerHouseholderSequence, private::Bool = false) =
    private ? (fieldnames(typeof(q))..., :basis, :factor, :tau, :nreflections) :
    (:basis, :factor, :tau, :blocksize, :nreflections, :transposed, :conjugated)
function Base.getindex(q::FaerHouseholderSequence{T}, i::Int, j::Int) where {T}
    checkbounds(q, i, j)
    lock(_matrix_lock) do
        _handle(q)
        unit=zeros(T, q.n)
        unit[j]=one(T)
        return (q*unit)[i]
    end
end
function _householder_coerce(q::FaerHouseholderSequence{T}, ::Type{T}) where {T}
    _handle(q)
    return q
end
function _householder_coerce(q::FaerHouseholderSequence, ::Type{T}) where {T<:_NativeScalar}
    return _sequence_build(
        T,
        q.basis,
        q.tau,
        q.blocksize;
        check = false,
        transposed = q.transposed,
        conjugated = q.conjugated,
    )
end
_householder_native_apply!(q::FaerHouseholderSequence, a, right) =
    _native_call(RustCrate.sequence_apply, a, q, right, q.transposed, q.conjugated)
Base.:\(q::FaerHouseholderSequence, a::_HouseholderArray) = q'*a
function _sequence_reblock_sizes(old, b, prev)
    b=Int(b)
    b>0 || throw(ArgumentError("Householder blocksize must be positive"))
    p=isnothing(prev) ? gcd(old, b) : Int(prev)
    p>0 && old%p==0 && b%p==0 ||
        throw(ArgumentError("prev_block_size must positively divide old and new blocksize"))
    return b, p
end
"""
    reblock(Q, blocksize; prev_block_size=gcd(Q.blocksize,blocksize), team=nothing)

Return an independent sequence, reusing validated previous blocks stored in Q.
`prev_block_size` must positively divide both old and new block sizes. Factors
and basis stay native; orientation is preserved. The native upgrade may allocate
internal reduction buffers; it has no external scratch API.
"""
function reblock(
    q::FaerHouseholderSequence,
    b::Integer;
    prev_block_size = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        _handle(q)
        _native_call(
            RustCrate.sequence_validate,
            q,
            UInt(q.n),
            UInt(q.k),
            UInt(q.blocksize),
            _tag(eltype(q)),
        )
        b, prev=_sequence_reblock_sizes(q.blocksize, b, prev_block_size)
        Base.checked_mul(b, q.k)
        h=RustCrate.OwnedHouseholderSequence()
        args=(
            h,
            q,
            UInt(b),
            UInt(prev),
            UInt(q.n),
            UInt(q.k),
            UInt(q.blocksize),
            _tag(eltype(q)),
        )
        _native_call(RustCrate.sequence_reblock_from, args..., team)
        return FaerHouseholderSequence{eltype(q),typeof(h)}(
            h,
            q.n,
            q.k,
            b,
            q.transposed,
            q.conjugated,
        )
    end
end
"""Rebuild the compact factors transactionally; previously exported basis/factor matrices remain independent."""
function reblock!(q::FaerHouseholderSequence, b::Integer; kwargs...)
    lock(_matrix_lock) do
        replacement=reblock(q, b; kwargs...)
        q.handle=replacement.handle
        q.blocksize=replacement.blocksize
        return q
    end
end
# Native destination action; see `_householder_into!`.
function _householder_into_native!(
    h::FaerHouseholderSequence,
    p,
    d,
    q,
    e,
    right,
    al,
    be,
    az,
    bz;
    scratch,
    team,
)
    _native_call(
        RustCrate.sequence_mul_into,
        p,
        d,
        q,
        e,
        h,
        right,
        h.transposed,
        h.conjugated,
        al,
        be,
        az,
        bz,
        scratch,
        team,
    )
end
