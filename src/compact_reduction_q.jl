abstract type AbstractFaerCompactQ{T} <: AbstractFaerHouseholder{T} end

function _handle(q::AbstractFaerCompactQ)
    h=getfield(q, :handle)
    isnothing(h) && throw(ArgumentError("compact reduction Q has been closed"))
    return h
end
_native_argument(q::AbstractFaerCompactQ) = _handle(q)
_is_native_owner(::AbstractFaerCompactQ) = true
Base.size(q::AbstractFaerCompactQ) = (q.n, q.n)
Base.size(q::AbstractFaerCompactQ, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? q.n : 1
Base.isopen(q::AbstractFaerCompactQ) =
    lock(()->!isnothing(getfield(q, :handle)), _matrix_lock)
function Base.close(q::AbstractFaerCompactQ)
    lock(_matrix_lock) do
        q.handle=nothing
    end
    return nothing
end
function _compact_q_validate(q::AbstractFaerCompactQ)
    _native_call(
        RustCrate.compact_tridiagonal_validate,
        q,
        UInt(q.n),
        UInt(q.k),
        UInt(q.blocksize),
        _tag(eltype(q)),
    )
    return nothing
end
_dense_rhs_validate(q::AbstractFaerCompactQ) = _compact_q_validate(q)
function _compact_q(
    ::Type{T},
    handle;
    transposed = false,
    conjugated = false,
    wrapper = FaerTridiagonalQ,
) where {T}
    info=zeros(UInt, 4)
    GC.@preserve info _native_call(
        RustCrate.compact_tridiagonal_info,
        handle,
        info,
    )
    info[4]==UInt(_tag(T)) ||
        throw(ArgumentError("compact Q returned an incompatible scalar"))
    return Core.apply_type(wrapper, T, typeof(handle))(
        handle,
        Int(info[1]),
        Int(info[2]),
        Int(info[3]),
        transposed,
        conjugated,
    )
end
function _compact_q_copy(
    q::AbstractFaerCompactQ{T};
    transposed = q.transposed,
    conjugated = q.conjugated,
    duplicate = false,
) where {T}
    lock(_matrix_lock) do
        _compact_q_validate(q)
        h=duplicate ? _native_call(RustCrate.OwnedCompactTridiagonal, q) : _handle(q)
        return _compact_q(T, h; transposed, conjugated, wrapper = _compact_q_wrapper(q))
    end
end
Base.copy(q::AbstractFaerCompactQ) = _compact_q_copy(q; duplicate = true)
Base.deepcopy_internal(q::AbstractFaerCompactQ, d::IdDict) = get!(() -> copy(q), d, q)
Base.transpose(q::AbstractFaerCompactQ) = _compact_q_copy(q; transposed = !q.transposed)
Base.conj(q::AbstractFaerCompactQ) = _compact_q_copy(q; conjugated = !q.conjugated)
Base.adjoint(q::AbstractFaerCompactQ) =
    _compact_q_copy(q; transposed = !q.transposed, conjugated = !q.conjugated)
Base.inv(q::AbstractFaerCompactQ) = adjoint(q)
function Base.getproperty(q::AbstractFaerCompactQ{T}, name::Symbol) where {T}
    if name in (:basis, :factor, :phases)
        return lock(_matrix_lock) do
            _compact_q_validate(q)
            kind=name===:basis ? Int32(0) : name===:factor ? Int32(1) : Int32(2)
            out=name===:phases ? Vector{T}(undef, q.n) :
                Matrix{T}(undef, name===:basis ? (q.k, q.k) : (q.blocksize, q.k))
            GC.@preserve out _native_call(
                RustCrate.compact_tridiagonal_component,
                q,
                UInt(q.n),
                UInt(q.k),
                UInt(q.blocksize),
                _tag(T),
                kind,
                Ptr{UInt8}(pointer(out)),
                UInt(length(out)),
            )
            return out
        end
    elseif name===:nreflections
        return getfield(q, :k)
    end
    return getfield(q, name)
end
Base.propertynames(q::AbstractFaerCompactQ, private::Bool = false) =
    private ? (fieldnames(typeof(q))..., :basis, :factor, :phases, :nreflections) :
    (:basis, :factor, :phases, :nreflections, :blocksize, :transposed, :conjugated)
function Base.getindex(q::AbstractFaerCompactQ{T}, i::Int, j::Int) where {T}
    checkbounds(q, i, j)
    lock(_matrix_lock) do
        _compact_q_validate(q)
        x=zeros(T, q.n)
        x[j]=one(T)
        return (q*x)[i]
    end
end
function _householder_coerce(q::AbstractFaerCompactQ{T}, ::Type{T}) where {T}
    _compact_q_validate(q)
    return q
end
function _householder_coerce(q::AbstractFaerCompactQ, ::Type{T}) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        _compact_q_validate(q)
        h=_native_call(RustCrate.OwnedCompactTridiagonal, q, _tag(T))
        return _compact_q(
            T,
            h;
            transposed = q.transposed,
            conjugated = q.conjugated,
            wrapper = _compact_q_wrapper(q),
        )
    end
end
_householder_workspace_functions(::AbstractFaerCompactQ) =
    (RustCrate.compact_tridiagonal_apply_requirements, RustCrate.compact_tridiagonal_workspace)
_householder_workspace_metadata(q::AbstractFaerCompactQ) =
    (UInt(q.n), UInt(q.k), UInt(q.blocksize), _tag(eltype(q)))
_householder_orientation(q::AbstractFaerCompactQ) = (q.transposed, q.conjugated)
function _householder_native_apply!(q::AbstractFaerCompactQ, a, right)
    req=_householder_workspace_query(
        q,
        right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2),
        right,
    )
    work=FaerScratch(req.bytes; alignment = req.alignment)
    try
        _native_call(
            RustCrate.compact_tridiagonal_workspace,
            a,
            q,
            work,
            _householder_workspace_metadata(q)...,
            right,
            q.transposed,
            q.conjugated,
            nothing,
        )
        return a
    finally
        close(work)
    end
end
Base.:\(q::AbstractFaerCompactQ, a::_HouseholderArray) = q'*a

# Queries validate the original native owner, then select the target scalar's
# kernel layout directly. They never allocate a promoted copy of the reflectors.
function _compact_q_requirement(q::AbstractFaerCompactQ, T, extent, right; team = nothing)
    _iterative_resource_check(nothing, team)
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.compact_tridiagonal_apply_requirements,
        q,
        _householder_workspace_metadata(q)...,
        _tag(T),
        UInt(extent),
        right,
        q.transposed,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
_householder_workspace_query(q::AbstractFaerCompactQ, extent, right; team = nothing) =
    _compact_q_requirement(q, eltype(q), extent, right; team)
function _householder_resource_check(
    q::AbstractFaerCompactQ,
    a,
    P,
    right;
    scratch = nothing,
    team = nothing,
)
    extent=right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2)
    req=_compact_q_requirement(q, P, extent, right; team)
    if !isnothing(scratch)
        scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        cap=scratch_capacity(scratch)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "Householder scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
    end
    return req
end
