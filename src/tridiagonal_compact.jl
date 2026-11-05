"""
    FaerTridiagonalQ

Owned implicit Q returned by `tridiagonalize(A; compact=true)`. Products and
in-place Householder operations apply saved native reflectors and complex phases
without constructing dense Q. `Matrix(Q)` explicitly materializes it.

`basis`, `factor` and `phases` return independent Julia arrays of the base representation:
Q = diag(1, HouseholderSequence(basis,factor)) * Diagonal(phases).
Orientation is recorded by `transposed`/`conjugated`. Transpose, adjoint and conj
share immutable native storage; closing one Julia wrapper leaves the others open.
Copy/deepcopy duplicate the native storage. Kernel scratch and teams are supplied
independently for later applications; the construction policy is not retained.
"""
mutable struct FaerTridiagonalQ{T,H} <: AbstractFaerHouseholder{T}
    handle::Union{Nothing,H}
    n::Int
    k::Int
    blocksize::Int
    transposed::Bool
    conjugated::Bool
end
function _handle(q::FaerTridiagonalQ)
    h=getfield(q, :handle)
    isnothing(h) && throw(ArgumentError("compact tridiagonal Q has been closed"))
    return h
end
_native_argument(q::FaerTridiagonalQ) = _handle(q)
_is_native_owner(::FaerTridiagonalQ) = true
Base.size(q::FaerTridiagonalQ) = (q.n, q.n)
Base.size(q::FaerTridiagonalQ, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? q.n : 1
Base.isopen(q::FaerTridiagonalQ) = lock(()->!isnothing(getfield(q, :handle)), _matrix_lock)
function Base.close(q::FaerTridiagonalQ)
    lock(_matrix_lock) do
        q.handle=nothing
    end
    return nothing
end
function _compact_q_validate(q::FaerTridiagonalQ)
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
_dense_rhs_validate(q::FaerTridiagonalQ) = _compact_q_validate(q)
function _compact_q(::Type{T}, handle; transposed = false, conjugated = false) where {T}
    info=zeros(UInt, 4)
    GC.@preserve info _native_call(
        RustCrate.compact_tridiagonal_info,
        handle,
        pointer(info),
    )
    info[4]==UInt(_tag(T)) ||
        throw(ArgumentError("compact Q returned an incompatible scalar"))
    return FaerTridiagonalQ{T,typeof(handle)}(
        handle,
        Int(info[1]),
        Int(info[2]),
        Int(info[3]),
        transposed,
        conjugated,
    )
end
function _compact_q_copy(
    q::FaerTridiagonalQ{T};
    transposed = q.transposed,
    conjugated = q.conjugated,
    duplicate = false,
) where {T}
    lock(_matrix_lock) do
        _compact_q_validate(q)
        h=duplicate ? _native_call(RustCrate.OwnedCompactTridiagonal, q) : _handle(q)
        return _compact_q(T, h; transposed, conjugated)
    end
end
Base.copy(q::FaerTridiagonalQ) = _compact_q_copy(q; duplicate = true)
Base.deepcopy_internal(q::FaerTridiagonalQ, d::IdDict) = get!(() -> copy(q), d, q)
Base.transpose(q::FaerTridiagonalQ) = _compact_q_copy(q; transposed = !q.transposed)
Base.conj(q::FaerTridiagonalQ) = _compact_q_copy(q; conjugated = !q.conjugated)
Base.adjoint(q::FaerTridiagonalQ) =
    _compact_q_copy(q; transposed = !q.transposed, conjugated = !q.conjugated)
Base.inv(q::FaerTridiagonalQ) = adjoint(q)
function Base.getproperty(q::FaerTridiagonalQ{T}, name::Symbol) where {T}
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
Base.propertynames(q::FaerTridiagonalQ, private::Bool = false) =
    private ? (fieldnames(typeof(q))..., :basis, :factor, :phases, :nreflections) :
    (:basis, :factor, :phases, :nreflections, :blocksize, :transposed, :conjugated)
function Base.getindex(q::FaerTridiagonalQ{T}, i::Int, j::Int) where {T}
    checkbounds(q, i, j)
    lock(_matrix_lock) do
        _compact_q_validate(q)
        x=zeros(T, q.n)
        x[j]=one(T)
        return (q*x)[i]
    end
end
function _householder_coerce(q::FaerTridiagonalQ{T}, ::Type{T}) where {T}
    _compact_q_validate(q)
    return q
end
function _householder_coerce(q::FaerTridiagonalQ, ::Type{T}) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        _compact_q_validate(q)
        h=_native_call(RustCrate.OwnedCompactTridiagonal, q, _tag(T))
        return _compact_q(T, h; transposed = q.transposed, conjugated = q.conjugated)
    end
end
_householder_workspace_functions(::FaerTridiagonalQ) = (
    RustCrate.compact_tridiagonal_apply_requirements,
    RustCrate.compact_tridiagonal_workspace,
    RustCrate.compact_tridiagonal_workspace_team,
)
_householder_workspace_metadata(q::FaerTridiagonalQ) =
    (UInt(q.n), UInt(q.k), UInt(q.blocksize), _tag(eltype(q)))
_householder_orientation(q::FaerTridiagonalQ) = (q.transposed, q.conjugated)
function _householder_native_apply!(q::FaerTridiagonalQ, a, right)
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
        )
        return a
    finally
        close(work)
    end
end
Base.:\(q::FaerTridiagonalQ, a::_HouseholderArray) = q'*a

function _compact_tridiagonal_eigen(
    f;
    recursion_threshold::Integer = 128,
    sortby = nothing,
    scratch = nothing,
    team = nothing,
)
    _iterative_resource_check(scratch, team)
    _compact_q_validate(f.Q)
    req=scratch_requirements(LinearAlgebra.eigen, f; recursion_threshold, team)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    original=nothing
    vectors=nothing
    success=false
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "saved compact tridiagonal eigen scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        E=tridiagonal_eigen(
            f.H.dv,
            f.H.ev;
            recursion_threshold,
            sortby,
            scratch = work,
            team,
        )
        original=E.vectors
        vectors=_coerce(original, eltype(f.Q))
        LinearAlgebra.lmul!(f.Q, vectors; scratch = work, team)
        success=true
        return LinearAlgebra.Eigen(E.values, vectors)
    finally
        isnothing(original) || (success && original===vectors) || close(original)
        success || isnothing(vectors) || vectors===original || close(vectors)
        isnothing(scratch) && close(work)
    end
end

# Queries validate the original native owner, then select the target scalar's
# kernel layout directly. They never allocate a promoted copy of the reflectors.
function _compact_q_requirement(q::FaerTridiagonalQ, T, extent, right; team = nothing)
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
        pointer(info),
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
_householder_workspace_query(q::FaerTridiagonalQ, extent, right; team = nothing) =
    _compact_q_requirement(q, eltype(q), extent, right; team)
function _householder_resource_check(
    q::FaerTridiagonalQ,
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
