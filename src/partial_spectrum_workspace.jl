const _PartialSpectrumFunction=Union{
    typeof(partial_eigen),
    typeof(partial_self_adjoint_eigen),
    typeof(partial_svd),
}
_partial_kind(::typeof(partial_eigen)) = Int32(0)
_partial_kind(::typeof(partial_self_adjoint_eigen)) = Int32(1)
_partial_kind(::typeof(partial_svd)) = Int32(2)
function _partial_validate(
    op::FaerLinearOperator{T},
    k,
    kind;
    v0 = nothing,
    tol = sqrt(eps(_realtype(T))),
    min_dim = 0,
    max_dim = 0,
    max_restarts = 1000,
) where {T}
    _handle(op)
    m, n=size(op)
    kind!=2 && m!=n && throw(DimensionMismatch("eigen operator must be square"))
    # Check the current native owner before sizing or reading Julia start buffers.
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.partial_requirements,
        op,
        kind,
        UInt(0),
        UInt(m),
        UInt(n),
        _tag(T),
        UInt(0),
        UInt(0),
        UInt(0),
        pointer(info),
    )
    0<=k<=min(m, n) || throw(ArgumentError("requested count must be in 0:min(size(op)...)"))
    min_dim>=0 && max_dim>=0 && max_restarts>=0 ||
        throw(ArgumentError("subspace dimensions and restarts must be nonnegative"))
    isfinite(tol) && tol>=0 || throw(ArgumentError("tol must be finite and nonnegative"))
    tolerance=convert(_realtype(T), tol)
    isfinite(tolerance) || throw(ArgumentError("tol exceeds native precision range"))
    k=Int(k)
    limit=kind==2 ? min(m, n) : n
    effective_min=min(max(min_dim, 32, k), limit)
    effective_max=min(max(max_dim, 64, 2k), limit)
    effective_min<=effective_max ||
        throw(ArgumentError("min_dim exceeds the effective maximum projection dimension"))
    isnothing(v0)||_dense_rhs_validate(v0)
    start=_partial_start(T, v0)
    if !isnothing(start)
        length(start)==n ||
            throw(DimensionMismatch("v0 length must equal operator column count"))
        # A nonfinite entry makes the norm nonfinite; Rust normalizes the start.
        startnorm=n>0 ? LinearAlgebra.norm(start) : zero(_realtype(T))
        isfinite(startnorm) ||
            throw(ArgumentError("v0 must be finite with a finite norm"))
        n>0 && iszero(startnorm) && throw(ArgumentError("v0 must have finite nonzero norm"))
    end
    return m, n, k, tolerance, start
end
# Native vectors and views, faerview borrows and same-scalar strided Julia
# vectors are read in place by Rust; other start vectors are converted once.
_partial_start(::Type, ::Nothing) = nothing
function _partial_start(::Type{T}, v0) where {T}
    if eltype(v0)===T
        v0 isa Union{FaerVector,FaerBorrowedVector} && return v0
        v0 isa FaerView && ndims(v0)==1 && return v0
        v0 isa StridedVector && _julia_borrowable(v0) && return FaerBorrowedArray(v0)
    end
    return FaerBorrowedArray(Vector{T}(v0))
end
function _partial_workspace_query(op, kind, k; min_dim, max_dim, max_restarts, team)
    _iterative_resource_check(nothing, team)
    info=zeros(UInt, 2)
    args=(
        op,
        kind,
        UInt(k),
        UInt(size(op, 1)),
        UInt(size(op, 2)),
        _tag(eltype(op)),
        UInt(min_dim),
        UInt(max_dim),
        UInt(max_restarts),
        pointer(info),
    )
    GC.@preserve info begin
        isnothing(team) ? _native_call(RustCrate.partial_requirements, args...) :
        _native_call(RustCrate.partial_requirements_team, args..., team)
    end
    # The native requirement includes the independent residual applications.
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(partial_eigen|partial_self_adjoint_eigen|partial_svd, op, k; min_dim=0, max_dim=0, max_restarts=1000, team=nothing)

Query faer's native partial kernel, including its full-subspace operator
materialization and dense EVD/SVD branch, and the native independently
recomputed residual applications. Queries do not apply operators. Only the
returned Julia arrays (and the one-time conversion of a `v0` that cannot be read
in place) allocate in Julia. Use identical projection options and team for
execution.
"""
function scratch_requirements(
    fun::_PartialSpectrumFunction,
    op::FaerLinearOperator,
    k::Integer;
    v0 = nothing,
    tol = sqrt(eps(_realtype(eltype(op)))),
    min_dim::Integer = 0,
    max_dim::Integer = 0,
    max_restarts::Integer = 1000,
    team = nothing,
)
    lock(_matrix_lock) do
        kind=_partial_kind(fun)
        _partial_validate(op, k, kind; v0, tol, min_dim, max_dim, max_restarts)
        _partial_workspace_query(op, kind, Int(k); min_dim, max_dim, max_restarts, team)
    end
end
function FaerScratch(
    fun::_PartialSpectrumFunction,
    op::FaerLinearOperator,
    k::Integer;
    kwargs...,
)
    req=scratch_requirements(fun, op, k; kwargs...)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    scratch::FaerScratch,
    fun::_PartialSpectrumFunction,
    op::FaerLinearOperator,
    k::Integer;
    kwargs...,
)
    req=scratch_requirements(fun, op, k; kwargs...)
    reserve!(scratch, req.bytes; alignment = req.alignment)
end
function _partial_native(op, args, work, team)
    task=current_task()
    sticky=task.sticky
    task.sticky=true
    try
        isnothing(team) ? _native_call(RustCrate.partial_workspace, op, args..., work) :
        _native_call(RustCrate.partial_workspace_team, op, args..., work, team)
    finally
        task.sticky=sticky
    end
end
